"""
    [DDP VERSION] Train DGN for Guide Vane 2.5D Flow Field.
    
    Why DDP?
    - Solves 'AttributeError' by avoiding DataParallel's fragile scatter/gather.
    - Each GPU loads and collates its own mini-batch independently.
    - Uses standard dgn.loader.Collater without modification.
    
    Run with:
        python train_dgn_guidevane_ddp.py --gpus 0,1
"""

import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.data import Subset
from tqdm import tqdm
import time
import glob
import re

import dgn4cfd as dgn
# Use the library's original DataLoader/Collater - DDP makes this safe!
from dgn4cfd.loader import DataLoader 

try:
    from guide_vane_dataset import GuideVane25D
except ImportError:
    print("[WARN] guide_vane_dataset.py not found.")

# ============================================================================
# 1. SCALER (Unchanged)
# ============================================================================
class ScaleGuideVane:
    def __init__(self, stats_dict):
        self.v_min = stats_dict['v_min']
        self.v_max = stats_dict['v_max']
        self.v_center = (self.v_max + self.v_min) * 0.5
        self.v_scale  = (self.v_max - self.v_min) * 0.5
        self.p_mean = stats_dict['p_mean']
        self.p_std  = stats_dict['p_std']
        if self.p_std < 1e-6: self.p_std = 1.0

    def __call__(self, graph):
        # Scale Target
        if hasattr(graph, 'target'):
            target = graph.target.clone()
            target[:, 0:3] = (target[:, 0:3] - self.v_center) / self.v_scale
            target[:, 3]   = (target[:, 3]   - self.p_mean) / self.p_std
            target[:, 0:3] = torch.clamp(target[:, 0:3], -1.1, 1.1)
            target[:, 3]   = torch.clamp(target[:, 3], -10.0, 10.0)
            graph.target = target

        # Scale Condition
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 1:4] = torch.clamp(loc[:, 1:4], -1.1, 1.1)
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            loc[:, 4]   = torch.clamp(loc[:, 4], -10.0, 10.0)
            
            loc[:, -4:-1] = (loc[:, -4:-1] - self.v_center) / self.v_scale
            loc[:, -4:-1] = torch.clamp(loc[:, -4:-1], -1.1, 1.1)
            loc[:, -1]    = (loc[:, -1]   - self.p_mean) / self.p_std
            loc[:, -1]    = torch.clamp(loc[:, -1], -10.0, 10.0)
            
            if hasattr(graph, 'bound'):
                is_boundary = (graph.bound == 1) | (graph.bound == 2)
                loc[~is_boundary, -4:] = 0.0
            graph.loc = loc
            
            if hasattr(graph, 'boundary_values'):
                bv = graph.boundary_values.clone()
                bv[:, 0:3] = (bv[:, 0:3] - self.v_center) / self.v_scale
                bv[:, 3]   = (bv[:, 3]   - self.p_mean) / self.p_std
                if hasattr(graph, 'bound'):
                    bv[~is_boundary] = 0.0
                graph.boundary_values = bv

        return graph

# ============================================================================
# 2. DDP SETUP
# ============================================================================
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# ============================================================================
# 3. HELPER: NOISE INJECTION
# ============================================================================
def inject_noise_to_batch(batch, diffusion_process, step_sampler, device):
    """
    Injects noise into a whole collated batch.
    """
    # 1. Sample Time Steps
    # In DDP, each GPU has 'batch.num_graphs' graphs.
    # We sample t for these graphs.
    t_batch, sample_weights = step_sampler.sample(batch.num_graphs, device)
    
    # 2. Add Noise using the model's diffusion process
    # DGN's diffusion_process expects (field_start, r, batch_vector)
    x_start = batch.target
    x_t, noise = diffusion_process(
        field_start=x_start, 
        r=t_batch, 
        batch=batch.batch
    )
    
    # 3. Inject into Batch object
    batch.r = t_batch          
    batch.field_r = x_t        
    batch.noise = noise        
    batch.field_start = x_start
    
    return batch, sample_weights

# ============================================================================
# 4. TRAINING LOOP (Per Process)
# ============================================================================
def train_worker(rank, world_size, gpu_ids, config, args):
    setup(rank, world_size)
    
    # Map logical rank to physical GPU ID
    physical_gpu_id = gpu_ids[rank]
    device = torch.device(f'cuda:{physical_gpu_id}')
    torch.cuda.set_device(device)
    
    is_master = (rank == 0)
    if is_master:
        print(f"[DDP] Starting training on {world_size} GPUs.")

    # --- Load Data (Each process loads dataset) ---
    # Note: GuideVane25D uses mmap or shared memory logic, so this is efficient.
    if is_master: print("1. Loading Dataset...")
    
    # Load raw data for stats (or load stats file)
    raw_data_np = np.load(config['dataset_path'], mmap_mode='r')
    num_cases, num_steps = raw_data_np.shape[0], raw_data_np.shape[1]
    
    val_case_idx = config['val_case_idx'] if config['val_case_idx'] >= 0 else num_cases + config['val_case_idx']
    train_case_indices = np.delete(np.arange(num_cases), val_case_idx)
    
    # Compute stats (only needed if not saved, but for safety let's assume passed in config or pre-calced)
    # Here we recalculate or load. For speed in DDP, usually better to load pre-calced.
    # We will do a quick calc on a subset or just load if exists.
    stats_path = './checkpoints_dgn/scaler_stats.pt'
    if is_master and not os.path.exists(stats_path):
        os.makedirs('./checkpoints_dgn', exist_ok=True)
        # Load fully to calc stats
        temp_data = np.load(config['dataset_path'])
        train_subset = temp_data[train_case_indices]
        u, v, w = train_subset[..., 3], train_subset[..., 4], train_subset[..., 5]
        p = train_subset[..., 2]
        scaler_stats = {
            'v_min': float(min(u.min(), v.min(), w.min())),
            'v_max': float(max(u.max(), v.max(), w.max())),
            'p_mean': float(p.mean()),
            'p_std':  float(p.std())
        }
        torch.save(scaler_stats, stats_path)
        del temp_data, train_subset
    
    # Sync processes to ensure stats file is ready
    dist.barrier()
    scaler_stats = torch.load(stats_path, map_location='cpu')
    
    # Data Tensor
    data_tensor = torch.from_numpy(np.load(config['dataset_path'])).float()

    # Transforms
    transform = transforms.Compose([
        ScaleGuideVane(scaler_stats),
        dgn.transforms.MeshCoarsening(num_scales=4, rel_pos_scaling=[0.1, 0.2, 0.4,0.8], scalar_rel_pos=True),
    ])

    dataset = GuideVane25D(
        dataset_path = config['dataset_path'],
        mesh_path    = config['mesh_path'],
        data_tensor  = data_tensor,
        transform    = transform,
        anchor_rate  = 16,
    )

    def get_indices(cases):
        idx = []
        for c in cases: idx.extend(range(c*num_steps, (c+1)*num_steps))
        return idx
        
    train_set = Subset(dataset, get_indices(train_case_indices))
    val_set   = Subset(dataset, get_indices([val_case_idx]))
    
    # --- Distributed Samplers ---
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    # Val sampler is optional, but good for consistent eval across devices
    val_sampler   = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    
    # --- DataLoaders (Use dgn.DataLoader or standard) ---
    # IMPORTANT: We use dgn.DataLoader because it uses dgn.loader.Collater internally
    # which handles the multi-scale graph collation (edge_index_2 shifts).
    # Since DDP creates local batches, this Collater works perfectly fine!
    train_loader = DataLoader(
        train_set, 
        batch_size=config['batch_size'], 
        sampler=train_sampler, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=config['batch_size'], 
        sampler=val_sampler, 
        num_workers=4, 
        pin_memory=True
    )

    # --- Model ---
    arch = {
        'in_node_features': 4, 'cond_node_features': 13, 'cond_edge_features': 2,
        'depths': config['depths'], 'fnns_width': config['width'],
        'aggr': 'sum', 'dropout': 0.1, 'dim': 3, 'scalar_rel_pos': True
    }
    
    diffusion_process = dgn.nn.diffusion.DiffusionProcess(num_steps=1000, schedule_type='linear')
    
    model = dgn.nn.DiffusionGraphNet(
        diffusion_process=diffusion_process,
        learnable_variance=True,
        arch=arch
    )
    
    # Sync Batch Norm if present (DGN uses LayerNorm/InstanceNorm usually, but good practice)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    
    # Wrap in DDP
    # find_unused_parameters=True needed if some graph branches aren't visited, usually False for DGN
    ddp_model = DDP(model, device_ids=[physical_gpu_id], output_device=physical_gpu_id)

    # --- Optimizer & Loss ---
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )
    criterion = dgn.nn.losses.HybridLoss()
    step_sampler = dgn.nn.diffusion.ImportanceStepSampler(num_diffusion_steps=diffusion_process.num_steps)
    
    scaler = GradScaler(enabled=True)
    if is_master:
        writer = SummaryWriter(config['tensor_board'])

    # --- Resume Logic (Only load on master, then broadcast? No, load on all) ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    # We rely on map_location to load to correct GPU
    checkpoints = glob.glob(os.path.join(config['folder'], f"{config['name']}_epoch_*.pt"))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda p: int(re.search(r'epoch_(\d+).pt', p).group(1)))
        if is_master: print(f"[Resume] Found checkpoint: {latest_ckpt}")
        
        map_location = {'cuda:0': f'cuda:{physical_gpu_id}'}
        ckpt = torch.load(latest_ckpt, map_location=device)
        
        # Load weights (handle 'module.' prefix from previous DDP/DP saves)
        # DDP wraps model in 'module.', so direct load is usually fine if saved from DDP
        # ddp_model.load_state_dict(ckpt['model_state_dict'])
        ddp_model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'loss' in ckpt: best_val_loss = ckpt['loss']
        if 'scaler_state_dict' in ckpt: scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        if is_master: print(f"[Resume] Resumed from Epoch {start_epoch}")

    # --- Training Loop ---
    if is_master: print("="*80)
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        
        ddp_model.train()
        train_loss_sum = torch.zeros(1, device=device)
        num_batches = torch.zeros(1, device=device)
        
        if is_master:
            pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [Tr]", leave=False)
        else:
            pbar = train_loader
            
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 1. Inject Noise (Locally on GPU)
            # DGN's diffusion process is stateless, so we can use the instance from `model`
            # Note: access `model.diffusion_process` via `ddp_model.module`
            batch, sample_weights = inject_noise_to_batch(
                batch, 
                ddp_model.module.diffusion_process, 
                step_sampler, 
                device
            )
            
            # 2. Forward
            with autocast(enabled=True):
                # DDP forward automatically handles gradient sync
                # We assume criterion handles the model call: criterion(model, graph)
                # But criterion expects a plain model or DDP model. 
                # dgn.nn.losses.HybridLoss calls model(graph). DDP model supports this.
                
                # However, HybridLoss needs to access `model.diffusion_process` etc.
                # If we pass `ddp_model` to `criterion`, `criterion` might try to access attributes.
                # Standard DDP wraps attributes in `.module`.
                # Let's check HybridLoss source: it calls `model.learnable_variance`.
                # DDP wrapper does NOT automatically proxy attributes.
                
                # FIX: We pass `ddp_model` for the forward pass call inside Loss, 
                # BUT we need a way for Loss to access metadata.
                # Alternatively, we calculate output manually and compute loss terms manually.
                # Or, we Monkey-Patch the DDP model to proxy attributes? No.
                
                # Better approach for DGN + DDP:
                # Manually run forward and loss logic here, or ensure criterion handles unwrapping.
                # The provided dgn.losses.HybridLoss checks `model.learnable_variance`.
                # DDP model won't have this.
                
                # WORKAROUND: Pass the underlying module to criterion for attribute access,
                # but pass the DDP model for the forward call?
                # HybridLoss code: `model_noise, model_v = model(graph)`
                
                # We will subclass HybridLoss locally to handle DDP unwrapping.
                pass

            # [Local Fix for HybridLoss with DDP]
            # We execute the logic of HybridLoss inline or via a compatible wrapper
            with autocast(enabled=True):
                # 2a. Forward
                model_noise, model_v = ddp_model(batch)
                
                # 2b. MSE
                se = (model_noise - batch.noise)**2
                mse_term = dgn.nn.losses.batch_wise_mean(se, batch.batch)
                
                # 2c. VLB (Disable gradient for VLB calc to save memory/compute if needed, usually needed for v)
                # Access underlying model for utils
                raw_model = ddp_model.module
                true_mean, true_var = raw_model.diffusion_process.get_posterior_mean_and_variance(
                    batch.field_start, batch.field_r, batch.batch, batch.r
                )
                
                frozen_out = (model_noise.detach(), model_v)
                pred_mean, pred_var = raw_model.get_posterior_mean_and_variance_from_output(frozen_out, batch)
                
                vlb_term = criterion.lambda_vlb * criterion.vlb_loss(
                    batch, true_mean, true_var, pred_mean, pred_var
                )
                
                loss_per_graph = mse_term + vlb_term
                
                # Importance Sampling Weighting
                loss = (loss_per_graph * sample_weights).mean()

            # 3. Backward
            scaler.scale(loss).backward()
            
            if config['grad_clip']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config['grad_clip']['limit'])
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update Sampler (Gather losses from all ranks for better sampler update? Optional.)
            # For simplicity, update local sampler with local losses.
            # Ideally, one should gather all losses to update a global sampler, but local updates work okay.
            step_sampler.update(batch.r, loss_per_graph.detach())
            
            train_loss_sum += loss.item()
            num_batches += 1
            
            if is_master:
                pbar.set_postfix({'L': f"{loss.item():.5f}"})

        # --- Aggregate Train Stats ---
        dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        avg_train_loss = (train_loss_sum / num_batches).item()

        # --- Validation ---
        # Run on all ranks to cover full val set
        ddp_model.eval()
        val_loss_sum = torch.zeros(1, device=device)
        val_batches = torch.zeros(1, device=device)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch, _ = inject_noise_to_batch(batch, ddp_model.module.diffusion_process, step_sampler, device)
                
                with autocast(enabled=True):
                    # Same logic as train
                    model_noise, model_v = ddp_model(batch)
                    se = (model_noise - batch.noise)**2
                    mse_term = dgn.nn.losses.batch_wise_mean(se, batch.batch)
                    
                    raw_model = ddp_model.module
                    true_mean, true_var = raw_model.diffusion_process.get_posterior_mean_and_variance(
                        batch.field_start, batch.field_r, batch.batch, batch.r
                    )
                    frozen_out = (model_noise, model_v) # No detach needed in eval
                    pred_mean, pred_var = raw_model.get_posterior_mean_and_variance_from_output(frozen_out, batch)
                    vlb_term = criterion.lambda_vlb * criterion.vlb_loss(
                        batch, true_mean, true_var, pred_mean, pred_var
                    )
                    val_l = (mse_term + vlb_term).mean()
                
                val_loss_sum += val_l.item()
                val_batches += 1
        
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches, op=dist.ReduceOp.SUM)
        avg_val_loss = (val_loss_sum / val_batches).item()

        # --- Logging & Saving (Master Only) ---
        if is_master:
            epoch_time = time.time() - epoch_start
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d} | Time: {epoch_time:.1f}s | Tr: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {curr_lr:.2e}")
            
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
            
            # Scheduler step (based on val loss)
            # Note: scheduler is on all ranks, so all need the val_loss.
            # But ReduceLROnPlateau doesn't sync internal state automatically.
            # We should step on all ranks using the synced avg_val_loss.
        
        # Step scheduler on all ranks
        scheduler.step(avg_val_loss)

        if is_master:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(), # Save underlying model
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_val_loss
            }
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(save_dict, f"{config['folder']}/{config['name']}_best.pt")
                
            if (epoch + 1) % config['chk_interval'] == 0:
                torch.save(save_dict, f"{config['folder']}/{config['name']}_epoch_{epoch+1}.pt")
    
    if is_master:
        writer.close()
        print("Training Finished.")
    
    cleanup()

# ============================================================================
# 5. MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpus', type=str, default='0', help="Comma separated GPU indices, e.g. '0,1'")
    argparser.add_argument('--name', type=str, default='DGN_GuideVane_Physical_DDP')
    args = argparser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(',')]
    world_size = len(gpu_ids)
    
    print(f"Starting DDP training on GPUs: {gpu_ids}")

    # Configuration
    config = {
        'dataset_path': 'sta_dataset_2.5D.npy',
        'mesh_path':    'sta_datamesh.npy',
        'folder':       './checkpoints_dgn_ddp',
        'tensor_board': './boards_dgn_ddp',
        'name':         args.name,
        'depths':       [2, 2, 2, 2],
        'width':        128,
        'batch_size':   4, # Batch size PER GPU
        'epochs':       5000,
        'lr':           1e-4,
        'val_case_idx': -1,
        'chk_interval': 10,
        'grad_clip':    {"limit": 1.0}
    }

    # Spawn processes
    mp.spawn(
        train_worker,
        args=(world_size, gpu_ids, config, args),
        nprocs=world_size,
        join=True
    )