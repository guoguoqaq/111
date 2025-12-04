"""
    [RESUME SUPPORTED] Train DGN for Guide Vane 2.5D Flow Field (Physical Space).
    
    Features:
    - Resume Training: Auto-load latest checkpoint.
    - Device: Defaults to GPU 1 (Second Graphics Card).
    - Custom Loop: Full control over diffusion steps and validation.
    
    Run with:
        python train_dgn_guidevane.py --gpu 1
"""

import torch
import argparse
import numpy as np
import os
import glob
import re
import time
from torchvision import transforms
from torch.utils.data import Subset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dgn4cfd as dgn
try:
    from guide_vane_dataset import GuideVane25D
except ImportError:
    raise ImportError("Please ensure 'guide_vane_dataset.py' is in the current directory.")

torch.multiprocessing.set_sharing_strategy('file_system')

# ============================================================================
# 1. SCALER (Copy from previous working version)
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
        # 1. Scale Target [vr, vt, vz, p]
        if hasattr(graph, 'target'):
            graph.target[:, 0:3] = (graph.target[:, 0:3] - self.v_center) / self.v_scale
            graph.target[:, 3]   = (graph.target[:, 3]   - self.p_mean) / self.p_std
            graph.target[:, 0:3] = torch.clamp(graph.target[:, 0:3], -1.1, 1.1)
            graph.target[:, 3]   = torch.clamp(graph.target[:, 3], -10.0, 10.0)

        # 2. Scale Condition (Loc)
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            # Interp fields (1-4)
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 1:4] = torch.clamp(loc[:, 1:4], -1.1, 1.1)
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            loc[:, 4]   = torch.clamp(loc[:, 4], -10.0, 10.0)
            
            # Boundary Values (Last 4)
            loc[:, -4:-1] = (loc[:, -4:-1] - self.v_center) / self.v_scale
            loc[:, -4:-1] = torch.clamp(loc[:, -4:-1], -1.1, 1.1)
            loc[:, -1]    = (loc[:, -1]   - self.p_mean) / self.p_std
            loc[:, -1]    = torch.clamp(loc[:, -1], -10.0, 10.0)
            
            # Clean non-boundary areas (Optional but recommended)
            if hasattr(graph, 'bound'):
                is_boundary = (graph.bound == 1) | (graph.bound == 2)
                loc[~is_boundary, -4:] = 0.0
                
            graph.loc = loc
            
            # Boundary Values Attribute
            if hasattr(graph, 'boundary_values'):
                bv = graph.boundary_values.clone()
                bv[:, 0:3] = (bv[:, 0:3] - self.v_center) / self.v_scale
                bv[:, 3]   = (bv[:, 3]   - self.p_mean) / self.p_std
                if hasattr(graph, 'bound'):
                    bv[~is_boundary] = 0.0
                graph.boundary_values = bv

        return graph

# ============================================================================
# 2. CUSTOM TRAIN LOOP (With Resume Functionality)
# ============================================================================
def custom_dgn_train_loop(model, train_settings, train_loader, val_loader=None):
    print(f"\n[INFO] Starting DGN Training on device: {train_settings['device']}...")
    
    device = train_settings['device']
    model = model.to(device)
    
    # Init Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['lr'])
    criterion = train_settings['training_loss'] # HybridLoss
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )
    
    use_amp = True # Mixed Precision
    scaler = GradScaler(enabled=use_amp)
    
    writer = SummaryWriter(train_settings['tensor_board'])
    
    # Diffusion Step Sampler
    step_sampler = train_settings['step_sampler'](num_diffusion_steps=model.diffusion_process.num_steps)

    # --- RESUME LOGIC ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    os.makedirs(train_settings['folder'], exist_ok=True)
    pattern = os.path.join(train_settings['folder'], f"{train_settings['name']}_epoch_*.pt")
    checkpoints = glob.glob(pattern)
    
    if checkpoints:
        # Find latest checkpoint by epoch number
        latest_ckpt = max(checkpoints, key=lambda p: int(re.search(r'epoch_(\d+).pt', p).group(1)))
        print(f"[Resume] Found checkpoint: {latest_ckpt}")
        
        try:
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
            start_epoch = ckpt['epoch'] + 1
            if 'loss' in ckpt: best_val_loss = ckpt['loss']
            if 'scaler_state_dict' in ckpt: scaler.load_state_dict(ckpt['scaler_state_dict'])
            if 'scheduler_state_dict' in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            
            print(f"[Resume] Successfully resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        print("[Init] No checkpoint found. Starting from scratch.")

    # --- MAIN LOOP ---
    print("="*80)
    
    for epoch in range(start_epoch, train_settings['epochs']):
        epoch_start = time.time()
        
        # === TRAIN ===
        model.train()
        train_loss_sum = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [Tr]", leave=False)
        for graph in pbar:
            graph = graph.to(device)
            optimizer.zero_grad()
            
            # 1. Prepare Data for Diffusion
            # DGN operates on physical data, so x_start is the clean target field
            x_start = graph.target 
            
            # 2. Sample Time Steps
            t_batch, _ = step_sampler.sample(graph.num_graphs, device)
            
            # 3. Add Noise (Forward Diffusion)
            # Returns: x_t (noisy field), noise (true noise)
            x_t, noise = model.diffusion_process(field_start=x_start, r=t_batch, batch=graph.batch)
            
            # 4. Inject into Graph
            graph.r = t_batch          # Time condition
            graph.field_r = x_t        # Noisy input
            graph.noise = noise        # Target for loss (if predicting noise)
            graph.field_start = x_start # Target for loss (if predicting start)
            
            # 5. Forward & Loss
            with autocast(enabled=use_amp):
                # model(graph) is called inside criterion usually, or we pass model to criterion
                loss = criterion(model, graph).mean()
            
            # 6. Backward
            scaler.scale(loss).backward()
            
            if train_settings['grad_clip']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_settings['grad_clip']['limit'])
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            num_batches += 1
            pbar.set_postfix({'L': f"{loss.item():.5f}"})
            
        avg_train_loss = train_loss_sum / max(1, num_batches)

        # === VALIDATION ===
        avg_val_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for graph in val_loader:
                    graph = graph.to(device)
                    x_start = graph.target
                    t_batch, _ = step_sampler.sample(graph.num_graphs, device)
                    x_t, noise = model.diffusion_process(x_start, t_batch, graph.batch)
                    
                    graph.r, graph.field_r = t_batch, x_t
                    graph.noise, graph.field_start = noise, x_start
                    
                    with autocast(enabled=use_amp):
                        val_l = criterion(model, graph).mean()
                    val_loss_sum += val_l.item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / max(1, val_batches)

        # === LOGGING & SAVING ===
        epoch_time = time.time() - epoch_start
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:4d} | Time: {epoch_time:.1f}s | Tr: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {curr_lr:.2e}")
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        
        scheduler.step(avg_val_loss)
        
        # Save Dictionary
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss
        }
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(save_dict, f"{train_settings['folder']}/{train_settings['name']}_best.pt")
            
        # Save Regular Checkpoint
        if (epoch + 1) % train_settings['chk_interval'] == 0:
            torch.save(save_dict, f"{train_settings['folder']}/{train_settings['name']}_epoch_{epoch+1}.pt")
            
    writer.close()
    print("Training Finished.")

# ============================================================================
# 3. MAIN
# ============================================================================
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # 默认设备改为 1 (第二张显卡)
    argparser.add_argument('--gpu', type=int, default=1, help="GPU index (default: 1 for 2nd card)")
    argparser.add_argument('--name', type=str, default='DGN_GuideVane_Physical')
    args = argparser.parse_args()

    # Seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = {
        'dataset_path': 'sta_dataset_2.5D.npy',
        'mesh_path':    'sta_datamesh.npy',
        'depths':       [2, 2, 2, 2],
        'width':        128,
        'batch_size':   4,
        'epochs':       5000,
        'lr':           1e-4,
        'val_case_idx': -1,
    }

    # Device
    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu')
    print(f"Using device: {device}")

    # --- Load Data & Stats ---
    print("1. Computing Stats...")
    raw_data_np = np.load(config['dataset_path'])
    num_cases, num_steps = raw_data_np.shape[0], raw_data_np.shape[1]
    
    val_case_idx = config['val_case_idx'] if config['val_case_idx'] >= 0 else num_cases + config['val_case_idx']
    train_case_indices = np.delete(np.arange(num_cases), val_case_idx)
    
    # Calc Stats on Train Set
    train_subset = raw_data_np[train_case_indices]
    u, v, w = train_subset[..., 3], train_subset[..., 4], train_subset[..., 5]
    p = train_subset[..., 2]
    scaler_stats = {
        'v_min': float(min(u.min(), v.min(), w.min())),
        'v_max': float(max(u.max(), v.max(), w.max())),
        'p_mean': float(p.mean()),
        'p_std':  float(p.std())
    }
    print(f"   Stats: {scaler_stats}")
    
    os.makedirs('./checkpoints_dgn', exist_ok=True)
    torch.save(scaler_stats, './checkpoints_dgn/scaler_stats.pt')
    
    data_tensor = torch.from_numpy(raw_data_np).float()
    del raw_data_np, train_subset

    # --- Dataset & Transform ---
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

    # Split
    def get_indices(cases):
        idx = []
        for c in cases: idx.extend(range(c*num_steps, (c+1)*num_steps))
        return idx
        
    train_set = Subset(dataset, get_indices(train_case_indices))
    val_set   = Subset(dataset, get_indices([val_case_idx]))
    
    train_loader = dgn.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader   = dgn.DataLoader(val_set,   batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # --- Model ---
    # In: 4 (vr,vt,vz,p), Cond: 13 (loc), Edge: 2
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

    # --- Settings & Run ---
    train_settings = dgn.nn.TrainingSettings(
        name          = args.name,
        folder        = './checkpoints_dgn',
        tensor_board  = './boards_dgn',
        chk_interval  = 10,
        training_loss = dgn.nn.losses.HybridLoss(),
        epochs        = config['epochs'],
        batch_size    = config['batch_size'],
        lr            = config['lr'],
        grad_clip     = {"epoch": 0, "limit": 1.0},
        step_sampler  = dgn.nn.diffusion.ImportanceStepSampler,
        device        = device,
    )

    # Call Custom Loop
    custom_dgn_train_loop(model, train_settings, train_loader, val_loader)