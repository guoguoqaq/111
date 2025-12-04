"""
    [FINAL OPTIMIZED VERSION]
    Train LDGN for Guide Vane 2.5D Flow Field.
    
    Features:
    - Case-wise Split & Validation
    - Resume Training Support
    - **In-Memory Latent Pre-computation**: Solves OOM issues, restores high Batch Size.
    - Auto-fix AE Checkpoint compatibility.
"""

import torch
import argparse
import os
import numpy as np
import time
import glob
import re
import shutil
from torch.utils.data import Dataset, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import dgn4cfd as dgn
try:
    from guide_vane_dataset import GuideVane25D
except ImportError:
    raise ImportError("Please ensure 'guide_vane_dataset.py' is in the current directory.")

torch.multiprocessing.set_sharing_strategy('file_system')

# ============================================================================
# 1. UTILITIES & FIXERS
# ============================================================================

def check_and_fix_checkpoint(ckpt_path, arch_config):
    """自动检测并修复 AE Checkpoint 的键名和架构配置"""
    print(f"[Check] Inspecting checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # 尝试加载
    try:
        # weights_only=False 是必须的，因为我们要加载 arch 字典
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    modified = False
    
    # 1. 修复架构参数
    if 'arch' not in checkpoint:
        print("   -> Injecting missing 'arch' config...")
        checkpoint['arch'] = arch_config
        modified = True
    
    # 2. 修复权重键名 (dgn4cfd 需要 'weights')
    if 'weights' not in checkpoint and 'model_state_dict' in checkpoint:
        print("   -> Mapping 'model_state_dict' to 'weights'...")
        checkpoint['weights'] = checkpoint['model_state_dict']
        modified = True

    # 3. 修复优化器键名
    if 'optimiser' not in checkpoint and 'optimizer_state_dict' in checkpoint:
        checkpoint['optimiser'] = checkpoint['optimizer_state_dict']
        modified = True

    if modified:
        # 备份原文件
        backup_path = ckpt_path + '.bak'
        if not os.path.exists(backup_path):
            shutil.copy(ckpt_path, backup_path)
            print(f"   -> Backup created at {backup_path}")
        
        torch.save(checkpoint, ckpt_path)
        print("[Check] Checkpoint fixed and saved.")
    else:
        print("[Check] Checkpoint is valid.")

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
        # 1. Scale Target (保持不变)
        if hasattr(graph, 'target'):
            graph.target[:, 0:3] = (graph.target[:, 0:3] - self.v_center) / self.v_scale
            graph.target[:, 3]   = (graph.target[:, 3]   - self.p_mean) / self.p_std
            graph.target[:, 0:3] = torch.clamp(graph.target[:, 0:3], -1.1, 1.1)
            graph.target[:, 3]   = torch.clamp(graph.target[:, 3], -10.0, 10.0)

        # 2. Scale Condition (Loc) - 修正 Masking 逻辑
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            
            # --- 常规流场特征 ---
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 1:4] = torch.clamp(loc[:, 1:4], -1.1, 1.1)
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            loc[:, 4]   = torch.clamp(loc[:, 4], -10.0, 10.0)
            
            # --- [关键修正] 边界值特征 ---
            # 先归一化
            loc[:, -4:-1] = (loc[:, -4:-1] - self.v_center) / self.v_scale
            loc[:, -1]    = (loc[:, -1]   - self.p_mean) / self.p_std
            
            loc[:, -4:-1] = torch.clamp(loc[:, -4:-1], -1.1, 1.1)
            loc[:, -1]    = torch.clamp(loc[:, -1], -10.0, 10.0)

            
            if hasattr(graph, 'bound'):
                # 1=Inlet, 2=Outlet (根据您的定义调整)
                is_boundary = (graph.bound == 1) | (graph.bound == 2)
                # 将非边界区域的最后4个通道（边界条件通道）置为0
                loc[~is_boundary, -4:] = 0.0
            
            graph.loc = loc
            
            # 处理 boundary_values 
            if hasattr(graph, 'boundary_values'):
                bv = graph.boundary_values.clone()
                bv[:, 0:3] = (bv[:, 0:3] - self.v_center) / self.v_scale
                bv[:, 3]   = (bv[:, 3]   - self.p_mean) / self.p_std
                
                if hasattr(graph, 'bound'):
                    mask = (graph.bound == 1) | (graph.bound == 2)
                    bv[~mask] = 0.0
                
                graph.boundary_values = bv

        return graph

# ============================================================================
# 2. LATENT DATASET & PRE-COMPUTATION (核心优化)
# ============================================================================

class LatentGraphDataset(Dataset):
    """简单的内存数据集，直接存储 Latent Graph 列表"""
    def __init__(self, graph_list):
        self.graphs = graph_list
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]

def precompute_latents(raw_dataset, ae_model, device, batch_size=16):
    """
    [Fixed] Pre-compute Latent Representations.
    Added 'num_nodes' to suppress PyG warnings.
    """
    print(f"\n[Pre-compute] Encoding {len(raw_dataset)} samples into Latent Space...")
    print(f"             Batch Size: {batch_size} (Inference only)")
    
    loader = dgn.DataLoader(raw_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    processed_graphs = []
    
    ae_model.eval()
    
    with torch.no_grad():
        for graph in tqdm(loader, desc="Encoding"):
            graph = graph.to(device)
            num_graphs_in_batch = graph.num_graphs
            
            if hasattr(graph, 'target'):
                # 1. AE Encode
                outputs = ae_model.encode(graph, graph.target)
                
                # 2. 提取 Latent 特征 (移回 CPU)
                x_latent = outputs[0].cpu()      
                c_latent = outputs[3][-1].cpu()  
                e_latent = outputs[4][-1].cpu()  
                edge_index = outputs[5][-1].cpu()
                batch_vec = outputs[6][-1].cpu() 
                
                
                #检查是否数据泄露
                correlation = torch.corrcoef(torch.stack([
                    x_latent.flatten()[:1000],
                    c_latent.flatten()[:1000]
                ]))[0, 1]
                print('检查数据是否泄露')
                if abs(correlation) > 0.7:
                    print(f"⚠️  WARNING: High correlation {correlation:.3f} between x_latent and c_latent!")
                    print("   -> Condition encoder may be leaking target information")
                
                
                
                # 3. 手动拆分 Batch
                node_counts = torch.bincount(batch_vec)
                is_static_mesh = (node_counts == node_counts[0]).all()
                
                if is_static_mesh:
                    # === 快速路径 (Static Mesh) ===
                    num_nodes = node_counts[0].item()
                    
                    edge_batch = batch_vec[edge_index[0]]
                    edge_counts = torch.bincount(edge_batch)
                    num_edges = edge_counts[0].item()
                    
                    x_splits = x_latent.split(num_nodes)
                    c_splits = c_latent.split(num_nodes)
                    e_splits = e_latent.split(num_edges)
                    
                    mask_0 = (edge_batch == 0)
                    edge_index_template = edge_index[:, mask_0].clone()
                    
                    for i in range(num_graphs_in_batch):
                        g = dgn.Graph(
                            x_latent=x_splits[i],
                            c_latent=c_splits[i],
                            e_latent=e_splits[i],
                            edge_index=edge_index_template.clone(),
                            num_nodes=num_nodes  # [新增] 显式指定节点数
                        )
                        processed_graphs.append(g)
                        
                else:
                    # === 通用路径 (Dynamic Mesh) ===
                    for i in range(num_graphs_in_batch):
                        node_mask = (batch_vec == i)
                        x_i = x_latent[node_mask]
                        c_i = c_latent[node_mask]
                        
                        edge_mask = node_mask[edge_index[0]]
                        e_i = e_latent[edge_mask]
                        edge_index_i = edge_index[:, edge_mask]
                        
                        node_indices = torch.where(node_mask)[0]
                        offset = node_indices[0]
                        edge_index_local = edge_index_i - offset
                        
                        g = dgn.Graph(
                            x_latent=x_i,
                            c_latent=c_i,
                            e_latent=e_i,
                            edge_index=edge_index_local,
                            num_nodes=x_i.size(0)  # [新增] 显式指定节点数
                        )
                        processed_graphs.append(g)

    print(f"[Pre-compute] Done. {len(processed_graphs)} latent graphs cached in RAM.")
    return LatentGraphDataset(processed_graphs)

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def custom_ldgn_train_loop(model, train_settings, train_loader, val_loader):
    print("\n[INFO] Starting LDGN Training (High-Performance Mode)...")
    
    device = train_settings['device']
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['lr'])
    criterion = train_settings['training_loss']
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )
    
    use_amp = train_settings['mixed_precision']
    scaler = GradScaler(enabled=use_amp)
    
    writer = SummaryWriter(train_settings['tensor_board']) if train_settings['tensor_board'] else None
    step_sampler = train_settings['step_sampler'](num_diffusion_steps=model.diffusion_process.num_steps)
    
    # --- 断点续训逻辑 ---
    start_epoch = 0
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    os.makedirs(train_settings['folder'], exist_ok=True)
    pattern = os.path.join(train_settings['folder'], f"{train_settings['name']}_epoch_*.pt")
    checkpoints = glob.glob(pattern)
    
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda p: int(re.search(r'epoch_(\d+).pt', p).group(1)))
        print(f"[Resume] Loading checkpoint: {latest_ckpt}")
        try:
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            if 'loss' in ckpt: best_val_loss = ckpt['loss']
            if 'scaler_state_dict' in ckpt and use_amp: scaler.load_state_dict(ckpt['scaler_state_dict'])
            if 'scheduler_state_dict' in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"[Resume] Resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"[Resume] Error loading checkpoint: {e}. Starting fresh.")
    
    # --- 训练循环 ---
    for epoch in range(start_epoch, train_settings['epochs']):
        epoch_start = time.time()
        train_loss = 0.0
        num_batches = 0
        
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 取一个验证batch
                val_graph = next(iter(val_loader)).to(device)
                
                # 1. 检查预测的噪声幅度
                x_t, noise_gt = model.diffusion_process(
                    val_graph.x_latent, 
                    torch.tensor([500]*val_graph.num_graphs).to(device), 
                    val_graph.batch
                )
                val_graph.r = torch.tensor([500]*val_graph.num_graphs).to(device)
                val_graph.field_r = x_t
                
                noise_pred = model(val_graph)
                if isinstance(noise_pred, tuple):
                    noise_pred = noise_pred[0]
                
                print(f"\n[Epoch {epoch+1}] Noise Diagnosis:")
                print(f"  GT Noise   -> Mean: {noise_gt.mean():.4f}, Std: {noise_gt.std():.4f}")
                print(f"  Pred Noise -> Mean: {noise_pred.mean():.4f}, Std: {noise_pred.std():.4f}")
                print(f"  Latent     -> Mean: {val_graph.x_latent.mean():.4f}, Std: {val_graph.x_latent.std():.4f}")
        
        
        # Train
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [Tr]", leave=False)
        for graph in pbar:
            graph = graph.to(device)
            optimizer.zero_grad()
            
            # 此时 graph 已经是 Latent 数据，无需再次 Encode
            x_start = graph.x_latent 
            t_batch, _ = step_sampler.sample(graph.num_graphs, device)
            
            x_t, noise = model.diffusion_process(field_start=x_start, r=t_batch, batch=graph.batch)
            
            # 注入数据
            graph.r = t_batch           
            graph.field_r = x_t         
            graph.noise = noise         
            graph.field_start = x_start 
            
            with autocast(enabled=use_amp):
                loss = criterion(model, graph).mean()
            
            scaler.scale(loss).backward()
            
            if train_settings['grad_clip']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_settings['grad_clip']['limit'])
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'L': f"{loss.item():.5f}"})
        
        avg_train_loss = train_loss / max(1, num_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for graph in val_loader:
                graph = graph.to(device)
                t_batch, _ = step_sampler.sample(graph.num_graphs, device)
                x_t, noise = model.diffusion_process(graph.x_latent, t_batch, graph.batch)
                graph.r, graph.field_r, graph.noise, graph.field_start = t_batch, x_t, noise, graph.x_latent
                
                with autocast(enabled=use_amp):
                    val_loss += criterion(model, graph).mean().item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(1, val_batches)
        
        # Log & Save
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:4d} | Time: {epoch_time:.1f}s | Tr: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {lr:.2e}")
        
        if writer:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        
        scheduler.step(avg_val_loss)
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(save_dict, f"{train_settings['folder']}/{train_settings['name']}_best.pt")
        else:
            patience_counter += 1
            
        if (epoch + 1) % train_settings['chk_interval'] == 0:
            torch.save(save_dict, f"{train_settings['folder']}/{train_settings['name']}_epoch_{epoch+1}.pt")
            
        if patience_counter >= patience:
            print("Early stopping.")
            break

    if writer: writer.close()

# ============================================================================
# 4. MAIN
# ============================================================================

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=1)
    argparser.add_argument('--name', type=str, default='LDGN_GuideVane_Final')
    args = argparser.parse_args()

    # 路径配置
    ae_checkpoint_path = 'checkpoints_ae/VGAE_GuideVane_2.5D_best.pt'
    stats_path         = 'checkpoints_ae/scaler_stats.pt'
    dataset_path       = 'sta_dataset_2.5D.npy'
    
    # 实验参数
    config = {
        'batch_size': 32,   # 
        'epochs': 5000,
        'lr': 5e-5,
        'val_case_idx': -1, 
    }

    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu')
    
    # 1. 加载 AE 配置并修复
    # 定义 AE 架构 (用于 check_and_fix)
    ae_arch_def = {
        'in_node_features': 4, 'cond_node_features': 13, 'cond_edge_features': 2,
        'latent_node_features': 4, 'depths': [2, 2, 2], 'fnns_width': 128,       
        'fnns_depth': 2, 'aggr': 'sum', 'dropout': 0.0, 'dim': 3, 'scalar_rel_pos': True,'norm_latents': True
    }
    check_and_fix_checkpoint(ae_checkpoint_path, ae_arch_def)

    # 2. 加载 AE 模型
    print("Loading AE Model...")
    try:
        ae_ckpt = torch.load(ae_checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ae_ckpt = torch.load(ae_checkpoint_path, map_location=device)
        
    ae_model = dgn.nn.VGAE(arch=ae_ckpt['arch']).to(device)
    ae_model.load_state_dict(ae_ckpt['weights'])
    ae_model.eval()

    # 3. 准备原始 Dataset (用于预计算)
    print("Initializing Raw Dataset...")
    scaler_stats = torch.load(stats_path)
    transform = transforms.Compose([
        ScaleGuideVane(scaler_stats),
        dgn.transforms.Copy('target', 'field'),
    ])
    pre_transform = transforms.Compose([
        # 建议改为 3，并去掉多余的 scaling 参数
        dgn.transforms.MeshCoarsening(num_scales=3, rel_pos_scaling=[0.1, 0.2, 0.4], scalar_rel_pos=True),
    ])
    
    # [优化] 直接传入 Tensor，避免 Dataset 内部重复读取
    raw_data_np = np.load(dataset_path)
    data_tensor = torch.from_numpy(raw_data_np).float()
    del raw_data_np
    
    raw_dataset = GuideVane25D(
        dataset_path = dataset_path,
        mesh_path    = 'sta_datamesh.npy',
        data_tensor  = data_tensor,
        transform    = transform,
        pre_transform= pre_transform,
        anchor_rate  = 16,
    )
    
    # 4. [核心步骤] 执行预计算 (内存换速度/显存)
    latent_dataset = precompute_latents(raw_dataset, ae_model, device, batch_size=8)
    
    # 5. 划分数据集 (基于 Latent Data)
    # 注意：raw_dataset 和 latent_dataset 索引是一一对应的
    num_cases, num_steps = raw_dataset.num_cases, raw_dataset.num_steps
    
    val_case_idx = config['val_case_idx'] if config['val_case_idx'] >= 0 else num_cases + config['val_case_idx']
    train_case_indices = np.delete(np.arange(num_cases), val_case_idx)
    
    def get_indices(case_indices):
        idx = []
        for c in case_indices:
            idx.extend(range(c * num_steps, (c + 1) * num_steps))
        return idx

    train_set = Subset(latent_dataset, get_indices(train_case_indices))
    val_set   = Subset(latent_dataset, get_indices([val_case_idx]))
    
    # 6. DataLoader
    # 现在加载的是极小的 Latent 数据，num_workers=0 即可，非常快
    train_loader = dgn.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader   = dgn.DataLoader(val_set,   batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # 7. Build LDGN
    ldgn_config = {
        'in_node_features': 4, 'cond_node_features': 128, 'cond_edge_features': 128,
        'depths': [6], 'fnns_width': 128, 'aggr': 'sum', 'dropout': 0.1, 'dim': 3,
    }
    
    diffusion_process = dgn.nn.diffusion.DiffusionProcess(num_steps=1000, schedule_type='linear')
    model = dgn.nn.LatentDiffusionGraphNet(
        autoencoder_checkpoint = ae_checkpoint_path,
        diffusion_process      = diffusion_process,
        learnable_variance     = True,
        arch                   = ldgn_config
    ).to(device)
    # 绑定 AE 仅为了推理时的 decode，训练时不用
    model.autoencoder = ae_model 

    # 8. Train
    train_settings = dgn.nn.TrainingSettings(
        name          = args.name,
        folder        = './checkpoints_ldgn',
        tensor_board  = './boards_ldgn',
        chk_interval  = 50,
        training_loss = dgn.nn.losses.HybridLoss(lambda_vlb=0.01),
        epochs        = config['epochs'],
        batch_size    = config['batch_size'],
        lr            = config['lr'],
        grad_clip     = {"epoch": 0, "limit": 1.0},
        step_sampler  = dgn.nn.diffusion.ImportanceStepSampler,
        mixed_precision = True, 
        device        = device,
    )

    custom_ldgn_train_loop(model, train_settings, train_loader, val_loader)