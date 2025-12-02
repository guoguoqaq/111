"""
    Train a Variational Graph Autoencoder (VGAE) for Guide Vane 2.5D Flow Field Reconstruction.
    This is the First Stage of the LDGN-2.5D pipeline.
    
    Prerequisites:
        - Ensure 'guide_vane_dataset.py' (containing GuideVane25D class) is in the same directory or python path.
        - Ensure 'sta_dataset_2.5D.npy' and 'data_mesh.npy' are available.
    
    Run with:
        python train_ae_guidevane.py --gpu 0
"""

import torch
import argparse
import os
import numpy as np
from torchvision import transforms

import dgn4cfd as dgn
# 假设你将上一轮的数据集代码保存为了 guide_vane_dataset.py
try:
    from guide_vane_dataset import GuideVane25D
except ImportError:
    raise ImportError("Please save the GuideVane25D dataset code as 'guide_vane_dataset.py' in the current directory.")

torch.multiprocessing.set_sharing_strategy('file_system')

# ============================================================================
# CUSTOM TRANSFORM FOR 4-CHANNEL 2.5D DATA
# ============================================================================
class ScaleGuideVane:
    """
    混合归一化策略：
    - 速度 (Velocity): Min-Max Scaling -> [-1, 1]
    - 压力 (Pressure): Standard Normalization (Z-Score) -> mean=0, std=1
    """
    def __init__(self, velocity_range: tuple, pressure_stats: tuple):
        """
        Args:
            velocity_range: (min, max) 速度的物理范围
            pressure_stats: (mean, std) 压力的均值和标准差
        """
        # --- 速度处理 (Min-Max) ---
        self.v_min, self.v_max = velocity_range
        self.v_center = (self.v_max + self.v_min) * 0.5
        self.v_scale  = (self.v_max - self.v_min) * 0.5
        
        # --- 压力处理 (Z-Score) ---
        self.p_mean, self.p_std = pressure_stats
        # 防止除以0 (虽然物理上不太可能)
        if self.p_std < 1e-6: self.p_std = 1.0

    def __call__(self, graph):
        # 1. Scale Target: [vr, vtheta, vz, p]
        if hasattr(graph, 'target'):
            # 速度分量 (Min-Max)
            graph.target[:, 0:3] = (graph.target[:, 0:3] - self.v_center) / self.v_scale
            # 压力分量 (Z-Score)
            graph.target[:, 3]   = (graph.target[:, 3]   - self.p_mean) / self.p_std
            
            # 截断 (Clamping)
            # 速度截断到 [-1.1, 1.1]
            graph.target[:, 0:3] = torch.clamp(graph.target[:, 0:3], -1.1, 1.1)
            # 压力截断：Z-Score后，±5σ 涵盖了99.9999%的数据，截断异常值防止梯度爆炸
            # 注意：不要截断得太死，否则还是会丢失极值信息
            graph.target[:, 3]   = torch.clamp(graph.target[:, 3], -10.0, 10.0)

        # 2. Scale Condition: Interpolated Field
        # cond indices: 1-3 (velocity), 4 (pressure)
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            
            # [原有] 预插值速度 (Indices 1-4)
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            
            # === [新增] 归一化新增的 Boundary Values (假设它们在最后 4 位) ===
            # loc 结构: [dist(1), interp(4), mask(2), period(2), boundary(4)]
            # boundary indices: -4, -3, -2, -1
            
            # 速度 (vr, vt, vz) -> indices -4, -3, -2
            loc[:, -4:-1] = (loc[:, -4:-1] - self.v_center) / self.v_scale
            # 压力 (p) -> index -1
            loc[:, -1]    = (loc[:, -1]   - self.p_mean) / self.p_std
            
            # 截断 (Clamping)
            loc[:, -4:-1] = torch.clamp(loc[:, -4:-1], -1.1, 1.1)
            loc[:, -1]    = torch.clamp(loc[:, -1], -10.0, 10.0)
            
            # 重新赋值
            graph.loc = loc
            
            # [重要] 同时归一化挂载在 graph 上的 boundary_values 属性
            # 这对于 Inference 时的 Dirichlet v_0 很重要
            if hasattr(graph, 'boundary_values'):
                bv = graph.boundary_values.clone()
                bv[:, 0:3] = (bv[:, 0:3] - self.v_center) / self.v_scale
                bv[:, 3]   = (bv[:, 3]   - self.p_mean) / self.p_std
                # 对于边界上的 0 值区域（非边界），归一化后可能变成非 0 (因为减了 mean)
                # 我们需要把非边界区域重新置 0，以免引入噪声
                # 使用 graph.bound 来掩码
                if hasattr(graph, 'bound'):
                    mask = (graph.bound == 1) | (graph.bound == 2)
                    bv[~mask] = 0.0
                
                graph.boundary_values = bv

        return graph

# ============================================================================
# CUSTOM TRAINING LOOP WITH DETAILED PROGRESS MONITORING
# ============================================================================



import os
import glob
import re
import torch

def custom_train_loop(model, train_settings, train_loader, val_loader=None, stats_to_save=None):
    """
    保留原有断点续训功能的升级版训练循环
    新增: 验证集评估, ReduceLROnPlateau 调度, 最佳模型保存
    """
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import time
    from torch.cuda.amp import autocast, GradScaler

    # --- 1. 初始化设置 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['lr'])
    criterion = train_settings['training_loss']
    
    # 学习率调度器 (基于验证集 Loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )
    
    use_amp = train_settings['mixed_precision']
    scaler = GradScaler(enabled=use_amp)

    writer = SummaryWriter(train_settings['tensor_board']) if train_settings['tensor_board'] else None
    device = train_settings['device']
    model = model.to(device)

    # --- 2. 自动断点续训逻辑 (完全保留您原有的逻辑) ---
    start_epoch = 0
    checkpoint_dir = train_settings['folder']
    model_name = train_settings['name']
    best_val_loss = float('inf') # 记录最佳验证集Loss
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    pattern = os.path.join(checkpoint_dir, f"{model_name}_epoch_*.pt")
    checkpoints = glob.glob(pattern)
    
    if checkpoints:
        def extract_epoch(path):
            match = re.search(r'epoch_(\d+).pt', path)
            return int(match.group(1)) if match else -1
        latest_ckpt = max(checkpoints, key=extract_epoch)
        print(f"\n[Resume] Found latest checkpoint: {latest_ckpt}")
        
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'scaler_state_dict' in checkpoint and use_amp:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"[Resume] Successfully resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        print("\n[Init] No checkpoint found. Starting training from scratch.")

    # --- 3. 训练主循环 ---
    print("=" * 80)
    print(f"Start Epoch: {start_epoch + 1} / {train_settings['epochs']}")
    print(f"Training Cases: {len(train_loader.dataset)} | Validation Cases: {len(val_loader.dataset) if val_loader else 0}")
    print("=" * 80)

    for epoch in range(start_epoch, train_settings['epochs']):
        # === Training Phase ===
        model.train()
        epoch_start_time = time.time()
        
        # 进度条
        epoch_pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [Train]", leave=False)
        total_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(epoch_pbar):
            data = data.to(device)
            optimizer.zero_grad()

            try:
                with autocast(enabled=use_amp):
                    loss = criterion(model, data)

                scaler.scale(loss).backward()
                
                if train_settings['grad_clip'] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_settings['grad_clip']["limit"])

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                num_batches += 1
                
                epoch_pbar.set_postfix({'Loss': f'{loss.item():.5f}'})

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\nWARNING: OOM at batch {batch_idx}. Skipping.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_train_loss = total_loss / max(num_batches, 1)

        # === Validation Phase (新增) ===
        avg_val_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    with autocast(enabled=use_amp):
                        # 验证集仅计算 Loss，不反向传播
                        v_loss = criterion(model, data)
                    val_loss_sum += v_loss.item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / max(val_batches, 1)

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印日志
        print(f"Epoch {epoch+1} | Time: {epoch_time:.1f}s | Tr Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        # TensorBoard
        if writer:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)

        # === 学习率调整 ===
        scheduler.step(avg_val_loss)

        # === 保存逻辑 ===
        # 1. 保存最佳模型 (Best Model)
        if val_loader is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = f"{train_settings['folder']}/{train_settings['name']}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_val_loss,
                'scaler_stats': stats_to_save # 同时保存统计量，方便推理时调用
            }, best_ckpt_path)
            # print(f"  * Best model saved (Val Loss: {avg_val_loss:.6f})")

        # 2. 定期保存 Checkpoint (Resuming用途)
        if (epoch + 1) % train_settings['chk_interval'] == 0:
            checkpoint_path = f"{train_settings['folder']}/{train_settings['name']}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    if writer: writer.close()
    print("Training completed!")

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0)
    argparser.add_argument('--name', type=str, default='VGAE_GuideVane_2.5D')
    args = argparser.parse_args()

    # Initial seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Experiment Configuration
    config = {
        'name':                 args.name,
        'dataset_path':         'sta_dataset_2.5D.npy',
        'mesh_path':            'sta_datamesh.npy',
        'fnns_width':           128,
        'latent_node_features': 8,  #原始2
        'kl_reg':               1e-5,
        'depths':               [2, 2, 2],#原始[1,1,1]
        'batch_size':           3,         
        'epochs':               2000,
        'lr':                   1e-4,
        'anchor_rate':          16,
        'val_case_idx':         -1,  # [新增] 指定倒数第1个案例作为验证集
    }


    # 1. Training Settings
    train_settings = dgn.nn.TrainingSettings(
        name          = config['name'],
        folder        = './checkpoints_ae',
        tensor_board  = './boards',
        chk_interval  = 10,  # 每10个epoch保存一次，增加进度报告频率
        training_loss = dgn.nn.losses.VaeLoss(kl_reg=config['kl_reg']),
        epochs        = config['epochs'],
        batch_size    = config['batch_size'],
        lr            = config['lr'],
        grad_clip     = {"epoch": 0, "limit": 1.0},
        scheduler     = {"factor": 0.5, "patience": 100, "loss": 'training'},
        stopping      = 1e-8,
        mixed_precision = True,
        device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
    )

    # 2. Data Pipeline 优化
    # === [优化] 数据加载与划分逻辑 ===
    print("1. Loading raw data to determine Split & Stats...")
    raw_data_np = np.load(config['dataset_path']) # [Cases, Time, Nodes, Feats]
    
    num_cases, num_steps = raw_data_np.shape[0], raw_data_np.shape[1]
    all_case_indices = np.arange(num_cases)
    
    # 确定验证集索引 (例如: 最后一个案例)
    val_case_idx = config['val_case_idx'] if config['val_case_idx'] >= 0 else num_cases + config['val_case_idx']
    train_case_indices = np.delete(all_case_indices, val_case_idx)
    val_case_indices = np.array([val_case_idx])
    
    print(f"   Total Cases: {num_cases}")
    print(f"   Training Cases: {train_case_indices}")
    print(f"   Validation Case: {val_case_indices} (Strictly Isolated)")

    # === [关键] 仅基于训练集计算统计量 ===
    print("2. Calculating statistics (Train Set Only)...")
    train_data_subset = raw_data_np[train_case_indices]
    
    u, v, w = train_data_subset[..., 3], train_data_subset[..., 4], train_data_subset[..., 5]
    v_min = float(min(u.min(), v.min(), w.min()))
    v_max = float(max(u.max(), v.max(), w.max()))
    
    p = train_data_subset[..., 2]
    p_mean = float(p.mean())
    p_std  = float(p.std())
    
    scaler_stats = {'v_min': v_min, 'v_max': v_max, 'p_mean': p_mean, 'p_std': p_std}
    print(f"   Stats Computed: {scaler_stats}")
    
    # 保存统计量供后续使用 (LDGN/Inference)
    os.makedirs(train_settings['folder'], exist_ok=True)
    torch.save(scaler_stats, os.path.join(train_settings['folder'], 'scaler_stats.pt'))

    # 转为 Tensor
    data_tensor = torch.from_numpy(raw_data_np).float()
    del raw_data_np, train_data_subset 
    import gc; gc.collect()

    # === 构建 Dataset ===
    pre_transform = transforms.Compose([
        dgn.transforms.MeshCoarsening(
            num_scales=3, rel_pos_scaling=[0.1, 0.2, 0.4], scalar_rel_pos=True, 
        ),
    ]) 
    
    transform = transforms.Compose([
        ScaleGuideVane( # 使用计算好的统计量
            velocity_range=(v_min, v_max), 
            pressure_stats=(p_mean, p_std)
        ),
        dgn.transforms.Copy('target', 'field'), 
    ])

    full_dataset = GuideVane25D(
        dataset_path = config['dataset_path'],
        mesh_path    = config['mesh_path'],
        data_tensor  = data_tensor,
        transform    = transform,
        pre_transform= pre_transform,
        anchor_rate  = config['anchor_rate'],
    )

    # === [关键] 使用 Subset 进行物理划分 ===
    from torch.utils.data import Subset
    
    def get_flat_indices(case_indices, n_steps):
        indices = []
        for c in case_indices:
            indices.extend(range(c * n_steps, (c + 1) * n_steps))
        return indices

    train_indices = get_flat_indices(train_case_indices, num_steps)
    val_indices   = get_flat_indices(val_case_indices, num_steps)
    
    train_set = Subset(full_dataset, train_indices)
    val_set   = Subset(full_dataset, val_indices)
    
    # 创建 Loader
    train_loader = dgn.DataLoader(train_set, batch_size=train_settings['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = dgn.DataLoader(val_set,   batch_size=train_settings['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # === 模型构建与训练 ===
    # (这部分保持原有的 arch 设置)
    sample = full_dataset[0]
    arch = {
        'in_node_features': 4, 
        'cond_node_features': sample.loc.shape[1], 
        'cond_edge_features': 2,
        'latent_node_features': config['latent_node_features'],
        'depths': config['depths'], 
        'fnns_depth': 2, 
        'fnns_width': config['fnns_width'],
        'norm_latents': True,
        'aggr': 'sum', 
        'dropout': 0.0, 
        'dim': 3, 
        'scalar_rel_pos': True,
    }

    print("Building VGAE Model...")
    model = dgn.nn.VGAE(arch = arch)

    # 调用修改后的训练循环
    custom_train_loop(
        model, 
        train_settings, 
        train_loader, 
        val_loader=val_loader, # 传入验证集
        stats_to_save=scaler_stats # 传入统计量以便保存到 checkpoint
    )
    print("Training finished.")
    # ========================================
    # 🔬 Latent Space 质量检查
    # ========================================
    print("\n" + "="*80)
    print("🔬 Latent Space Quality Check")
    print("="*80)
    
    model.eval()
    latent_stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': []
    }
    
    with torch.no_grad():
        for i, graph in enumerate(val_loader):
            if i >= 10:  # 只检查前10个验证样本
                break
            
            graph = graph.to(device)
            
            # 编码到潜在空间
            z_latent, mean, logvar = model.node_encoder(
                graph,
                graph.field,
                # ... 需要传入 c_latent_list 和 e_latent_list
                # 简化版: 直接用 encode
            )
            
            # 完整编码
            outputs = model.encode(graph, graph.target)
            z = outputs[0]  # [N, latent_dim]
            
            latent_stats['mean'].append(z.mean().item())
            latent_stats['std'].append(z.std().item())
            latent_stats['min'].append(z.min().item())
            latent_stats['max'].append(z.max().item())
    
    # 统计分析
    import numpy as np
    for key in latent_stats:
        latent_stats[key] = np.array(latent_stats[key])
    
    print(f"\n📊 Latent Space Statistics (Validation Set):")
    print(f"   Mean: {latent_stats['mean'].mean():.4f} ± {latent_stats['mean'].std():.4f}")
    print(f"   Std:  {latent_stats['std'].mean():.4f} ± {latent_stats['std'].std():.4f}")
    print(f"   Range: [{latent_stats['min'].mean():.4f}, {latent_stats['max'].mean():.4f}]")
    
    # 健康度判断
    mean_avg = latent_stats['mean'].mean()
    std_avg = latent_stats['std'].mean()
    
    print(f"\n✅ Health Check:")
    if abs(mean_avg) < 0.5 and 0.5 < std_avg < 1.5:
        print("   ✓ Latent distribution is healthy (close to N(0,1))")
    else:
        print(f"   ✗ WARNING: Latent distribution deviates from N(0,1)")
        print(f"     Consider adjusting KL regularization weight")
    
    # 可视化 (可选)
    if latent_stats['mean'].size > 0:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(latent_stats['mean'], bins=20, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Target (0)')
        axes[0, 0].set_title('Latent Mean Distribution')
        axes[0, 0].legend()
        
        axes[0, 1].hist(latent_stats['std'], bins=20, edgecolor='black')
        axes[0, 1].axvline(1, color='red', linestyle='--', label='Target (1)')
        axes[0, 1].set_title('Latent Std Distribution')
        axes[0, 1].legend()
        
        axes[1, 0].scatter(latent_stats['mean'], latent_stats['std'], alpha=0.6)
        axes[1, 0].axhline(1, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Mean')
        axes[1, 0].set_ylabel('Std')
        axes[1, 0].set_title('Mean vs Std')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 单个样本的潜在向量分布
        with torch.no_grad():
            sample_graph = next(iter(val_loader)).to(device)
            sample_z = model.encode(sample_graph, sample_graph.target)[0]
            axes[1, 1].hist(sample_z.cpu().flatten().numpy(), bins=50, edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Single Sample Latent Distribution')
            axes[1, 1].set_xlabel('Latent Value')
        
        plt.tight_layout()
        plt.savefig(f'{train_settings["folder"]}/latent_space_quality.png', dpi=150)
        print(f"\n📈 Visualization saved to {train_settings['folder']}/latent_space_quality.png")
        plt.close()
    
    print("="*80 + "\n")
