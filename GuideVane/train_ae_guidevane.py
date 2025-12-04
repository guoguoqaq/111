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
    混合归一化策略 [Updated with Masking]
    - 速度 (Velocity): Min-Max Scaling -> [-1, 1]
    - 压力 (Pressure): Standard Normalization (Z-Score) -> mean=0, std=1
    - 边界条件 (Boundary): 归一化后对非边界区域进行 Mask 置零
    """
    def __init__(self, velocity_range: tuple, pressure_stats: tuple):
        self.v_min, self.v_max = velocity_range
        self.v_center = (self.v_max + self.v_min) * 0.5
        self.v_scale  = (self.v_max - self.v_min) * 0.5
        self.p_mean, self.p_std = pressure_stats
        if self.p_std < 1e-6: self.p_std = 1.0

    def __call__(self, graph):
        # 1. Scale Target (保持不变)
        if hasattr(graph, 'target'):
            graph.target[:, 0:3] = (graph.target[:, 0:3] - self.v_center) / self.v_scale
            graph.target[:, 3]   = (graph.target[:, 3]   - self.p_mean) / self.p_std
            graph.target[:, 0:3] = torch.clamp(graph.target[:, 0:3], -1.1, 1.1)
            graph.target[:, 3]   = torch.clamp(graph.target[:, 3], -10.0, 10.0)

        # 2. Scale Condition (修正部分)
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            
            # --- 常规流场特征 (Indices 1-4) ---
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            
            # --- 边界条件特征 (Indices -4 到 -1) ---
            # 先进行归一化
            loc[:, -4:-1] = (loc[:, -4:-1] - self.v_center) / self.v_scale
            loc[:, -1]    = (loc[:, -1]   - self.p_mean) / self.p_std
            
            # 截断防止极值
            loc[:, -4:-1] = torch.clamp(loc[:, -4:-1], -1.1, 1.1)
            loc[:, -1]    = torch.clamp(loc[:, -1], -10.0, 10.0)
            
            # 将非边界节点（内部节点和墙壁）的边界条件特征强制重置为 0。
            if hasattr(graph, 'bound'):
                # 假设: 1=Inlet, 2=Outlet. 仅保留这两个区域的值。
                is_boundary = (graph.bound == 1) | (graph.bound == 2)
                
                # 对 loc 的最后 4 个通道，在 非(~) is_boundary 区域置 0
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
# CUSTOM TRAINING LOOP WITH DETAILED PROGRESS MONITORING
# ============================================================================



import os
import glob
import re
import torch

import torch.nn.functional as F

class KLAnnealingVaeLoss(torch.nn.Module):
    """
    支持 KL Annealing 的 VAE 损失函数。
    返回: (total_loss, metrics_dict)
    """
    def __init__(self, target_kl_reg: float, anneal_epochs: int = 50):
        super().__init__()
        self.target_kl_reg = target_kl_reg
        self.anneal_epochs = anneal_epochs
        self.current_kl_weight = 0.0
        
    def update_kl_weight(self, current_epoch):
        """根据当前 Epoch 更新 KL 权重 (线性增长)"""
        if self.anneal_epochs <= 0:
            self.current_kl_weight = self.target_kl_reg
        else:
            # 线性增长: 0 -> target
            ratio = min(1.0, current_epoch / self.anneal_epochs)
            self.current_kl_weight = self.target_kl_reg * ratio

    def forward(self, model, graph):
        # 1. Inference
        v, _, mean, logvar = model(graph)
        
        # 2. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(v, graph.target)
        
        # 3. KL Divergence
        # sum(1 + log(var) - mu^2 - var)
        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # 4. Total Loss
        total_loss = recon_loss + self.current_kl_weight * kl_div
        
        # 返回 Loss 和 分量字典
        return total_loss, {
            "Recon": recon_loss.item(),
            "KL_Div": kl_div.item(),
            "KL_Weight": self.current_kl_weight
        }

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
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
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
        
        # === [Modification] 更新 KL 权重 ===
        if hasattr(criterion, 'update_kl_weight'):
            criterion.update_kl_weight(epoch)
        
        # === Training Phase ===
        model.train()
        epoch_start_time = time.time()
        
        # 记录分项 Loss
        train_metrics = {"Loss": 0.0, "Recon": 0.0, "KL": 0.0}
        num_batches = 0
        
        # 进度条
        epoch_pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [Train]", leave=False)


        for batch_idx, data in enumerate(epoch_pbar):
            data = data.to(device)
            optimizer.zero_grad()

            try:
                with autocast(enabled=use_amp):
                    loss, metrics = criterion(model, data)

                scaler.scale(loss).backward()
                
                if train_settings['grad_clip'] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_settings['grad_clip']["limit"])

                scaler.step(optimizer)
                scaler.update()

                # 累加指标
                train_metrics["Loss"]  += loss.item()
                train_metrics["Recon"] += metrics["Recon"]
                train_metrics["KL"]    += metrics["KL_Div"]
                num_batches += 1
                
                # 进度条显示当前状态
                epoch_pbar.set_postfix({
                    'L': f'{loss.item():.4f}', 
                    'R': f'{metrics["Recon"]:.4f}',
                    'KL': f'{metrics["KL_Div"]:.4f}',
                    'W': f'{metrics["KL_Weight"]:.1e}'
                })

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\nWARNING: OOM at batch {batch_idx}. Skipping.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e


        # 计算平均值
        avg_train = {k: v / max(num_batches, 1) for k, v in train_metrics.items()}
        current_kl_w = criterion.current_kl_weight if hasattr(criterion, 'current_kl_weight') else 0

        # === Validation Phase ===
        avg_val_loss = 0.0
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    with autocast(enabled=train_settings['mixed_precision']):
                        v_loss, _ = criterion(model, data) # 只关心总 Loss 用于调度
                    val_loss_sum += v_loss.item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / max(val_batches, 1)

        # === Logging & Saving ===
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印详细日志：包含 Recon 和 KL
        print(f"Ep {epoch+1} | T: {epoch_time:.0f}s | "
              f"Tr Loss: {avg_train['Loss']:.5f} (R: {avg_train['Recon']:.5f}, KL: {avg_train['KL']:.1f}) | "
              f"KL_W: {current_kl_w:.1e} | Val: {avg_val_loss:.5f}")

        if writer:
            writer.add_scalar('Loss/Total', avg_train['Loss'], epoch)
            writer.add_scalar('Loss/Recon', avg_train['Recon'], epoch)
            writer.add_scalar('Loss/KL', avg_train['KL'], epoch)
            writer.add_scalar('Param/KL_Weight', current_kl_w, epoch)
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)

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
                'loss': avg_train['Loss'],
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
        'latent_node_features': 4,  #原始2
        'kl_reg':               1e-4,
        'kl_anneal_epochs':     50, # 前50个epoch线性增加权重
        'depths':               [2, 2, 2],#原始[1,1,1]
        'batch_size':           3,         
        'epochs':               500,
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
        training_loss = KLAnnealingVaeLoss(
            target_kl_reg = config['kl_reg'], 
            anneal_epochs = config['kl_anneal_epochs']
        ),
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