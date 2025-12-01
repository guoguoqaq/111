"""
    Visualization & Evaluation Script for Guide Vane AE Results.
    [Updated] Added quantitative metrics (MSE, RMSE, MAE, MRE) calculation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import argparse
import os
import atexit
from torchvision import transforms
import dgn4cfd as dgn

# 导入 Dataset 和必要的类
try:
    from guide_vane_dataset import GuideVane25D
except ImportError:
    raise ImportError("Please ensure 'guide_vane_dataset.py' is in the current directory.")

# ============================================================================
# 1. UTILS & METRICS
# ============================================================================

class ScaleGuideVane:
    def __init__(self, stats_dict):
        self.stats = stats_dict
        self.v_min = stats_dict['v_min']
        self.v_max = stats_dict['v_max']
        self.v_center = (self.v_max + self.v_min) * 0.5
        self.v_scale  = (self.v_max - self.v_min) * 0.5
        self.p_mean = stats_dict['p_mean']
        self.p_std  = stats_dict['p_std']
        if self.p_std < 1e-6: self.p_std = 1.0

    def __call__(self, graph):
        if hasattr(graph, 'target'):
            graph.target[:, 0:3] = (graph.target[:, 0:3] - self.v_center) / self.v_scale
            graph.target[:, 3]   = (graph.target[:, 3]   - self.p_mean) / self.p_std
            graph.target[:, 0:3] = torch.clamp(graph.target[:, 0:3], -1.1, 1.1)
            graph.target[:, 3]   = torch.clamp(graph.target[:, 3], -10.0, 10.0)
        
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 1:4] = torch.clamp(loc[:, 1:4], -1.1, 1.1)
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            loc[:, 4]   = torch.clamp(loc[:, 4], -10.0, 10.0)
            graph.loc = loc
        return graph

def denormalize(tensor, stats_dict):
    v_min, v_max = stats_dict['v_min'], stats_dict['v_max']
    p_mean, p_std = stats_dict['p_mean'], stats_dict['p_std']
    v_center = (v_max + v_min) * 0.5
    v_scale  = (v_max - v_min) * 0.5
    
    recon = tensor.clone()
    recon[:, 0:3] = recon[:, 0:3] * v_scale + v_center
    recon[:, 3]   = recon[:, 3] * p_std + p_mean
    return recon

def calculate_metrics(target, recon, name="Field"):
    """
    计算物理场重构精度指标
    Args:
        target: 真实值 (numpy array)
        recon: 重构值 (numpy array)
        name: 物理量名称
    Returns:
        dict: 包含 MSE, RMSE, MAE, MRE 的字典
    """
    diff = recon - target
    abs_diff = np.abs(diff)
    
    # 1. MSE (Mean Squared Error)
    mse = np.mean(diff**2)
    
    # 2. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # 3. MAE (Mean Absolute Error)
    mae = np.mean(abs_diff)
    
    # 4. MRE (Mean Relative Error) - 平均相对误差
    # 加上 epsilon 防止除以 0，通常取物理量量纲的一个极小值
    epsilon = 1e-6 
    rel_err = abs_diff / (np.abs(target) + epsilon)
    mre = np.mean(rel_err)
    
    # 5. Peak Error (最大误差)
    peak_err = np.max(abs_diff)

    return {
        f"{name}_MSE": mse,
        f"{name}_RMSE": rmse,
        f"{name}_MAE": mae,
        f"{name}_MRE": mre,
        f"{name}_Peak": peak_err
    }

def print_metrics(metrics_dict, title="Metrics"):
    print(f"\n[{title}]")
    print(f"{'Metric':<15} | {'Value':<12}")
    print("-" * 30)
    for k, v in metrics_dict.items():
        suffix = ""
        if "MRE" in k: suffix = " (dimless)"
        print(f"{k:<15} | {v:.6f}{suffix}")
    print("-" * 30)

# ============================================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================================

def get_cartesian_coords(graph):
    r = graph.pos[:, 0]
    sin_t = graph.pos[:, 1]
    cos_t = graph.pos[:, 2]
    x = r * cos_t
    y = r * sin_t
    return x.cpu().numpy(), y.cpu().numpy()

def visualize_sample(model, dataset, sample_idx, device, stats, output_dir):
    print(f"\nProcessing sample index: {sample_idx}...")
    
    graph = dataset[sample_idx].to(device)
    
    model.eval()
    with torch.no_grad():
        recon_norm = model(graph)[0]
    
    # 反归一化获取物理值
    target_phys = denormalize(graph.target, stats).cpu().numpy()
    recon_phys  = denormalize(recon_norm, stats).cpu().numpy()
    
    # 坐标准备
    x, y = get_cartesian_coords(graph)
    triangles = dataset.cells.numpy()
    triangulation = mtri.Triangulation(x, y, triangles)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ================= 精度指标计算 =================
    # 1. 速度大小 (Magnitude)
    vel_target_mag = np.sqrt(target_phys[:, 0]**2 + target_phys[:, 1]**2)
    vel_recon_mag  = np.sqrt(recon_phys[:, 0]**2 + recon_phys[:, 1]**2)
    metrics_vel = calculate_metrics(vel_target_mag, vel_recon_mag, "Velocity")
    
    # 2. 压力 (Pressure)
    press_target = target_phys[:, 3]
    press_recon  = recon_phys[:, 3]
    metrics_press = calculate_metrics(press_target, press_recon, "Pressure")
    
    # 打印并保存指标
    print_metrics(metrics_vel, "Velocity Metrics (m/s)")
    print_metrics(metrics_press, "Pressure Metrics (Pa)")
    
    # 保存到 log 文件
    log_path = os.path.join(output_dir, "metrics_log.txt")
    with open(log_path, "a") as f:
        f.write(f"\n=== Sample {sample_idx} ===\n")
        f.write(str(metrics_vel) + "\n")
        f.write(str(metrics_press) + "\n")
    print(f"Metrics saved to {log_path}")

    # ================= 绘图部分 =================
    
    # 1. Geometry (保持不变)
    bound_mask = graph.bound.cpu().numpy()
    anchor_mask = graph.anchor_mask.cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 8))
    plt.tripcolor(triangulation, np.zeros_like(x), cmap='Greys', alpha=0.1, shading='gouraud')
    plt.scatter(x[bound_mask==1], y[bound_mask==1], c='green', s=10, label='Inlet')
    plt.scatter(x[bound_mask==2], y[bound_mask==2], c='blue', s=10, label='Outlet')
    plt.scatter(x[bound_mask==3], y[bound_mask==3], c='black', s=5, label='Wall')
    plt.scatter(x[anchor_mask==1], y[anchor_mask==1], c='red', s=30, marker='x', label='Anchors')
    plt.title(f'Geometry (Sample {sample_idx})')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'vis_{sample_idx}_geometry.png'), dpi=200)
    plt.close()
    
    # 2. Velocity Visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    v_min, v_max = vel_target_mag.min(), vel_target_mag.max()
    
    ax = axes[0]
    tc = ax.tripcolor(triangulation, vel_target_mag, cmap='jet', shading='gouraud', vmin=v_min, vmax=v_max)
    ax.set_title('Ground Truth (Velocity)')
    ax.axis('equal')
    plt.colorbar(tc, ax=ax)
    
    ax = axes[1]
    tc = ax.tripcolor(triangulation, vel_recon_mag, cmap='jet', shading='gouraud', vmin=v_min, vmax=v_max)
    # 在标题中直接展示核心指标
    ax.set_title(f'AE Reconstruction\nRMSE: {metrics_vel["Velocity_RMSE"]:.4f} m/s | MRE: {metrics_vel["Velocity_MRE"]:.2%}')
    ax.axis('equal')
    plt.colorbar(tc, ax=ax)
    
    # 相对误差分布
    epsilon = 1e-1
    rel_error = np.abs(vel_recon_mag - vel_target_mag) / (np.abs(vel_target_mag) + epsilon)
    ax = axes[2]
    tc_err = ax.tripcolor(triangulation, rel_error, cmap='inferno', shading='gouraud', vmin=0, vmax=0.2)
    ax.set_title('Relative Error Distribution')
    ax.axis('equal')
    cbar = plt.colorbar(tc_err, ax=ax)
    cbar.set_label('Relative Error (0 - 20%)')
    
    plt.suptitle(f'Velocity Results (Sample {sample_idx})', fontsize=16)
    plt.savefig(os.path.join(output_dir, f'vis_{sample_idx}_velocity.png'), dpi=200)
    plt.close()
    
    # 3. Pressure Visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    p_min, p_max = press_target.min(), press_target.max()
    
    ax = axes[0]
    tc = ax.tripcolor(triangulation, press_target, cmap='viridis', shading='gouraud', vmin=p_min, vmax=p_max)
    ax.set_title('Ground Truth (Pressure)')
    ax.axis('equal')
    plt.colorbar(tc, ax=ax)
    
    ax = axes[1]
    tc = ax.tripcolor(triangulation, press_recon, cmap='viridis', shading='gouraud', vmin=p_min, vmax=p_max)
    # 标题指标
    ax.set_title(f'AE Reconstruction\nRMSE: {metrics_press["Pressure_RMSE"]:.4f} Pa | MAE: {metrics_press["Pressure_MAE"]:.4f}')
    ax.axis('equal')
    plt.colorbar(tc, ax=ax)
    
    abs_error_p = np.abs(press_recon - press_target)
    ax = axes[2]
    tc_err = ax.tripcolor(triangulation, abs_error_p, cmap='magma', shading='gouraud')
    ax.set_title('Absolute Error Distribution')
    ax.axis('equal')
    plt.colorbar(tc_err, ax=ax)
    
    plt.suptitle(f'Pressure Results (Sample {sample_idx})', fontsize=16)
    plt.savefig(os.path.join(output_dir, f'vis_{sample_idx}_pressure.png'), dpi=200)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

# ============================================================================
# 3. MAIN (Auto-compatible)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to .npy file")
    parser.add_argument('--mesh_path', type=str, default='sta_datamesh.npy')
    parser.add_argument('--model_path', type=str, default='checkpoints_ae/VGAE_GuideVane_CaseSplit_best.pt')
    parser.add_argument('--stats_path', type=str, default='checkpoints_ae/scaler_stats.pt')
    parser.add_argument('--sample_idx', type=int, default=0, help="Index of the sample to visualize")
    parser.add_argument('--output_dir', type=str, default='./vis_results')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu')
    
    scaler_stats = torch.load(args.stats_path)
    
    # 智能处理 3D 数据文件 (兼容测试集)
    raw_data = np.load(args.dataset_path)
    final_dataset_path = args.dataset_path
    is_temp_file = False
    
    if raw_data.ndim == 3:
        print("[Info] Detected 3D data [T, N, V]. Converting to 4D [1, T, N, V]...")
        data_4d = raw_data[np.newaxis, ...] 
        temp_filename = args.dataset_path.replace('.npy', '_temp_4d.npy')
        np.save(temp_filename, data_4d)
        final_dataset_path = temp_filename
        is_temp_file = True
        data_tensor = torch.from_numpy(data_4d).float()
    else:
        data_tensor = torch.from_numpy(raw_data).float()

    if is_temp_file:
        def cleanup():
            if os.path.exists(final_dataset_path):
                os.remove(final_dataset_path)
        atexit.register(cleanup)
    
    print("Initializing Dataset...")
    pre_transform = transforms.Compose([
        dgn.transforms.MeshCoarsening(num_scales=4, rel_pos_scaling=[0.1, 0.2, 0.4, 0.8], scalar_rel_pos=True),
    ])
    transform = transforms.Compose([
        ScaleGuideVane(scaler_stats),
        dgn.transforms.Copy('target', 'field'),
    ])
    
    dataset = GuideVane25D(
        dataset_path=final_dataset_path,
        mesh_path=args.mesh_path,
        data_tensor=data_tensor,
        transform=transform,
        pre_transform=pre_transform,
        anchor_rate=16
    )
    
    print("Loading Model...")
    sample = dataset[0]
    arch = {
        'in_node_features': 4, 'cond_node_features': sample.loc.shape[1], 'cond_edge_features': 2,
        'latent_node_features': 2, 'depths': [1, 1, 1], 'fnns_depth': 2, 'fnns_width': 128,
        'aggr': 'sum', 'dim': 3, 'scalar_rel_pos': True,
    }
    model = dgn.nn.VGAE(arch=arch).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    visualize_sample(model, dataset, args.sample_idx, device, scaler_stats, args.output_dir)
    # python vis_ae_guidevane.py --dataset_path sta_dataset_2.5D_test.npy --model_path checkpoints/VGAE_GuideVane_2.5D_epoch_190.pt --sample_idx 0