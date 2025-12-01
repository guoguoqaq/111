"""
    [FIXED V3] Inference & Evaluation Script for Guide Vane LDGN.
    Fixes:
    1. Added missing Colorbars for all plots.
    2. Inherits all previous fixes (Tuple output, In-place graph modification).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import argparse
import os
import shutil
from tqdm import tqdm
from torchvision import transforms

import dgn4cfd as dgn
try:
    from guide_vane_dataset import GuideVane25D
except ImportError:
    raise ImportError("Please ensure 'guide_vane_dataset.py' is in the current directory.")

# ============================================================================
# 1. UTILITIES & FIXERS
# ============================================================================

def check_and_fix_checkpoint(ckpt_path, arch_config):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    modified = False
    if 'arch' not in checkpoint:
        checkpoint['arch'] = arch_config
        modified = True
    if 'weights' not in checkpoint and 'model_state_dict' in checkpoint:
        checkpoint['weights'] = checkpoint['model_state_dict']
        modified = True
    
    if modified:
        backup_path = ckpt_path + '.bak'
        if not os.path.exists(backup_path):
            shutil.copy(ckpt_path, backup_path)
        torch.save(checkpoint, ckpt_path)
        print(f"[Info] Fixed checkpoint: {ckpt_path}")

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
            # 原有的处理
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 1:4] = torch.clamp(loc[:, 1:4], -1.1, 1.1)
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            loc[:, 4]   = torch.clamp(loc[:, 4], -10.0, 10.0)
            
            # === [必须补充] 处理新增的 boundary_vals (最后4位) ===
            # loc 结构: [dist(1), interp(4), mask(2), period(2), boundary(4)]
            # 倒数4位是 boundary_vals
            loc[:, -4:-1] = (loc[:, -4:-1] - self.v_center) / self.v_scale
            loc[:, -1]    = (loc[:, -1]   - self.p_mean) / self.p_std
            # 同样截断
            loc[:, -4:-1] = torch.clamp(loc[:, -4:-1], -1.1, 1.1)
            loc[:, -1]    = torch.clamp(loc[:, -1], -10.0, 10.0)
            
            graph.loc = loc
            
            # === [必须补充] 同时处理 graph.boundary_values 属性 ===
            # 这是 decode 时用的 v_0
            if hasattr(graph, 'boundary_values'):
                bv = graph.boundary_values.clone()
                bv[:, 0:3] = (bv[:, 0:3] - self.v_center) / self.v_scale
                bv[:, 3]   = (bv[:, 3]   - self.p_mean) / self.p_std
                
                # 非边界区域置零 (防止归一化产生的均值偏移)
                if hasattr(graph, 'bound'):
                    mask = (graph.bound == 1) | (graph.bound == 2)
                    bv[~mask] = 0.0
                
                graph.boundary_values = bv
                
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
    diff = recon - target
    abs_diff = np.abs(diff)
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff)
    epsilon = 1e-6 
    rel_err = abs_diff / (np.abs(target) + epsilon)
    mre = np.mean(rel_err)
    return {f"{name}_RMSE": rmse, f"{name}_MAE": mae, f"{name}_MRE": mre}

# ============================================================================
# 2. SAMPLER & GENERATION LOGIC
# ============================================================================

class DDPMSampler:
    def __init__(self, model, num_steps=1000, device='cuda'):
        self.model = model
        self.num_steps = num_steps
        self.device = device
        
        beta_start, beta_end = 1e-4, 0.02
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @torch.no_grad()
    def p_sample(self, graph, x, t, t_index):
        graph.field_r = x
        graph.r = t
        
        model_out = self.model(graph) 
        if isinstance(model_out, tuple):
            model_out = model_out[0]
        
        beta_t = self.betas[t_index]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_index]
        
        model_mean = sqrt_recip_alpha_t * (x - beta_t * model_out / sqrt_one_minus_alpha_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, graph, shape):
        x = torch.randn(shape, device=self.device)
        for i in tqdm(reversed(range(0, self.num_steps)), desc="Sampling", leave=False):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            t_batch = t.repeat(shape[0])
            x = self.p_sample(graph, x, t_batch, i)
        return x

def encode_conditions_only(graph, ae_model):
    """
    WARNING: This function modifies 'graph' in-place (downsampling edge_index).
    """
    c = torch.cat([f for f in [graph.get('loc'), graph.get('glob'), graph.get('omega')] if f is not None], dim=1)
    e = torch.cat([f for f in [graph.get('edge_attr'), graph.get('edge_cond')] if f is not None], dim=1)
    
    # Run Conditional Encoder
    c_list, e_list, edge_list, batch_list = ae_model.cond_encoder(graph, c, e, graph.edge_index)
    
    # Attach latent level features
    graph.c_latent = c_list[-1]
    graph.e_latent = e_list[-1]
    graph.edge_index = edge_list[-1] # Graph is now Latent Topology!
    graph.batch = batch_list[-1]
    return graph

def generate_flow_field(model, graph, device):
    graph = graph.to(device)
    
    # [CRITICAL] Backup High-Res Topology
    edge_index_high = graph.edge_index_original
    batch_high = graph.batch_original

    # 1. Encode Conditions
    with torch.no_grad():
        graph = encode_conditions_only(graph, model.autoencoder)
    
    # 2. Sample Latent
    num_latent_nodes = graph.c_latent.size(0)
    latent_dim = model.arch['in_node_features']
    
    sampler = DDPMSampler(model, num_steps=model.diffusion_process.num_steps, device=device)
    z_gen = sampler.sample(graph, shape=(num_latent_nodes, latent_dim))
    
    # 3. Decode
    # [CRITICAL] Restore High-Res Topology
    graph.edge_index = edge_index_high
    graph.batch = batch_high
    
    c_raw = torch.cat([f for f in [graph.get('loc'), graph.get('glob'), graph.get('omega')] if f is not None], dim=1)
    e_raw = torch.cat([f for f in [graph.get('edge_attr'), graph.get('edge_cond')] if f is not None], dim=1)
    
    # Re-run encoder to get skip connection lists
    c_list, e_list, edge_list, batch_list = model.autoencoder.cond_encoder(graph, c_raw, e_raw, graph.edge_index)
    
    with torch.no_grad():
        # 获取 Dirichlet Mask (我们在 Dataset 里已经设好了包含 Inlet/Outlet)
        dirichlet_mask = graph.dirichlet_mask if hasattr(graph, 'dirichlet_mask') else None
        
        # === [核心修改] 构造 v_0 (初始值/边界值) ===
        # 我们使用 graph.boundary_values，它包含了归一化后的真实边界值
        if hasattr(graph, 'boundary_values'):
            v_0 = graph.boundary_values.to(device)
        else:
            # Fallback (仅包含 0)
            v_0 = torch.zeros((graph.num_nodes, model.autoencoder.num_fields), device=device)
        
        # 调用解码器
        # Decode 函数内部逻辑：output = torch.where(dirichlet_mask, v_0, output)
        # 这保证了在 Inlet/Outlet 节点，输出值被强制替换为 v_0 中的值
        recon_field = model.autoencoder.decode(
            graph, z_gen, c_list, e_list, edge_list, batch_list, 
            dirichlet_mask=dirichlet_mask, v_0=v_0
        )
        
    return recon_field

# ============================================================================
# 3. VISUALIZATION (Fixed Colorbars)
# ============================================================================

def get_cartesian_coords(graph):
    r = graph.pos[:, 0]
    sin_t = graph.pos[:, 1]
    cos_t = graph.pos[:, 2]
    x = r * cos_t
    y = r * sin_t
    return x.cpu().numpy(), y.cpu().numpy()

def visualize_generation(sample_idx, target_phys, recon_phys, graph, output_dir, metrics):
    x, y = get_cartesian_coords(graph)
    triangles = graph.cells.cpu().numpy()
    triangulation = mtri.Triangulation(x, y, triangles)
    
    vel_target = np.sqrt(target_phys[:, 0]**2 + target_phys[:, 1]**2)
    vel_recon  = np.sqrt(recon_phys[:, 0]**2 + recon_phys[:, 1]**2)
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    vmin, vmax = vel_target.min(), vel_target.max()
    
    # 1. Ground Truth
    tc0 = ax[0].tripcolor(triangulation, vel_target, cmap='jet', vmin=vmin, vmax=vmax, shading='gouraud')
    ax[0].set_title("Ground Truth (Velocity)")
    ax[0].axis('equal')
    plt.colorbar(tc0, ax=ax[0], fraction=0.046, pad=0.04) # [FIX] Added Colorbar
    
    # 2. LDGN Generation
    tc1 = ax[1].tripcolor(triangulation, vel_recon, cmap='jet', vmin=vmin, vmax=vmax, shading='gouraud')
    ax[1].set_title(f"LDGN Generation\nRMSE: {metrics['Velocity_RMSE']:.3f}")
    ax[1].axis('equal')
    plt.colorbar(tc1, ax=ax[1], fraction=0.046, pad=0.04) # [FIX] Added Colorbar
    
    # 3. Relative Error
    err = np.abs(vel_recon - vel_target) / (np.abs(vel_target) + 0.1)
    tc2 = ax[2].tripcolor(triangulation, err, cmap='inferno', vmin=0, vmax=0.2, shading='gouraud')
    ax[2].set_title("Relative Error")
    ax[2].axis('equal')
    plt.colorbar(tc2, ax=ax[2], fraction=0.046, pad=0.04) # [FIX] Added Colorbar
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_{sample_idx}_velocity.png", dpi=200)
    plt.close()

# ============================================================================
# 4. MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--mesh_path', type=str, default='sta_datamesh.npy')
    parser.add_argument('--ae_path', type=str, default='checkpoints_ae/VGAE_GuideVane_CaseSplit_best.pt')
    parser.add_argument('--ldgn_path', type=str, default='checkpoints_ldgn/LDGN_GuideVane_Final_best.pt')
    parser.add_argument('--stats_path', type=str, default='checkpoints_ae/scaler_stats.pt')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./inference_results')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    scaler_stats = torch.load(args.stats_path)

    # 1. Configs
    ae_config = {
        'in_node_features': 4, 
        'cond_node_features': 13,  # <--- 修改这里：从 9 改为 13
        'cond_edge_features': 2,
        'latent_node_features': 8, # <--- 修改这里：从 2 改为 8 (见问题B)
        'depths': [2, 2, 2],       # <--- 修改这里：从 [1,1,1] 改为 [2,2,2] (见问题C)
        'fnns_width': 128, 
        'fnns_depth': 2,           # 补全可能缺失的参数
        'aggr': 'sum', 
        'dropout': 0.0, 
        'dim': 3, 
        'scalar_rel_pos': True,
        'norm_latents': True       # Train AE 中设置了 True
    }
    ldgn_config = {
        'in_node_features': 8,      # 必须匹配 AE 的 latent_node_features
        'cond_node_features': 128,  # 这是 LDGN 内部映射后的维度，通常训练默认也是128
        'cond_edge_features': 128,
        'depths': [8],              # <--- 修改这里：必须匹配 train_ldgn 中的设置
        'fnns_width': 128, 
        'aggr': 'sum', 
        'dropout': 0.1,             # 训练时设了 0.1，推理时虽然 eval() 会忽略 dropout，但架构加载需要一致
        'dim': 3
    }

    # 2. Fix & Load Models
    check_and_fix_checkpoint(args.ae_path, ae_config)
    
    print("Loading Models...")
    diffusion_process = dgn.nn.diffusion.DiffusionProcess(num_steps=1000, schedule_type='linear')
    model = dgn.nn.LatentDiffusionGraphNet(
        autoencoder_checkpoint = args.ae_path,
        diffusion_process      = diffusion_process,
        learnable_variance     = True,
        arch                   = ldgn_config
    ).to(device)
    
    try:
        ldgn_ckpt = torch.load(args.ldgn_path, map_location=device, weights_only=False)
    except TypeError:
        ldgn_ckpt = torch.load(args.ldgn_path, map_location=device)
    model.load_state_dict(ldgn_ckpt['model_state_dict'])
    
    ae_ckpt = torch.load(args.ae_path, map_location=device)
    if 'weights' in ae_ckpt:
        model.autoencoder.load_state_dict(ae_ckpt['weights'])
    else:
        model.autoencoder.load_state_dict(ae_ckpt['model_state_dict'])
    
    model.eval()
    model.autoencoder.eval()

    # 3. Data
    print("Loading Data...")
    raw_data = np.load(args.dataset_path)
    if raw_data.ndim == 3:
        raw_data = raw_data[np.newaxis, ...]
    
    dataset = GuideVane25D(
        dataset_path=args.dataset_path, 
        mesh_path=args.mesh_path,
        data_tensor=torch.from_numpy(raw_data).float(),
        transform=transforms.Compose([
            ScaleGuideVane(scaler_stats),
            dgn.transforms.Copy('target', 'field')
        ]),
        pre_transform = transforms.Compose([
            # 建议改为 3，并去掉多余的 scaling 参数
            dgn.transforms.MeshCoarsening(num_scales=3, rel_pos_scaling=[0.1, 0.2, 0.4], scalar_rel_pos=True),
        ]),
        anchor_rate=16
    )
    
    loader = dgn.DataLoader(dataset, batch_size=1, shuffle=False)

    # 4. Inference
    print(f"Starting Inference on {args.num_samples} samples...")
    metrics_log = []
    
    for i, graph in enumerate(loader):
        if i >= args.num_samples: break
        
        # [CRITICAL] Save High-Res Topology for this sample
        graph.edge_index_original = graph.edge_index.clone()
        if hasattr(graph, 'batch'):
            graph.batch_original = graph.batch.clone()
        else:
            graph.batch_original = torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.x.device)

        recon_norm = generate_flow_field(model, graph, device)
        
        target_phys = denormalize(graph.target, scaler_stats).cpu().numpy()
        recon_phys = denormalize(recon_norm, scaler_stats).cpu().numpy()
        
        v_target = np.sqrt(target_phys[:,0]**2 + target_phys[:,1]**2)
        v_recon = np.sqrt(recon_phys[:,0]**2 + recon_phys[:,1]**2)
        m_vel = calculate_metrics(v_target, v_recon, "Velocity")
        
        p_target = target_phys[:,3]
        p_recon = recon_phys[:,3]
        m_pre = calculate_metrics(p_target, p_recon, "Pressure")
        
        print(f"Sample {i}: Vel RMSE={m_vel['Velocity_RMSE']:.4f}, Pressure RMSE={m_pre['Pressure_RMSE']:.4f}")
        metrics_log.append({**m_vel, **m_pre})
        
        graph.cells = dataset.cells 
        visualize_generation(i, target_phys, recon_phys, graph, args.output_dir, m_vel)

    print("\n=== Average Metrics ===")
    if metrics_log:
        avg_metrics = {}
        for k in metrics_log[0].keys():
            avg_metrics[k] = np.mean([m[k] for m in metrics_log])
            print(f"{k}: {avg_metrics[k]:.6f}")
        
# python inference_ldgn_guidevane.py --dataset_path sta_dataset_2.5D_test.npy --ae_path checkpoints_ae/VGAE_GuideVane_2.5D_epoch_1060.pt --ldgn_path checkpoints_ldgn/LDGN_GuideVane_Final_best.pt --num_samples 5