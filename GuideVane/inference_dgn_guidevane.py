"""
    Inference & Evaluation Script for Guide Vane DGN (Physical Space).
    
    Features:
    - Loads trained DGN model.
    - Performs DDPM sampling on physical mesh.
    - Computes RMSE/MAE/MRE metrics.
    - Generates comparison plots (GT vs Pred vs Error).
    - Applies Hard Dirichlet Constraints at t=0.

    Run with:
        python inference_dgn_guidevane.py --gpu 0 --epoch 5000
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
from torch.utils.data import Subset

import dgn4cfd as dgn
try:
    from guide_vane_dataset import GuideVane25D
except ImportError:
    raise ImportError("Please ensure 'guide_vane_dataset.py' is in the current directory.")

# ============================================================================
# 1. UTILITIES & SCALER
# ============================================================================

class ScaleGuideVane:
    """
    Consistent Scaler with Training (Includes Boundary Fixes)
    """
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
        # 1. Scale Target
        if hasattr(graph, 'target'):
            graph.target[:, 0:3] = (graph.target[:, 0:3] - self.v_center) / self.v_scale
            graph.target[:, 3]   = (graph.target[:, 3]   - self.p_mean) / self.p_std
            graph.target[:, 0:3] = torch.clamp(graph.target[:, 0:3], -1.1, 1.1)
            graph.target[:, 3]   = torch.clamp(graph.target[:, 3], -10.0, 10.0)
        
        # 2. Scale Condition (Loc)
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            # Interp fields
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 1:4] = torch.clamp(loc[:, 1:4], -1.1, 1.1)
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            loc[:, 4]   = torch.clamp(loc[:, 4], -10.0, 10.0)
            
            # Boundary Values (Last 4)
            loc[:, -4:-1] = (loc[:, -4:-1] - self.v_center) / self.v_scale
            loc[:, -4:-1] = torch.clamp(loc[:, -4:-1], -1.1, 1.1)
            loc[:, -1]    = (loc[:, -1]   - self.p_mean) / self.p_std
            loc[:, -1]    = torch.clamp(loc[:, -1], -10.0, 10.0)

            # Clean non-boundary areas
            if hasattr(graph, 'bound'):
                is_boundary = (graph.bound == 1) | (graph.bound == 2)
                loc[~is_boundary, -4:] = 0.0
            
            graph.loc = loc
            
            # Boundary Values Attribute (for hard constraint)
            if hasattr(graph, 'boundary_values'):
                bv = graph.boundary_values.clone()
                bv[:, 0:3] = (bv[:, 0:3] - self.v_center) / self.v_scale
                bv[:, 3]   = (bv[:, 3]   - self.p_mean) / self.p_std
                if hasattr(graph, 'bound'):
                    bv[~is_boundary] = 0.0
                graph.boundary_values = bv

        return graph

def denormalize(tensor, stats_dict):
    """ Inverse scaling to get physical units """
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
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    
    # Relative Error (avoid div by zero)
    epsilon = 1e-6 
    rel_err = np.abs(diff) / (np.abs(target) + epsilon)
    mre = np.mean(rel_err)
    
    return {f"{name}_RMSE": rmse, f"{name}_MAE": mae, f"{name}_MRE": mre}

# ============================================================================
# 2. DGN SAMPLER
# ============================================================================

class DDPMSamplerPhysical:
    def __init__(self, model, num_steps=1000, device='cuda'):
        self.model = model
        self.num_steps = num_steps
        self.device = device
        
        # Standard DDPM constants
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
        # Inject noisy field into graph
        graph.field_r = x
        graph.r = t
        
        # Predict noise (assuming model predicts epsilon)
        # Note: DGN output format can vary, here we assume it matches 'field_r' shape
        model_out = self.model(graph) 
        if isinstance(model_out, tuple):
            model_out = model_out[0]
        
        # DDPM Sampling Step
        beta_t = self.betas[t_index]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_index]
        
        # Mean
        model_mean = sqrt_recip_alpha_t * (x - beta_t * model_out / sqrt_one_minus_alpha_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, graph):
        # Start from Gaussian Noise
        x = torch.randn_like(graph.target, device=self.device)
        
        # Iterative Denoising
        for i in tqdm(reversed(range(0, self.num_steps)), desc="DGN Sampling", leave=False):
            t = torch.full((graph.num_graphs,), i, device=self.device, dtype=torch.long)
            # Expand t for nodes if necessary, but DGN expects graph.r as batch vector usually
            # DGN implementation usually handles `graph.r` as (Batch_Size,)
            t_batch = t # graph.r expects [Batch_Size]
            
            x = self.p_sample(graph, x, t_batch, i)
            
        return x

# ============================================================================
# 3. VISUALIZATION
# ============================================================================

def get_cartesian_coords(graph):
    r = graph.pos[:, 0]
    sin_t = graph.pos[:, 1]
    cos_t = graph.pos[:, 2]
    x = r * cos_t
    y = r * sin_t
    return x.cpu().numpy(), y.cpu().numpy()

def visualize_result(case_idx, target_phys, recon_phys, graph, output_dir, metrics):
    x, y = get_cartesian_coords(graph)
    triangles = graph.cells.cpu().numpy()
    triangulation = mtri.Triangulation(x, y, triangles)
    
    # Extract Scalar Fields
    vel_target = np.sqrt(target_phys[:, 0]**2 + target_phys[:, 1]**2)
    vel_recon  = np.sqrt(recon_phys[:, 0]**2 + recon_phys[:, 1]**2)
    
    pres_target = target_phys[:, 3]
    pres_recon  = recon_phys[:, 3]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Velocity
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    vmin, vmax = vel_target.min(), vel_target.max()
    
    tc0 = ax[0].tripcolor(triangulation, vel_target, cmap='jet', vmin=vmin, vmax=vmax, shading='gouraud')
    ax[0].set_title("Ground Truth (Velocity)")
    ax[0].axis('equal')
    plt.colorbar(tc0, ax=ax[0])

    tc1 = ax[1].tripcolor(triangulation, vel_recon, cmap='jet', vmin=vmin, vmax=vmax, shading='gouraud')
    ax[1].set_title(f"DGN Prediction (RMSE: {metrics['Velocity_RMSE']:.3f})")
    ax[1].axis('equal')
    plt.colorbar(tc1, ax=ax[1])

    err = np.abs(vel_recon - vel_target)
    tc2 = ax[2].tripcolor(triangulation, err, cmap='inferno', shading='gouraud')
    ax[2].set_title("Absolute Error")
    ax[2].axis('equal')
    plt.colorbar(tc2, ax=ax[2])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case_{case_idx}_velocity.png", dpi=150)
    plt.close()

    # Plot Pressure
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    vmin, vmax = pres_target.min(), pres_target.max()
    
    tc0 = ax[0].tripcolor(triangulation, pres_target, cmap='jet', vmin=vmin, vmax=vmax, shading='gouraud')
    ax[0].set_title("Ground Truth (Pressure)")
    ax[0].axis('equal')
    plt.colorbar(tc0, ax=ax[0])

    tc1 = ax[1].tripcolor(triangulation, pres_recon, cmap='jet', vmin=vmin, vmax=vmax, shading='gouraud')
    ax[1].set_title(f"DGN Prediction (RMSE: {metrics['Pressure_RMSE']:.3f})")
    ax[1].axis('equal')
    plt.colorbar(tc1, ax=ax[1])

    err = np.abs(pres_recon - pres_target)
    tc2 = ax[2].tripcolor(triangulation, err, cmap='inferno', shading='gouraud')
    ax[2].set_title("Absolute Error")
    ax[2].axis('equal')
    plt.colorbar(tc2, ax=ax[2])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case_{case_idx}_pressure.png", dpi=150)
    plt.close()

# ============================================================================
# 4. MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_path', type=str, default='sta_dataset_2.5D.npy')
    parser.add_argument('--mesh_path', type=str, default='sta_datamesh.npy')
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to DGN checkpoint (e.g., checkpoints_dgn/DGN_GuideVane_Physical_epoch_5000.pt)")
    parser.add_argument('--stats_path', type=str, default='checkpoints_dgn/scaler_stats.pt')
    parser.add_argument('--output_dir', type=str, default='./inference_results_dgn')
    parser.add_argument('--num_samples', type=int, default=1, help="Number of test cases to evaluate")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load Stats
    print("Loading Statistics...")
    scaler_stats = torch.load(args.stats_path)
    
    # 2. Config (Must match Training!)
    # Based on your previous successful training setup
    arch = {
        'in_node_features':   4, 
        'cond_node_features': 13, 
        'cond_edge_features': 2,
        'depths':             [2, 2, 2, 2],
        'fnns_width':         128,
        'aggr':               'sum',
        'dropout':            0.1,
        'dim':                3,
        'scalar_rel_pos':     True,
    }

    # 3. Build Model
    print("Building Model...")
    diffusion_process = dgn.nn.diffusion.DiffusionProcess(num_steps=1000, schedule_type='linear')
    model = dgn.nn.DiffusionGraphNet(
        diffusion_process=diffusion_process,
        learnable_variance=True,
        arch=arch
    ).to(device)

    # 4. Load Checkpoint
    print(f"Loading weights from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 5. Load Data (Test Mode)
    # Important: Apply the same 4-scale transform fix!
    print("Loading Dataset...")
    raw_data = np.load(args.dataset_path)
    data_tensor = torch.from_numpy(raw_data).float()
    
    # Test Split: e.g., Last case
    num_cases, num_steps = raw_data.shape[0], raw_data.shape[1]
    # Assuming testing on the validation case (last one)
    test_case_idx = num_cases - 1 
    start_idx = test_case_idx * num_steps
    end_idx = (test_case_idx + 1) * num_steps
    
    # Transforms
    transform = transforms.Compose([
        ScaleGuideVane(scaler_stats),
        dgn.transforms.MeshCoarsening(
            num_scales=4,  # <--- MATCHING TRAINING CONFIG
            rel_pos_scaling=[0.1, 0.2, 0.4, 0.8], 
            scalar_rel_pos=True
        ),
    ])

    dataset = GuideVane25D(
        dataset_path=args.dataset_path, 
        mesh_path=args.mesh_path,
        data_tensor=data_tensor,
        transform=transform,
        anchor_rate=16,
        preload=False # Save memory
    )
    
    # Subset
    test_indices = list(range(start_idx, end_idx))
    test_set = Subset(dataset, test_indices)
    test_loader = dgn.DataLoader(test_set, batch_size=1, shuffle=False) # Batch 1 for inference

    # 6. Inference Loop
    print(f"Starting Inference on Case {test_case_idx}...")
    metrics_log = []
    sampler = DDPMSamplerPhysical(model, num_steps=1000, device=device)
    
    # Limit samples
    count = 0
    
    with torch.no_grad():
        for i, graph in enumerate(test_loader):
            if count >= args.num_samples: break
            graph = graph.to(device)
            
            # --- SAMPLING ---
            gen_field = sampler.sample(graph)
            
            # --- HARD CONSTRAINT (Boundary Fix) ---
            # Replace boundary predictions with Ground Truth (v_0)
            if hasattr(graph, 'bound'):
                is_boundary = (graph.bound == 1) | (graph.bound == 2)
                # If we have normalized boundary values in 'boundary_values' attribute
                if hasattr(graph, 'boundary_values'):
                    gen_field[is_boundary] = graph.boundary_values[is_boundary]
                else:
                    # Fallback: use target (since it's normalized GT)
                    gen_field[is_boundary] = graph.target[is_boundary]

            # --- DENORMALIZE ---
            target_phys = denormalize(graph.target, scaler_stats).cpu().numpy()
            recon_phys  = denormalize(gen_field, scaler_stats).cpu().numpy()
            
            # --- METRICS ---
            # Velocity (Mag)
            v_target = np.sqrt(target_phys[:,0]**2 + target_phys[:,1]**2)
            v_recon = np.sqrt(recon_phys[:,0]**2 + recon_phys[:,1]**2)
            m_vel = calculate_metrics(v_target, v_recon, "Velocity")
            
            # Pressure
            p_target = target_phys[:,3]
            p_recon = recon_phys[:,3]
            m_pre = calculate_metrics(p_target, p_recon, "Pressure")
            
            print(f"Sample {i}: Vel RMSE={m_vel['Velocity_RMSE']:.4f}, Pres RMSE={m_pre['Pressure_RMSE']:.4f}")
            metrics_log.append({**m_vel, **m_pre})
            
            # --- PLOTTING ---
            # Restore cells for plotting (Dataset specific)
            graph.cells = dataset.cells 
            visualize_result(i, target_phys, recon_phys, graph, args.output_dir, m_vel)
            count += 1

    # 7. Final Report
    print("\n=== Average Metrics ===")
    if metrics_log:
        avg_metrics = {}
        for k in metrics_log[0].keys():
            vals = [m[k] for m in metrics_log]
            avg_metrics[k] = np.mean(vals)
            print(f"{k}: {avg_metrics[k]:.6f}")
    
    print(f"\nResults saved to {args.output_dir}")