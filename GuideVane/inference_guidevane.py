"""
    Inference Script for Guide Vane 2.5D Flow Field Reconstruction (LDGN)
    
    Workflow:
    1. Load Training Data -> Calculate Normalization Stats (v_min, v_max, p_mean, p_std)
    2. Load Trained Models (AE + LDGN)
    3. Load Test Data (using Training Stats for normalization)
    4. Run Inference (Sampling)
    5. Inverse Normalize -> Save/Plot Results
    
    Run with:
        python inference_guidevane.py --gpu 0 --ae_path checkpoints/YOUR_AE.pt --ldgn_path checkpoints_ldgn/YOUR_LDGN.pt
"""

import torch
import argparse
import numpy as np
import os
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

import dgn4cfd as dgn
try:
    from guide_vane_dataset import GuideVane25D
except ImportError:
    raise ImportError("Please ensure 'guide_vane_dataset.py' is in the current directory.")

# 复用 Scale 类 (必须与训练时完全一致)
class ScaleGuideVane:
    def __init__(self, velocity_range: tuple, pressure_stats: tuple):
        self.v_min, self.v_max = velocity_range
        self.v_center = (self.v_max + self.v_min) * 0.5
        self.v_scale  = (self.v_max - self.v_min) * 0.5
        self.p_mean, self.p_std = pressure_stats
        if self.p_std < 1e-6: self.p_std = 1.0

    def __call__(self, graph):
        # Scale Target (if present)
        if hasattr(graph, 'target') and graph.target is not None:
            graph.target[:, 0:3] = (graph.target[:, 0:3] - self.v_center) / self.v_scale
            graph.target[:, 3]   = (graph.target[:, 3]   - self.p_mean) / self.p_std
            graph.target[:, 0:3] = torch.clamp(graph.target[:, 0:3], -1.1, 1.1)
            graph.target[:, 3]   = torch.clamp(graph.target[:, 3], -10.0, 10.0)

        # Scale Condition (Loc)
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            loc[:, 1:4] = (loc[:, 1:4] - self.v_center) / self.v_scale
            loc[:, 1:4] = torch.clamp(loc[:, 1:4], -1.1, 1.1)
            loc[:, 4]   = (loc[:, 4]   - self.p_mean) / self.p_std
            loc[:, 4]   = torch.clamp(loc[:, 4], -10.0, 10.0)
            graph.loc = loc
        return graph

# 反归一化工具函数 (将模型输出恢复为物理值)
def inverse_scale(field_norm, v_min, v_max, p_mean, p_std):
    """
    Args:
        field_norm: [N, 4] Tensor (vr, vtheta, vz, p) normalized
    Returns:
        field_physical: [N, 4] Tensor physical units
    """
    v_center = (v_max + v_min) * 0.5
    v_scale  = (v_max - v_min) * 0.5
    
    field_phys = field_norm.clone()
    
    # Unscale Velocity (channels 0, 1, 2)
    field_phys[:, 0:3] = field_norm[:, 0:3] * v_scale + v_center
    
    # Unscale Pressure (channel 3)
    field_phys[:, 3]   = field_norm[:, 3] * p_std + p_mean
    
    return field_phys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ae_path', type=str, required=True, help='Path to AE checkpoint')
    parser.add_argument('--ldgn_path', type=str, required=True, help='Path to LDGN checkpoint')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples per test case')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu')
    
    # =========================================================
    # 1. 现场计算统计量 (解决不重新训练的问题)
    # =========================================================
    train_data_path = 'examples/GuideVane/sta_dataset_2.5D.npy'
    print(f"[Step 1] Loading TRAINING data from {train_data_path} to compute stats...")
    
    # 加载训练数据
    train_raw = np.load(train_data_path)
    # 计算统计量
    u_train, v_train, w_train = train_raw[..., 3], train_raw[..., 4], train_raw[..., 5]
    p_train = train_raw[..., 2]
    
    v_min = float(min(u_train.min(), v_train.min(), w_train.min()))
    v_max = float(max(u_train.max(), v_train.max(), w_train.max()))
    p_mean = float(p_train.mean())
    p_std  = float(p_train.std())
    
    print(f"   >>> Stats Calculated: V_range=[{v_min:.4f}, {v_max:.4f}], P_mean={p_mean:.4f}, P_std={p_std:.4f}")
    
    # 释放训练数据内存
    del train_raw, u_train, v_train, w_train, p_train

    # =========================================================
    # 2. 加载模型
    # =========================================================
    print(f"[Step 2] Loading Models...")
    
    # 加载 AE 配置 (保持与训练一致)
    ae_config = {
        'in_node_features': 4, 'cond_node_features': 9, 'cond_edge_features': 2,
        'latent_node_features': 4, 'depths': [1, 1, 1], 'fnns_width': 128,
        'fnns_depth': 2, 'aggr': 'sum', 'dropout': 0.0, 'dim': 3, 'scalar_rel_pos': True,
    }
    
    # 加载 LDGN 配置 (保持与训练一致)
    ldgn_config = {
        'in_node_features': 4, 'cond_node_features': 128, 'cond_edge_features': 128,
        'depths': [2, 2], 'fnns_width': 128, 'aggr': 'sum', 'dropout': 0.1,
    }
    
    diffusion_process = dgn.nn.diffusion.DiffusionProcess(num_steps=1000, schedule_type='linear')
    
    # 初始化并加载权重
    model = dgn.nn.LatentDiffusionGraphNet(
        autoencoder_checkpoint = args.ae_path,
        diffusion_process      = diffusion_process,
        learnable_variance     = True,
        arch                   = ldgn_config
    ).to(device)
    
    # 加载 LDGN 权重
    print(f"   Loading LDGN weights from {args.ldgn_path}")
    checkpoint = torch.load(args.ldgn_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # =========================================================
    # 3. 准备测试数据 (应用训练集的统计量)
    # =========================================================
    test_data_path = 'examples/GuideVane/sta_dataset_2.5D_test.npy'
    mesh_path      = 'examples/GuideVane/sta_datamesh.npy'
    
    print(f"[Step 3] Loading TEST data from {test_data_path}...")
    
    # 定义变换
    pre_transform = transforms.Compose([
        dgn.transforms.MeshCoarsening(num_scales=4, rel_pos_scaling=[0.1, 0.2, 0.4, 0.8], scalar_rel_pos=True),
    ])
    
    # 注意：这里我们不需要 LatentTransform，因为推理时我们是从头生成，或者根据需求进行 encode
    # 通常 LDGN 推理只需要几何条件 (loc, edge_index 等)，不需要 field
    # 但为了方便获得这些条件，我们还是加载完整 dataset，并在推理时只取条件部分
    transform = transforms.Compose([
        ScaleGuideVane(velocity_range=(v_min, v_max), pressure_stats=(p_mean, p_std)),
        dgn.transforms.Copy('target', 'field'),
    ])

    test_dataset = GuideVane25D(
        dataset_path = test_data_path,
        mesh_path    = mesh_path,
        transform    = transform,
        pre_transform= pre_transform,
        anchor_rate  = 16,
        preload      = True,
        # 注意：推理不需要缓存文件，或者使用独立的缓存名
        use_cache    = False 
    )
    
    test_loader = dgn.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # =========================================================
    # 4. 执行推理
    # =========================================================
    print(f"[Step 4] Starting Inference on {len(test_dataset)} cases...")
    
    predictions = []
    ground_truths = []
    
    # 创建保存目录
    os.makedirs('results/inference', exist_ok=True)

    with torch.no_grad():
        for i, graph in enumerate(tqdm(test_loader)):
            graph = graph.to(device)
            
            # --- A. 采样 (Sampling) ---
            # LDGN 的 sample 方法通常只需要图结构和条件
            # sample_n 会自动调用 model.sample 并处理多次采样
            # 返回形状通常是: [Num_Nodes, Num_Samples, Num_Fields]
            
            # 这里的 sample_n 是 LatentDiffusionGraphNet 的方法
            # 它内部会先在 Latent Space 采样，然后用 AE Decoder 解码到物理空间
            samples_norm = model.sample_n(
                num_samples = args.num_samples,
                graph       = graph,
                batch_size  = args.num_samples # 并行采样
            ) 
            
            # --- B. 反归一化 (Inverse Scaling) ---
            # samples_norm: [N, S, 4]
            # 我们需要对每个样本进行反归一化
            samples_phys = torch.zeros_like(samples_norm)
            for s in range(args.num_samples):
                samples_phys[:, s, :] = inverse_scale(
                    samples_norm[:, s, :], v_min, v_max, p_mean, p_std
                )
            
            # GT 反归一化 (用于对比)
            gt_norm = graph.target # [N, 4]
            gt_phys = inverse_scale(gt_norm, v_min, v_max, p_mean, p_std)
            
            # --- C. 保存结果 (示例：保存第一个案例) ---
            if i == 0:
                # 转为 Numpy 保存
                res_dict = {
                    'pos': graph.pos.cpu().numpy(),
                    'ground_truth': gt_phys.cpu().numpy(),
                    'prediction_mean': samples_phys.mean(dim=1).cpu().numpy(),
                    'prediction_std': samples_phys.std(dim=1).cpu().numpy(),
                    'samples': samples_phys.cpu().numpy()
                }
                np.save('results/inference/case_0_prediction.npy', res_dict)
                print(f"   Saved Case 0 result to results/inference/case_0_prediction.npy")
            
            # 收集误差统计等 (此处省略)
            
    print("Inference Completed.")