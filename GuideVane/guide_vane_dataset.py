import torch
import numpy as np
import scipy.spatial
from torch_geometric.data import Data
from typing import Optional
import os

import dgn4cfd as dgn
from dgn4cfd.graph import Graph
from dgn4cfd.transforms import cells_to_edge_index

class GuideVane25D(torch.utils.data.Dataset):
    """
    核主泵导叶 2.5D 流场数据集 (极速查表版)
    
    优化策略:
    1. 静态网格共享: 内存中只存一份网格拓扑。
    2. 静态插值映射: 预计算 [N] 的最近邻索引表，运行时零计算量。
    3. 显存保护: 数据随用随取，利用 512GB 内存优势做高速缓存。
    """

    def __init__(
        self,
        dataset_path: str,
        mesh_path: str,
        data_tensor: Optional[torch.Tensor] = None, # [新增] 允许传入预处理好的Tensor
        T: Optional[int] = None,
        transform = None,
        anchor_rate: int = 16,
        blades_num: int = 13,
        pre_transform = None,
        **kwargs 
    ):
        self.dataset_path = dataset_path
        self.mesh_path = mesh_path
        self.transform = transform
        self.anchor_rate = anchor_rate
        self.blades_num = blades_num
        self.pre_transform = pre_transform 
        
        print("Initializing dataset...")

        # 1. 静态几何处理 (保持不变)
        self._process_static_geometry()
        
        # 2. 静态粗化结构 (保持不变)
        if self.pre_transform:
            self.static_ms_attributes = self._compute_static_multiscale_structure()
        else:
            self.static_ms_attributes = {}

        # 3. 插值表 (保持不变)
        self._compute_static_interpolation_map()

        # 4. 数据加载逻辑优化 [核心修改]
        if data_tensor is not None:
            print("Using pre-loaded data tensor (Shared Memory).")
            self.data = data_tensor
            self.num_cases = self.data.shape[0]
            self.num_steps = self.data.shape[1]
        else:
            print(f"Loading raw data from disk: {self.dataset_path}...")
            # 建议使用 mmap_mode='r' 以节省内存，除非显存足够且追求极致速度
            # 如果内存吃紧，请改用: raw_data = np.load(self.dataset_path, mmap_mode='r')
            raw_data = np.load(self.dataset_path) 
            if T is not None:
                raw_data = raw_data[:, :T, :, :]
            self.num_cases = raw_data.shape[0]
            self.num_steps = raw_data.shape[1]
            # 转换为 float32 这是一个内存峰值点
            self.data = torch.from_numpy(raw_data).float()
        
        print(f"Dataset ready. Total samples: {self.num_cases * self.num_steps}")

    def _process_static_geometry(self):
        # 加载网格和坐标
        cells = np.load(self.mesh_path)
        self.cells = torch.from_numpy(cells).long()
        
        temp_data = np.load(self.dataset_path, mmap_mode='r')
        coords = temp_data[0, 0, :, 0:2]
        self.num_nodes = coords.shape[0]
        self.coords = torch.from_numpy(coords).float()
        
        # 极坐标转换
        x, y = self.coords[:, 0], self.coords[:, 1]
        self.r = torch.sqrt(x**2 + y**2)
        self.theta = torch.atan2(y, x)
        self.sin_t = torch.sin(self.theta)
        self.cos_t = torch.cos(self.theta)
        self.pos_polar = torch.stack([self.r, self.sin_t, self.cos_t], dim=1)
        
        # 周期性特征
        self.period_sin = torch.sin(self.blades_num * self.theta).unsqueeze(1)
        self.period_cos = torch.cos(self.blades_num * self.theta).unsqueeze(1)

        # 图连接
        self.edge_index = cells_to_edge_index(self.cells, pos=self.coords)
        row, col = self.edge_index
        self.edge_attr = self.coords[col] - self.coords[row]
        
        # 边界识别
        r_np = self.r.numpy()
        bound_np = np.zeros(self.num_nodes, dtype=int)
        r_min, r_max = r_np.min(), r_np.max()
        inlet_mask = r_np < (r_min + (r_max-r_min)*0.02)
        outlet_mask = r_np > (r_max - (r_max-r_min)*0.02)
        bound_np[inlet_mask] = 1
        bound_np[outlet_mask] = 2
        
        # 拓扑找壁面
        edges = np.concatenate([cells[:,[0,1]], cells[:,[1,2]], cells[:,[2,0]]], axis=0)
        edges.sort(axis=1)
        edges_bytes = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * edges.shape[1])))
        _, counts = np.unique(edges_bytes, return_counts=True)
        boundary_indices = np.unique(edges[np.where(counts==1)[0]])
        is_boundary = np.zeros(self.num_nodes, dtype=bool)
        is_boundary[boundary_indices] = True
        
        wall_mask = is_boundary & (~inlet_mask) & (~outlet_mask)
        bound_np[wall_mask] = 3
        self.bound = torch.from_numpy(bound_np).long()
        self.interior_indices = torch.where(self.bound == 0)[0]
        self.boundary_mask = (self.bound > 0).float().unsqueeze(1)
        
        self.inlet_indices = torch.where(self.bound == 1)[0]
        self.outlet_indices = torch.where(self.bound == 2)[0]
        self.io_mask = (self.bound == 1) | (self.bound == 2)
        
        # 壁面距离
        if wall_mask.sum() > 0:
            tree = scipy.spatial.cKDTree(coords[wall_mask])
            d, _ = tree.query(coords)
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            self.wall_dist = torch.from_numpy(d).float().unsqueeze(1)
        else:
            self.wall_dist = torch.zeros(self.num_nodes, 1)

    def _compute_static_multiscale_structure(self):
        """计算一次静态粗化结构"""
        # 创建模板图
        template = Graph(
            pos=self.pos_polar,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            batch=torch.zeros(self.num_nodes, dtype=torch.long)
        )
        # 应用粗化
        template = self.pre_transform(template)
        
        # 提取生成的静态属性
        static_attrs = {}
        for key, value in template:
            # 过滤掉动态变化的或基础的属性，保留 _2, _3, idx... 等粗化属性
            if key not in ['pos', 'edge_index', 'edge_attr', 'batch', 'ptr', 'x', 'target']:
                static_attrs[key] = value
        return static_attrs

    def _compute_static_interpolation_map(self):
        """核心：预计算 '哪个节点对应哪个锚点' 的索引表"""
        num_interior = len(self.interior_indices)
        num_anchors = max(1, num_interior // self.anchor_rate)
        
        # 固定随机种子选择锚点 (工程上位置是固定的)
        rng = np.random.default_rng(42)
        # 注意：这里选的是 interior_indices 中的下标
        selected_local_indices = rng.choice(len(self.interior_indices), num_anchors, replace=False)
        # 映射回全局节点索引
        self.anchor_global_indices = self.interior_indices[selected_local_indices].long()
        
        # 生成 Anchor Mask
        self.anchor_mask = torch.zeros(self.num_nodes, 1)
        self.anchor_mask[self.anchor_global_indices] = 1.0
        
        # KDTree 搜索最近邻
        anchor_coords = self.coords[self.anchor_global_indices].numpy()
        all_coords = self.coords.numpy()
        
        tree = scipy.spatial.cKDTree(anchor_coords)
        # query 返回的 idx 是 anchor_coords 列表中的索引 (0 ~ num_anchors-1)
        _, nearest_anchor_idx = tree.query(all_coords) 
        
        # 保存这个索引表 [N]
        self.nearest_anchor_idx_map = torch.from_numpy(nearest_anchor_idx).long()

    def __len__(self):
        return self.num_cases * self.num_steps

    def __getitem__(self, idx):
        """极速生成函数: 全程无计算，只有查表"""
        case_idx = idx // self.num_steps
        time_idx = idx % self.num_steps
        
        # 1. 提取数据帧
        frame = self.data[case_idx, time_idx] # [N, 6]
        p, u, v, w = frame[:, 2], frame[:, 3], frame[:, 4], frame[:, 5]
        
        # 2. 坐标变换 (向量化操作，微秒级)
        v_r = u * self.cos_t + v * self.sin_t
        v_theta = -u * self.sin_t + v * self.cos_t
        
        # 目标: [vr, vtheta, vz, p]
        target = torch.stack([v_r, v_theta, w, p], dim=1)
        
        boundary_vals = torch.zeros_like(target)
        boundary_vals[self.io_mask] = target[self.io_mask]
        
        
        # 3. 生成预插值场 (查表法!)
        # 3.1 取出当前时刻所有锚点的真实值 [Num_Anchors, 4]
        current_anchor_vals = target[self.anchor_global_indices]
        
        # 3.2 根据预计算的映射表，直接把锚点值广播到全场 [N, 4]
        interpolated_vals = current_anchor_vals[self.nearest_anchor_idx_map]
        
        # 4. 组装条件特征
        loc = torch.cat([
            self.wall_dist, 
            interpolated_vals, 
            self.anchor_mask, 
            self.boundary_mask, 
            self.period_sin, 
            self.period_cos,
            boundary_vals  # <--- 新增特征
        ], dim=1)
        
        full_dirichlet_mask = (self.bound > 0).bool().unsqueeze(1) | self.anchor_mask.bool()
        
        # 5. 组装 Graph
        graph = Graph(
            x=target, 
            pos=self.pos_polar, 
            target=target,
            edge_index=self.edge_index, 
            edge_attr=self.edge_attr, 
            loc=loc, 
            bound=self.bound,
            anchor_mask=self.anchor_mask.bool(),
            dirichlet_mask=full_dirichlet_mask,
            # [新增] 将边界值单独存一个属性，方便推理时作为 v_0 使用
            boundary_values=boundary_vals 
        ).clone()
        
        # 6. 挂载静态粗化结构 (Level 1+)
        for key, val in self.static_ms_attributes.items():
            # === 核心修正 ===
            # 原代码: setattr(graph, key, val)  <-- 危险！会被原地修改污染
            # 修改为:
            setattr(graph, key, val.clone())  
            # ================
            
        # 7. 动态变换
        if self.transform:
            graph = self.transform(graph)
            
        return graph