"""
导叶流场数据预处理模块
包含网格连接、特征缩放、锚点处理等变换
"""

import torch
import numpy as np
from typing import List, Dict, Any

from .abstract_transform import AbstractTransform
from ..graph import Graph


class ConnectGuideVaneMesh(AbstractTransform):
    """构建导叶网格连接关系"""

    def __init__(self, k_neighbors: int = 6):
        """
        Args:
            k_neighbors: KNN邻居数量
        """
        self.k_neighbors = k_neighbors

    def __call__(self, graph: Graph) -> Graph:
        """基于坐标构建KNN连接"""
        pos = graph.pos  # [N, 2]
        N = pos.shape[0]

        # 计算距离矩阵
        distances = torch.cdist(pos, pos)  # [N, N]

        # 排除自连接
        distances.fill_diagonal_(float('inf'))

        # 找到k个最近邻居
        _, knn_indices = torch.topk(distances, self.k_neighbors, dim=1, largest=False)

        # 构建边索引 (双向连接)
        edge_indices = []
        for i in range(N):
            for j in knn_indices[i]:
                edge_indices.append([i, j])
                edge_indices.append([j, i])  # 双向连接

        edge_index = torch.tensor(edge_indices).t().contiguous()  # [2, 2*N*k]

        # 计算边属性 (相对位置)
        edge_attr = pos[edge_index[1]] - pos[edge_index[0]]  # [E, 2]

        graph.edge_index = edge_index
        graph.edge_attr = edge_attr

        return graph


class ScaleGuideVaneAttr(AbstractTransform):
    """缩放导叶相关属性"""

    def __init__(self,
                 velocity_range: tuple = (-5.0, 5.0),
                 distance_range: tuple = (0.0, 1.0),
                 edge_scale: float = 0.1):
        """
        Args:
            velocity_range: 速度场缩放范围
            distance_range: 距离特征缩放范围
            edge_scale: 边属性缩放因子
        """
        self.velocity_vmin, self.velocity_vmax = velocity_range
        self.distance_vmin, self.distance_vmax = distance_range
        self.edge_scale = edge_scale

    def __call__(self, graph: Graph) -> Graph:
        # 缩放目标速度场
        if hasattr(graph, 'target'):
            target = graph.target.clone()
            target = 2 * (target - self.velocity_vmin) / (self.velocity_vmax - self.velocity_vmin) - 1.0
            graph.target = torch.clamp(target, -1.0, 1.0)

        # 缩放距离特征
        if hasattr(graph, 'loc'):
            loc = graph.loc.clone()
            if loc.max() > loc.min():  # 避免除零
                loc = 2 * (loc - self.distance_vmin) / (self.distance_vmax - self.distance_vmin) - 1.0
            graph.loc = torch.clamp(loc, -1.0, 1.0)

        # 缩放边属性
        if hasattr(graph, 'edge_attr'):
            graph.edge_attr = graph.edge_attr * self.edge_scale

        return graph


class AddAnchorConstraints(AbstractTransform):
    """添加锚点约束"""

    def __init__(self,
                 anchor_rates: List[int] = [4, 16, 32],
                 constraint_weight: float = 10.0):
        """
        Args:
            anchor_rates: 锚点下采样率列表
            constraint_weight: 约束权重
        """
        self.anchor_rates = anchor_rates
        self.constraint_weight = constraint_weight

    def __call__(self, graph: Graph) -> Graph:
        if not hasattr(graph, 'anchors'):
            return graph

        # 处理锚点信息
        anchor_masks = {}
        anchor_values = {}

        for rate in self.anchor_rates:
            anchor_key = f'rate_{rate}'
            if anchor_key in graph.anchors:
                anchor_info = graph.anchors[anchor_key]
                indices = anchor_info['indices']

                # 创建锚点掩码
                mask = torch.zeros(graph.num_nodes, 1, dtype=torch.bool)
                mask[indices] = True
                anchor_masks[f'anchor_mask_{rate}'] = mask

                # 获取锚点值
                values = graph.target.clone()
                anchor_values[f'anchor_values_{rate}'] = values[mask.squeeze()]

        # 将锚点信息添加到图中
        graph.anchor_masks = anchor_masks
        graph.anchor_values = anchor_values
        graph.constraint_weight = self.constraint_weight

        return graph


class AddGuideVaneDirichletMask(AbstractTransform):
    """添加导叶Dirichlet边界条件掩码"""

    def __init__(self, num_features: int = 1, dirichlet_boundary_ids: List[int] = [1, 2, 3]):
        """
        Args:
            num_features: 特征数量
            dirichlet_boundary_ids: Dirichlet边界ID列表
        """
        self.num_features = num_features
        self.dirichlet_boundary_ids = dirichlet_boundary_ids

    def __call__(self, graph: Graph) -> Graph:
        N = graph.num_nodes

        # 创建Dirichlet掩码
        dirichlet_mask = torch.zeros(N, self.num_features, dtype=torch.bool)

        for boundary_id in self.dirichlet_boundary_ids:
            mask = (graph.bound == boundary_id).unsqueeze(1)
            dirichlet_mask += mask

        graph.dirichlet_mask = dirichlet_mask

        # 获取Dirichlet边界值 (使用目标值作为边界值)
        if hasattr(graph, 'target'):
            graph.dirichlet_values = graph.target.clone()
            # 只在边界位置保留值，其他位置置零
            graph.dirichlet_values = graph.dirichlet_values * dirichlet_mask.float()

        return graph


class GuideVaneMeshCoarsening(AbstractTransform):
    """导叶网格粗化 (多尺度处理)"""

    def __init__(self,
                 num_scales: int = 4,
                 rel_pos_scaling: List[float] = None,
                 scalar_rel_pos: bool = True):
        """
        Args:
            num_scales: 粗化层数
            rel_pos_scaling: 相对位置缩放因子
            scalar_rel_pos: 是否使用标量相对位置
        """
        self.num_scales = num_scales
        self.rel_pos_scaling = rel_pos_scaling or [0.1, 0.2, 0.4, 0.8]
        self.scalar_rel_pos = scalar_rel_pos

    def __call__(self, graph: Graph) -> Graph:
        """简化的网格粗化实现"""
        # 这里实现一个简化版本，实际应用中需要更复杂的粗化算法
        N = graph.num_nodes

        # 简单的均匀采样粗化
        for scale in range(1, self.num_scales):
            # 计算当前尺度的采样点数
            num_coarse = max(10, N // (2 ** scale))

            # 均匀采样
            indices = torch.linspace(0, N-1, num_coarse, dtype=torch.long)

            # 创建粗化网格属性
            coarse_pos = graph.pos[indices]

            # 存储粗化信息
            setattr(graph, f'coarse_pos_{scale}', coarse_pos)
            setattr(graph, f'coarse_indices_{scale}', indices)

        return graph


class GuideVaneEdgeFlowDirection(AbstractTransform):
    """添加导叶流动方向信息"""

    def __call__(self, graph: Graph) -> Graph:
        """计算主流方向投影"""
        if not hasattr(graph, 'edge_attr'):
            return graph

        edge_attr = graph.edge_attr  # [E, 2]

        # 假设主流方向为x方向 (可以根据实际几何调整)
        flow_direction = torch.tensor([1.0, 0.0], device=edge_attr.device)

        # 计算边向量的单位向量
        edge_norms = torch.norm(edge_attr, dim=1, keepdim=True)
        edge_unit_vectors = edge_attr / (edge_norms + 1e-8)

        # 计算主流方向投影 (点积)
        flow_projection = torch.sum(edge_unit_vectors * flow_direction, dim=1, keepdim=True)

        graph.edge_cond = flow_projection

        return graph


# 组合变换
def create_guide_vane_transforms(
    k_neighbors: int = 6,
    velocity_range: tuple = (-5.0, 5.0),
    edge_scale: float = 0.1,
    anchor_rates: List[int] = [4, 16, 32],
    use_dirichlet: bool = True,
    num_scales: int = 3
):
    """创建导叶数据预处理管道"""
    transforms = []

    # 1. 网格连接
    transforms.append(ConnectGuideVaneMesh(k_neighbors=k_neighbors))

    # 2. 特征缩放
    transforms.append(ScaleGuideVaneAttr(
        velocity_range=velocity_range,
        edge_scale=edge_scale
    ))

    # 3. 流动方向信息
    transforms.append(GuideVaneEdgeFlowDirection())

    # 4. 锚点约束
    if anchor_rates:
        transforms.append(AddAnchorConstraints(anchor_rates=anchor_rates))

    # 5. Dirichlet边界条件
    if use_dirichlet:
        transforms.append(AddGuideVaneDirichletMask())

    # 6. 网格粗化
    if num_scales > 1:
        transforms.append(GuideVaneMeshCoarsening(num_scales=num_scales))

    return transforms