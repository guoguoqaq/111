"""
导叶流场数据集类
基于椭圆流场案例修改，适配导叶速度场重构任务
"""

import torch
import numpy as np
from typing import Union, List, Optional
from pathlib import Path

from ..graph import Graph
from ..loader import DataLoader
from ..transforms import AbstractTransform


class GuideVaneVelocityField(torch.utils.data.Dataset):
    """
    导叶速度场数据集

    数据格式：
    - sta_data_mesh.npy: [M, 3] 三角网格拓扑 (节点1, 节点2, 节点3)
    - sta_dataset.npy: [B, T, N, 3+] (案例数, 时间步, 节点数, [x, y, velocity, ...])
    """

    def __init__(
        self,
        mesh_path: Union[str, Path],
        dataset_path: Union[str, Path],
        T: Optional[int] = None,
        transform: Optional[AbstractTransform] = None,
        anchor_downsampling_rates: List[int] = [4, 16, 32],
        preload: bool = False
    ):
        """
        初始化导叶数据集

        Args:
            mesh_path: 三角网格文件路径
            dataset_path: 数据集文件路径
            T: 时间步数，None表示使用所有时间步
            transform: 数据变换
            anchor_downsampling_rates: 锚点下采样率列表
            preload: 是否预加载数据
        """
        self.mesh_path = Path(mesh_path)
        self.dataset_path = Path(dataset_path)
        self.T = T
        self.transform = transform
        self.anchor_downsampling_rates = anchor_downsampling_rates
        self.preload = preload

        # 加载网格拓扑
        self._load_mesh()

        # 加载数据集
        self._load_dataset()

        # 识别边界类型
        self._identify_boundaries()

        # 计算局部几何特征
        self._compute_geometric_features()

    def _load_mesh(self):
        """加载三角网格拓扑"""
        self.mesh_topology = np.load(self.mesh_path)  # [M, 3]
        print(f"加载网格拓扑: {self.mesh_topology.shape}")

    def _load_dataset(self):
        """加载流场数据"""
        dataset = np.load(self.dataset_path)  # [B, T, N, 3+]
        self.num_cases = dataset.shape[0]
        self.num_time_steps = dataset.shape[1]
        self.num_nodes = dataset.shape[2]
        self.num_features = dataset.shape[3]

        # 提取坐标和速度场
        self.coordinates = dataset[0, 0, :, :2]  # [N, 2] 假设坐标不随时间变化
        self.velocity_data = dataset[:, :, :, 2]  # [B, T, N] 速度标量场

        if self.T is not None:
            self.velocity_data = self.velocity_data[:, :self.T, :]
            self.num_time_steps = min(self.T, self.num_time_steps)

        print(f"加载数据集: {self.num_cases}案例, {self.num_time_steps}时间步, {self.num_nodes}节点")

        # 预处理数据
        if self.preload:
            self.processed_data = []
            for case_idx in range(self.num_cases):
                for time_idx in range(self.num_time_steps):
                    graph = self._create_graph(case_idx, time_idx)
                    if self.transform:
                        graph = self.transform(graph)
                    self.processed_data.append(graph)

    def _identify_boundaries(self):
        """
        识别导叶流场边界类型
        基于节点坐标和网格拓扑自动识别
        """
        N = self.num_nodes
        self.boundaries = np.zeros(N, dtype=int)  # 0:内部, 1:入口, 2:出口, 3:壁面

        # 获取所有节点的坐标
        coords = self.coordinates

        # 识别入口和出口 (基于x坐标的极值)
        x_coords = coords[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()

        # 容差设置 (可根据实际几何调整)
        inlet_tolerance = 0.05 * (x_max - x_min)
        outlet_tolerance = 0.05 * (x_max - x_min)

        # 入口边界 (x坐标最小)
        inlet_mask = x_coords <= (x_min + inlet_tolerance)
        self.boundaries[inlet_mask] = 1

        # 出口边界 (x坐标最大)
        outlet_mask = x_coords >= (x_max - outlet_tolerance)
        self.boundaries[outlet_mask] = 2

        # 识别壁面 (基于几何特征，需要更复杂的算法)
        # 这里简化处理：非入口、出口的边界点
        wall_mask = self._identify_walls()
        self.boundaries[wall_mask] = 3

        # 统计边界类型
        boundary_counts = np.bincount(self.boundaries)
        print(f"边界识别结果: 内部{boundary_counts[0]}, 入口{boundary_counts[1]}, 出口{boundary_counts[2]}, 壁面{boundary_counts[3]}")

    def _identify_walls(self):
        """识别壁面节点 (导叶型面)"""
        # 基于网格拓扑识别边界边
        # 找到只属于一个三角形的边
        from collections import defaultdict

        edge_count = defaultdict(int)
        edge_to_nodes = {}

        for tri in self.mesh_topology:
            # 三角形的三条边
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for edge in edges:
                # 规范化边表示 (小节点在前)
                edge = tuple(sorted(edge))
                edge_count[edge] += 1
                edge_to_nodes[edge] = edge

        # 找到边界边 (只出现一次的边)
        boundary_edges = [edge_to_nodes[edge] for edge, count in edge_count.items() if count == 1]
        boundary_nodes = set()
        for edge in boundary_edges:
            boundary_nodes.update(edge)

        return np.array(list(boundary_nodes))

    def _compute_geometric_features(self):
        """计算局部几何特征"""
        N = self.num_nodes
        coords = self.coordinates

        # 计算到最近壁面的距离
        wall_coords = coords[self.boundaries == 3]
        self.wall_distances = np.zeros(N)

        if len(wall_coords) > 0:
            for i in range(N):
                # 计算到所有壁面节点的距离
                distances = np.linalg.norm(coords[i] - wall_coords, axis=1)
                self.wall_distances[i] = distances.min()

        # 归一化距离特征
        dist_min, dist_max = self.wall_distances.min(), self.wall_distances.max()
        if dist_max > dist_min:
            self.wall_distances_normalized = 2 * (self.wall_distances - dist_min) / (dist_max - dist_min) - 1
        else:
            self.wall_distances_normalized = self.wall_distances * 0  # 全零

    def _create_graph(self, case_idx: int, time_idx: int) -> Graph:
        """创建图对象"""
        graph = Graph()

        # 基础属性
        graph.pos = torch.from_numpy(self.coordinates).float()  # [N, 2]
        graph.bound = torch.from_numpy(self.boundaries).long()   # [N]

        # 目标速度场
        graph.target = torch.from_numpy(
            self.velocity_data[case_idx, time_idx, :]
        ).float().unsqueeze(1)  # [N, 1]

        # 几何特征
        graph.loc = torch.from_numpy(
            self.wall_distances_normalized
        ).float().unsqueeze(1)  # [N, 1]

        # 边界类型编码 (one-hot)
        graph.omega = torch.zeros(self.num_nodes, 4)  # 4种边界类型
        for i, bound_type in enumerate([0, 1, 2, 3]):
            graph.omega[self.boundaries == bound_type, i] = 1.0

        # 批次索引
        graph.batch = torch.zeros(self.num_nodes, dtype=torch.long)

        return graph

    def _create_anchors(self, graph: Graph, case_idx: int, time_idx: int) -> dict:
        """创建锚点信息"""
        anchors = {}

        for downsample_rate in self.anchor_downsampling_rates:
            # 计算锚点数量
            num_anchors = max(1, self.num_nodes // downsample_rate)

            # 均匀采样锚点
            anchor_indices = np.linspace(0, self.num_nodes-1, num_anchors, dtype=int)

            # 获取锚点信息
            anchor_positions = graph.pos[anchor_indices]
            anchor_velocities = graph.target[anchor_indices]

            anchors[f'rate_{downsample_rate}'] = {
                'indices': anchor_indices,
                'positions': anchor_positions,
                'velocities': anchor_velocities
            }

        return anchors

    def __len__(self):
        """数据集长度"""
        if self.preload:
            return len(self.processed_data)
        else:
            return self.num_cases * self.num_time_steps

    def __getitem__(self, idx):
        """获取数据样本"""
        if self.preload:
            return self.processed_data[idx]
        else:
            case_idx = idx // self.num_time_steps
            time_idx = idx % self.num_time_steps

            graph = self._create_graph(case_idx, time_idx)

            # 创建锚点
            graph.anchors = self._create_anchors(graph, case_idx, time_idx)

            if self.transform:
                graph = self.transform(graph)

            return graph


def create_guide_vane_dataloader(
    mesh_path: str,
    dataset_path: str,
    batch_size: int = 32,
    T: int = 10,
    transform=None,
    **kwargs
) -> DataLoader:
    """创建导叶数据加载器"""
    dataset = GuideVaneVelocityField(
        mesh_path=mesh_path,
        dataset_path=dataset_path,
        T=T,
        transform=transform,
        **kwargs
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=None  # 使用默认的collate_fn
    )


# 使用示例
if __name__ == "__main__":
    # 示例用法
    dataset = GuideVaneVelocityField(
        mesh_path="sta_data_mesh.npy",
        dataset_path="sta_dataset.npy",
        T=10,
        preload=False
    )

    print(f"数据集大小: {len(dataset)}")

    # 测试数据加载
    sample = dataset[0]
    print(f"图属性: pos={sample.pos.shape}, bound={sample.bound.shape}")
    print(f"目标速度: {sample.target.shape}")
    print(f"几何特征: {sample.loc.shape}")
    print(f"边界编码: {sample.omega.shape}")