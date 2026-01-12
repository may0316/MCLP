import numpy as np
import torch
import torch_geometric as pyg
import time
import pickle
from typing import Tuple, List, Optional

def build_graph_from_points(points, dist=None, return_dist=False, distance_metric='euclidean'):
    """
    从点构建图
    points: 节点坐标 [n, dim]
    dist: 预计算的距离矩阵 [n, n]
    return_dist: 是否返回距离矩阵
    distance_metric: 距离度量方式
    """
    if dist is None:
        dist_func = _get_distance_func(distance_metric)
        dist = dist_func(points, points, points.device)
    
    # 计算图构建阈值（动态调整，基于数据范围）
    diameter = dist.max().item()
    threshold = 0.05 * diameter  # 使用直径的5%作为阈值
    
    edge_indices = torch.nonzero(dist <= threshold, as_tuple=False).transpose(0, 1)
    edge_attrs = dist[torch.nonzero(dist <= threshold, as_tuple=True)]
    edge_attrs = 1/(edge_attrs+1e-3)
    dist_row = torch.sum(dist, dim=1)
    dist_row = dist_row.unsqueeze(1)
    
    g = pyg.data.Data(x=points, edge_index=edge_indices, edge_attr=edge_attrs, dist_row=dist_row)
    
    if return_dist:
        return g, dist
    else:
        return g

def _get_distance_func(distance):
    """获取距离计算函数"""
    if distance == 'euclidean':
        return _pairwise_euclidean
    elif distance == 'cosine':
        return _pairwise_cosine
    elif distance == 'manhattan':
        return _pairwise_manhattan
    else:
        raise NotImplementedError(f"Distance metric '{distance}' not implemented")

def _pairwise_euclidean(data1, data2, device=torch.device('cpu')):
    """计算欧几里得距离"""
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=-2)
    B = data2.unsqueeze(dim=-3)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1)
    dis = torch.sqrt(dis)
    return dis

def _pairwise_manhattan(data1, data2, device=torch.device('cpu')):
    """计算曼哈顿距离"""
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=-2)
    B = data2.unsqueeze(dim=-3)
    dis = torch.abs(A - B)
    dis = dis.sum(dim=-1)
    return dis

def _pairwise_cosine(data1, data2, device=torch.device('cpu')):
    """计算余弦距离"""
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=-2)
    B = data2.unsqueeze(dim=-3)
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)
    cosine = A_normalized * B_normalized
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze(-1)
    return cosine_dis

class MCLPDatasetGenerator:
    """MCLP数据集生成器，组合多种分布"""
    
    def __init__(self, 
                 num_nodes: int = 500,
                 dim: int = 2,
                 num_instances: int = 800,
                 coord_range: Tuple[float, float] = (0.0, 10.0),
                 device: torch.device = torch.device('cpu')):
        """
        初始化生成器
        
        参数:
            num_nodes: 每个实例的节点数
            dim: 维度
            num_instances: 实例数量
            coord_range: 坐标范围 (min, max)
            device: 计算设备
        """
        self.num_nodes = num_nodes
        self.dim = dim
        self.num_instances = num_instances
        self.min_val, self.max_val = coord_range
        self.range_size = self.max_val - self.min_val
        self.device = device
        
    def generate_uniform_distribution(self) -> torch.Tensor:
        """生成均匀分布的点"""
        return torch.rand(self.num_nodes, self.dim, device=self.device) * self.range_size + self.min_val
    
    def generate_clustered_distribution(self, num_clusters: int = 8) -> torch.Tensor:
        """生成聚类分布的点"""
        # 生成聚类中心
        cluster_centers = torch.rand(num_clusters, self.dim, device=self.device) * self.range_size + self.min_val
        
        points_list = []
        points_per_cluster = self.num_nodes // num_clusters
        remaining = self.num_nodes
        
        for i, center in enumerate(cluster_centers):
            if i == num_clusters - 1:
                cluster_size = remaining
            else:
                cluster_size = points_per_cluster
                remaining -= cluster_size
            
            # 聚类内的点呈正态分布
            std = self.range_size * 0.08  # 标准差为范围的8%
            cluster_points = center + torch.randn(cluster_size, self.dim, device=self.device) * std
            cluster_points = torch.clamp(cluster_points, self.min_val, self.max_val)
            points_list.append(cluster_points)
        
        points = torch.cat(points_list, dim=0)
        return points
    
    def generate_line_distribution(self) -> torch.Tensor:
        """生成线状分布的点（模拟街道）"""
        # 随机选择一条线
        if self.dim == 2:
            # 随机生成直线方程: y = kx + b
            k = torch.randn(1, device=self.device).item() * 2
            b = torch.rand(1, device=self.device).item() * self.range_size + self.min_val
            
            # 在x轴上均匀采样
            x = torch.rand(self.num_nodes, 1, device=self.device) * self.range_size + self.min_val
            y = k * x + b + torch.randn(self.num_nodes, 1, device=self.device) * (self.range_size * 0.05)
            
            points = torch.cat([x, y], dim=1)
            points = torch.clamp(points, self.min_val, self.max_val)
        else:
            # 高维情况使用主成分分析方向
            points = torch.rand(self.num_nodes, self.dim, device=self.device) * self.range_size + self.min_val
            # 使数据在第一主成分方向上有更强的相关性
            for i in range(1, self.dim):
                points[:, i] = points[:, 0] * 0.7 + points[:, i] * 0.3
        
        return points
    
    def generate_grid_distribution(self) -> torch.Tensor:
        """生成网格分布的点"""
        if self.dim == 2:
            # 计算网格大小
            grid_size = int(np.sqrt(self.num_nodes))
            actual_nodes = grid_size * grid_size
            
            # 创建网格
            x = torch.linspace(self.min_val + 0.1*self.range_size, 
                              self.max_val - 0.1*self.range_size, 
                              grid_size, device=self.device)
            y = torch.linspace(self.min_val + 0.1*self.range_size,
                              self.max_val - 0.1*self.range_size,
                              grid_size, device=self.device)
            
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            
            # 添加随机扰动
            points += torch.randn_like(points) * (self.range_size * 0.02)
            
            # 如果点数不够，补充随机点
            if len(points) < self.num_nodes:
                extra = torch.rand(self.num_nodes - len(points), 2, device=self.device) * self.range_size + self.min_val
                points = torch.cat([points, extra], dim=0)
            
            return points
        else:
            return self.generate_uniform_distribution()
    
    def generate_ring_distribution(self) -> torch.Tensor:
        """生成环状分布的点"""
        if self.dim == 2:
            center = torch.tensor([self.min_val + self.range_size/2, 
                                  self.min_val + self.range_size/2], device=self.device)
            
            # 生成半径（在中间范围）
            min_radius = self.range_size * 0.2
            max_radius = self.range_size * 0.4
            radii = torch.rand(self.num_nodes, device=self.device) * (max_radius - min_radius) + min_radius
            
            # 生成角度
            angles = torch.rand(self.num_nodes, device=self.device) * 2 * np.pi
            
            # 转换为直角坐标
            x = center[0] + radii * torch.cos(angles)
            y = center[1] + radii * torch.sin(angles)
            
            points = torch.stack([x, y], dim=1)
            
            # 添加一些随机点填充中心
            num_center = self.num_nodes // 5
            center_points = torch.rand(num_center, 2, device=self.device) * (self.range_size * 0.3) + \
                           center - self.range_size * 0.15
            points = torch.cat([points, center_points], dim=0)[:self.num_nodes]
            
            return points
        else:
            return self.generate_uniform_distribution()
    
    def generate_mixed_distribution(self, method_weights=None) -> torch.Tensor:
        """
        生成混合分布的点
        
        参数:
            method_weights: 各种生成方法的权重
        """
        if method_weights is None:
            method_weights = {
                'uniform': 0.25,
                'clustered': 0.25,
                'line': 0.20,
                'grid': 0.15,
                'ring': 0.15
            }
        
        # 选择生成方法
        methods = list(method_weights.keys())
        weights = list(method_weights.values())
        
        # 根据权重随机选择方法
        method_idx = torch.multinomial(torch.tensor(weights), 1).item()
        method = methods[method_idx]
        
        # 调用对应的方法
        if method == 'uniform':
            return self.generate_uniform_distribution()
        elif method == 'clustered':
            # 随机选择聚类数量
            num_clusters = torch.randint(3, 12, (1,)).item()
            return self.generate_clustered_distribution(num_clusters)
        elif method == 'line':
            return self.generate_line_distribution()
        elif method == 'grid':
            return self.generate_grid_distribution()
        elif method == 'ring':
            return self.generate_ring_distribution()
        else:
            return self.generate_uniform_distribution()
    
    def generate_instance(self, instance_id: int, 
                         distribution_type: str = 'mixed') -> Tuple[str, torch.Tensor]:
        """
        生成一个实例
        
        参数:
            instance_id: 实例ID
            distribution_type: 分布类型，可以是 'uniform', 'clustered', 'line', 'grid', 'ring', 'mixed'
        """
        # 设置随机种子确保可重复性
        seed = 42 + instance_id * 1000
        torch.random.manual_seed(seed)
        
        if distribution_type == 'uniform':
            points = self.generate_uniform_distribution()
        elif distribution_type == 'clustered':
            points = self.generate_clustered_distribution()
        elif distribution_type == 'line':
            points = self.generate_line_distribution()
        elif distribution_type == 'grid':
            points = self.generate_grid_distribution()
        elif distribution_type == 'ring':
            points = self.generate_ring_distribution()
        elif distribution_type == 'mixed':
            points = self.generate_mixed_distribution()
        else:
            raise ValueError(f"未知的分布类型: {distribution_type}")
        
        return f"mclp_{instance_id:04d}", points
    
    def generate_dataset(self, distribution_type: str = 'mixed') -> List[Tuple[str, torch.Tensor]]:
        """
        生成完整的数据集
        
        参数:
            distribution_type: 分布类型
        """
        dataset = []
        
        print(f"正在生成MCLP数据集...")
        print(f"参数: 节点数={self.num_nodes}, 实例数={self.num_instances}, 坐标范围=[{self.min_val}, {self.max_val}]")
        print(f"分布类型: {distribution_type}")
        
        for i in range(self.num_instances):
            if i % 100 == 0:
                print(f"  已生成 {i}/{self.num_instances} 个实例")
            
            name, points = self.generate_instance(i, distribution_type)
            dataset.append((name, points))
        
        print(f"数据集生成完成！共生成 {len(dataset)} 个实例")
        return dataset

# ============================================================================
# 为了向后兼容，保留旧的API函数
# ============================================================================

def get_random_data(num_data, dim, seed, device, num_instances=1):
    """
    保持向后兼容的函数（用于pre_mclp.py）
    参数:
        num_data: 节点数
        dim: 维度
        seed: 随机种子
        device: 设备
        num_instances: 实例数量（默认1，保持旧行为）
    """
    torch.random.manual_seed(seed)
    dataset = [(f'rand{_}', torch.rand(num_data, dim, device=device)) for _ in range(num_instances)]
    return dataset

def get_starbucks_data(device):
    """旧的星巴克数据加载函数"""
    dataset = []
    areas = ['london', 'newyork', 'shanghai', 'seoul']
    for area in areas:
        try:
            with open(f'data/starbucks/{area}.csv', encoding='utf-8-sig') as f:
                locations = []
                for l in f.readlines():
                    l_str = l.strip().split(',')
                    if l_str[0] == 'latitude' and l_str[1] == 'longitude':
                        continue
                    n1, n2 = float(l_str[0]) / 365 * 400, float(l_str[1]) / 365 * 400
                    locations.append((n1, n2))
            dataset.append((area, torch.tensor(locations, device=device)))
        except FileNotFoundError:
            print(f"警告: 找不到文件 data/starbucks/{area}.csv")
    return dataset

def save_dataset(dataset: List[Tuple[str, torch.Tensor]], file_path: str):
    """保存数据集到文件"""
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)
    print(f"数据集已保存到: {file_path}")

def load_dataset(file_path: str) -> List[Tuple[str, torch.Tensor]]:
    """从文件加载数据集"""
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    print(f"从 {file_path} 加载了 {len(dataset)} 个实例")
    return dataset

def analyze_dataset(dataset: List[Tuple[str, torch.Tensor]], 
                   max_instances_to_analyze: int = 5):
    """分析数据集的基本统计信息"""
    print("\n" + "="*60)
    print("数据集分析报告")
    print("="*60)
    
    if not dataset:
        print("数据集为空！")
        return
    
    total_instances = len(dataset)
    print(f"总实例数: {total_instances}")
    
    # 分析前几个实例
    for i in range(min(max_instances_to_analyze, total_instances)):
        name, points = dataset[i]
        
        # 计算基本统计
        dist_matrix = _pairwise_euclidean(points, points, points.device)
        diameter = dist_matrix.max().item()
        avg_dist = dist_matrix.mean().item()
        std_dist = dist_matrix.std().item()
        
        # MCLP相关指标
        coverage_radius = 0.15 * diameter  # MCLP中通常使用直径的15%作为覆盖半径
        graph_threshold = 0.05 * diameter  # 图构建阈值
        
        print(f"\n实例 {name}:")
        print(f"  坐标范围: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
              f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  直径: {diameter:.3f}")
        print(f"  平均距离: {avg_dist:.3f} ± {std_dist:.3f}")
        print(f"  MCLP覆盖半径: {coverage_radius:.3f} (直径的15%)")
        print(f"  图构建阈值: {graph_threshold:.3f} (直径的5%)")
        print(f"  是否适合MCLP: {'✓' if coverage_radius > graph_threshold else '✗'}")
        
        # 计算点的密度
        area = (points[:, 0].max() - points[:, 0].min()) * (points[:, 1].max() - points[:, 1].min())
        density = len(points) / (area + 1e-10)
        print(f"  点密度: {density:.3f} 点/单位面积")

# 主程序：生成数据集
if __name__ == "__main__":
    device = torch.device('cpu')
    
    print("="*60)
    print("MCLP数据集生成器")
    print("="*60)
    
    # 1. 生成用于预训练的数据（1个实例，[0,1]范围，保持与pre_mclp.py兼容）
    print("\n1. 生成预训练数据集...")
    pretrain_dataset = get_random_data(500, 2, 0, device, num_instances=1)
    save_dataset(pretrain_dataset, 'pretrain_dataset.pkl')
    analyze_dataset(pretrain_dataset)
    
    # 2. 生成用于MCLP求解的数据（800个实例，[0,10]范围，多样化分布）
    print("\n2. 生成MCLP求解数据集...")
    generator = MCLPDatasetGenerator(
        num_nodes=500,
        dim=2,
        num_instances=800,
        coord_range=(0.0, 10.0),
        device=device
    )
    
    mclp_dataset = generator.generate_dataset(distribution_type='mixed')
    save_dataset(mclp_dataset, 'dataset_800.pkl')
    analyze_dataset(mclp_dataset, max_instances_to_analyze=5)
    
    print("\n" + "="*60)
    print("所有数据集生成完成！")
    print("="*60)
    print("生成的文件:")
    print("  1. pretrain_dataset.pkl - 用于预训练（1个实例）")
    print("  2. dataset_800.pkl - 用于MCLP求解（800个实例）")
    print("\n使用方法:")
    print("  - pre_mclp.py: 使用 pretrain_dataset.pkl 或 get_random_data()")
    print("  - mclp.py: 使用 dataset_800.pkl")