import numpy as np
import torch
import torch_geometric as pyg
import time
import pickle
from typing import Tuple, List, Optional, Dict, Any
from sklearn.cluster import KMeans

# ========== 距离计算函数 ==========
def _get_distance_func(distance_metric: str):
    """获取距离计算函数"""
    if distance_metric == 'euclidean':
        return _pairwise_euclidean
    elif distance_metric == 'cosine':
        return _pairwise_cosine
    elif distance_metric == 'manhattan':
        return _pairwise_manhattan
    else:
        raise NotImplementedError(f"距离度量 '{distance_metric}' 未实现")

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

# ========== MCLP数据集生成器 ==========
class MCLPDatasetGenerator:
    """MCLP数据集生成器，模拟北京六环内的地理空间特征"""
    
    def __init__(self, 
                 num_nodes: int = 500,
                 dim: int = 2,
                 num_instances: int = 800,
                 coord_range: Tuple[float, float] = (0.0, 100.0),
                 device: torch.device = torch.device('cpu'),
                 include_population_data: bool = True,
                 include_road_network: bool = True,
                 generate_regions: bool = True,
                 urban_ratio: float = 0.6):
        self.num_nodes = num_nodes
        self.dim = dim
        self.num_instances = num_instances
        self.min_val, self.max_val = coord_range
        self.range_size = self.max_val - self.min_val
        self.device = device
        self.include_population_data = include_population_data
        self.include_road_network = include_road_network
        self.generate_regions = generate_regions
        self.urban_ratio = urban_ratio
        
    def generate_beijing_like_distribution(self):
        """生成模拟北京六环内分布的点"""
        # 简化版本：生成随机点，但加入一些聚集效果
        n = self.num_nodes
        
        # 生成几个聚集中心
        centers = []
        for _ in range(5):
            center = torch.tensor([
                self.min_val + torch.rand(1).item() * self.range_size,
                self.min_val + torch.rand(1).item() * self.range_size
            ], device=self.device)
            centers.append(center)
        
        points_list = []
        labels_list = []
        
        # 为每个中心生成点
        points_per_center = n // len(centers)
        for i, center in enumerate(centers):
            # 生成围绕中心的点
            cluster_points = center + torch.randn(points_per_center, 2, device=self.device) * (self.range_size * 0.1)
            cluster_points = torch.clamp(cluster_points, self.min_val, self.max_val)
            points_list.append(cluster_points)
            labels_list.append(torch.ones(points_per_center, device=self.device, dtype=torch.long) * (i + 1))
        
        # 生成一些随机点
        remaining = n - len(points_list) * points_per_center
        if remaining > 0:
            random_points = torch.rand(remaining, 2, device=self.device) * self.range_size + self.min_val
            points_list.append(random_points)
            labels_list.append(torch.zeros(remaining, device=self.device, dtype=torch.long))
        
        # 合并所有点
        points = torch.cat(points_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # 打乱
        indices = torch.randperm(len(points), device=self.device)
        points = points[indices]
        labels = labels[indices]
        
        return points, labels
    
    def generate_population_weights(self, points, labels=None):
        """生成人口密度权重"""
        n = len(points)
        # 生成基础权重
        weights = torch.rand(n, device=self.device) * 0.7 + 0.1  # 范围[0.1, 0.8]
        
        # 如果有标签，调整权重
        if labels is not None:
            for i in range(1, 6):  # 假设有5个类别
                mask = (labels == i)
                if mask.any():
                    weights[mask] = weights[mask] * (1 + i * 0.1)  # 类别越高权重越高
        
        # 归一化到[0, 0.9]
        weights = torch.clamp(weights / weights.max() * 0.9, 0, 0.9)
        return weights
    
    def generate_road_weights(self, points, labels=None):
        """生成道路可达性权重"""
        n = len(points)
        weights = torch.rand(n, device=self.device) * 0.08 + 0.01  # 范围[0.01, 0.09]
        
        # 添加一些道路网络效果
        # 模拟几条主干道
        for _ in range(3):
            # 随机直线
            p1 = torch.tensor([self.min_val + torch.rand(1).item() * self.range_size,
                              self.min_val + torch.rand(1).item() * self.range_size], device=self.device)
            p2 = torch.tensor([self.min_val + torch.rand(1).item() * self.range_size,
                              self.min_val + torch.rand(1).item() * self.range_size], device=self.device)
            
            # 计算点到直线的距离
            line_vec = p2 - p1
            line_len = torch.norm(line_vec)
            if line_len > 0:
                line_vec = line_vec / line_len
                points_vec = points - p1
                proj_len = torch.matmul(points_vec, line_vec)
                proj_len = torch.clamp(proj_len, 0, line_len)
                nearest_points = p1 + proj_len.unsqueeze(1) * line_vec
                dist = torch.norm(points - nearest_points, dim=1)
                
                # 距离越近权重越高
                road_weight = torch.exp(-dist / (self.range_size * 0.1)) * 0.02
                weights += road_weight
        
        # 归一化到[0, 0.1]
        weights = torch.clamp(weights / weights.max() * 0.1, 0, 0.1)
        return weights
    
    def generate_regions_labels(self, points):
        """生成区域标签"""
        points_np = points.cpu().numpy()
        n_regions = min(6, len(points) // 10)  # 确保有足够点
        kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
        region_labels = kmeans.fit_predict(points_np)
        return torch.tensor(region_labels, device=self.device)
    
    def generate_instance(self, instance_id: int):
        """生成一个完整的MCLP实例"""
        seed = 42 + instance_id * 1000
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. 生成空间点分布
        points, spatial_labels = self.generate_beijing_like_distribution()
        
        # 2. 生成权重数据
        population_weights = None
        road_weights = None
        
        if self.include_population_data:
            population_weights = self.generate_population_weights(points, spatial_labels)
        
        if self.include_road_network:
            road_weights = self.generate_road_weights(points, spatial_labels)
        
        # 3. 计算总权重
        if population_weights is not None and road_weights is not None:
            total_weights = population_weights + road_weights
        elif population_weights is not None:
            total_weights = population_weights
        elif road_weights is not None:
            total_weights = road_weights
        else:
            total_weights = torch.ones(len(points), device=self.device) * 0.5
        
        # 4. 生成区域标签
        region_labels = None
        if self.generate_regions:
            region_labels = self.generate_regions_labels(points)
        
        # 5. 计算距离矩阵
        distance_matrix = _pairwise_euclidean(points, points, self.device)
        
        # 6. 计算覆盖半径
        diameter = distance_matrix.max().item()
        coverage_radius = diameter * 0.15
        
        # 7. 构建实例信息
        instance_info = {
            'name': f'mclp_beijing_{instance_id:04d}',
            'instance_id': instance_id,
            'points': points,
            'spatial_labels': spatial_labels,
            'population_weights': population_weights,
            'road_weights': road_weights,
            'total_weights': total_weights,
            'region_labels': region_labels,
            'distance_matrix': distance_matrix,
            'coverage_radius': coverage_radius,
            'diameter': diameter,
            'num_nodes': len(points),
            'generation_time': time.time()
        }
        
        return instance_info
    
    def generate_dataset(self, start_id: int = 0):
        """生成完整的数据集"""
        dataset = []
        
        print(f"正在生成MCLP数据集...")
        print(f"参数: 节点数={self.num_nodes}, 实例数={self.num_instances}")
        print("-" * 60)
        
        start_time = time.time()
        
        for i in range(start_id, start_id + self.num_instances):
            if i % 20 == 0 and i > start_id:
                elapsed = time.time() - start_time
                print(f"  已生成 {i - start_id}/{self.num_instances} 个实例")
            
            instance = self.generate_instance(i)
            dataset.append(instance)
        
        total_time = time.time() - start_time
        print(f"\n数据集生成完成！")
        print(f"共生成 {len(dataset)} 个实例，总耗时: {total_time:.2f}秒")
        
        return dataset

# ========== 图构建函数 ==========
def build_mclp_graph(instance, graph_threshold=None, distance_metric='euclidean'):
    """为MCLP问题构建图"""
    points = instance['points']
    
    # 使用预计算的距离矩阵或重新计算
    if 'distance_matrix' in instance:
        dist = instance['distance_matrix']
    else:
        dist_func = _get_distance_func(distance_metric)
        dist = dist_func(points, points, points.device)
    
    # 确定图构建阈值
    if graph_threshold is None:
        diameter = dist.max().item()
        graph_threshold = diameter * 0.05
    
    # 构建边
    edge_indices = torch.nonzero(dist <= graph_threshold, as_tuple=False).transpose(0, 1)
    
    # 计算边权重
    edge_attrs = dist[edge_indices[0], edge_indices[1]]
    edge_attrs = 1.0 / (edge_attrs + 1e-6)
    
    # 构建节点特征
    node_features_list = [points]
    
    if instance.get('population_weights') is not None:
        node_features_list.append(instance['population_weights'].unsqueeze(1))
    
    if instance.get('road_weights') is not None:
        node_features_list.append(instance['road_weights'].unsqueeze(1))
    
    if instance.get('total_weights') is not None:
        node_features_list.append(instance['total_weights'].unsqueeze(1))
    
    node_features = torch.cat(node_features_list, dim=1)
    
    # 计算距离行的和
    dist_row_sum = torch.sum(dist, dim=1, keepdim=True)
    
    # 创建图数据对象
    graph_data = pyg.data.Data(
        x=node_features,
        edge_index=edge_indices,
        edge_attr=edge_attrs.unsqueeze(1),
        pos=points,
        dist_row_sum=dist_row_sum,
        population_weights=instance.get('population_weights'),
        road_weights=instance.get('road_weights'),
        total_weights=instance.get('total_weights'),
        spatial_labels=instance.get('spatial_labels'),
        region_labels=instance.get('region_labels'),
        distance_matrix=dist,
        coverage_radius=instance.get('coverage_radius', graph_threshold * 3),
        graph_threshold=graph_threshold,
        diameter=dist.max().item(),
        instance_name=instance['name'],
        instance_id=instance['instance_id']
    )
    
    return graph_data

# ========== 数据集工具函数 ==========
def save_dataset(dataset, file_path: str):
    """保存数据集到文件"""
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"数据集已保存到: {file_path}")
    print(f"  实例数: {len(dataset)}")

def load_dataset(file_path: str):
    """从文件加载数据集"""
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    
    print(f"从 {file_path} 加载了 {len(dataset)} 个实例")
    
    # 转换回CPU
    for instance in dataset:
        for key, value in instance.items():
            if torch.is_tensor(value):
                instance[key] = value.cpu()
    
    return dataset

# ========== 其他函数 ==========
def analyze_dataset(dataset, max_instances_to_analyze=3):
    """分析数据集"""
    if not dataset:
        print("数据集为空")
        return
    
    print(f"数据集包含 {len(dataset)} 个实例")
    for i, instance in enumerate(dataset[:max_instances_to_analyze]):
        print(f"实例 {i}: {instance['name']}, 节点数: {len(instance['points'])}")

def get_random_data(num_data, dim, seed, device, num_instances=1):
    """生成随机数据"""
    torch.random.manual_seed(seed)
    dataset = []
    
    for i in range(num_instances):
        points = torch.rand(num_data, dim, device=device)
        dataset.append({
            'name': f'rand_{i:04d}',
            'instance_id': i,
            'points': points,
            'num_nodes': num_data,
            'distance_matrix': _pairwise_euclidean(points, points, device),
            'total_weights': torch.ones(num_data, device=device) * 0.5
        })
    
    return dataset

# ========== 主程序 ==========
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成测试数据集
    print("\n生成测试数据集...")
    test_generator = MCLPDatasetGenerator(
        num_nodes=200,
        num_instances=50,
        device=device
    )
    
    test_dataset = test_generator.generate_dataset()
    save_dataset(test_dataset, 'mclp_beijing_test_50.pkl')
    
    print("\n数据集生成完成！")