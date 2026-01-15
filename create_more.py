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

# ========== MCLP数据集生成器（增强版） ==========
class MCLPDatasetGenerator:
    """MCLP数据集生成器，特别针对文旅场景优化"""
    
    def __init__(self, 
                 num_nodes: int = 200,
                 dim: int = 2,
                 num_instances: int = 100,
                 coord_range: Tuple[float, float] = (0.0, 100.0),
                 device: torch.device = torch.device('cpu'),
                 include_population_data: bool = True,
                 include_road_network: bool = True,
                 include_tourism_features: bool = True,
                 generate_regions: bool = True,
                 urban_ratio: float = 0.6,
                 tourism_hotspots: int = 8):
        self.num_nodes = num_nodes
        self.dim = dim
        self.num_instances = num_instances
        self.min_val, self.max_val = coord_range
        self.range_size = self.max_val - self.min_val
        self.device = device
        self.include_population_data = include_population_data
        self.include_road_network = include_road_network
        self.include_tourism_features = include_tourism_features
        self.generate_regions = generate_regions
        self.urban_ratio = urban_ratio
        self.tourism_hotspots = tourism_hotspots
    def generate_tourism_distribution(self):
        """生成文旅场景分布：景点、道路、休息区"""
        n = self.num_nodes
        
        # 生成景点热点
        hotspots = []
        for _ in range(self.tourism_hotspots):
            hotspot = torch.tensor([
                self.min_val + torch.rand(1).item() * self.range_size,
                self.min_val + torch.rand(1).item() * self.range_size
            ], device=self.device)
            hotspots.append(hotspot)
        
        points_list = []
        labels_list = []
        tourism_features_list = []
        
        # 确保总节点数为num_nodes
        total_generated = 0
        
        # 为每个热点生成点（景点区域）
        points_per_hotspot = max(1, n // (len(hotspots) * 2))
        for i, hotspot in enumerate(hotspots):
            if total_generated >= n:
                break
                
            # 景点区域：密度较高
            n_cluster = min(points_per_hotspot, n - total_generated)
            cluster_points = hotspot + torch.randn(n_cluster, 2, device=self.device) * (self.range_size * 0.05)
            cluster_points = torch.clamp(cluster_points, self.min_val, self.max_val)
            points_list.append(cluster_points)
            labels_list.append(torch.ones(n_cluster, device=self.device, dtype=torch.long) * (i + 1))
            total_generated += n_cluster
            
            # 景点特征：高热度
            features = torch.ones(n_cluster, 3, device=self.device)
            features[:, 0] = 0.8 + torch.rand(n_cluster, device=self.device) * 0.2  # 景点热度
            features[:, 1] = 0.6 + torch.rand(n_cluster, device=self.device) * 0.4  # 交通便利度
            features[:, 2] = 0.7 + torch.rand(n_cluster, device=self.device) * 0.3  # 设施需求度
            tourism_features_list.append(features)
        
        # 生成道路连接线
        if total_generated < n:
            n_road = min(n // 4, n - total_generated)
            road_points = []
            
            for _ in range(3):  # 3条主要道路
                if len(road_points) >= n_road:
                    break
                    
                p1 = torch.tensor([self.min_val + torch.rand(1).item() * self.range_size,
                                self.min_val + torch.rand(1).item() * self.range_size], device=self.device)
                p2 = torch.tensor([self.min_val + torch.rand(1).item() * self.range_size,
                                self.min_val + torch.rand(1).item() * self.range_size], device=self.device)
                
                # 沿道路生成点
                n_points_per_road = max(1, n_road // 3)
                for t in torch.linspace(0, 1, n_points_per_road):
                    point = p1 * (1 - t) + p2 * t
                    noise = torch.randn(2, device=self.device) * (self.range_size * 0.02)
                    road_points.append(point + noise)
            
            if road_points:
                n_actual = min(len(road_points), n - total_generated)
                road_points_tensor = torch.stack(road_points[:n_actual])
                road_points_tensor = torch.clamp(road_points_tensor, self.min_val, self.max_val)
                points_list.append(road_points_tensor)
                labels_list.append(torch.zeros(n_actual, device=self.device, dtype=torch.long))
                total_generated += n_actual
                
                # 道路特征：高交通便利度
                features = torch.ones(n_actual, 3, device=self.device)
                features[:, 0] = 0.3 + torch.rand(n_actual, device=self.device) * 0.3
                features[:, 1] = 0.8 + torch.rand(n_actual, device=self.device) * 0.2
                features[:, 2] = 0.5 + torch.rand(n_actual, device=self.device) * 0.3
                tourism_features_list.append(features)
        
        # 生成一般区域（填充剩余点）
        if total_generated < n:
            n_general = n - total_generated
            general_points = torch.rand(n_general, 2, device=self.device) * self.range_size + self.min_val
            points_list.append(general_points)
            labels_list.append(torch.ones(n_general, device=self.device, dtype=torch.long) * -1)
            
            # 一般区域特征
            features = torch.ones(n_general, 3, device=self.device)
            features[:, 0] = torch.rand(n_general, device=self.device) * 0.4
            features[:, 1] = 0.3 + torch.rand(n_general, device=self.device) * 0.4
            features[:, 2] = 0.3 + torch.rand(n_general, device=self.device) * 0.4
            tourism_features_list.append(features)
            total_generated += n_general
        
        # 合并
        points = torch.cat(points_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        tourism_features = torch.cat(tourism_features_list, dim=0)
        
        # 确保节点数正确
        assert len(points) == n, f"节点数不一致: {len(points)} != {n}"
        
        # 打乱
        indices = torch.randperm(len(points), device=self.device)
        points = points[indices]
        labels = labels[indices]
        tourism_features = tourism_features[indices]
        
        return points, labels, tourism_features
    
    def generate_population_weights(self, points, labels=None, tourism_features=None):
        """生成人口密度权重（文旅场景优化）"""
        n = len(points)
        weights = torch.ones(n, device=self.device) * 0.3  # 基础权重
        
        # 如果有旅游特征，使用景点热度
        if tourism_features is not None:
            weights += tourism_features[:, 0] * 0.4  # 景点热度贡献
            weights += tourism_features[:, 2] * 0.3  # 设施需求度贡献
        
        # 如果有标签，调整权重
        if labels is not None:
            for i in range(1, self.tourism_hotspots + 1):
                mask = (labels == i)
                if mask.any():
                    weights[mask] = weights[mask] * 1.5  # 景点区域权重更高
        
        # 归一化到[0.1, 0.9]
        weights = torch.clamp(weights, 0.1, 0.9)
        if weights.max() > weights.min():
            weights = (weights - weights.min()) / (weights.max() - weights.min()) * 0.8 + 0.1
        
        return weights
    
    def generate_road_weights(self, points, labels=None, tourism_features=None):
        """生成道路可达性权重"""
        n = len(points)
        weights = torch.ones(n, device=self.device) * 0.2
        
        # 如果有旅游特征，使用交通便利度
        if tourism_features is not None:
            weights += tourism_features[:, 1] * 0.5  # 交通便利度贡献
        
        # 添加道路网络效果
        for _ in range(3):
            p1 = torch.tensor([self.min_val + torch.rand(1).item() * self.range_size,
                              self.min_val + torch.rand(1).item() * self.range_size], device=self.device)
            p2 = torch.tensor([self.min_val + torch.rand(1).item() * self.range_size,
                              self.min_val + torch.rand(1).item() * self.range_size], device=self.device)
            
            line_vec = p2 - p1
            line_len = torch.norm(line_vec)
            if line_len > 0:
                line_vec = line_vec / line_len
                points_vec = points - p1
                proj_len = torch.matmul(points_vec, line_vec)
                proj_len = torch.clamp(proj_len, 0, line_len)
                nearest_points = p1 + proj_len.unsqueeze(1) * line_vec
                dist = torch.norm(points - nearest_points, dim=1)
                
                road_weight = torch.exp(-dist / (self.range_size * 0.05)) * 0.2
                weights += road_weight
        
        # 归一化
        weights = torch.clamp(weights, 0.1, 0.9)
        if weights.max() > weights.min():
            weights = (weights - weights.min()) / (weights.max() - weights.min()) * 0.8 + 0.1
        
        return weights
    
    def generate_regions_labels(self, points):
        """生成区域标签"""
        points_np = points.cpu().numpy()
        n_regions = min(8, len(points) // 20)
        kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
        region_labels = kmeans.fit_predict(points_np)
        return torch.tensor(region_labels, device=self.device)
    
    def generate_instance(self, instance_id: int):
        """生成一个完整的MCLP实例"""
        seed = 42 + instance_id * 1000
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. 生成文旅场景分布
        points, spatial_labels, tourism_features = self.generate_tourism_distribution()
        
        # 2. 生成权重数据
        population_weights = None
        road_weights = None
        
        if self.include_population_data:
            population_weights = self.generate_population_weights(points, spatial_labels, tourism_features)
        
        if self.include_road_network:
            road_weights = self.generate_road_weights(points, spatial_labels, tourism_features)
        
        # 3. 计算总权重（文旅场景加权）
        total_weights = torch.ones(len(points), device=self.device) * 0.5
        
        if population_weights is not None:
            total_weights = total_weights * 0.3 + population_weights * 0.4
        
        if road_weights is not None:
            total_weights = total_weights + road_weights * 0.3
        
        if tourism_features is not None:
            total_weights = total_weights + tourism_features[:, 2] * 0.4  # 设施需求度
        
        # 归一化总权重
        total_weights = torch.clamp(total_weights, 0.1, 1.0)
        
        # 4. 生成区域标签
        region_labels = None
        if self.generate_regions:
            region_labels = self.generate_regions_labels(points)
        
        # 5. 计算距离矩阵
        distance_matrix = _pairwise_euclidean(points, points, self.device)
        
        # 6. 计算覆盖半径（基于文旅场景调整）
        avg_dist = distance_matrix[distance_matrix > 0].mean().item()
        coverage_radius = avg_dist * 0.25  # 更合理的覆盖半径
        
        # 7. 构建实例信息
        instance_info = {
            'name': f'mclp_tourism_{instance_id:04d}',
            'instance_id': instance_id,
            'points': points,
            'spatial_labels': spatial_labels,
            'tourism_features': tourism_features,
            'population_weights': population_weights,
            'road_weights': road_weights,
            'total_weights': total_weights,
            'region_labels': region_labels,
            'distance_matrix': distance_matrix,
            'coverage_radius': coverage_radius,
            'avg_distance': avg_dist,
            'diameter': distance_matrix.max().item(),
            'num_nodes': len(points),
            'generation_time': time.time()
        }
        
        return instance_info
    
    def generate_dataset(self, start_id: int = 0):
        """生成完整的数据集"""
        dataset = []
        
        print(f"正在生成文旅MCLP数据集...")
        print(f"参数: 节点数={self.num_nodes}, 实例数={self.num_instances}, 景点数={self.tourism_hotspots}")
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

# ========== 图构建函数（优化版） ==========
def build_mclp_graph(instance, graph_threshold=None, distance_metric='euclidean'):
    """为MCLP问题构建图（增强特征）"""
    points = instance['points']
    
    # 使用预计算的距离矩阵或重新计算
    if 'distance_matrix' in instance:
        dist = instance['distance_matrix']
    else:
        dist_func = _get_distance_func(distance_metric)
        dist = dist_func(points, points, points.device)
    
    # 确定图构建阈值（基于文旅场景）
    if graph_threshold is None:
        avg_dist = dist[dist > 0].mean().item()
        graph_threshold = avg_dist * 0.3
    
    # 构建边（双向连接）
    edge_mask = dist <= graph_threshold
    edge_indices = torch.nonzero(edge_mask, as_tuple=False).t()
    
    # 计算边权重（距离的倒数，归一化）
    edge_dists = dist[edge_indices[0], edge_indices[1]]
    edge_attrs = 1.0 / (edge_dists + 1e-6)
    edge_attrs = edge_attrs / edge_attrs.max() if edge_attrs.max() > 0 else edge_attrs
    
    # 构建节点特征（增强）
    node_features_list = [points]
    
    if instance.get('tourism_features') is not None:
        node_features_list.append(instance['tourism_features'])
    
    if instance.get('population_weights') is not None:
        node_features_list.append(instance['population_weights'].unsqueeze(1))
    
    if instance.get('road_weights') is not None:
        node_features_list.append(instance['road_weights'].unsqueeze(1))
    
    if instance.get('total_weights') is not None:
        node_features_list.append(instance['total_weights'].unsqueeze(1))
    
    node_features = torch.cat(node_features_list, dim=1)
    
    # 计算节点重要性特征
    dist_row_sum = torch.sum(dist, dim=1, keepdim=True)
    dist_row_norm = dist_row_sum / dist_row_sum.max() if dist_row_sum.max() > 0 else dist_row_sum
    
    # 计算度数特征
    if edge_indices.shape[1] > 0:
        degree = torch.zeros(len(points), dtype=torch.long, device=points.device)
        unique, counts = torch.unique(edge_indices[0], return_counts=True)
        degree[unique] = counts
        degree = degree.float().unsqueeze(1)
        degree_norm = degree / degree.max() if degree.max() > 0 else degree
    else:
        degree = torch.ones(len(points), 1, device=points.device)
        degree_norm = degree
    
    # 创建图数据对象
    graph_data = pyg.data.Data(
        x=node_features,
        edge_index=edge_indices,
        edge_attr=edge_attrs.unsqueeze(1),
        pos=points,
        dist_row_sum=dist_row_norm,
        degree=degree_norm,
        tourism_features=instance.get('tourism_features'),
        population_weights=instance.get('population_weights'),
        road_weights=instance.get('road_weights'),
        total_weights=instance.get('total_weights'),
        spatial_labels=instance.get('spatial_labels'),
        region_labels=instance.get('region_labels'),
        distance_matrix=dist,
        coverage_radius=instance.get('coverage_radius', graph_threshold * 2),
        graph_threshold=graph_threshold,
        diameter=dist.max().item(),
        avg_distance=instance.get('avg_distance', dist[dist > 0].mean().item()),
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
        print(f"实例 {i}: {instance['name']}, 节点数: {len(instance['points'])}, 覆盖半径: {instance.get('coverage_radius', 'N/A'):.2f}")

# ========== 主程序 ==========
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成文旅场景数据集
    print("\n生成文旅MCLP数据集...")
    test_generator = MCLPDatasetGenerator(
        num_nodes=200,
        num_instances=50,
        device=device,
        include_tourism_features=True,
        tourism_hotspots=8
    )
    
    test_dataset = test_generator.generate_dataset()
    save_dataset(test_dataset, 'mclp_tourism_test_50.pkl')
    
    # 分析数据集
    analyze_dataset(test_dataset)
    
    print("\n数据集生成完成！")