# generate_synthetic_data.py
import torch
import numpy as np
import time
import pickle
import os
from typing import Tuple
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _pairwise_euclidean(data1, data2, device=torch.device('cpu')):
    """计算欧几里得距离"""
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=-2)
    B = data2.unsqueeze(dim=-3)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1)
    dis = torch.sqrt(dis)
    return dis

class SyntheticDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self,
                 num_nodes: int = 200,
                 num_instances: int = 100,
                 coord_range: Tuple[float, float] = (0.0, 2000.0),
                 device: torch.device = torch.device('cpu'),
                 service_radius: float = 200.0,
                 scenic_areas: int = 5,
                 rest_areas: int = 3):
        self.num_nodes = num_nodes
        self.num_instances = num_instances
        self.min_val, self.max_val = coord_range
        self.range_size = self.max_val - self.min_val
        self.device = device
        self.service_radius = service_radius
        self.scenic_areas = scenic_areas
        self.rest_areas = rest_areas

    def generate_scene_distribution(self):
        """生成更真实的文旅场景分布"""
        n = self.num_nodes
        
        # 生成景点中心
        scenic_centers = []
        for i in range(self.scenic_areas):
            while True:
                center = torch.tensor([
                    self.min_val + torch.rand(1).item() * self.range_size,
                    self.min_val + torch.rand(1).item() * self.range_size
                ], device=self.device)
                
                if len(scenic_centers) == 0:
                    break
                min_dist = min([torch.norm(center - c).item() for c in scenic_centers])
                if min_dist > self.service_radius * 1.5:
                    break
            scenic_centers.append(center)
        
        # 生成休息区中心
        rest_centers = []
        for _ in range(self.rest_areas):
            center = torch.tensor([
                self.min_val + torch.rand(1).item() * self.range_size,
                self.min_val + torch.rand(1).item() * self.range_size
            ], device=self.device)
            rest_centers.append(center)
        
        points_list = []
        demand_weights_list = []
        scenic_labels_list = []
        
        total_generated = 0
        
        # 1. 景点区域
        points_per_scenic = max(12, n // (self.scenic_areas * 2))
        for i, center in enumerate(scenic_centers):
            if total_generated >= n:
                break
            
            n_cluster = min(points_per_scenic, n - total_generated)
            cluster_points = center + torch.randn(n_cluster, 2, device=self.device) * (self.range_size * 0.05)
            cluster_points = torch.clamp(cluster_points, self.min_val, self.max_val)
            points_list.append(cluster_points)
            scenic_labels_list.append(torch.ones(n_cluster, device=self.device, dtype=torch.long) * (i + 1))
            
            weights = 15.0 + torch.randn(n_cluster, device=self.device) * 3.0
            weights = torch.clamp(weights, 8.0, 25.0)
            demand_weights_list.append(weights)
            total_generated += n_cluster
        
        # 2. 休息区区域
        if total_generated < n:
            points_per_rest = max(8, n // (self.rest_areas * 3))
            for center in rest_centers:
                if total_generated >= n:
                    break
                
                n_cluster = min(points_per_rest, n - total_generated)
                cluster_points = center + torch.randn(n_cluster, 2, device=self.device) * (self.range_size * 0.06)
                cluster_points = torch.clamp(cluster_points, self.min_val, self.max_val)
                points_list.append(cluster_points)
                scenic_labels_list.append(torch.zeros(n_cluster, device=self.device, dtype=torch.long))
                
                weights = 8.0 + torch.randn(n_cluster, device=self.device) * 2.0
                weights = torch.clamp(weights, 4.0, 12.0)
                demand_weights_list.append(weights)
                total_generated += n_cluster
        
        # 3. 道路沿线
        if total_generated < n:
            n_road = min(n // 4, n - total_generated)
            road_points = []
            
            num_roads = np.random.randint(2, 4)
            for _ in range(num_roads):
                if len(road_points) >= n_road:
                    break
                    
                p1 = torch.tensor([self.min_val + torch.rand(1).item() * self.range_size,
                                   self.min_val + torch.rand(1).item() * self.range_size], device=self.device)
                p2 = torch.tensor([self.min_val + torch.rand(1).item() * self.range_size,
                                   self.min_val + torch.rand(1).item() * self.range_size], device=self.device)
                
                n_points_per_road = max(4, n_road // num_roads)
                for t in torch.linspace(0, 1, n_points_per_road):
                    point = p1 * (1 - t) + p2 * t
                    noise = torch.randn(2, device=self.device) * (self.range_size * 0.01)
                    road_points.append(point + noise)
            
            if road_points:
                n_actual = min(len(road_points), n - total_generated)
                road_points_tensor = torch.stack(road_points[:n_actual])
                road_points_tensor = torch.clamp(road_points_tensor, self.min_val, self.max_val)
                points_list.append(road_points_tensor)
                scenic_labels_list.append(torch.zeros(n_actual, device=self.device, dtype=torch.long))
                
                weights = 4.0 + torch.randn(n_actual, device=self.device) * 1.0
                weights = torch.clamp(weights, 2.0, 6.0)
                demand_weights_list.append(weights)
                total_generated += n_actual
        
        # 4. 一般区域
        if total_generated < n:
            n_general = n - total_generated
            general_points = torch.rand(n_general, 2, device=self.device) * self.range_size + self.min_val
            points_list.append(general_points)
            scenic_labels_list.append(torch.ones(n_general, device=self.device, dtype=torch.long) * -1)
            
            weights = 1.0 + torch.rand(n_general, device=self.device) * 1.0
            demand_weights_list.append(weights)
        
        points = torch.cat(points_list, dim=0)
        demand_weights = torch.cat(demand_weights_list, dim=0)
        scenic_labels = torch.cat(scenic_labels_list, dim=0)
        
        assert len(points) == n
        
        indices = torch.randperm(len(points), device=self.device)
        points = points[indices]
        demand_weights = demand_weights[indices]
        scenic_labels = scenic_labels[indices]
        
        return points, demand_weights, scenic_labels
    
    def generate_instance(self, instance_id: int):
        """生成一个完整的MCLP实例"""
        seed = 42 + instance_id * 1000
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        
        points, demand_weights, scenic_labels = self.generate_scene_distribution()
        
        distance_matrix = _pairwise_euclidean(points, points, self.device)
        coverage_matrix = (distance_matrix <= self.service_radius).float()
        
        points_np = points.cpu().numpy()
        n_regions = min(8, len(points) // 15)
        if n_regions >= 2:
            kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
            region_labels = kmeans.fit_predict(points_np)
            region_labels = torch.tensor(region_labels, device=self.device)
        else:
            region_labels = torch.zeros(len(points), device=self.device, dtype=torch.long)
        
        instance = {
            'name': f'tourism_mclp_{instance_id:04d}',
            'instance_id': instance_id,
            'points': points,
            'demand_weights': demand_weights,
            'scenic_labels': scenic_labels,
            'region_labels': region_labels,
            'distance_matrix': distance_matrix,
            'coverage_matrix': coverage_matrix,
            'service_radius': self.service_radius,
            'num_nodes': len(points),
            'generation_time': time.time()
        }
        
        return instance
    
    def generate_dataset(self, start_id: int = 0, save_path=None):
        """生成完整的数据集"""
        dataset = []
        
        print(f"生成模拟文旅MCLP数据集...")
        print(f"参数: 节点数={self.num_nodes}, 实例数={self.num_instances}")
        print(f"服务半径={self.service_radius}米, 景点数={self.scenic_areas}")
        print("-" * 50)
        
        start_time = time.time()
        
        for i in range(start_id, start_id + self.num_instances):
            instance = self.generate_instance(i)
            dataset.append(instance)
            
            if (i + 1) % 10 == 0:
                print(f"已生成 {i+1}/{self.num_instances} 个实例")
        
        total_time = time.time() - start_time
        print(f"\n数据集生成完成！")
        print(f"总耗时: {total_time:.2f}秒")
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"数据集已保存到: {save_path}")
        
        return dataset

def main():
    """主函数：生成模拟数据并保存"""
    print("=" * 60)
    print("模拟数据生成器")
    print("=" * 60)
    
    # 生成模拟数据集
    generator = SyntheticDataGenerator(
        num_nodes=150,
        num_instances=40,
        coord_range=(0.0, 2500.0),
        service_radius=250.0,
        scenic_areas=4,
        rest_areas=2,
        device=device
    )
    
    # 生成并保存
    dataset = generator.generate_dataset(save_path='synthetic_dataset.pkl')
    
    print("\n模拟数据生成完成！")

if __name__ == "__main__":
    main()