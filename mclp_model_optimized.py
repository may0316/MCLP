# mclp_model_optimized.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric
import numpy as np
import pickle
import os
import time
from typing import Tuple, List, Optional
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== 距离计算函数 ==========
def _pairwise_euclidean(data1, data2, device=torch.device('cpu')):
    """计算欧几里得距离"""
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=-2)
    B = data2.unsqueeze(dim=-3)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1)
    dis = torch.sqrt(dis)
    return dis


# ========== 改进1：更合理的数据生成 ==========
class ImprovedTourismMCLPDatasetGenerator:
    """改进的文旅MCLP数据集生成器"""
    
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
    
    def generate_dataset(self, start_id: int = 0):
        """生成完整的数据集"""
        dataset = []
        
        print(f"生成改进的文旅MCLP数据集...")
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
        
        return dataset


# ========== 图构建函数 ==========
def build_mclp_graph(instance, device=None):
    """构建MCLP图结构"""
    points = instance['points']
    
    if device is None:
        device = points.device if torch.is_tensor(points) else torch.device('cpu')
    
    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float).to(device)
    else:
        points = points.to(device)
    
    service_radius = instance['service_radius']
    demand_weights = instance['demand_weights']
    if not torch.is_tensor(demand_weights):
        demand_weights = torch.tensor(demand_weights, dtype=torch.float).to(device)
    else:
        demand_weights = demand_weights.to(device)
    
    N = len(points)
    
    if 'distance_matrix' in instance:
        dist = instance['distance_matrix']
        if not torch.is_tensor(dist):
            dist = torch.tensor(dist, dtype=torch.float).to(device)
        else:
            dist = dist.to(device)
    else:
        dist = _pairwise_euclidean(points, points, device)
    
    cover_mask = (dist <= service_radius) & (dist > 0)
    edge_index = torch.nonzero(cover_mask, as_tuple=False).t()
    
    if edge_index.shape[1] > 0:
        edge_dist = dist[edge_index[0], edge_index[1]]
        edge_weight = torch.exp(-edge_dist / service_radius).unsqueeze(1)
    else:
        edge_weight = torch.zeros(0, 1, device=device)
    
    # degree encoding
    weighted_degree = torch.zeros(N, device=device)
    for i in range(N):
        covered_nodes = cover_mask[i]
        if covered_nodes.any():
            weighted_degree[i] = demand_weights[covered_nodes].sum()
    
    if weighted_degree.max() > 0:
        weighted_degree_norm = weighted_degree / (weighted_degree.max() + 1e-6)
    else:
        weighted_degree_norm = weighted_degree
    
    # distance encoding
    dist_potential = torch.zeros(N, device=device)
    for i in range(N):
        dist_potential[i] = torch.sum(
            demand_weights * torch.exp(-dist[i] / service_radius)
        )
    
    if dist_potential.max() > 0:
        dist_potential_norm = dist_potential / (dist_potential.max() + 1e-6)
    else:
        dist_potential_norm = dist_potential
    
    # 节点特征
    node_feats = [points, demand_weights.unsqueeze(1)]
    
    x = torch.cat(node_feats, dim=1)
    
    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        pos=points,
        demand_weights=demand_weights,
        weighted_degree=weighted_degree_norm.unsqueeze(1),
        dist_potential=dist_potential_norm.unsqueeze(1),
        distance_matrix=dist,
        coverage_mask=cover_mask,
        service_radius=service_radius,
        instance_name=instance.get('name', f'instance_{id(instance)}'),
        num_nodes=N
    )
    
    return graph


# ========== 改进2：增强的GCN编码器（修复维度问题）==========
class EnhancedMCLPEncoder(nn.Module):
    """增强的MCLP编码器 - 修复维度问题"""
    
    def __init__(self, in_channels, hidden_channels=256, out_channels=128, dropout=0.2):
        super().__init__()
        
        # 修复维度问题：确保每层输出维度正确
        self.conv1 = GATConv(in_channels, hidden_channels // 4, heads=4, concat=True)
        # conv1输出: hidden_channels // 4 * 4 = hidden_channels
        
        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        # conv2输出: hidden_channels // 4 * 4 = hidden_channels
        
        self.conv3 = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        # conv3输出: out_channels
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # 改进的特征处理器
        self.degree_processor = nn.Sequential(
            nn.Linear(1, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.dist_processor = nn.Sequential(
            nn.Linear(1, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(3 * out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
    def forward(self, x, edge_index, edge_weight, weighted_degree, dist_potential):
        # GAT编码
        h = self.conv1(x, edge_index)
        h = F.relu(self.bn1(h))
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.relu(self.bn2(h))
        h = self.dropout(h)
        
        h = self.conv3(h, edge_index)
        h = F.relu(self.bn3(h))
        
        if weighted_degree.dim() == 1:
            weighted_degree = weighted_degree.unsqueeze(1)
        if dist_potential.dim() == 1:
            dist_potential = dist_potential.unsqueeze(1)
        
        deg_feat = self.degree_processor(weighted_degree)
        dist_feat = self.dist_processor(dist_potential)
        
        # 融合特征
        combined = torch.cat([h, deg_feat, dist_feat], dim=1)
        output = self.fusion(combined)
        
        return output


# ========== 改进3：MCLP模型 ==========
class ImprovedMCLPModel(nn.Module):
    """改进的MCLP模型"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.encoder = EnhancedMCLPEncoder(input_dim, hidden_dim, output_dim)
        
        # 设施得分头
        self.facility_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 覆盖预测头
        self.coverage_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 价值预测头
        self.value_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        
        edge_weight = None
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            edge_weight = graph.edge_attr.squeeze()
        
        weighted_degree = graph.weighted_degree
        dist_potential = graph.dist_potential
        
        emb = self.encoder(x, edge_index, edge_weight, weighted_degree, dist_potential)
        
        facility_scores = self.facility_head(emb).squeeze()
        coverage_pred = self.coverage_head(emb).squeeze()
        value_pred = self.value_head(emb).squeeze()
        
        return emb, facility_scores, coverage_pred, value_pred


# ========== 改进4：求解器 ==========
class ImprovedMCLPSolver:
    """改进的MCLP求解器"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def initialize_model(self, input_dim, hidden_dim=256, output_dim=128):
        """初始化模型"""
        self.model = ImprovedMCLPModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=5e-4, 
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        print(f"改进模型初始化: 输入维度={input_dim}")
        return self.model
    
    def improved_mclp_loss(self, facility_scores, coverage_pred, value_pred, graph, K, alpha=0.3, beta=0.1):
        """改进的损失函数"""
        dist = graph.distance_matrix
        R = graph.service_radius
        demand_weights = graph.demand_weights
        
        N = len(facility_scores)
        
        # 使用退火温度
        temperature = max(0.2, 1.0 - 0.5 * (K / N))
        p = F.gumbel_softmax(facility_scores.unsqueeze(0), tau=temperature, hard=False, dim=1).squeeze()
        p = p * K
        
        # 覆盖概率计算
        coverage_prob = (dist <= R).float()
        
        # 每个节点被覆盖的概率
        facility_probs = p.unsqueeze(0)
        covered_prob = torch.max(facility_probs * coverage_prob, dim=1)[0]
        covered_prob = torch.clamp(covered_prob, 0, 1)
        
        # 主损失：最大化加权覆盖
        weighted_coverage = torch.sum(demand_weights * covered_prob)
        coverage_loss = -weighted_coverage / (torch.sum(demand_weights) + 1e-8)
        
        # 设施数量约束
        facility_count_loss = (p.sum() - K) ** 2 * 0.05
        
        # 辅助任务1：覆盖预测
        coverage_target = (graph.coverage_mask.float().sum(dim=1) > 0).float()
        aux_loss1 = F.binary_cross_entropy(coverage_pred, coverage_target)
        
        # 辅助任务2：价值预测
        value_target = torch.sum(demand_weights.unsqueeze(1) * coverage_prob, dim=0)
        value_target = value_target / (value_target.max() + 1e-8)
        aux_loss2 = F.mse_loss(value_pred, value_target)
        
        # 设施分散度奖励
        if len(p[p > 0.1]) > 1:
            selected_probs = p / (p.sum() + 1e-8)
            facility_locations = graph.pos
            pairwise_dist = torch.cdist(facility_locations, facility_locations)
            dispersion = torch.sum(
                selected_probs.unsqueeze(1) * selected_probs.unsqueeze(0) * pairwise_dist
            )
            dispersion_reward = -beta * dispersion / (R * K)
        else:
            dispersion_reward = 0
        
        total_loss = (coverage_loss + 
                     facility_count_loss + 
                     alpha * (aux_loss1 + aux_loss2) + 
                     dispersion_reward)
        
        return total_loss, weighted_coverage.item()
    
    def train_on_instance(self, graph, K, epochs=20):
        """在单个实例上训练"""
        self.model.train()
        graph = graph.to(self.device)
        
        losses = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            _, facility_scores, coverage_pred, value_pred = self.model(graph)
            loss, coverage = self.improved_mclp_loss(
                facility_scores, coverage_pred, value_pred, graph, K
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            losses.append(loss.item())
        
        return losses
    
    @torch.no_grad()
    def solve(self, graph, K, num_trials=10):
        """改进的求解算法"""
        self.model.eval()
        graph = graph.to(self.device)
        
        best_coverage = 0
        best_selection = None
        
        for trial in range(num_trials):
            _, facility_scores, _, _ = self.model(graph)
            
            noise_scale = 0.1 * (1 - trial / num_trials)
            if trial > 0:
                noise = torch.randn_like(facility_scores) * noise_scale
                scores = facility_scores + noise
            else:
                scores = facility_scores
            
            if trial < num_trials // 2:
                selected = self._greedy_selection(graph, scores, K)
            else:
                selected = self._greedy_with_swap(graph, scores, K)
            
            coverage = self._calculate_coverage(graph, selected)
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_selection = selected
        
        if best_selection is not None:
            best_selection = self._remove_redundant(graph, best_selection, K)
            best_coverage = self._calculate_coverage(graph, best_selection)
        
        return best_selection, best_coverage
    
    def _greedy_with_swap(self, graph, scores, K):
        """带交换的贪心选择"""
        N = graph.num_nodes
        dist = graph.distance_matrix
        R = graph.service_radius
        demand_weights = graph.demand_weights
        
        selected = []
        covered = torch.zeros(N, dtype=torch.bool, device=self.device)
        remaining = list(range(N))
        
        # 第一阶段：贪心选择
        for _ in range(K):
            best_gain = -float('inf')
            best_node = -1
            
            for node in remaining:
                newly_covered = (dist[node] <= R) & (~covered)
                gain = torch.sum(demand_weights[newly_covered]).item()
                
                combined_gain = gain + 0.02 * scores[node].item()
                
                if combined_gain > best_gain:
                    best_gain = combined_gain
                    best_node = node
            
            if best_node != -1:
                selected.append(best_node)
                newly_covered = (dist[best_node] <= R) & (~covered)
                covered = covered | newly_covered
                remaining.remove(best_node)
        
        # 第二阶段：交换改进
        improved = True
        while improved:
            improved = False
            current_coverage = torch.sum(demand_weights[covered]).item()
            
            for i in range(len(selected)):
                for j in remaining:
                    new_selected = selected.copy()
                    new_selected[i] = j
                    new_coverage = self._calculate_coverage(graph, new_selected)
                    
                    if new_coverage > current_coverage * 1.01:
                        selected = new_selected
                        covered = torch.zeros(N, dtype=torch.bool, device=self.device)
                        for node in selected:
                            covered = covered | (dist[node] <= R)
                        remaining = [x for x in range(N) if x not in selected]
                        improved = True
                        break
                if improved:
                    break
        
        return torch.tensor(selected, device=self.device)
    
    def _remove_redundant(self, graph, selected, K):
        """移除冗余设施"""
        if len(selected) <= K:
            return selected
        
        selected_list = selected.tolist()
        dist = graph.distance_matrix
        R = graph.service_radius
        
        facility_coverage = {}
        for i, node in enumerate(selected_list):
            covered = (dist[node] <= R).nonzero().squeeze().tolist()
            if isinstance(covered, int):
                covered = [covered]
            facility_coverage[node] = set(covered)
        
        while len(selected_list) > K:
            redundancy_scores = []
            for node in selected_list:
                other_coverage = set()
                for other in selected_list:
                    if other != node:
                        other_coverage.update(facility_coverage[other])
                redundant = len(facility_coverage[node] & other_coverage)
                redundancy_scores.append((redundant, node))
            
            redundancy_scores.sort(reverse=True)
            selected_list.remove(redundancy_scores[0][1])
        
        return torch.tensor(selected_list, device=self.device)
    
    def _greedy_selection(self, graph, scores, K):
        """基础贪心选择"""
        N = graph.num_nodes
        dist = graph.distance_matrix
        R = graph.service_radius
        demand_weights = graph.demand_weights
        
        selected = []
        covered = torch.zeros(N, dtype=torch.bool, device=self.device)
        remaining = list(range(N))
        
        for _ in range(K):
            best_gain = -float('inf')
            best_node = -1
            
            for node in remaining:
                newly_covered = (dist[node] <= R) & (~covered)
                gain = torch.sum(demand_weights[newly_covered]).item()
                
                combined_gain = gain + 0.02 * scores[node].item()
                
                if combined_gain > best_gain:
                    best_gain = combined_gain
                    best_node = node
            
            if best_node != -1:
                selected.append(best_node)
                newly_covered = (dist[best_node] <= R) & (~covered)
                covered = covered | newly_covered
                remaining.remove(best_node)
        
        return torch.tensor(selected, device=self.device)
    
    def _calculate_coverage(self, graph, selected_indices):
        """计算覆盖的总需求"""
        if len(selected_indices) == 0:
            return 0.0
        
        dist = graph.distance_matrix
        R = graph.service_radius
        demand_weights = graph.demand_weights
        
        min_dist = torch.min(dist[:, selected_indices], dim=1)[0]
        covered_mask = (min_dist <= R)
        
        return torch.sum(demand_weights[covered_mask]).item()
    
    def save_model(self, path='improved_mclp_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, path)
        print(f"模型已保存: {path}")
    
    def load_model(self, path='improved_mclp_model.pth'):
        if self.model is None:
            raise ValueError("请先初始化模型")
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"模型已加载: {path}")
        else:
            print(f"模型文件不存在: {path}")


# ========== 训练函数 ==========
def train_improved_model(dataset, val_instances=None, K_range=[3, 5, 8, 10, 12], epochs_per_instance=20):
    """训练改进的MCLP模型"""
    
    sample_instance = dataset[0]
    sample_graph = build_mclp_graph(sample_instance)
    input_dim = sample_graph.x.shape[1]
    print(f"检测到的输入维度: {input_dim}")
    
    solver = ImprovedMCLPSolver(device=device)
    solver.initialize_model(input_dim=input_dim, hidden_dim=256, output_dim=128)
    
    print(f"\n开始训练...")
    print(f"训练实例数: {len(dataset)}")
    print(f"输入维度: {input_dim}")
    print("-" * 50)
    
    all_losses = []
    best_val_coverage = 0
    
    for epoch, instance in enumerate(dataset):
        graph = build_mclp_graph(instance, device=device)
        
        if epoch < len(dataset) // 3:
            K = np.random.choice(K_range[:3])
        elif epoch < 2 * len(dataset) // 3:
            K = np.random.choice(K_range[1:4])
        else:
            K = np.random.choice(K_range[2:])
        
        losses = solver.train_on_instance(graph, K=K, epochs=epochs_per_instance)
        all_losses.extend(losses)
        
        if val_instances and (epoch + 1) % 5 == 0:
            val_coverage = validate_improved_model(solver, val_instances)
            print(f"Epoch {epoch+1}: 验证覆盖率 = {val_coverage:.1f}%")
            
            if val_coverage > best_val_coverage:
                best_val_coverage = val_coverage
                solver.save_model('best_improved_model.pth')
        
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(all_losses[-10:])
            print(f"Epoch {epoch+1}/{len(dataset)} | 平均损失: {avg_loss:.4f}")
    
    print(f"\n训练完成！最佳验证覆盖率: {best_val_coverage:.1f}%")
    return solver


def validate_improved_model(solver, val_instances, K=8):
    """验证改进模型"""
    coverages = []
    
    for instance in val_instances[:3]:
        graph = build_mclp_graph(instance, device=device)
        selected, coverage = solver.solve(graph, K=K, num_trials=5)
        
        total_demand = torch.sum(graph.demand_weights).item()
        coverage_pct = (coverage / total_demand) * 100
        coverages.append(coverage_pct)
    
    return np.mean(coverages) if coverages else 0


# ========== 主程序 ==========
def main():
    print("=" * 60)
    print("改进的文旅MCLP自监督学习求解")
    print("=" * 60)
    
    # 1. 生成数据集
    print("\n1. 生成改进的文旅MCLP数据集...")
    generator = ImprovedTourismMCLPDatasetGenerator(
        num_nodes=150,
        num_instances=40,
        coord_range=(0.0, 2500.0),
        service_radius=250.0,
        scenic_areas=4,
        rest_areas=2,
        device=device
    )
    
    dataset = generator.generate_dataset()
    
    # 2. 划分训练集和验证集
    train_dataset = dataset[:30]
    val_dataset = dataset[30:]
    
    print(f"\n训练集: {len(train_dataset)}个实例")
    print(f"验证集: {len(val_dataset)}个实例")
    
    # 3. 训练改进的模型
    solver = train_improved_model(
        train_dataset,
        val_instances=val_dataset,
        K_range=[3, 5, 8, 10, 12],
        epochs_per_instance=15
    )
    
    # 4. 保存最终模型
    solver.save_model('final_improved_model.pth')
    
    # 5. 全面测试
    print("\n4. 测试改进模型...")
    test_coverages = {3: [], 5: [], 8: [], 10: [], 12: []}
    
    for i, instance in enumerate(val_dataset[:5]):
        graph = build_mclp_graph(instance, device=device)
        total_demand = torch.sum(graph.demand_weights).item()
        
        for K in [3, 5, 8, 10, 12]:
            selected, coverage = solver.solve(graph, K=K, num_trials=8)
            coverage_pct = (coverage / total_demand) * 100
            
            test_coverages[K].append(coverage_pct)
            print(f"实例{i}, K={K}: 覆盖率={coverage_pct:.1f}%, 设施数={len(selected)}")
    
    print("\n" + "-" * 50)
    print("各K值平均覆盖率:")
    for K in [3, 5, 8, 10, 12]:
        avg = np.mean(test_coverages[K])
        print(f"K={K}: {avg:.1f}%")
    
    overall_avg = np.mean([c for covs in test_coverages.values() for c in covs])
    print(f"\n总体平均覆盖率: {overall_avg:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()