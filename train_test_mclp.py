# train_test_mclp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric
import numpy as np
import pandas as pd
import pickle
import os
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

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

# ========== 增强的GCN编码器 ==========
class EnhancedMCLPEncoder(nn.Module):
    """增强的MCLP编码器"""
    
    def __init__(self, in_channels, hidden_channels=256, out_channels=128, dropout=0.2):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels // 4, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
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
        
        self.fusion = nn.Sequential(
            nn.Linear(3 * out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
    def forward(self, x, edge_index, edge_weight, weighted_degree, dist_potential):
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
        
        combined = torch.cat([h, deg_feat, dist_feat], dim=1)
        output = self.fusion(combined)
        
        return output

# ========== MCLP模型 ==========
class ImprovedMCLPModel(nn.Module):
    """改进的MCLP模型"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.encoder = EnhancedMCLPEncoder(input_dim, hidden_dim, output_dim)
        
        self.facility_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.coverage_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
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

# ========== 求解器 ==========
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
        print(f"模型初始化: 输入维度={input_dim}")
        return self.model
    
    def improved_mclp_loss(self, facility_scores, coverage_pred, value_pred, graph, K, alpha=0.3, beta=0.1):
        """改进的损失函数"""
        dist = graph.distance_matrix
        R = graph.service_radius
        demand_weights = graph.demand_weights
        
        N = len(facility_scores)
        
        temperature = max(0.2, 1.0 - 0.5 * (K / N))
        p = F.gumbel_softmax(facility_scores.unsqueeze(0), tau=temperature, hard=False, dim=1).squeeze()
        p = p * K
        
        coverage_prob = (dist <= R).float()
        
        facility_probs = p.unsqueeze(0)
        covered_prob = torch.max(facility_probs * coverage_prob, dim=1)[0]
        covered_prob = torch.clamp(covered_prob, 0, 1)
        
        weighted_coverage = torch.sum(demand_weights * covered_prob)
        coverage_loss = -weighted_coverage / (torch.sum(demand_weights) + 1e-8)
        
        facility_count_loss = (p.sum() - K) ** 2 * 0.05
        
        coverage_target = (graph.coverage_mask.float().sum(dim=1) > 0).float()
        aux_loss1 = F.binary_cross_entropy(coverage_pred, coverage_target)
        
        value_target = torch.sum(demand_weights.unsqueeze(1) * coverage_prob, dim=0)
        value_target = value_target / (value_target.max() + 1e-8)
        aux_loss2 = F.mse_loss(value_pred, value_target)
        
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
    
    def save_model(self, path='model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, path)
        print(f"模型已保存: {path}")
    
    def load_model(self, path='model.pth'):
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

# ========== 真实数据加载函数 ==========
def load_real_data_from_csv(csv_path, service_radius=0.05):
    """从CSV文件加载真实数据"""
    df = pd.read_csv(csv_path)
    print(f"加载真实数据: {len(df)} 个点位")
    
    points = torch.tensor(df[['x', 'y']].values, dtype=torch.float)
    demand_weights = torch.tensor(df['demand'].values, dtype=torch.float)
    
    # 归一化坐标
    center = points.mean(dim=0)
    scale = points.std(dim=0).max()
    points = (points - center) / (scale + 1e-8)
    
    # 生成场景标签
    scenic_labels = []
    for t in df['type']:
        type_str = str(t)
        if '风景名胜' in type_str or '风景区' in type_str or '公园' in type_str:
            scenic_labels.append(1)
        elif '度假' in type_str or '休闲' in type_str:
            scenic_labels.append(0)
        else:
            scenic_labels.append(-1)
    scenic_labels = torch.tensor(scenic_labels, dtype=torch.long)
    
    instance = {
        'name': 'real_data',
        'points': points,
        'demand_weights': demand_weights,
        'scenic_labels': scenic_labels,
        'service_radius': service_radius,
        'num_nodes': len(points)
    }
    
    print(f"真实数据实例构建完成: {instance['num_nodes']}个节点")
    return instance

def create_real_dataset_from_csv(csv_path, num_instances=20, service_radius=0.05):
    """从CSV创建多个训练实例（通过随机采样）"""
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 个点位")
    
    dataset = []
    
    for i in range(num_instances):
        sampled_df = df.sample(frac=0.8, random_state=42+i)
        
        points = torch.tensor(sampled_df[['x', 'y']].values, dtype=torch.float)
        demand_weights = torch.tensor(sampled_df['demand'].values, dtype=torch.float)
        
        # 归一化
        center = points.mean(dim=0)
        scale = points.std(dim=0).max()
        points = (points - center) / (scale + 1e-8)
        
        scenic_labels = []
        for t in sampled_df['type']:
            type_str = str(t)
            if '风景名胜' in type_str or '风景区' in type_str or '公园' in type_str:
                scenic_labels.append(1)
            elif '度假' in type_str or '休闲' in type_str:
                scenic_labels.append(0)
            else:
                scenic_labels.append(-1)
        scenic_labels = torch.tensor(scenic_labels, dtype=torch.long)
        
        instance = {
            'name': f'real_instance_{i:04d}',
            'instance_id': i,
            'points': points,
            'demand_weights': demand_weights,
            'scenic_labels': scenic_labels,
            'service_radius': service_radius,
            'num_nodes': len(points)
        }
        
        dataset.append(instance)
        
        if (i+1) % 5 == 0:
            print(f"已创建 {i+1}/{num_instances} 个真实实例")
    
    return dataset

# ========== 训练函数 ==========
def train_model(dataset, val_ratio=0.2, K_range=[3, 5, 8, 10, 12, 15], epochs_per_instance=15):
    """训练MCLP模型"""
    
    # 划分训练集和验证集
    n_train = int(len(dataset) * (1 - val_ratio))
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:]
    
    print(f"\n训练集: {len(train_dataset)} 个实例")
    print(f"验证集: {len(val_dataset)} 个实例")
    
    # 获取输入维度
    sample_graph = build_mclp_graph(train_dataset[0], device=device)
    input_dim = sample_graph.x.shape[1]
    print(f"输入特征维度: {input_dim}")
    
    # 初始化求解器
    solver = ImprovedMCLPSolver(device=device)
    solver.initialize_model(input_dim=input_dim, hidden_dim=256, output_dim=128)
    
    print(f"\n开始训练...")
    print(f"K值范围: {K_range}")
    print(f"每个实例训练轮数: {epochs_per_instance}")
    print("-" * 50)
    
    all_losses = []
    best_val_coverage = 0
    
    for epoch, instance in enumerate(train_dataset):
        graph = build_mclp_graph(instance, device=device)
        
        # 动态选择K值
        if epoch < len(train_dataset) // 3:
            K = np.random.choice(K_range[:3])
        elif epoch < 2 * len(train_dataset) // 3:
            K = np.random.choice(K_range[2:5])
        else:
            K = np.random.choice(K_range[3:])
        
        # 训练
        losses = solver.train_on_instance(graph, K=K, epochs=epochs_per_instance)
        all_losses.extend(losses)
        
        # 定期验证
        if (epoch + 1) % 5 == 0 and len(val_dataset) > 0:
            val_coverages = []
            for val_instance in val_dataset[:2]:
                val_graph = build_mclp_graph(val_instance, device=device)
                _, val_coverage = solver.solve(val_graph, K=8, num_trials=3)
                total_demand = torch.sum(val_graph.demand_weights).item()
                val_pct = (val_coverage / total_demand) * 100
                val_coverages.append(val_pct)
            
            avg_val_coverage = np.mean(val_coverages)
            print(f"Epoch {epoch+1}: 验证覆盖率 = {avg_val_coverage:.1f}%")
            
            if avg_val_coverage > best_val_coverage:
                best_val_coverage = avg_val_coverage
                solver.save_model('best_model.pth')
                print(f"  → 保存最佳模型 (覆盖率: {avg_val_coverage:.1f}%)")
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(all_losses[-10:])
            print(f"Epoch {epoch+1}/{len(train_dataset)} | 平均损失: {avg_loss:.4f}")
    
    print(f"\n训练完成！最佳验证覆盖率: {best_val_coverage:.1f}%")
    solver.save_model('final_model.pth')
    
    return solver

# ========== 测试函数 ==========
def test_model(solver, test_instances, K_values=[3, 5, 8, 10, 12, 15]):
    """测试模型"""
    print("\n" + "=" * 50)
    print("模型测试")
    print("=" * 50)
    
    results = {}
    
    for K in K_values:
        results[K] = []
    
    for i, instance in enumerate(test_instances[:5]):  # 测试前5个实例
        graph = build_mclp_graph(instance, device=device)
        total_demand = torch.sum(graph.demand_weights).item()
        
        print(f"\n测试实例 {i+1}: {instance['name']}, 节点数={instance['num_nodes']}")
        
        for K in K_values:
            selected, coverage = solver.solve(graph, K=K, num_trials=8)
            coverage_pct = (coverage / total_demand) * 100
            results[K].append(coverage_pct)
            print(f"  K={K:2d}: 覆盖率={coverage_pct:5.1f}%")
    
    print("\n" + "-" * 50)
    print("各K值平均覆盖率:")
    for K in K_values:
        avg = np.mean(results[K])
        print(f"K={K:2d}: {avg:5.1f}%")
    
    overall_avg = np.mean([c for covs in results.values() for c in covs])
    print(f"\n总体平均覆盖率: {overall_avg:.1f}%")
    
    return results

# ========== 可视化函数 ==========
def visualize_solution(graph, selected_indices, K, coverage, save_path=None):
    """可视化MCLP求解结果"""
    points = graph.pos.cpu().numpy()
    selected = selected_indices.cpu().numpy()
    demand = graph.demand_weights.cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    
    sizes = demand * 20
    
    scatter = plt.scatter(points[:, 0], points[:, 1], 
                         c='lightblue', s=sizes, alpha=0.5, 
                         label='All POIs')
    
    plt.scatter(points[selected, 0], points[selected, 1], 
               c='red', s=200, marker='*', edgecolors='black',
               linewidths=1, label=f'Selected Facilities (K={K})')
    
    R = graph.service_radius
    for i, idx in enumerate(selected[:min(5, len(selected))]):
        circle = plt.Circle(points[idx], R, color='red', 
                           fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
    
    total_demand = torch.sum(graph.demand_weights).item()
    coverage_pct = (coverage / total_demand) * 100
    
    plt.title(f'MCLP Solution - Coverage: {coverage:.1f} ({coverage_pct:.1f}%)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(scatter, label='Demand Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ========== 主函数 ==========
def main():
    print("=" * 60)
    print("MCLP模型训练与测试 (可选数据集)")
    print("=" * 60)
    
    # 选择数据集
    print("\n请选择数据集类型:")
    print("1. 模拟数据 (需要先运行 generate_synthetic_data.py 生成)")
    print("2. 真实数据 (使用 tourism_poi_beijing.csv)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    dataset = None
    
    if choice == '1':
        # 使用模拟数据
        print("\n使用模拟数据...")
        synthetic_path = 'synthetic_dataset.pkl'
        
        if not os.path.exists(synthetic_path):
            print(f"错误: 模拟数据文件 {synthetic_path} 不存在!")
            print("请先运行 generate_synthetic_data.py 生成模拟数据")
            return
        
        with open(synthetic_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"加载了 {len(dataset)} 个模拟实例")
        
        # 模拟数据的服务半径是250.0（米）
        for instance in dataset:
            instance['service_radius'] = 250.0
    
    elif choice == '2':
        # 使用真实数据
        print("\n使用真实数据...")
        csv_path = 'tourism_poi_beijing.csv'
        
        if not os.path.exists(csv_path):
            print(f"错误: 真实数据文件 {csv_path} 不存在!")
            return
        
        # 创建多个训练实例
        dataset = create_real_dataset_from_csv(
            csv_path,
            num_instances=30,  # 创建30个实例
            service_radius=0.05  # 服务半径约5.5公里
        )
    
    else:
        print("无效选择，退出程序")
        return
    
    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.7)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    print(f"\n训练集: {len(train_dataset)} 个实例")
    print(f"测试集: {len(test_dataset)} 个实例")
    
    # 训练模型
    solver = train_model(
        train_dataset,
        val_ratio=0.2,  # 从训练集中再分20%做验证
        K_range=[3, 5, 8, 10, 12, 15],
        epochs_per_instance=15
    )
    
    # 测试模型
    test_model(solver, test_dataset, K_values=[3, 5, 8, 10, 12, 15])
    
    # 如果是真实数据，可视化一个结果
    if choice == '2':
        print("\n可视化真实数据的一个求解结果...")
        full_instance = load_real_data_from_csv('tourism_poi_beijing.csv', service_radius=0.05)
        full_graph = build_mclp_graph(full_instance, device=device)
        selected, coverage = solver.solve(full_graph, K=15, num_trials=10)
        visualize_solution(full_graph, selected, 15, coverage, save_path='real_solution.png')

if __name__ == "__main__":
    main()