# train_test_mclp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import torch_geometric
import numpy as np
import pandas as pd
import pickle
import os
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
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


# ========== 文旅场景MCLP图构建 ==========
def build_tourism_mclp_graph(demand_points, facility_candidates, demand_weights,
                             service_radius, scenic_labels=None, device=None):
    """
    构建文旅MCLP图结构（分离需求点和设施点）

    Args:
        demand_points: 需求点坐标 [n_demand, 2]
        facility_candidates: 设施候选点坐标 [n_facility, 2]
        demand_weights: 需求点权重 [n_demand]
        service_radius: 服务半径
        scenic_labels: 景点标签 [n_demand]
        device: 设备
    """
    if device is None:
        device = demand_points.device if torch.is_tensor(demand_points) else torch.device('cpu')

    if not torch.is_tensor(demand_points):
        demand_points = torch.tensor(demand_points, dtype=torch.float).to(device)
        facility_candidates = torch.tensor(facility_candidates, dtype=torch.float).to(device)
        demand_weights = torch.tensor(demand_weights, dtype=torch.float).to(device)
    else:
        demand_points = demand_points.to(device)
        facility_candidates = facility_candidates.to(device)
        demand_weights = demand_weights.to(device)

    n_demand = len(demand_points)
    n_facility = len(facility_candidates)

    # 计算需求点到设施点的距离矩阵 [n_demand, n_facility]
    dist_matrix = _pairwise_euclidean(demand_points, facility_candidates, device)

    # 覆盖关系
    coverage_matrix = (dist_matrix <= service_radius).float()

    # 构建二分图边：需求点 <-> 设施点
    covered_pairs = torch.nonzero(coverage_matrix, as_tuple=False)

    if len(covered_pairs) > 0:
        edge_index = torch.stack([
            covered_pairs[:, 0],  # 需求点索引
            covered_pairs[:, 1] + n_demand  # 设施点索引（偏移）
        ], dim=0)

        edge_dist = dist_matrix[covered_pairs[:, 0], covered_pairs[:, 1]]
        edge_weight = torch.exp(-edge_dist / service_radius).unsqueeze(1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_weight = torch.zeros(0, 1, device=device)

    # 需求点的特征
    demand_features = [demand_points, demand_weights.unsqueeze(1)]
    if scenic_labels is not None:
        if not torch.is_tensor(scenic_labels):
            scenic_labels = torch.tensor(scenic_labels, dtype=torch.float).unsqueeze(1).to(device)
        else:
            scenic_labels = scenic_labels.float().unsqueeze(1).to(device)
        demand_features.append(scenic_labels)

    demand_x = torch.cat(demand_features, dim=1)

    # 设施点的特征
    # 1. 坐标
    # 2. 该设施点覆盖的需求总权重（加权度）
    facility_covered_weights = torch.zeros(n_facility, device=device)
    for j in range(n_facility):
        covered_demand = coverage_matrix[:, j] > 0
        if covered_demand.any():
            facility_covered_weights[j] = demand_weights[covered_demand].sum()

    # 归一化
    if facility_covered_weights.max() > 0:
        facility_covered_weights_norm = facility_covered_weights / facility_covered_weights.max()
    else:
        facility_covered_weights_norm = facility_covered_weights

    # 3. 设施点的潜在需求（距离衰减）
    facility_potential = torch.zeros(n_facility, device=device)
    for j in range(n_facility):
        distances = dist_matrix[:, j]
        potential = torch.sum(demand_weights * torch.exp(-distances / service_radius))
        facility_potential[j] = potential

    if facility_potential.max() > 0:
        facility_potential_norm = facility_potential / facility_potential.max()
    else:
        facility_potential_norm = facility_potential

    facility_x = torch.cat([
        facility_candidates,
        facility_covered_weights_norm.unsqueeze(1),
        facility_potential_norm.unsqueeze(1)
    ], dim=1)

    # 合并节点特征
    x = torch.cat([demand_x, facility_x], dim=0)

    # 节点类型标记
    node_type = torch.cat([
        torch.zeros(n_demand, device=device),
        torch.ones(n_facility, device=device)
    ], dim=0)

    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        demand_pos=demand_points,
        facility_pos=facility_candidates,
        demand_weights=demand_weights,
        facility_covered_weights=facility_covered_weights,
        coverage_matrix=coverage_matrix,
        dist_matrix=dist_matrix,
        service_radius=service_radius,
        node_type=node_type,
        n_demand=n_demand,
        n_facility=n_facility,
        instance_name='tourism_mclp'
    )

    return graph


# ========== 改进的数据集创建 ==========
def create_improved_tourism_dataset(csv_path, num_instances=20,
                                    service_radius=0.1,  # 服务半径
                                    n_facility_candidates=100,  # 候选点数量
                                    facility_ratio=0.5):  # 设施候选点比例
    """
    改进的数据集创建
    """
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 个点位")

    # 分析数据分布
    coords = df[['x', 'y']].values
    print(f"坐标范围: X[{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}], "
          f"Y[{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")

    # 计算实际距离范围
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    print(f"实际范围: 经度差={x_range:.4f}° (~{x_range * 111:.1f}km), "
          f"纬度差={y_range:.4f}° (~{y_range * 111:.1f}km)")

    dataset = []

    for i in range(num_instances):
        # 使用不同的随机种子
        seed = 42 + i * 10
        np.random.seed(seed)

        # 随机采样70-90%作为需求点
        sample_ratio = np.random.uniform(0.7, 0.9)
        sampled_df = df.sample(frac=sample_ratio, random_state=seed)

        demand_points = sampled_df[['x', 'y']].values
        demand_weights = sampled_df['demand'].values

        # 归一化坐标
        center = demand_points.mean(axis=0)
        scale = max(demand_points.std(axis=0).max(), 0.1)
        demand_points_norm = (demand_points - center) / scale

        # 设施候选点：包括从所有点中随机选择 + 聚类中心
        n_facility = max(20, int(len(demand_points) * facility_ratio))

        # 方法1：随机选择
        random_indices = np.random.choice(len(df), min(n_facility // 2, len(df)), replace=False)
        random_facilities = df.iloc[random_indices][['x', 'y']].values

        # 方法2：KMeans聚类中心
        if len(demand_points) > n_facility // 2:
            kmeans = KMeans(n_clusters=n_facility // 2, random_state=seed, n_init=10)
            kmeans.fit(demand_points)
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_centers = demand_points[np.random.choice(len(demand_points),
                                                             n_facility // 2, replace=True)]

        # 合并设施候选点
        facility_points = np.vstack([random_facilities, cluster_centers])
        facility_points_norm = (facility_points - center) / scale

        # 生成场景标签
        scenic_labels = []
        for t in sampled_df['type']:
            type_str = str(t)
            if '风景名胜' in type_str or '公园' in type_str:
                scenic_labels.append(1.0)
            elif '度假' in type_str or '休闲' in type_str:
                scenic_labels.append(0.8)
            elif '博物馆' in type_str or '纪念馆' in type_str:
                scenic_labels.append(0.6)
            else:
                scenic_labels.append(0.3)
        scenic_labels = np.array(scenic_labels)

        instance = {
            'name': f'tourism_instance_{i:04d}',
            'instance_id': i,
            'demand_points': demand_points_norm,
            'facility_candidates': facility_points_norm,
            'demand_weights': demand_weights / demand_weights.max(),  # 归一化需求权重
            'scenic_labels': scenic_labels,
            'service_radius': service_radius,
            'n_demand': len(demand_points),
            'n_facility': len(facility_points)
        }

        dataset.append(instance)

        if (i + 1) % 5 == 0:
            print(f"已创建 {i + 1}/{num_instances} 个实例")

    return dataset


# ========== 改进的MCLP模型 ==========
class ImprovedTourismMCLPEncoder(nn.Module):
    """改进的文旅MCLP编码器"""

    def __init__(self, in_channels, hidden_channels=256, out_channels=128, dropout=0.1):
        super().__init__()

        # 使用GCN和GAT的组合
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gat1 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # 注意力机制
        self.attention = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)

    def forward(self, x, edge_index):
        # GCN层
        h = self.gcn1(x, edge_index)
        h = F.relu(self.bn1(h))
        h = self.dropout(h)

        # GAT层
        h = self.gat1(h, edge_index)
        h = F.relu(self.bn2(h))
        h = self.dropout(h)

        h = self.gat2(h, edge_index)
        h = F.relu(self.bn3(h))

        # 自注意力（捕捉全局关系）
        h = h.unsqueeze(0)  # [1, N, dim]
        h, _ = self.attention(h, h, h)
        h = h.squeeze(0)

        return h


class ImprovedTourismMCLPModel(nn.Module):
    """改进的文旅MCLP模型"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()

        self.encoder = ImprovedTourismMCLPEncoder(input_dim, hidden_dim, output_dim)

        # 设施点选择头（增强）
        self.facility_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 需求点覆盖预测头
        self.coverage_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 价值预测头
        self.value_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 区域聚类头（新增）
        self.cluster_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # 最多8个聚类
        )

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index

        emb = self.encoder(x, edge_index)

        demand_emb = emb[:graph.n_demand]
        facility_emb = emb[graph.n_demand:]

        facility_scores = self.facility_head(facility_emb).squeeze()
        coverage_pred = self.coverage_head(demand_emb).squeeze()
        value_pred = self.value_head(emb).squeeze()
        cluster_logits = self.cluster_head(emb)

        return demand_emb, facility_emb, facility_scores, coverage_pred, value_pred, cluster_logits


# ========== 改进的求解器 ==========
class ImprovedTourismMCLPSolver:
    """改进的文旅MCLP求解器"""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def initialize_model(self, input_dim, hidden_dim=256, output_dim=128):
        self.model = ImprovedTourismMCLPModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        # 移除verbose参数
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        print(f"改进的文旅MCLP模型初始化: 输入维度={input_dim}")
        return self.model

    def improved_loss(self, facility_scores, coverage_pred, value_pred, cluster_logits,
                      graph, K, alpha=0.2, beta=0.1, gamma=0.05):
        """改进的损失函数"""
        coverage_matrix = graph.coverage_matrix
        demand_weights = graph.demand_weights

        n_facility = graph.n_facility
        n_demand = graph.n_demand

        # 1. 主目标：加权覆盖
        # 使用Softmax + Top-k近似
        temperature = max(0.1, 1.0 - 0.8 * (K / n_facility))
        facility_probs = F.softmax(facility_scores / temperature, dim=0)

        # 期望覆盖
        covered_prob = torch.zeros(n_demand, device=self.device)
        for j in range(n_facility):
            covered_prob += facility_probs[j] * K * coverage_matrix[:, j]

        covered_prob = torch.clamp(covered_prob, 0, 1)

        weighted_coverage = torch.sum(demand_weights * covered_prob)
        total_demand = torch.sum(demand_weights)
        coverage_loss = -torch.log(weighted_coverage / (total_demand + 1e-8) + 1e-8)

        # 2. 设施数量约束（使用熵正则化）
        facility_entropy = -torch.sum(facility_probs * torch.log(facility_probs + 1e-8))
        entropy_loss = -facility_entropy * 0.01  # 鼓励多样性

        # 3. 辅助任务：覆盖预测
        with torch.no_grad():
            # 使用贪心选择
            selected = self._greedy_selection(graph, facility_scores, K)
            coverage_target = torch.zeros(n_demand, device=self.device)
            for idx in selected:
                coverage_target = torch.max(coverage_target, coverage_matrix[:, idx])

        aux_loss1 = F.binary_cross_entropy(coverage_pred, coverage_target)

        # 4. 辅助任务：价值预测
        with torch.no_grad():
            value_target = torch.zeros(len(value_pred), device=self.device)
            value_target[:n_demand] = demand_weights / (total_demand + 1e-8)

            # 简化版Shapley值近似
            for j in range(n_facility):
                # 计算边际贡献的近似
                base_coverage = torch.zeros(n_demand, device=self.device)
                for idx in selected:
                    base_coverage = torch.max(base_coverage, coverage_matrix[:, idx])

                with_j = torch.max(base_coverage, coverage_matrix[:, j])
                marginal = torch.sum(demand_weights * (with_j - base_coverage)).item()
                value_target[n_demand + j] = max(0, marginal) / (total_demand + 1e-8)

        aux_loss2 = F.mse_loss(value_pred, value_target)

        # 5. 聚类损失（鼓励区域多样性）
        cluster_loss = 0
        if n_demand > 10:
            try:
                # 使用KMeans得到伪标签
                with torch.no_grad():
                    from sklearn.cluster import KMeans
                    demand_pos_np = graph.demand_pos.cpu().numpy()
                    kmeans = KMeans(n_clusters=min(8, n_demand // 10), random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(demand_pos_np)
                    cluster_labels = torch.tensor(cluster_labels, device=self.device)

                cluster_loss = F.cross_entropy(cluster_logits[:n_demand], cluster_labels)
            except:
                cluster_loss = 0

        # 6. 分散度奖励
        with torch.no_grad():
            _, top_indices = torch.topk(facility_scores, min(K, n_facility))
            if len(top_indices) > 1:
                facility_pos = graph.facility_pos
                selected_pos = facility_pos[top_indices]
                pairwise_dist = torch.cdist(selected_pos, selected_pos)
                mask = 1 - torch.eye(len(selected_pos), device=self.device)
                avg_dist = (pairwise_dist * mask).sum() / (len(selected_pos) * (len(selected_pos) - 1) + 1e-8)
                dispersion_reward = beta * avg_dist / (graph.service_radius * 2 + 1e-8)
            else:
                dispersion_reward = 0

        total_loss = (coverage_loss + entropy_loss +
                      alpha * (aux_loss1 + aux_loss2) +
                      gamma * cluster_loss -
                      dispersion_reward)

        return total_loss, weighted_coverage.item()

    def _calculate_coverage_by_indices(self, graph, indices):
        """根据索引计算覆盖"""
        if len(indices) == 0:
            return 0.0
        coverage_matrix = graph.coverage_matrix
        demand_weights = graph.demand_weights
        covered = torch.zeros(graph.n_demand, dtype=torch.bool, device=self.device)
        for idx in indices:
            covered = covered | (coverage_matrix[:, idx] > 0)
        return torch.sum(demand_weights[covered]).item()

    def train_on_instance(self, graph, K, epochs=20):
        self.model.train()
        graph = graph.to(self.device)

        losses = []

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            (_, _, facility_scores,
             coverage_pred, value_pred,
             cluster_logits) = self.model(graph)

            loss, coverage = self.improved_loss(
                facility_scores, coverage_pred, value_pred, cluster_logits,
                graph, K
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())

        if self.scheduler is not None:
            self.scheduler.step(np.mean(losses))

        return losses

    @torch.no_grad()
    def solve(self, graph, K, num_trials=20):
        """改进的求解算法"""
        self.model.eval()
        graph = graph.to(self.device)

        coverage_matrix = graph.coverage_matrix
        demand_weights = graph.demand_weights
        n_facility = graph.n_facility

        best_coverage = 0
        best_selection = None

        # 确保K不超过设施点数量
        K = min(K, n_facility)

        for trial in range(num_trials):
            _, _, facility_scores, _, _, _ = self.model(graph)

            # 策略1：贪心选择（基础）
            selected1 = self._greedy_selection(graph, facility_scores, K)
            coverage1 = self._calculate_coverage(graph, selected1)

            # 策略2：带噪声的贪心
            noise = torch.randn_like(facility_scores) * 0.1
            selected2 = self._greedy_selection(graph, facility_scores + noise, K)
            coverage2 = self._calculate_coverage(graph, selected2)

            # 策略3：Top-K
            _, top_indices = torch.topk(facility_scores, K)
            selected3 = top_indices
            coverage3 = self._calculate_coverage(graph, selected3)

            # 策略4：聚类辅助选择
            selected4 = None
            coverage4 = 0
            if n_facility > K * 2:
                try:
                    from sklearn.cluster import KMeans
                    facility_pos = graph.facility_pos.cpu().numpy()
                    kmeans = KMeans(n_clusters=K, random_state=trial, n_init=10)
                    cluster_labels = kmeans.fit_predict(facility_pos)

                    selected4_list = []
                    facility_scores_np = facility_scores.cpu().numpy()
                    for c in range(K):
                        cluster_indices = np.where(cluster_labels == c)[0]
                        if len(cluster_indices) > 0:
                            best_in_cluster = cluster_indices[np.argmax(facility_scores_np[cluster_indices])]
                            selected4_list.append(best_in_cluster)
                    if len(selected4_list) == K:
                        selected4 = torch.tensor(selected4_list, device=self.device)
                        coverage4 = self._calculate_coverage(graph, selected4)
                except:
                    pass

            # 记录最佳
            candidates = [(coverage1, selected1), (coverage2, selected2), (coverage3, selected3)]
            if selected4 is not None:
                candidates.append((coverage4, selected4))

            for cov, sel in candidates:
                if cov > best_coverage:
                    best_coverage = cov
                    best_selection = sel

        return best_selection, best_coverage

    def _greedy_selection(self, graph, scores, K):
        """贪心选择（带边际增益）"""
        coverage_matrix = graph.coverage_matrix
        demand_weights = graph.demand_weights
        n_facility = graph.n_facility

        K = min(K, n_facility)

        selected = []
        covered = torch.zeros(graph.n_demand, dtype=torch.bool, device=self.device)

        # 考虑边际增益
        remaining = list(range(n_facility))

        for _ in range(K):
            best_gain = -1
            best_node = -1

            for node in remaining:
                newly_covered = (coverage_matrix[:, node] > 0) & (~covered)
                gain = torch.sum(demand_weights[newly_covered]).item()

                # 结合模型得分
                combined_gain = gain + 0.05 * scores[node].item()

                if combined_gain > best_gain:
                    best_gain = combined_gain
                    best_node = node

            if best_node != -1:
                selected.append(best_node)
                newly_covered = (coverage_matrix[:, best_node] > 0) & (~covered)
                covered = covered | newly_covered
                remaining.remove(best_node)

        # 如果还不够，补全得分最高的
        if len(selected) < K:
            remaining_scores = [(i, scores[i].item()) for i in remaining]
            remaining_scores.sort(key=lambda x: x[1], reverse=True)
            for i, _ in remaining_scores[:K - len(selected)]:
                selected.append(i)

        return torch.tensor(selected, device=self.device)

    def _calculate_coverage(self, graph, selected_indices):
        """计算覆盖的总需求"""
        if len(selected_indices) == 0:
            return 0.0

        coverage_matrix = graph.coverage_matrix
        demand_weights = graph.demand_weights

        covered = torch.zeros(graph.n_demand, dtype=torch.bool, device=self.device)
        for idx in selected_indices:
            covered = covered | (coverage_matrix[:, idx] > 0)

        return torch.sum(demand_weights[covered]).item()

    def save_model(self, path='improved_tourism_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"模型已保存: {path}")

    def load_model(self, path='improved_tourism_model.pth'):
        if self.model is None:
            raise ValueError("请先初始化模型")

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型已加载: {path}")
        else:
            print(f"模型文件不存在: {path}")


# ========== 训练函数 ==========
def train_improved_model(dataset, val_ratio=0.2, K_range=[5, 8, 10, 12, 15, 20],
                         epochs_per_instance=20):
    """训练改进的文旅MCLP模型"""

    n_train = int(len(dataset) * (1 - val_ratio))
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:]

    print(f"\n训练集: {len(train_dataset)} 个实例")
    print(f"验证集: {len(val_dataset)} 个实例")

    # 获取输入维度
    sample = train_dataset[0]
    sample_graph = build_tourism_mclp_graph(
        sample['demand_points'],
        sample['facility_candidates'],
        sample['demand_weights'],
        sample['service_radius'],
        sample.get('scenic_labels'),
        device=device
    )
    input_dim = sample_graph.x.shape[1]
    print(f"输入特征维度: {input_dim}")

    # 初始化求解器
    solver = ImprovedTourismMCLPSolver(device=device)
    solver.initialize_model(input_dim=input_dim, hidden_dim=256, output_dim=128)

    print(f"\n开始训练...")
    print(f"K值范围: {K_range}")
    print(f"每个实例训练轮数: {epochs_per_instance}")
    print("-" * 50)

    all_losses = []
    best_val_coverage = 0

    for epoch, instance in enumerate(train_dataset):
        graph = build_tourism_mclp_graph(
            instance['demand_points'],
            instance['facility_candidates'],
            instance['demand_weights'],
            instance['service_radius'],
            instance.get('scenic_labels'),
            device=device
        )

        # 根据实例规模动态选择K
        n_demand = instance['n_demand']
        n_facility = instance['n_facility']
        suggested_K = min(max(5, n_demand // 20), n_facility // 2)

        available_K = [k for k in K_range if k <= suggested_K]
        if len(available_K) == 0:
            available_K = [min(suggested_K, 5)]

        if epoch < len(train_dataset) // 3:
            K = np.random.choice(available_K)
        else:
            K = np.random.choice([k for k in K_range if k <= n_facility])

        # 训练
        losses = solver.train_on_instance(graph, K=K, epochs=epochs_per_instance)
        all_losses.extend(losses)

        # 验证
        if (epoch + 1) % 5 == 0 and len(val_dataset) > 0:
            val_coverages = []
            for val_instance in val_dataset[:3]:
                val_graph = build_tourism_mclp_graph(
                    val_instance['demand_points'],
                    val_instance['facility_candidates'],
                    val_instance['demand_weights'],
                    val_instance['service_radius'],
                    val_instance.get('scenic_labels'),
                    device=device
                )
                _, val_coverage = solver.solve(val_graph, K=10, num_trials=5)
                total_demand = torch.sum(val_graph.demand_weights).item()
                val_pct = (val_coverage / total_demand) * 100
                val_coverages.append(val_pct)

            avg_val_coverage = np.mean(val_coverages)
            print(f"Epoch {epoch + 1}: 验证覆盖率 = {avg_val_coverage:.1f}%")

            if avg_val_coverage > best_val_coverage:
                best_val_coverage = avg_val_coverage
                solver.save_model('best_improved_model.pth')
                print(f"  → 保存最佳模型 (覆盖率: {avg_val_coverage:.1f}%)")

        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(all_losses[-20:]) if len(all_losses) >= 20 else np.mean(all_losses)
            print(f"Epoch {epoch + 1}/{len(train_dataset)} | 平均损失: {avg_loss:.4f}")

    print(f"\n训练完成！最佳验证覆盖率: {best_val_coverage:.1f}%")
    solver.save_model('final_improved_model.pth')

    return solver


# ========== 主函数 ==========
def main():
    print("=" * 60)
    print("改进的文旅场景MCLP模型")
    print("=" * 60)

    csv_path = 'tourism_poi_beijing.csv'

    if not os.path.exists(csv_path):
        print(f"错误: 数据文件 {csv_path} 不存在!")
        return

    # 创建改进的数据集
    print("\n创建改进的文旅MCLP数据集...")
    dataset = create_improved_tourism_dataset(
        csv_path=csv_path,
        num_instances=30,
        service_radius=0.1,  # 增大到0.15（约16.5公里）
        n_facility_candidates=120,
        facility_ratio=0.6
    )

    print(f"\n数据集大小: {len(dataset)} 个实例")
    print(f"示例实例: 需求点={dataset[0]['n_demand']}, 设施候选={dataset[0]['n_facility']}")

    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.7)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    print(f"\n训练集: {len(train_dataset)} 个实例")
    print(f"测试集: {len(test_dataset)} 个实例")

    # 训练模型
    solver = train_improved_model(
        train_dataset,
        val_ratio=0.2,
        K_range=[5, 8, 10, 12, 15, 20, 25],
        epochs_per_instance=20
    )

    # 测试模型
    print("\n" + "=" * 50)
    print("改进模型测试")
    print("=" * 50)

    results = {}
    for K in [5, 10, 15, 20, 25]:
        results[K] = []

    for i, instance in enumerate(test_dataset[:5]):
        graph = build_tourism_mclp_graph(
            instance['demand_points'],
            instance['facility_candidates'],
            instance['demand_weights'],
            instance['service_radius'],
            instance.get('scenic_labels'),
            device=device
        )
        total_demand = torch.sum(graph.demand_weights).item()

        print(f"\n测试实例 {i + 1}: 需求点={instance['n_demand']}, 设施候选={instance['n_facility']}")

        for K in [5, 10, 15, 20, 25]:
            selected, coverage = solver.solve(graph, K=min(K, instance['n_facility']), num_trials=10)
            coverage_pct = (coverage / total_demand) * 100
            results[K].append(coverage_pct)
            print(f"  K={K:2d}: 覆盖率={coverage_pct:5.1f}%")

    print("\n" + "-" * 50)
    print("各K值平均覆盖率:")
    for K in [5, 10, 15, 20, 25]:
        avg = np.mean(results[K])
        print(f"K={K:2d}: {avg:5.1f}%")

    overall_avg = np.mean([c for covs in results.values() for c in covs])
    print(f"\n总体平均覆盖率: {overall_avg:.1f}%")


if __name__ == "__main__":
    main()