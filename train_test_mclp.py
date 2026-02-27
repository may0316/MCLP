# train_test_mclp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch_geometric
import numpy as np
import pandas as pd
import pickle
import os
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

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


# ========== 从真实数据创建数据集 ==========
def create_tourism_dataset_from_csv(csv_path, num_instances=20,
                                   service_radius=0.05,  # 约5.5公里（经纬度单位）
                                   n_facility_candidates=50,
                                   facility_ratio=0.3):  # 设施候选点数量 = 需求点数量 * facility_ratio
    """
    从CSV创建文旅MCLP数据集（分离需求点和设施点）

    根据论文中的方法：
    - 需求点：从POI中采样
    - 设施点：从POI中随机选择（模拟建筑/可用场地）
    - 需求量：使用CSV中的demand列
    """
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 个点位")

    dataset = []

    for i in range(num_instances):
        # 随机采样80%作为需求点
        sampled_df = df.sample(frac=0.8, random_state=42 + i)
        demand_points = sampled_df[['x', 'y']].values
        demand_weights = sampled_df['demand'].values

        # 归一化坐标（重要：使距离计算合理）
        center = demand_points.mean(axis=0)
        scale = demand_points.std(axis=0).max()
        demand_points_norm = (demand_points - center) / (scale + 1e-8)

        # 从所有点中随机选择设施候选点
        # 设施点数量 = 需求点数量 * facility_ratio
        n_facility = max(10, int(len(demand_points) * facility_ratio))
        facility_indices = np.random.choice(len(df), n_facility, replace=False)
        facility_points = df.iloc[facility_indices][['x', 'y']].values
        facility_points_norm = (facility_points - center) / (scale + 1e-8)

        # 生成场景标签（用于特征）
        scenic_labels = []
        for t in sampled_df['type']:
            type_str = str(t)
            if '风景名胜' in type_str or '风景区' in type_str or '公园' in type_str:
                scenic_labels.append(1.0)  # 高需求区域
            elif '度假' in type_str or '休闲' in type_str:
                scenic_labels.append(0.5)  # 中等需求区域
            else:
                scenic_labels.append(0.0)  # 一般区域
        scenic_labels = np.array(scenic_labels)

        instance = {
            'name': f'tourism_instance_{i:04d}',
            'instance_id': i,
            'demand_points': demand_points_norm,
            'facility_candidates': facility_points_norm,
            'demand_weights': demand_weights,
            'scenic_labels': scenic_labels,
            'service_radius': service_radius,
            'n_demand': len(demand_points),
            'n_facility': len(facility_points)
        }

        dataset.append(instance)

        if (i + 1) % 5 == 0:
            print(f"已创建 {i + 1}/{num_instances} 个实例")

    return dataset


# ========== 文旅MCLP编码器 ==========
class TourismMCLPEncoder(nn.Module):
    """文旅MCLP编码器（处理需求点和设施点的二分图）"""

    def __init__(self, in_channels, hidden_channels=256, out_channels=128, dropout=0.2):
        super().__init__()

        # 使用GAT处理异构图
        self.conv1 = GATConv(in_channels, hidden_channels // 4, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels, out_channels, heads=1, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(self.bn1(h))
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = F.relu(self.bn2(h))
        h = self.dropout(h)

        h = self.conv3(h, edge_index)
        h = F.relu(self.bn3(h))

        return h


# ========== 文旅MCLP模型 ==========
class TourismMCLPModel(nn.Module):
    """文旅MCLP模型"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()

        self.encoder = TourismMCLPEncoder(input_dim, hidden_dim, output_dim)

        # 设施点选择头
        self.facility_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 需求点覆盖预测头
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

        emb = self.encoder(x, edge_index)

        # 分离需求点和设施点的嵌入
        demand_emb = emb[:graph.n_demand]
        facility_emb = emb[graph.n_demand:]

        facility_scores = self.facility_head(facility_emb).squeeze()
        coverage_pred = self.coverage_head(demand_emb).squeeze()
        value_pred = self.value_head(emb).squeeze()

        return demand_emb, facility_emb, facility_scores, coverage_pred, value_pred


# ========== 文旅MCLP求解器 ==========
class TourismMCLPSolver:
    """文旅MCLP求解器"""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def initialize_model(self, input_dim, hidden_dim=256, output_dim=128):
        """初始化模型"""
        self.model = TourismMCLPModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        print(f"文旅MCLP模型初始化: 输入维度={input_dim}")
        return self.model

    def tourism_mclp_loss(self, facility_scores, coverage_pred, value_pred, graph, K, alpha=0.3, beta=0.1):
        """文旅MCLP损失函数（自监督）"""
        coverage_matrix = graph.coverage_matrix
        demand_weights = graph.demand_weights

        n_facility = graph.n_facility

        # 使用Gumbel-Softmax进行可微选择
        temperature = max(0.2, 1.0 - 0.5 * (K / n_facility))
        p = F.gumbel_softmax(facility_scores.unsqueeze(0), tau=temperature, hard=False, dim=1).squeeze()
        p = p * K

        # 计算覆盖概率
        facility_probs = p.unsqueeze(0)  # [1, n_facility]
        covered_prob = torch.max(facility_probs * coverage_matrix, dim=1)[0]
        covered_prob = torch.clamp(covered_prob, 0, 1)

        # 主目标：加权覆盖
        weighted_coverage = torch.sum(demand_weights * covered_prob)
        total_demand = torch.sum(demand_weights)
        coverage_loss = -weighted_coverage / (total_demand + 1e-8)

        # 设施数量约束
        facility_count_loss = (p.sum() - K) ** 2 * 0.05

        # 辅助任务1：预测覆盖
        with torch.no_grad():
            _, top_indices = torch.topk(facility_scores, min(K, n_facility))
            selected_mask = torch.zeros(n_facility, device=self.device)
            selected_mask[top_indices] = 1.0
            coverage_target = (torch.max(selected_mask.unsqueeze(0) * coverage_matrix, dim=1)[0] > 0).float()

        aux_loss1 = F.binary_cross_entropy(coverage_pred, coverage_target)

        # 辅助任务2：预测价值
        with torch.no_grad():
            value_target = torch.zeros(len(value_pred), device=self.device)
            # 需求点的价值 = 其需求权重
            value_target[:graph.n_demand] = demand_weights / (total_demand + 1e-8)
            # 设施点的价值 = 其覆盖的需求权重
            for j in range(n_facility):
                covered = coverage_matrix[:, j] > 0
                if covered.any():
                    value_target[graph.n_demand + j] = demand_weights[covered].sum() / (total_demand + 1e-8)

        aux_loss2 = F.mse_loss(value_pred, value_target)

        # 分散度奖励（避免设施过于集中）
        if len(top_indices) > 1:
            facility_pos = graph.facility_pos
            selected_pos = facility_pos[top_indices]

            pairwise_dist = torch.cdist(selected_pos, selected_pos)
            mask = 1 - torch.eye(len(selected_pos), device=self.device)
            avg_dist = (pairwise_dist * mask).sum() / (len(selected_pos) * (len(selected_pos) - 1) + 1e-8)

            # 奖励大的平均距离
            dispersion_reward = -beta * (graph.service_radius * 2) / (avg_dist + 1e-8)
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

            _, _, facility_scores, coverage_pred, value_pred = self.model(graph)
            loss, coverage = self.tourism_mclp_loss(facility_scores, coverage_pred, value_pred, graph, K)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            losses.append(loss.item())

        return losses

    @torch.no_grad()
    def solve(self, graph, K, num_trials=10):
        """求解MCLP问题"""
        self.model.eval()
        graph = graph.to(self.device)

        coverage_matrix = graph.coverage_matrix
        demand_weights = graph.demand_weights
        n_facility = graph.n_facility

        best_coverage = 0
        best_selection = None

        for trial in range(num_trials):
            _, _, facility_scores, _, _ = self.model(graph)

            # 添加噪声进行探索
            noise_scale = 0.1 * (1 - trial / num_trials)
            if trial > 0:
                noise = torch.randn_like(facility_scores) * noise_scale
                scores = facility_scores + noise
            else:
                scores = facility_scores

            # 贪心选择
            selected = []
            covered = torch.zeros(graph.n_demand, dtype=torch.bool, device=self.device)

            # 按得分排序后贪心
            sorted_indices = torch.argsort(scores, descending=True)

            for idx in sorted_indices:
                if len(selected) >= K:
                    break

                newly_covered = (coverage_matrix[:, idx] > 0) & (~covered)
                if newly_covered.any() or len(selected) < K:
                    selected.append(idx.item())
                    covered = covered | (coverage_matrix[:, idx] > 0)

            # 如果还不够，补全
            if len(selected) < K:
                for idx in range(n_facility):
                    if idx not in selected and len(selected) < K:
                        selected.append(idx)

            selected_tensor = torch.tensor(selected, device=self.device)

            # 计算覆盖
            covered_mask = torch.zeros(graph.n_demand, dtype=torch.bool, device=self.device)
            for idx in selected:
                covered_mask = covered_mask | (coverage_matrix[:, idx] > 0)
            coverage = torch.sum(demand_weights[covered_mask]).item()

            if coverage > best_coverage:
                best_coverage = coverage
                best_selection = selected_tensor

        return best_selection, best_coverage

    def save_model(self, path='tourism_mclp_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, path)
        print(f"模型已保存: {path}")

    def load_model(self, path='tourism_mclp_model.pth'):
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
def train_tourism_model(dataset, val_ratio=0.2, K_range=[3, 5, 8, 10, 12, 15], epochs_per_instance=15):
    """训练文旅MCLP模型"""

    # 划分训练集和验证集
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
    solver = TourismMCLPSolver(device=device)
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
                val_graph = build_tourism_mclp_graph(
                    val_instance['demand_points'],
                    val_instance['facility_candidates'],
                    val_instance['demand_weights'],
                    val_instance['service_radius'],
                    val_instance.get('scenic_labels'),
                    device=device
                )
                _, val_coverage = solver.solve(val_graph, K=8, num_trials=3)
                total_demand = torch.sum(val_graph.demand_weights).item()
                val_pct = (val_coverage / total_demand) * 100
                val_coverages.append(val_pct)

            avg_val_coverage = np.mean(val_coverages)
            print(f"Epoch {epoch + 1}: 验证覆盖率 = {avg_val_coverage:.1f}%")

            if avg_val_coverage > best_val_coverage:
                best_val_coverage = avg_val_coverage
                solver.save_model('best_tourism_model.pth')
                print(f"  → 保存最佳模型 (覆盖率: {avg_val_coverage:.1f}%)")

        # 打印进度
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(all_losses[-10:]) if len(all_losses) >= 10 else np.mean(all_losses)
            print(f"Epoch {epoch + 1}/{len(train_dataset)} | 平均损失: {avg_loss:.4f}")

    print(f"\n训练完成！最佳验证覆盖率: {best_val_coverage:.1f}%")
    solver.save_model('final_tourism_model.pth')

    return solver


# ========== 测试函数 ==========
def test_tourism_model(solver, test_instances, K_values=[3, 5, 8, 10, 12, 15]):
    """测试模型"""
    print("\n" + "=" * 50)
    print("文旅MCLP模型测试")
    print("=" * 50)

    results = {}

    for K in K_values:
        results[K] = []

    for i, instance in enumerate(test_instances[:5]):  # 测试前5个实例
        graph = build_tourism_mclp_graph(
            instance['demand_points'],
            instance['facility_candidates'],
            instance['demand_weights'],
            instance['service_radius'],
            instance.get('scenic_labels'),
            device=device
        )
        total_demand = torch.sum(graph.demand_weights).item()

        print(f"\n测试实例 {i + 1}: {instance['name']}, 需求点={instance['n_demand']}, 设施候选={instance['n_facility']}")

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
def visualize_tourism_solution(graph, selected_indices, K, coverage, save_path=None):
    """可视化文旅MCLP求解结果"""
    demand_points = graph.demand_pos.cpu().numpy()
    facility_points = graph.facility_pos.cpu().numpy()
    selected = selected_indices.cpu().numpy()
    demand_weights = graph.demand_weights.cpu().numpy()

    plt.figure(figsize=(14, 10))

    # 绘制需求点（颜色表示需求权重）
    scatter = plt.scatter(demand_points[:, 0], demand_points[:, 1],
                          c=demand_weights, cmap='YlOrRd', s=50, alpha=0.7,
                          label='Demand Points')
    plt.colorbar(scatter, label='Demand Weight')

    # 绘制所有候选设施点
    plt.scatter(facility_points[:, 0], facility_points[:, 1],
                c='lightblue', s=80, marker='s', alpha=0.5,
                label='Candidate Facilities')

    # 绘制选中的设施点
    plt.scatter(facility_points[selected, 0], facility_points[selected, 1],
                c='red', s=200, marker='*', edgecolors='black', linewidths=2,
                label=f'Selected Facilities (K={K})')

    # 绘制服务半径（只画前5个，避免太乱）
    R = graph.service_radius
    for i, idx in enumerate(selected[:min(5, len(selected))]):
        circle = plt.Circle(facility_points[idx], R, color='red',
                            fill=False, linestyle='--', alpha=0.3, linewidth=1.5)
        plt.gca().add_patch(circle)

    total_demand = torch.sum(graph.demand_weights).item()
    coverage_pct = (coverage / total_demand) * 100

    plt.title(f'Tourism MCLP Solution - Coverage: {coverage:.1f} ({coverage_pct:.1f}%)', fontsize=14)
    plt.xlabel('X (normalized)', fontsize=12)
    plt.ylabel('Y (normalized)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ========== 主函数 ==========
def main():
    print("=" * 60)
    print("文旅场景MCLP模型训练与测试")
    print("=" * 60)

    # 检查数据文件
    csv_path = 'tourism_poi_beijing.csv'

    if not os.path.exists(csv_path):
        print(f"错误: 数据文件 {csv_path} 不存在!")
        print("请确保 tourism_poi_beijing.csv 在当前目录下")
        return

    # 创建数据集
    print("\n从真实数据创建文旅MCLP数据集...")
    dataset = create_tourism_dataset_from_csv(
        csv_path=csv_path,
        num_instances=30,  # 创建30个实例
        service_radius=0.05,  # 服务半径（归一化后约对应5.5公里）
        n_facility_candidates=40,  # 每个实例的设施候选点数量
        facility_ratio=0.3  # 设施候选点数量 = 需求点数量 * 0.3
    )

    print(f"\n数据集大小: {len(dataset)} 个实例")
    print(f"每个实例: 需求点 ~{dataset[0]['n_demand']}, 设施候选 ~{dataset[0]['n_facility']}")

    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.7)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    print(f"\n训练集: {len(train_dataset)} 个实例")
    print(f"测试集: {len(test_dataset)} 个实例")

    # 训练模型
    solver = train_tourism_model(
        train_dataset,
        val_ratio=0.2,
        K_range=[3, 5, 8, 10, 12, 15],
        epochs_per_instance=15
    )

    # 测试模型
    test_tourism_model(solver, test_dataset, K_values=[3, 5, 8, 10, 12, 15])

    # 可视化一个结果
    print("\n可视化求解结果...")
    full_instance = dataset[0]  # 用第一个实例
    full_graph = build_tourism_mclp_graph(
        full_instance['demand_points'],
        full_instance['facility_candidates'],
        full_instance['demand_weights'],
        full_instance['service_radius'],
        full_instance.get('scenic_labels'),
        device=device
    )
    selected, coverage = solver.solve(full_graph, K=10, num_trials=10)
    visualize_tourism_solution(full_graph, selected, 10, coverage, save_path='tourism_solution.png')


if __name__ == "__main__":
    main()