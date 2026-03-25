# FLP.py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
import os
import pickle
import matplotlib.pyplot as plt
import sys
from config import MCLPConfig

# 设备设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 获取配置
if len(sys.argv) > 1:
    # 用法: python FLP.py graph_size p radius
    graph_size = int(sys.argv[1])
    p = int(sys.argv[2])
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    config = MCLPConfig(graph_size=graph_size, p=p, radius=radius)
else:
    # 使用默认配置
    config = MCLPConfig()

print(f"使用配置:\n{config}")

# 设备设置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class GCN_MCLP(torch.nn.Module):
    """MCLP的GCN编码器"""
    def __init__(self, in_channels, hidden_channels, out_channels, n_nodes):
        super(GCN_MCLP, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.be = nn.BatchNorm1d(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bd = nn.BatchNorm1d(out_channels)

        self.fc_distance = nn.Linear(1, out_channels)
        self.fc_degree = nn.Linear(1, out_channels)
        self.fc_fusion = nn.Linear(3 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, distance_features, degree_features):
        """
        Args:
            x: 节点特征 (n_nodes, in_channels)
            edge_index: 边索引
            edge_weight: 边权重
            distance_features: 距离特征 (n_nodes, 1)
            degree_features: 度特征 (n_nodes, 1)
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, 0)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn(x)
        x = F.relu(x)

        distance_enc = self.fc_distance(distance_features)
        distance_enc = self.be(distance_enc)
        distance_enc = F.relu(distance_enc)

        degree_enc = self.fc_degree(degree_features.float())
        degree_enc = self.bd(degree_enc)
        degree_enc = F.relu(degree_enc)

        x_concat = torch.cat((x, distance_enc, degree_enc), dim=1)
        x_concat = self.fc_fusion(x_concat)
        return x_concat


class MocoModel_MCLP(torch.nn.Module):
    """MoCo模型用于MCLP"""
    def __init__(self, dim_in, dim_hidden, dim_out, n_nodes, m=0.99, K=256):
        super().__init__()
        self.m = m
        self.K = K

        self.q_net = GCN_MCLP(dim_in, dim_hidden, dim_out, n_nodes)
        self.k_net = GCN_MCLP(dim_in, dim_hidden, dim_out, n_nodes)

        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim_out, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, idx, x, edge_index, edge_weight, distance_features, degree_features, batch_size):
        embs_q = self.q_net(x, edge_index, edge_weight, distance_features, degree_features)
        embs_q = F.normalize(embs_q, dim=1)
        
        if batch_size == x.shape[0]:
            return embs_q
            
        q = embs_q[idx * batch_size:(idx + 1) * batch_size, :]

        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, distance_features, degree_features)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch_size:(idx + 1) * batch_size, :]

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.07

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        self._dequeue_and_enqueue(k)

        return embs_q, logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


class MLP_Solver(torch.nn.Module):
    """MLP求解器，基于自监督学习的嵌入进行设施选择"""
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, embeddings):
        scores = self.fc(embeddings)
        return scores.squeeze(-1)


def compute_coverage_graph(points, radius, demand_weights, device):
    """根据覆盖半径构建图"""
    n_nodes = points.shape[0]
    
    diff = points.unsqueeze(1) - points.unsqueeze(0)
    dist_matrix = torch.norm(diff, dim=2)
    
    mask = dist_matrix <= radius
    mask.fill_diagonal_(True)
    edge_index = mask.nonzero(as_tuple=False).t()
    edge_weight = dist_matrix[edge_index[0], edge_index[1]].unsqueeze(1)
    
    covered_demand = (mask.float() * demand_weights.unsqueeze(0)).sum(dim=1).unsqueeze(1)
    
    return edge_index, edge_weight, covered_demand, dist_matrix


def compute_weighted_coverage_value(selected_facilities, dist_matrix, radius, demand_weights):
    """
    计算加权覆盖值：被至少一个设施覆盖的节点的需求权重之和
    """
    if len(selected_facilities) == 0:
        return 0
    
    min_dist_to_facility = dist_matrix[:, selected_facilities].min(dim=1)[0]
    covered = (min_dist_to_facility <= radius).float()
    weighted_coverage = (covered * demand_weights).sum()
    return weighted_coverage.item()


def rounding(x, K):
    """
    Fast pipage rounding for MCLP
    """
    x = x.clone()
    x = torch.clamp(x, 0, 1)
    
    if x.sum() != K:
        x = K * x / x.sum()
    
    sorted_indices = torch.argsort(x, descending=True)
    y = torch.zeros_like(x)
    y[sorted_indices[:K]] = 1
    
    return y


def compute_distance_features(points, selected_facilities, device):
    """计算每个节点到最近设施的距离"""
    if selected_facilities is None or len(selected_facilities) == 0:
        return torch.ones(points.shape[0], 1).to(device)
    
    n_nodes = points.shape[0]
    diff = points.unsqueeze(1) - points[selected_facilities].unsqueeze(0)
    dist_to_facilities = torch.norm(diff, dim=2)
    min_dist, _ = dist_to_facilities.min(dim=1)
    
    return min_dist.unsqueeze(1)


def solve_mclp_with_self_supervised(model_encoder, solver_mlp, points, demand_weights, 
                                    radius, p, num_epochs=500, lr=0.001, device=device):
    """
    使用自监督模型求解MCLP
    """
    n_nodes = points.shape[0]
    
    # 构建覆盖图
    edge_index, edge_weight, degree_features, dist_matrix = compute_coverage_graph(
        points, radius, demand_weights, device
    )
    
    # 节点特征（坐标）
    x = points
    
    # 初始距离特征（使用随机选择的设施）
    random_indices = torch.randperm(n_nodes)[:min(p, n_nodes)]
    distance_features = compute_distance_features(points, random_indices, device)
    
    # 获取嵌入
    with torch.no_grad():
        embeddings = model_encoder(x, edge_index, edge_weight, 
                                   distance_features, degree_features)
    
    # 优化MLP求解器
    optimizer = torch.optim.Adam(solver_mlp.parameters(), lr=lr, weight_decay=1e-4)
    
    best_solution = None
    best_weighted_coverage = 0
    best_simple_coverage = 0
    
    for epoch in range(num_epochs):
        # 前向传播
        scores = solver_mlp(embeddings)
        probs = torch.sigmoid(scores)
        
        # 规范化以确保和接近p
        if probs.sum() > 0:
            probs = p * probs / probs.sum()
        
        # 获取候选设施
        _, candidate_indices = torch.topk(probs, min(p, n_nodes))
        
        # 计算加权覆盖值（目标函数）
        weighted_coverage = compute_weighted_coverage_value(
            candidate_indices, dist_matrix, radius, demand_weights
        )
        
        # 也计算简单覆盖（用于参考）
        simple_coverage = compute_simple_coverage_value(
            candidate_indices, dist_matrix, radius
        )
        
        # 损失函数：最大化加权覆盖
        total_demand = demand_weights.sum().item()
        loss = -torch.tensor(weighted_coverage / total_demand, dtype=torch.float32, device=device)
        
        # 添加熵正则化鼓励探索
        entropy = -(probs * torch.log(probs + 1e-8)).mean()
        loss = loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 定期更新距离特征和嵌入
        if epoch % 50 == 0:
            with torch.no_grad():
                distance_features = compute_distance_features(points, candidate_indices, device)
                embeddings = model_encoder(x, edge_index, edge_weight,
                                          distance_features, degree_features)
        
        # 记录最佳解
        if weighted_coverage > best_weighted_coverage:
            best_weighted_coverage = weighted_coverage
            best_simple_coverage = simple_coverage
            best_solution = candidate_indices.clone()
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}, "
                  f"Weighted Coverage: {weighted_coverage:.2f}/{total_demand:.2f} "
                  f"({weighted_coverage/total_demand*100:.1f}%), "
                  f"Simple Coverage: {simple_coverage}/{n_nodes}")
    
    return best_solution, best_weighted_coverage, best_simple_coverage


def compute_simple_coverage_value(selected_facilities, dist_matrix, radius):
    """计算简单覆盖值（不计权重）"""
    if len(selected_facilities) == 0:
        return 0
    
    min_dist_to_facility = dist_matrix[:, selected_facilities].min(dim=1)[0]
    covered = (min_dist_to_facility <= radius).float()
    return covered.sum().item()


def load_mclp_data(data_path, device):
    """从pkl文件加载MCLP数据"""
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    instances = []
    for instance in dataset:
        points = instance['loc'].to(device)
        demand = instance['demand'].to(device)
        p = instance['p']
        radius = instance['radius']
        instances.append((points, demand, p, radius))
    
    return instances


def visualize_solution(points, facilities, radius, demand_weights, idx, config):
    """可视化MCLP解，节点大小表示需求权重"""
    plt.figure(figsize=(10, 8))
    
    points_np = points.cpu().numpy()
    facilities_np = facilities.cpu().numpy()
    demand_np = demand_weights.cpu().numpy()
    
    # 归一化节点大小（需求权重越大，节点越大）
    sizes = 50 + 200 * demand_np / demand_np.max()
    
    # 绘制所有点
    scatter = plt.scatter(points_np[:, 0], points_np[:, 1], 
                          c=demand_np, s=sizes, cmap='viridis', 
                          alpha=0.6, label='Demand points')
    plt.colorbar(scatter, label='Demand Weight')
    
    # 绘制设施点
    plt.scatter(points_np[facilities_np, 0], points_np[facilities_np, 1], 
                c='red', s=150, marker='s', edgecolors='black', 
                linewidths=2, label='Facilities')
    
    # 绘制覆盖圆
    for fidx in facilities_np:
        circle = plt.Circle((points_np[fidx, 0], points_np[fidx, 1]), radius, 
                           color='red', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
    
    total_demand = demand_np.sum()
    covered_demand = demand_np[(dist_to_facilities(points_np, facilities_np) <= radius).any(axis=1)].sum()
    
    plt.title(f'MCLP Solution (p={len(facilities)}, radius={radius})\n'
              f'Weighted Coverage: {covered_demand:.2f}/{total_demand:.2f} ({covered_demand/total_demand*100:.1f}%)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.savefig(f'mclp_solution_{config.graph_size}_{config.p}_{idx}.png', dpi=150)
    plt.show()
    plt.close()


def dist_to_facilities(points, facilities):
    """计算每个点到设施的最小距离"""
    n_nodes = points.shape[0]
    dist_matrix = np.zeros((n_nodes, len(facilities)))
    for i, f in enumerate(facilities):
        dist_matrix[:, i] = np.linalg.norm(points - points[f], axis=1)
    return dist_matrix


def main():
    # 检查数据文件是否存在
    if not os.path.exists(config.data_path):
        print(f"数据文件不存在: {config.data_path}")
        print("请先运行: python sponet_gen_data.py")
        return
    
    # 加载数据
    instances = load_mclp_data(config.data_path, device)
    print(f"加载了 {len(instances)} 个实例")
    
    # 加载预训练模型
    pretrained_path = config.pretrained_path
    n_nodes = config.graph_size
    model = MocoModel_MCLP(2, config.hidden_dim, config.out_dim, n_nodes).to(device)
    
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"加载预训练模型: {pretrained_path}")
        encoder = model.q_net
    else:
        print("未找到预训练模型，使用随机初始化")
        encoder = model.q_net
    
    # 冻结编码器
    for param in encoder.parameters():
        param.requires_grad = False
    
    # 测试每个实例
    results = []
    num_test = min(10, len(instances))
    
    for idx in range(num_test):
        points, demand, p, radius = instances[idx]
        total_demand = demand.sum().item()
        print(f"\n求解实例 {idx + 1}/{num_test}: n={points.shape[0]}, p={p}, radius={radius}, "
              f"总需求权重={total_demand:.4f}")
        
        start_time = time.time()
        
        # 创建MLP求解器
        solver = MLP_Solver(config.out_dim, config.hidden_dim // 4).to(device)
        
        # 求解
        solution, weighted_coverage, simple_coverage = solve_mclp_with_self_supervised(
            encoder, solver, points, demand, radius, p,
            num_epochs=config.num_epochs_solve, lr=config.lr, device=device
        )
        
        elapsed_time = time.time() - start_time
        
        coverage_rate = weighted_coverage / total_demand * 100
        print(f"  加权覆盖值: {weighted_coverage:.4f}/{total_demand:.4f} ({coverage_rate:.1f}%)")
        print(f"  简单覆盖节点数: {simple_coverage}/{points.shape[0]}")
        print(f"  选择设施: {solution.tolist()}")
        print(f"  求解时间: {elapsed_time:.2f}秒")
        
        results.append({
            'weighted_coverage': weighted_coverage,
            'total_demand': total_demand,
            'coverage_rate': coverage_rate,
            'simple_coverage': simple_coverage,
            'total_nodes': points.shape[0],
            'p': p,
            'radius': radius,
            'solution': solution,
            'time': elapsed_time
        })
        
        # 可视化
        if idx < 3:
            visualize_solution(points, solution, radius, demand, idx, config)
    
    # 输出汇总
    print("\n" + "="*50)
    print("汇总结果:")
    if results:
        avg_coverage_rate = np.mean([r['coverage_rate'] for r in results])
        avg_weighted_coverage = np.mean([r['weighted_coverage'] for r in results])
        avg_simple_coverage = np.mean([r['simple_coverage'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        
        print(f"平均加权覆盖率: {avg_coverage_rate:.1f}%")
        print(f"平均加权覆盖值: {avg_weighted_coverage:.4f}")
        print(f"平均简单覆盖节点数: {avg_simple_coverage:.1f}")
        print(f"平均求解时间: {avg_time:.2f}秒")
        print(f"最佳加权覆盖率: {max([r['coverage_rate'] for r in results]):.1f}%")
        print(f"最差加权覆盖率: {min([r['coverage_rate'] for r in results]):.1f}%")


if __name__ == "__main__":
    main()