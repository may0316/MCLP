import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data, DataLoader
import os
import create_more
import pickle
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
n_nodes = 500

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_nodes):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.be = nn.BatchNorm1d(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bd = nn.BatchNorm1d(out_channels)

        self.fc_0 = nn.Linear(1, out_channels)
        self.fc_1 = nn.Embedding(500, out_channels)
        self.fc_2 = nn.Linear(3 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, edges, degree):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, 0)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn(x)
        x = F.relu(x)

        edges = self.fc_0(edges)
        edges = self.be(edges)
        edges = F.relu(edges)

        degree = self.fc_1(degree)
        degree = self.bd(degree)
        degree = F.relu(degree)

        x_concat = torch.cat((x, edges, degree), dim=1)
        x_concat = self.fc_2(x_concat)
        return x_concat


class MocoModel(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_nodes, m=0.99, K=256):
        super().__init__()
        self.m = m
        self.K = K

        self.q_net = GCN(dim_in, dim_hidden, dim_out, n_nodes)
        self.k_net = GCN(dim_in, dim_hidden, dim_out, n_nodes)

        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim_out, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_w", torch.randn(n_nodes, K))
        self.queue_w = nn.functional.normalize(self.queue_w, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, idx, x, edge_index, edge_weight, edge, degree, batch):
        embs_q = self.q_net(x, edge_index, edge_weight, edge, degree)
        embs_q = F.normalize(embs_q, dim=1)
        
        if batch == x.shape[0]:
            return embs_q
        
        q = embs_q[idx * batch:(idx + 1) * batch, :]

        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, edge, degree)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch:(idx + 1) * batch, :]

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

        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


import torch.optim as optim

device = torch.device('cpu')
train_dataset = create_more.load_dataset('dataset_800.pkl')
model = MocoModel(2, 128, 64, n_nodes).to(device)
model.load_state_dict(torch.load('pre_mclp.pth'))
K = 50


class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), 
            nn.ReLU(), 
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, embs):
        mlp_embs = self.fc(embs)
        return mlp_embs


def mclp_objective(y, dist_all, coverage_radius):
    """
    原始MCLP目标函数（用于评估）
    y: 选择设施的二进制向量 (n_nodes,)
    dist_all: 所有节点对之间的距离矩阵 (n_nodes, n_nodes)
    coverage_radius: 覆盖半径
    """
    selected_indices = torch.where(y == 1)[0]
    
    if len(selected_indices) > 0:
        dist_to_selected = dist_all[:, selected_indices]
        min_dist_to_facility = torch.min(dist_to_selected, dim=1)[0]
        covered_nodes = (min_dist_to_facility <= coverage_radius).float()
        return covered_nodes.sum()
    else:
        return torch.tensor(0.0)


def differentiable_mclp_objective(y_probs, dist_all, coverage_radius, temperature=0.1):
    """
    可微分的MCLP目标函数近似
    y_probs: 选择设施的概率向量 (n_nodes,)
    dist_all: 距离矩阵 (n_nodes, n_nodes)
    coverage_radius: 覆盖半径
    temperature: 温度参数，控制平滑程度
    """
    n_nodes = y_probs.shape[0]
    
    # 确保y_probs是概率值（0-1之间）
    if y_probs.min() < 0 or y_probs.max() > 1:
        y_probs = torch.sigmoid(y_probs)
    
    # 添加小量避免除零和log(0)
    eps = 1e-8
    y_probs = y_probs.clamp(eps, 1-eps)
    
    # 使用对数概率，增加数值稳定性
    log_y_probs = torch.log(y_probs + eps)
    
    # 计算每个节点到各个设施的加权距离
    # 使用负距离，使得较近的设施有较高的权重
    neg_dist = -dist_all
    weighted_logits = neg_dist / temperature + log_y_probs.unsqueeze(0)
    
    # 使用logsumexp计算软最小距离
    log_weights = F.log_softmax(weighted_logits, dim=1)
    
    # 计算期望最小距离
    expected_min_dist = torch.sum(torch.exp(log_weights) * dist_all, dim=1)
    
    # 使用sigmoid近似指示函数：距离是否小于覆盖半径
    # 使用较大的斜率使sigmoid更接近阶跃函数
    slope = 50.0 / coverage_radius
    coverage_probs = torch.sigmoid(slope * (coverage_radius - expected_min_dist))
    
    # 返回总覆盖概率
    total_coverage = torch.sum(coverage_probs)
    
    return total_coverage


def greedy_mclp_rounding(probabilities, K, dist_all, coverage_radius, n_iterations=20):
    """
    贪心舍入算法用于MCLP
    """
    n_nodes = len(probabilities)
    probabilities_np = probabilities.detach().cpu().numpy()
    dist_all_np = dist_all.detach().cpu().numpy()
    
    best_coverage = -1
    best_selection = None
    
    for iteration in range(n_iterations):
        selected = np.zeros(n_nodes, dtype=bool)
        remaining_budget = K
        
        # 首轮选择：基于概率和潜在覆盖
        # 计算每个设施的潜在覆盖率（覆盖半径内的节点数）
        potential_coverage = np.sum(dist_all_np <= coverage_radius, axis=0)
        
        # 综合得分：概率 * 潜在覆盖率
        combined_scores = probabilities_np * (potential_coverage + 1)
        
        # 选择前K个
        top_indices = np.argsort(combined_scores)[-K:]
        selected[top_indices] = True
        
        # 局部优化：尝试替换设施以提高覆盖率
        for local_iter in range(10):
            improved = False
            
            # 尝试用未选择的设施替换已选择的设施
            selected_indices = np.where(selected)[0]
            unselected_indices = np.where(~selected)[0]
            
            for s_idx in selected_indices:
                for u_idx in unselected_indices:
                    # 临时替换
                    temp_selected = selected.copy()
                    temp_selected[s_idx] = False
                    temp_selected[u_idx] = True
                    
                    # 计算新覆盖
                    temp_selected_idx = np.where(temp_selected)[0]
                    dist_to_selected = dist_all_np[:, temp_selected_idx]
                    min_dist = np.min(dist_to_selected, axis=1)
                    new_coverage = np.sum(min_dist <= coverage_radius)
                    
                    # 当前覆盖
                    current_selected_idx = np.where(selected)[0]
                    dist_to_current = dist_all_np[:, current_selected_idx]
                    current_min_dist = np.min(dist_to_current, axis=1)
                    current_coverage = np.sum(current_min_dist <= coverage_radius)
                    
                    if new_coverage > current_coverage:
                        selected = temp_selected
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        # 评估这个选择
        selected_tensor = torch.tensor(selected.astype(float)).to(device)
        coverage = mclp_objective(selected_tensor, dist_all, coverage_radius)
        coverage_value = coverage.item()
        
        if coverage_value > best_coverage:
            best_coverage = coverage_value
            best_selection = selected.copy()
    
    return torch.tensor(best_selection.astype(float)).to(device), best_coverage


# 主训练和求解循环
for index, (_, points) in enumerate(train_dataset):
    start = time.time()
    
    # 构建图
    graph, dist_all = create_more.build_graph_from_points(points, None, True, 'euclidean')
    diameter = dist_all.max().item()
    
    # 设置覆盖半径
    coverage_radius = 0.15 * diameter
    print(f"\nInstance {index}: diameter={diameter:.4f}, coverage_radius={coverage_radius:.4f}")
    print(f"Total nodes: {n_nodes}, Facilities to place: {K}")
    
    # 计算度矩阵
    adj_matrix = to_dense_adj(graph.edge_index)
    adj_matrix = torch.squeeze(adj_matrix)
    degree = torch.sum(adj_matrix, dim=0).to(device)
    degree = degree.long()
    
    # 获取预训练嵌入
    with torch.no_grad():
        embs = model(0, graph.x, graph.edge_index, graph.edge_attr, 
                     graph.dist_row, degree, graph.x.shape[0])
    
    # 创建并训练MLP
    model_mlp = MLP(64, 32, 1).to(device)
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    # 训练MLP
    best_train_coverage = 0
    patience = 30
    patience_counter = 0
    
    print("Training MLP...")
    for t in range(500):
        model_mlp.train()
        
        # 前向传播
        logits = model_mlp(embs).squeeze()  # (n_nodes,)
        
        # 计算选择概率
        y_probs = torch.sigmoid(logits)
        
        # 使用可微分的目标函数计算覆盖率
        coverage = differentiable_mclp_objective(y_probs, dist_all, coverage_radius, temperature=0.2)
        
        # 设施数量约束：鼓励选择大约K个设施
        facility_count = torch.sum(y_probs)
        count_penalty = torch.abs(facility_count - K) / K
        
        # 熵正则化：鼓励清晰的决策
        entropy = -torch.sum(y_probs * torch.log(y_probs + 1e-10) + 
                            (1 - y_probs) * torch.log(1 - y_probs + 1e-10)) / n_nodes
        
        # 总损失：负覆盖率 + 数量惩罚 + 熵正则化
        loss = -coverage / n_nodes + 0.1 * count_penalty + 0.01 * entropy
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_mlp.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 定期评估
        if t % 10 == 0:
            with torch.no_grad():
                model_mlp.eval()
                logits_eval = model_mlp(embs).squeeze()
                probabilities = torch.sigmoid(logits_eval)
                
                # 使用贪心舍入进行评估
                y_rounded, coverage_val = greedy_mclp_rounding(
                    probabilities, K, dist_all, coverage_radius, n_iterations=10
                )
                
                coverage_percentage = coverage_val / n_nodes * 100
                
                if coverage_val > best_train_coverage:
                    best_train_coverage = coverage_val
                    patience_counter = 0
                    torch.save(model_mlp.state_dict(), f'mclp_best_model_{index}.pth')
                    best_probabilities = probabilities.clone()
                else:
                    patience_counter += 1
                
                if t % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Iteration {t}: loss={loss.item():.4f}, "
                          f"coverage={coverage_val}/{n_nodes} ({coverage_percentage:.1f}%), "
                          f"LR={current_lr:.6f}")
                
                if patience_counter >= patience:
                    print(f"  Early stopping at iteration {t}")
                    break
    
    # 最终评估
    print("\nFinal evaluation...")
    model_mlp.eval()
    with torch.no_grad():
        # 加载最佳模型
        if os.path.exists(f'mclp_best_model_{index}.pth'):
            model_mlp.load_state_dict(torch.load(f'mclp_best_model_{index}.pth'))
        
        # 获取最终概率
        probabilities = torch.sigmoid(model_mlp(embs).squeeze())
        
        # 使用贪心舍入获得最终解
        final_solution, final_coverage = greedy_mclp_rounding(
            probabilities, K, dist_all, coverage_radius, n_iterations=100
        )
        
        # 计算覆盖细节
        selected_indices = torch.where(final_solution == 1)[0].cpu().numpy()
        dist_to_selected = dist_all[:, selected_indices]
        min_dist_to_facility = torch.min(dist_to_selected, dim=1)[0]
        covered_nodes = (min_dist_to_facility <= coverage_radius).float()
        
        avg_distance = min_dist_to_facility.mean().item()
        max_distance = min_dist_to_facility.max().item()
        coverage_percentage = final_coverage / n_nodes * 100
        
        # 计算设施之间的平均距离（避免聚集）
        if len(selected_indices) > 1:
            facility_distances = dist_all[selected_indices][:, selected_indices]
            np.fill_diagonal(facility_distances.cpu().numpy(), np.inf)
            min_facility_dist = torch.min(facility_distances, dim=1)[0].mean().item()
        else:
            min_facility_dist = 0.0
    
    end = time.time()
    
    # 输出结果
    print(f"\n=== Instance {index} Results ===")
    print(f"Time elapsed: {end - start:.2f} seconds")
    print(f"Selected facilities ({len(selected_indices)}): {selected_indices}")
    print(f"Coverage: {final_coverage}/{n_nodes} nodes ({coverage_percentage:.1f}%)")
    print(f"Average distance to nearest facility: {avg_distance:.4f}")
    print(f"Maximum distance to nearest facility: {max_distance:.4f}")
    print(f"Average distance between facilities: {min_facility_dist:.4f}")
    print(f"Coverage radius: {coverage_radius:.4f}")
    
    # 可视化（可选）
    if index < 3:  # 只可视化前几个实例
        plt.figure(figsize=(12, 10))
        points_np = points.cpu().numpy()
        
        # 绘制所有节点
        plt.scatter(points_np[:, 0], points_np[:, 1], c='lightblue', s=20, 
                   label=f'All Nodes ({n_nodes})', alpha=0.5)
        
        # 标记被覆盖的节点
        covered_indices = torch.where(covered_nodes == 1)[0].cpu().numpy()
        plt.scatter(points_np[covered_indices, 0], points_np[covered_indices, 1], 
                   c='green', s=30, label=f'Covered ({len(covered_indices)})', alpha=0.8)
        
        # 标记未覆盖的节点
        uncovered_indices = torch.where(covered_nodes == 0)[0].cpu().numpy()
        if len(uncovered_indices) > 0:
            plt.scatter(points_np[uncovered_indices, 0], points_np[uncovered_indices, 1], 
                       c='red', s=30, label=f'Uncovered ({len(uncovered_indices)})', alpha=0.8)
        
        # 标记选择的设施
        plt.scatter(points_np[selected_indices, 0], points_np[selected_indices, 1], 
                   c='orange', s=150, marker='*', label=f'Facilities ({len(selected_indices)})', 
                   edgecolors='black', linewidths=2, zorder=5)
        
        # 为每个设施添加编号
        for i, idx in enumerate(selected_indices):
            plt.annotate(f'{i+1}', (points_np[idx, 0], points_np[idx, 1]), 
                        fontsize=10, fontweight='bold', ha='center', va='center', color='black')
        
        # 绘制覆盖范围
        for facility_idx in selected_indices:
            circle = plt.Circle((points_np[facility_idx, 0], points_np[facility_idx, 1]), 
                               coverage_radius, color='orange', alpha=0.08, linewidth=1, linestyle='--')
            plt.gca().add_patch(circle)
        
        plt.title(f'MCLP Solution - Instance {index}\nCoverage: {coverage_percentage:.1f}%', fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'mclp_solution_{index}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("=" * 50 + "\n")

print("MCLP solving completed!")