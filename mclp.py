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
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，支持中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号

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

# 加载数据集 - 使用新的数据集格式
print("正在加载MCLP数据集...")
try:
    # 尝试从文件加载dataset_800.pkl
    train_dataset = create_more.load_dataset('dataset_800.pkl')
    print(f"成功加载数据集，包含 {len(train_dataset)} 个实例")
except FileNotFoundError:
    print("未找到 dataset_800.pkl，正在生成新的数据集...")
    # 生成新的数据集
    generator = create_more.MCLPDatasetGenerator(
        num_nodes=500,
        dim=2,
        num_instances=800,
        coord_range=(0.0, 10.0),
        device=device
    )
    train_dataset = generator.generate_dataset(distribution_type='mixed')
    create_more.save_dataset(train_dataset, 'dataset_800.pkl')
    print(f"已生成并保存新的数据集，包含 {len(train_dataset)} 个实例")

# 分析数据集的前几个实例
print("\n数据集前3个实例的分析:")
for i in range(min(3, len(train_dataset))):
    name, points = train_dataset[i]
    dist_matrix = create_more._pairwise_euclidean(points, points, device)
    diameter = dist_matrix.max().item()
    coverage_radius = 0.15 * diameter
    print(f"  实例 {name}: 直径={diameter:.2f}, 覆盖半径={coverage_radius:.2f}, "
          f"坐标范围=[{points[:,0].min():.1f},{points[:,0].max():.1f}]x"
          f"[{points[:,1].min():.1f},{points[:,1].max():.1f}]")

# 加载预训练模型
print("\n正在加载预训练模型...")
model = MocoModel(2, 128, 64, n_nodes).to(device)
try:
    model.load_state_dict(torch.load('pre_mclp.pth'))
    print("预训练模型加载成功")
except FileNotFoundError:
    print("警告: 未找到 pre_mclp.pth，使用随机初始化的模型")

K = 50  # 要放置的设施数量


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


# 统计所有实例的结果
all_results = []

# 主训练和求解循环
print("\n" + "="*70)
print("开始MCLP求解")
print("="*70)

# 可以选择只处理部分实例进行测试
max_instances_to_process = min(10, len(train_dataset))  # 只处理前10个实例进行测试
print(f"将处理前 {max_instances_to_process} 个实例")

for index, (instance_name, points) in enumerate(train_dataset[:max_instances_to_process]):
    start = time.time()
    
    print(f"\n正在处理实例 {index}: {instance_name}")
    
    # 构建图
    graph, dist_all = create_more.build_graph_from_points(points, None, True, 'euclidean')
    diameter = dist_all.max().item()
    
    # 设置覆盖半径
    coverage_radius = 0.15 * diameter
    print(f"  直径: {diameter:.4f}")
    print(f"  覆盖半径: {coverage_radius:.4f} (直径的15%)")
    print(f"  总节点数: {n_nodes}, 需要放置设施数: {K}")
    
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
    
    print("  训练MLP...")
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
                    print(f"    迭代 {t}: 损失={loss.item():.4f}, "
                          f"覆盖率={coverage_val}/{n_nodes} ({coverage_percentage:.1f}%), "
                          f"学习率={current_lr:.6f}")
                
                if patience_counter >= patience:
                    print(f"    早停于迭代 {t}")
                    break
    
    # 最终评估
    print("  最终评估...")
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
        
        # 保存结果
        instance_result = {
            'instance_id': index,
            'instance_name': instance_name,
            'selected_facilities': selected_indices,
            'coverage': final_coverage,
            'coverage_percentage': coverage_percentage,
            'avg_distance': avg_distance,
            'max_distance': max_distance,
            'min_facility_dist': min_facility_dist,
            'coverage_radius': coverage_radius,
            'diameter': diameter
        }
        all_results.append(instance_result)
    
    end = time.time()
    elapsed_time = end - start
    
    # 输出结果
    print(f"\n=== 实例 {index} ({instance_name}) 结果 ===")
    print(f"  耗时: {elapsed_time:.2f} 秒")
    print(f"  选择的设施 ({len(selected_indices)}个): {selected_indices}")
    print(f"  覆盖率: {final_coverage}/{n_nodes} 节点 ({coverage_percentage:.1f}%)")
    print(f"  到最近设施的平均距离: {avg_distance:.4f}")
    print(f"  到最近设施的最大距离: {max_distance:.4f}")
    print(f"  设施间平均最小距离: {min_facility_dist:.4f}")
    print(f"  覆盖半径: {coverage_radius:.4f}")
    
    # 可视化（只可视化前3个实例）- 修复版
    if index < 3:
        try:
            plt.figure(figsize=(14, 12))
            points_np = points.cpu().numpy()
            
            # 绘制所有节点
            plt.scatter(points_np[:, 0], points_np[:, 1], c='lightblue', s=30, 
                       label=f'所有节点 ({n_nodes})', alpha=0.5, edgecolors='white', linewidth=0.5)
            
            # 标记被覆盖的节点
            covered_indices = torch.where(covered_nodes == 1)[0].cpu().numpy()
            if len(covered_indices) > 0:
                plt.scatter(points_np[covered_indices, 0], points_np[covered_indices, 1], 
                           c='green', s=50, label=f'已覆盖 ({len(covered_indices)})', alpha=0.7, 
                           edgecolors='darkgreen', linewidth=1)
            
            # 标记未覆盖的节点
            uncovered_indices = torch.where(covered_nodes == 0)[0].cpu().numpy()
            if len(uncovered_indices) > 0:
                plt.scatter(points_np[uncovered_indices, 0], points_np[uncovered_indices, 1], 
                           c='red', s=50, label=f'未覆盖 ({len(uncovered_indices)})', alpha=0.7,
                           edgecolors='darkred', linewidth=1)
            
            # 标记选择的设施
            if len(selected_indices) > 0:
                plt.scatter(points_np[selected_indices, 0], points_np[selected_indices, 1], 
                           c='gold', s=300, marker='*', label=f'设施 ({len(selected_indices)})', 
                           edgecolors='black', linewidths=3, zorder=10)
                
                # 为每个设施添加编号
                for i, idx in enumerate(selected_indices):
                    plt.annotate(f'{i+1}', (points_np[idx, 0], points_np[idx, 1]), 
                                fontsize=12, fontweight='bold', ha='center', va='center', 
                                color='black', bbox=dict(boxstyle="circle,pad=0.2", 
                                                        facecolor='white', 
                                                        edgecolor='black', 
                                                        alpha=0.8))
            
            # 绘制覆盖范围
            if len(selected_indices) > 0:
                for facility_idx in selected_indices:
                    circle = plt.Circle((points_np[facility_idx, 0], points_np[facility_idx, 1]), 
                                       coverage_radius, color='orange', alpha=0.1, 
                                       linewidth=2, linestyle='-', edgecolor='orange', zorder=5)
                    plt.gca().add_patch(circle)
            
            plt.title(f'MCLP解决方案 - 实例 {index}: {instance_name}\n'
                     f'覆盖率: {coverage_percentage:.1f}% | 覆盖半径: {coverage_radius:.2f} | '
                     f'设施数量: {len(selected_indices)}', 
                     fontsize=16, fontweight='bold', pad=20)
            
            plt.xlabel('X 坐标', fontsize=14)
            plt.ylabel('Y 坐标', fontsize=14)
            
            # 设置图例
            plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
            
            # 添加网格
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # 设置坐标轴范围
            x_min, x_max = points_np[:, 0].min(), points_np[:, 0].max()
            y_min, y_max = points_np[:, 1].min(), points_np[:, 1].max()
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)
            
            # 设置纵横比相等
            plt.gca().set_aspect('equal', adjustable='box')
            
            # 添加信息文本框
            info_text = f'节点总数: {n_nodes}\n'
            info_text += f'设施数量: {len(selected_indices)}\n'
            info_text += f'覆盖节点: {final_coverage}\n'
            info_text += f'覆盖率: {coverage_percentage:.1f}%\n'
            info_text += f'平均距离: {avg_distance:.3f}\n'
            info_text += f'最大距离: {max_distance:.3f}\n'
            info_text += f'覆盖半径: {coverage_radius:.3f}'
            
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(f'mclp_solution_{index}.png', dpi=300, bbox_inches='tight', facecolor='white')
            
            # 显示图像
            plt.show()
            
            print(f"  已保存可视化图像: mclp_solution_{index}.png")
            
        except Exception as e:
            print(f"  可视化生成失败: {e}")
            # 简单备份可视化
            plt.figure(figsize=(10, 8))
            points_np = points.cpu().numpy()
            plt.scatter(points_np[:, 0], points_np[:, 1], s=20, alpha=0.5)
            if len(selected_indices) > 0:
                plt.scatter(points_np[selected_indices, 0], points_np[selected_indices, 1], 
                           s=100, marker='*', c='red')
            plt.title(f'MCLP Solution {index}: {coverage_percentage:.1f}% coverage')
            plt.savefig(f'mclp_solution_simple_{index}.png')
            plt.close()
    
    print("="*70)

# 输出总体统计
print("\n" + "="*70)
print("MCLP求解完成 - 总体统计")
print("="*70)

if all_results:
    total_coverage = sum(r['coverage'] for r in all_results)
    avg_coverage_percentage = sum(r['coverage_percentage'] for r in all_results) / len(all_results)
    avg_avg_distance = sum(r['avg_distance'] for r in all_results) / len(all_results)
    
    print(f"处理的实例数量: {len(all_results)}")
    print(f"平均覆盖率: {avg_coverage_percentage:.1f}%")
    print(f"总覆盖节点数: {total_coverage}")
    print(f"平均到最近设施距离: {avg_avg_distance:.4f}")
    
    # 显示每个实例的简要结果
    print("\n各实例详细结果:")
    for result in all_results:
        print(f"  实例 {result['instance_id']} ({result['instance_name']}): "
              f"{result['coverage']}/{n_nodes} ({result['coverage_percentage']:.1f}%)")
    
    # 保存结果到文件
    results_summary = {
        'all_results': all_results,
        'summary': {
            'num_instances': len(all_results),
            'avg_coverage_percentage': avg_coverage_percentage,
            'total_coverage': total_coverage,
            'avg_avg_distance': avg_avg_distance,
            'K': K,
            'n_nodes': n_nodes
        }
    }
    
    with open('mclp_results_summary.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"\n详细结果已保存到: mclp_results_summary.pkl")

print("\nMCLP求解完成!")