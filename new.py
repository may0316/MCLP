# 使用生成的数据训练模型
# python new.py --data_path ./data/MCLP_20_4_0.30.pkl --epochs_per_instance 20
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
import argparse

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


# ========== 数据加载函数 ==========
def load_mclp_dataset(data_path):
    """
    加载由sponet_gen_data.py生成的MCLP模拟数据
    
    Args:
        data_path: .pkl文件路径
    
    Returns:
        list: 包含MCLP实例的列表
    """
    print(f"加载数据集: {data_path}")
    
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"数据集大小: {len(dataset)} 个实例")
    if len(dataset) > 0:
        print(f"示例实例: 节点数={dataset[0]['loc'].shape[0]}, "
              f"p={dataset[0]['p']}, radius={dataset[0]['radius']}")
    
    return dataset


def build_mclp_graph_from_instance(instance, demand_weights=None, device=torch.device('cpu')):
    """
    从模拟数据实例构建MCLP图结构
    
    关键修改：
    1. 所有需求点权重设为1
    2. 添加加权度编码（基于覆盖的需求点数量）
    3. 添加距离编码
    """
    # 从实例中获取数据
    loc = instance['loc']
    p = instance['p']
    radius = instance['radius']
    
    if not torch.is_tensor(loc):
        loc = torch.tensor(loc, dtype=torch.float).to(device)
    else:
        loc = loc.to(device)
    
    n_nodes = loc.shape[0]
    
    # ========== 所有需求点权重设为1 ==========
    demand_weights = torch.ones(n_nodes, device=device)
    
    # 计算距离矩阵 [n_demand, n_facility]
    dist_matrix = _pairwise_euclidean(loc, loc, device)
    
    # 覆盖关系（基于服务半径）
    coverage_matrix = (dist_matrix <= radius).float()
    
    # 构建图边：需求点 <-> 设施点
    covered_pairs = torch.nonzero(coverage_matrix, as_tuple=False)
    
    if len(covered_pairs) > 0:
        edge_index = torch.stack([
            covered_pairs[:, 0],  # 需求点索引
            covered_pairs[:, 1] + n_nodes  # 设施点索引（偏移）
        ], dim=0)
        
        # 边特征：距离（用于距离编码）
        edge_dist = dist_matrix[covered_pairs[:, 0], covered_pairs[:, 1]]
        edge_dist_norm = edge_dist / radius  # 归一化距离
        
        # 边权重：基于距离的衰减
        edge_weight = torch.exp(-edge_dist / radius).unsqueeze(1)
        
        # 边特征：包含归一化距离和权重
        edge_attr = torch.stack([edge_dist_norm, edge_weight.squeeze()], dim=1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_weight = torch.zeros(0, 1, device=device)
        edge_attr = torch.zeros(0, 2, device=device)
    
    # ========== 1. 需求点特征构建 ==========
    
    # 需求点的度（被多少设施覆盖）
    demand_degree = torch.zeros(n_nodes, device=device)
    for i in range(n_nodes):
        covering_facilities = coverage_matrix[i, :] > 0
        if covering_facilities.any():
            demand_degree[i] = covering_facilities.sum().float()
    
    # 需求点的距离编码：到最近设施的距离（归一化）
    demand_min_dist = torch.zeros(n_nodes, device=device)
    for i in range(n_nodes):
        distances_to_facilities = dist_matrix[i, :]
        demand_min_dist[i] = distances_to_facilities.min() / radius
    
    # 需求点特征拼接 [n_demand, feature_dim=5]
    demand_features = torch.cat([
        loc,  # 坐标 [n_demand, 2]
        demand_degree.unsqueeze(1),  # 度（覆盖的设施数量） [n_demand, 1]
        demand_min_dist.unsqueeze(1),  # 最近距离 [n_demand, 1]
        torch.ones(n_nodes, 1, device=device)  # 固定值1，表示需求点 [n_demand, 1]
    ], dim=1)  # [n_demand, 2+1+1+1=5]
    
    # ========== 2. 设施点特征构建 ==========
    
    # 设施点的度（覆盖的需求点数量）
    facility_degree = torch.zeros(n_nodes, device=device)
    for j in range(n_nodes):
        covered_demand = coverage_matrix[:, j] > 0
        if covered_demand.any():
            facility_degree[j] = covered_demand.sum().float()
    
    # 设施点的距离编码：到最近需求点的距离
    facility_min_dist = torch.zeros(n_nodes, device=device)
    for j in range(n_nodes):
        distances_to_demand = dist_matrix[:, j]
        facility_min_dist[j] = distances_to_demand.min() / radius
    
    # 设施点的潜在需求（距离衰减）
    facility_potential = torch.zeros(n_nodes, device=device)
    for j in range(n_nodes):
        distances = dist_matrix[:, j]
        facility_potential[j] = torch.sum(torch.exp(-distances / radius))
    
    # 归一化
    max_degree = facility_degree.max()
    facility_degree_norm = facility_degree / max_degree if max_degree > 0 else facility_degree
    
    max_potential = facility_potential.max()
    facility_potential_norm = facility_potential / max_potential if max_potential > 0 else facility_potential
    
    # 设施点特征拼接 [n_facility, feature_dim=5]
    facility_features = torch.cat([
        loc,  # 坐标 [n_facility, 2]
        facility_degree_norm.unsqueeze(1),  # 度（覆盖的需求点数量） [n_facility, 1]
        facility_min_dist.unsqueeze(1),  # 最近距离 [n_facility, 1]
        facility_potential_norm.unsqueeze(1),  # 潜在需求 [n_facility, 1]
    ], dim=1)  # [n_facility, 2+1+1+1=5]
    
    # 合并节点特征 - 现在都是5维
    x = torch.cat([demand_features, facility_features], dim=0)
    
    # 计算全局特征
    global_features = torch.tensor([
        p / n_nodes,  # 设施密度
        radius,  # 服务半径
        1.0,  # 平均需求权重（固定为1）
        0.0  # 需求权重标准差（固定为0）
    ], device=device)
    
    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        demand_pos=loc,
        facility_pos=loc,
        demand_weights=demand_weights,  # 全部为1
        facility_degree=facility_degree,  # 设施覆盖的需求点数量
        coverage_matrix=coverage_matrix,
        dist_matrix=dist_matrix,
        service_radius=radius,
        n_demand=n_nodes,
        n_facility=n_nodes,
        p=p,
        global_features=global_features
    )
    
    return graph


# ========== 目标函数值计算函数 ==========
def calculate_objective_value(graph, selected_indices):
    """
    计算MCLP的目标函数值：覆盖的总需求点数
    
    现在所有需求点权重为1，所以就是覆盖的需求点数量
    """
    if len(selected_indices) == 0:
        return 0.0
    
    coverage_matrix = graph.coverage_matrix
    
    covered = torch.zeros(graph.n_demand, dtype=torch.bool, device=coverage_matrix.device)
    for idx in selected_indices:
        covered = covered | (coverage_matrix[:, idx] > 0)
    
    # 目标函数值 = 被覆盖的需求点数量
    objective_value = covered.sum().item()
    
    return objective_value


def calculate_coverage_percentage(graph, selected_indices):
    """
    计算覆盖率百分比
    """
    if len(selected_indices) == 0:
        return 0.0
    
    objective_value = calculate_objective_value(graph, selected_indices)
    total_demand = graph.n_demand
    
    return (objective_value / total_demand) * 100 if total_demand > 0 else 0


# ========== 自监督MCLP模型 ==========
class MCLPEncoder(nn.Module):
    """MCLP编码器 - 考虑度和距离编码"""
    
    def __init__(self, in_channels, hidden_channels=256, out_channels=128, dropout=0.1):
        super().__init__()
        
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gat1 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)
        
        # 全局特征编码
        self.global_encoder = nn.Sequential(
            nn.Linear(4, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, out_channels // 4)
        )
    
    def forward(self, x, edge_index, global_features=None):
        h = self.gcn1(x, edge_index)
        h = F.relu(self.bn1(h))
        h = self.dropout(h)
        
        h = self.gat1(h, edge_index)
        h = F.relu(self.bn2(h))
        h = self.dropout(h)
        
        h = self.gat2(h, edge_index)
        h = F.relu(self.bn3(h))
        
        # 自注意力
        h = h.unsqueeze(0)
        h, _ = self.attention(h, h, h)
        h = h.squeeze(0)
        
        # 如果提供全局特征，将其添加到所有节点
        if global_features is not None:
            global_emb = self.global_encoder(global_features.unsqueeze(0))
            global_emb = global_emb.expand(h.size(0), -1)
            h = torch.cat([h, global_emb], dim=-1)
        
        return h


class SelfSupervisedMCLPModel(nn.Module):
    """自监督MCLP模型"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, global_dim=32):
        super().__init__()
        
        self.output_dim = output_dim
        self.global_dim = global_dim
        
        self.encoder = MCLPEncoder(input_dim, hidden_dim, output_dim)
        
        # 设施选择头
        self.facility_head = nn.Sequential(
            nn.Linear(output_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 自监督任务1：覆盖预测
        self.coverage_head = nn.Sequential(
            nn.Linear(output_dim + global_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 自监督任务2：边际收益预测
        self.marginal_head = nn.Sequential(
            nn.Linear(output_dim + global_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 自监督任务3：对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim + global_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
    
    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        global_features = graph.global_features
        
        emb = self.encoder(x, edge_index, global_features)
        
        # 分离需求和设施嵌入
        demand_emb = emb[:graph.n_demand]
        facility_emb = emb[graph.n_demand:]
        
        # 设施选择分数
        facility_scores = self.facility_head(facility_emb).squeeze()
        
        # 自监督任务输出
        coverage_pred = self.coverage_head(demand_emb).squeeze()  # 需求点被覆盖概率
        marginal_pred = self.marginal_head(facility_emb).squeeze()  # 设施边际收益
        
        # 对比学习投影
        demand_proj = self.projection_head(demand_emb)
        facility_proj = self.projection_head(facility_emb)
        
        return {
            'demand_emb': demand_emb,
            'facility_emb': facility_emb,
            'facility_scores': facility_scores,
            'coverage_pred': coverage_pred,
            'marginal_pred': marginal_pred,
            'demand_proj': demand_proj,
            'facility_proj': facility_proj
        }


# ========== 自监督MCLP求解器 ==========
class SelfSupervisedMCLPSolver:
    """自监督MCLP求解器"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def initialize_model(self, input_dim, hidden_dim=256, output_dim=128):
        self.model = SelfSupervisedMCLPModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        print(f"自监督MCLP模型初始化: 输入维度={input_dim}")
        return self.model
    
    def contrastive_loss(self, demand_proj, facility_proj, temperature=0.1):
        """对比学习损失：让能覆盖的需求-设施对更接近"""
        # 归一化
        demand_proj = F.normalize(demand_proj, dim=1)
        facility_proj = F.normalize(facility_proj, dim=1)
        
        # 相似度矩阵
        sim_matrix = torch.mm(demand_proj, facility_proj.t()) / temperature
        
        # 正样本对（假设每个设施应该与其覆盖的需求点更接近）
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        loss_d2f = F.cross_entropy(sim_matrix, labels)
        loss_f2d = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_d2f + loss_f2d) / 2
    
    def loss_function(self, model_outputs, graph, K, alpha=0.5, beta=0.3, gamma=0.1):
        """
        自监督损失函数
        
        Args:
            model_outputs: 模型输出字典
            graph: 图数据
            K: 要选择的设施数量
            alpha: 覆盖预测损失权重
            beta: 边际收益预测损失权重
            gamma: 对比学习损失权重
        """
        facility_scores = model_outputs['facility_scores']
        coverage_pred = model_outputs['coverage_pred']
        marginal_pred = model_outputs['marginal_pred']
        demand_proj = model_outputs['demand_proj']
        facility_proj = model_outputs['facility_proj']
        
        coverage_matrix = graph.coverage_matrix
        
        n_facility = graph.n_facility
        n_demand = graph.n_demand
        
        # ========== 主任务损失：基于贪心选择的监督 ==========
        # 使用带噪声的贪心选择生成伪标签
        with torch.no_grad():
            # 添加噪声以增加多样性
            noisy_scores = facility_scores + torch.randn_like(facility_scores) * 0.1
            selected = self._greedy_selection(graph, noisy_scores, K)
            
            # 构建设施选择的伪标签（1表示被选中）
            facility_labels = torch.zeros(n_facility, device=self.device)
            facility_labels[selected] = 1.0
            
            # 构建覆盖标签
            coverage_labels = torch.zeros(n_demand, device=self.device)
            for idx in selected:
                coverage_labels = torch.max(coverage_labels, coverage_matrix[:, idx])
            
            # 构建边际收益标签
            marginal_labels = torch.zeros(n_facility, device=self.device)
            current_coverage = torch.zeros(n_demand, device=self.device)
            
            # 计算每个设施的边际收益
            for idx in selected:
                new_coverage = torch.max(current_coverage, coverage_matrix[:, idx])
                marginal_gain = (new_coverage - current_coverage).sum().item()
                marginal_labels[idx] = marginal_gain / n_demand
                current_coverage = new_coverage
        
        # 设施选择损失（BCE）
        main_loss = F.binary_cross_entropy_with_logits(
            facility_scores, facility_labels, 
            pos_weight=torch.tensor([K / n_facility], device=self.device)
        )
        
        # 覆盖预测损失（BCE）
        coverage_loss = F.binary_cross_entropy(coverage_pred, coverage_labels)
        
        # 边际收益预测损失（MSE）
        marginal_loss = F.mse_loss(marginal_pred, marginal_labels)
        
        # 对比学习损失
        contrast_loss = self.contrastive_loss(demand_proj, facility_proj)
        
        # 总损失
        total_loss = (main_loss + 
                     alpha * coverage_loss + 
                     beta * marginal_loss + 
                     gamma * contrast_loss)
        
        # 计算当前目标值（用于监控）
        with torch.no_grad():
            current_objective = calculate_objective_value(graph, selected)
        
        return total_loss, current_objective, {
            'main_loss': main_loss.item(),
            'coverage_loss': coverage_loss.item(),
            'marginal_loss': marginal_loss.item(),
            'contrast_loss': contrast_loss.item()
        }
    
    def _greedy_selection(self, graph, scores, K):
        """贪心选择（基于边际收益）"""
        coverage_matrix = graph.coverage_matrix
        n_facility = graph.n_facility
        
        K = min(K, n_facility)
        
        selected = []
        covered = torch.zeros(graph.n_demand, dtype=torch.bool, device=self.device)
        remaining = list(range(n_facility))
        
        for _ in range(K):
            best_gain = -1
            best_node = -1
            
            for node in remaining:
                newly_covered = (coverage_matrix[:, node] > 0) & (~covered)
                # 边际增益（新覆盖的需求点数量）
                gain = newly_covered.sum().item()
                # 结合模型分数
                combined_gain = gain + 0.1 * scores[node].item()
                
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
    
    def train_on_instance(self, graph, K, epochs=20):
        """在单个实例上训练"""
        self.model.train()
        graph = graph.to(self.device)
        
        losses = []
        loss_components = {'main_loss': [], 'coverage_loss': [], 
                          'marginal_loss': [], 'contrast_loss': []}
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            
            model_outputs = self.model(graph)
            
            loss, objective, components = self.loss_function(
                model_outputs, graph, K
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            losses.append(loss.item())
            for k, v in components.items():
                loss_components[k].append(v)
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return losses, loss_components
    
    @torch.no_grad()
    def solve(self, graph, K, num_trials=20):
        """
        求解MCLP问题
        
        Returns:
            selected_indices: 选择的设施点索引
            objective_value: 目标函数值（覆盖的需求点数量）
        """
        self.model.eval()
        graph = graph.to(self.device)
        
        n_facility = graph.n_facility
        K = min(K, n_facility)
        
        best_objective = 0
        best_selection = None
        
        for trial in range(num_trials):
            model_outputs = self.model(graph)
            facility_scores = model_outputs['facility_scores']
            
            # 多种选择策略
            candidates = []
            
            # 策略1：贪心选择（使用模型分数）
            selected1 = self._greedy_selection(graph, facility_scores, K)
            objective1 = calculate_objective_value(graph, selected1)
            candidates.append((objective1, selected1))
            
            # 策略2：带噪声的贪心（增加探索）
            noise = torch.randn_like(facility_scores) * 0.2 * (1 - trial / num_trials)
            selected2 = self._greedy_selection(graph, facility_scores + noise, K)
            objective2 = calculate_objective_value(graph, selected2)
            candidates.append((objective2, selected2))
            
            # 策略3：Top-K
            _, top_indices = torch.topk(facility_scores, K)
            selected3 = top_indices
            objective3 = calculate_objective_value(graph, selected3)
            candidates.append((objective3, selected3))
            
            # 策略4：纯贪心（只基于边际收益）
            selected4 = []
            covered = torch.zeros(graph.n_demand, dtype=torch.bool, device=self.device)
            remaining = list(range(n_facility))
            
            for _ in range(K):
                if not remaining:
                    break
                # 只基于边际增益
                gains = []
                for node in remaining:
                    newly_covered = (graph.coverage_matrix[:, node] > 0) & (~covered)
                    gain = newly_covered.sum().item()
                    gains.append((gain, node))
                
                gains.sort(reverse=True)
                best_node = gains[0][1]
                selected4.append(best_node)
                newly_covered = (graph.coverage_matrix[:, best_node] > 0) & (~covered)
                covered = covered | newly_covered
                remaining.remove(best_node)
            
            if len(selected4) == K:
                selected4 = torch.tensor(selected4, device=self.device)
                objective4 = calculate_objective_value(graph, selected4)
                candidates.append((objective4, selected4))
            
            # 记录最佳
            for obj, sel in candidates:
                if obj > best_objective:
                    best_objective = obj
                    best_selection = sel
        
        return best_selection, best_objective
    
    def save_model(self, path='mclp_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        print(f"模型已保存: {path}")
    
    def load_model(self, path='mclp_model.pth'):
        if self.model is None:
            raise ValueError("请先初始化模型")
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"模型已加载: {path}")
        else:
            print(f"模型文件不存在: {path}")


# ========== 训练函数 ==========
def train_model(train_dataset, val_dataset, epochs_per_instance=20):
    """
    训练自监督MCLP模型
    """
    print(f"\n训练集: {len(train_dataset)} 个实例")
    print(f"验证集: {len(val_dataset)} 个实例")
    
    # 获取输入维度
    sample = train_dataset[0]
    sample_graph = build_mclp_graph_from_instance(sample, device=device)
    input_dim = sample_graph.x.shape[1]
    print(f"输入特征维度: {input_dim}")
    
    # 初始化求解器
    solver = SelfSupervisedMCLPSolver(device=device)
    solver.initialize_model(input_dim=input_dim, hidden_dim=256, output_dim=128)
    
    print(f"\n开始训练，每个实例训练轮数: {epochs_per_instance}")
    print("-" * 50)
    
    all_losses = []
    all_components = {'main_loss': [], 'coverage_loss': [], 
                     'marginal_loss': [], 'contrast_loss': []}
    best_val_objective = 0
    
    for epoch, instance in enumerate(train_dataset):
        graph = build_mclp_graph_from_instance(instance, device=device)
        K = instance['p']
        
        losses, components = solver.train_on_instance(graph, K=K, epochs=epochs_per_instance)
        all_losses.extend(losses)
        for k, v in components.items():
            all_components[k].extend(v)
        
        # 验证
        if (epoch + 1) % 5 == 0 and len(val_dataset) > 0:
            val_objectives = []
            
            for val_instance in val_dataset[:3]:
                val_graph = build_mclp_graph_from_instance(val_instance, device=device)
                _, obj_value = solver.solve(val_graph, K=val_instance['p'], num_trials=5)
                val_objectives.append(obj_value)
            
            avg_val_objective = np.mean(val_objectives)
            total_demand_avg = val_dataset[0]['loc'].shape[0]  # 所有实例节点数相同
            coverage_pct = (avg_val_objective / total_demand_avg) * 100
            
            # 计算平均损失组件
            avg_main = np.mean(all_components['main_loss'][-20:]) if len(all_components['main_loss']) >= 20 else np.mean(all_components['main_loss'])
            avg_cov = np.mean(all_components['coverage_loss'][-20:]) if len(all_components['coverage_loss']) >= 20 else np.mean(all_components['coverage_loss'])
            avg_marg = np.mean(all_components['marginal_loss'][-20:]) if len(all_components['marginal_loss']) >= 20 else np.mean(all_components['marginal_loss'])
            avg_cont = np.mean(all_components['contrast_loss'][-20:]) if len(all_components['contrast_loss']) >= 20 else np.mean(all_components['contrast_loss'])
            
            print(f"Epoch {epoch + 1}: 验证目标值 = {avg_val_objective:.2f} ({coverage_pct:.1f}%) | "
                  f"损失: main={avg_main:.4f}, cov={avg_cov:.4f}, marg={avg_marg:.4f}, cont={avg_cont:.4f}")
            
            if avg_val_objective > best_val_objective:
                best_val_objective = avg_val_objective
                solver.save_model('best_mclp_model.pth')
                print(f"  → 保存最佳模型 (目标值: {avg_val_objective:.2f})")
        
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(all_losses[-20:]) if len(all_losses) >= 20 else np.mean(all_losses)
            print(f"Epoch {epoch + 1}/{len(train_dataset)} | 平均总损失: {avg_loss:.4f}")
    
    print(f"\n训练完成！最佳验证目标函数值: {best_val_objective:.2f}")
    solver.save_model('final_mclp_model.pth')
    
    return solver


# ========== 测试函数 ==========
def test_model(solver, test_dataset):
    """
    测试模型性能
    """
    print("\n" + "=" * 60)
    print("模型测试")
    print("=" * 60)
    
    results = []
    
    for i, instance in enumerate(test_dataset):
        graph = build_mclp_graph_from_instance(instance, device=device)
        K = instance['p']
        
        start_time = time.time()
        selected, obj_value = solver.solve(graph, K=K, num_trials=20)
        solve_time = time.time() - start_time
        
        coverage_pct = calculate_coverage_percentage(graph, selected)
        total_demand = graph.n_demand
        
        # 计算选择的设施分布
        if len(selected) > 1:
            facility_pos = graph.facility_pos[selected].cpu().numpy()
            pairwise_dist = np.linalg.norm(facility_pos[:, None] - facility_pos[None, :], axis=-1)
            avg_facility_dist = np.mean(pairwise_dist[pairwise_dist > 0])
        else:
            avg_facility_dist = 0
        
        results.append({
            'instance_id': i,
            'n_nodes': graph.n_demand,
            'K': K,
            'objective_value': obj_value,
            'total_demand': total_demand,
            'coverage_pct': coverage_pct,
            'solve_time': solve_time,
            'avg_facility_dist': avg_facility_dist
        })
        
        print(f"实例 {i + 1}: 节点数={graph.n_demand}, K={K}, "
              f"目标值={obj_value:.2f}/{total_demand:.2f} ({coverage_pct:.1f}%), "
              f"设施平均距离={avg_facility_dist:.3f}, 时间={solve_time:.3f}s")
    
    avg_objective = np.mean([r['objective_value'] for r in results])
    avg_coverage = np.mean([r['coverage_pct'] for r in results])
    avg_time = np.mean([r['solve_time'] for r in results])
    avg_dist = np.mean([r['avg_facility_dist'] for r in results])
    
    print("\n" + "-" * 50)
    print("测试结果汇总:")
    print(f"平均目标函数值: {avg_objective:.2f}")
    print(f"平均覆盖率: {avg_coverage:.1f}%")
    print(f"平均设施间距: {avg_dist:.3f}")
    print(f"平均求解时间: {avg_time:.3f}s")
    
    return results


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='训练和测试自监督MCLP模型')
    parser.add_argument('--data_path', type=str, required=True,
                        help='MCLP数据文件路径 (.pkl)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例 (默认: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例 (默认: 0.15)')
    parser.add_argument('--epochs_per_instance', type=int, default=20,
                        help='每个实例的训练轮数 (默认: 20)')
    parser.add_argument('--test_only', action='store_true',
                        help='仅测试已有模型')
    parser.add_argument('--model_path', type=str, default='best_mclp_model.pth',
                        help='模型文件路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("自监督MCLP模型训练与测试")
    print("=" * 60)
    
    # 加载数据集
    dataset = load_mclp_dataset(args.data_path)
    
    # 划分训练集、验证集和测试集
    n_total = len(dataset)
    n_train = int(n_total * (1 - args.val_ratio - args.test_ratio))
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val
    
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]
    
    print(f"\n数据集划分 (总实例数: {n_total}):")
    print(f"训练集: {len(train_dataset)} 个实例 ({len(train_dataset)/n_total*100:.1f}%)")
    print(f"验证集: {len(val_dataset)} 个实例 ({len(val_dataset)/n_total*100:.1f}%)")
    print(f"测试集: {len(test_dataset)} 个实例 ({len(test_dataset)/n_total*100:.1f}%)")
    
    if args.test_only:
        print("\n测试模式: 加载已有模型")
        
        sample = test_dataset[0]
        sample_graph = build_mclp_graph_from_instance(sample, device=device)
        input_dim = sample_graph.x.shape[1]
        
        solver = SelfSupervisedMCLPSolver(device=device)
        solver.initialize_model(input_dim=input_dim)
        solver.load_model(args.model_path)
        
        results = test_model(solver, test_dataset)
        
    else:
        print("\n训练模式: 开始训练模型")
        
        solver = train_model(
            train_dataset,
            val_dataset,
            epochs_per_instance=args.epochs_per_instance
        )
        
        print("\n" + "=" * 60)
        print("在测试集上评估模型")
        print("=" * 60)
        results = test_model(solver, test_dataset)
    
    # 保存测试结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('mclp_test_results.csv', index=False)
    print("\n测试结果已保存到: mclp_test_results.csv")


if __name__ == "__main__":
    main()