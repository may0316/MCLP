import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import os
import math
# =====================================================
# 增强的GCN编码器
# =====================================================
class EnhancedGCNEncoder(nn.Module):
    """
    增强的GCN编码器，具有更强的特征提取能力
    """
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, dropout=0.2):
        super().__init__()
        
        # 三层GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # 增强的距离特征处理
        self.dist_processor = nn.Sequential(
            nn.Linear(1, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # 增强的度特征处理
        self.degree_processor = nn.Sequential(
            nn.Linear(1, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            out_channels, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True
        )
        
        # 特征合并层
        self.merge = nn.Sequential(
            nn.Linear(3 * out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # 残差连接
        self.residual = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight, dist_feat, degree_feat):
        # 残差连接
        x_res = self.residual(x)
        
        # GCN编码
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(self.bn1(h))
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(self.bn2(h))
        h = self.dropout(h)
        
        h = self.conv3(h, edge_index, edge_weight)
        h = F.relu(self.bn3(h))
        
        # 处理MCLP特征
        if dist_feat.dim() == 1:
            dist_feat = dist_feat.unsqueeze(1)
        if degree_feat.dim() == 1:
            degree_feat = degree_feat.unsqueeze(1)
            
        d_feat = self.dist_processor(dist_feat)
        deg_feat = self.degree_processor(degree_feat)
        
        # 应用注意力机制
        h_reshaped = h.unsqueeze(0)  # [1, N, out_channels]
        attn_output, _ = self.attention(h_reshaped, h_reshaped, h_reshaped)
        h_attn = attn_output.squeeze(0)
        
        # 合并特征
        combined = torch.cat([h_attn, d_feat, deg_feat], dim=1)
        output = self.merge(combined)
        
        # 残差连接
        output = output + x_res
        
        return output

# =====================================================
# 优化的MCLP模型
# =====================================================
class OptimizedMCLPModel(nn.Module):
    """
    优化的MCLP模型
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        
        # 编码器
        self.encoder = EnhancedGCNEncoder(
            input_dim, 
            hidden_dim, 
            output_dim,
            dropout=0.2
        )
        
        # 设施选择头（多层）
        self.facility_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 覆盖预测头（辅助任务）
        self.coverage_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, graph):
        # 获取图数据
        x = graph.x
        edge_index = graph.edge_index
        
        # 边权重
        edge_weight = None
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            edge_weight = graph.edge_attr.squeeze()
        
        # MCLP特征
        if hasattr(graph, 'dist_row_sum'):
            dist_feat = graph.dist_row_sum
        else:
            dist_feat = torch.zeros(x.shape[0], device=x.device)
            
        if hasattr(graph, 'degree'):
            degree_feat = graph.degree
        else:
            degree_feat = torch.zeros(x.shape[0], device=x.device)
        
        # 编码
        emb = self.encoder(x, edge_index, edge_weight, dist_feat, degree_feat)
        
        # 设施分数
        facility_score = self.facility_head(emb).squeeze()
        
        # 覆盖预测（辅助任务）
        coverage_pred = self.coverage_head(emb).squeeze()
        
        return emb, facility_score, coverage_pred

# =====================================================
# 优化的Wrapper
# =====================================================
class OptimizedSelfSupervisedMCLPWrapper:
    """
    优化的自监督MCLP包装器
    """
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def initialize_model(self, input_dim, hidden_dim=128, output_dim=64):
        """初始化模型"""
        self.model = OptimizedMCLPModel(input_dim, hidden_dim, output_dim).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-5
        )
        
        print(f"模型已初始化: 输入={input_dim}, 隐藏={hidden_dim}, 输出={output_dim}")
        
        return self.model
    
    def enhanced_coverage_loss(self, scores, coverage_pred, graph, K, alpha=0.3, beta=0.1):
        """增强的覆盖损失函数"""
        dist = graph.distance_matrix.to(self.device)
        R = graph.coverage_radius
        
        # 节点权重
        if hasattr(graph, 'total_weights') and graph.total_weights is not None:
            w = graph.total_weights.to(self.device)
        else:
            w = torch.ones(graph.num_nodes).to(self.device)
        
        N = scores.shape[0]
        
        # 使用Gumbel-Softmax获得概率分布
        temperature = max(0.1, 1.0 - 0.9 * min(1.0, len(scores) / (K * 10)))
        p = F.gumbel_softmax(scores.unsqueeze(0), tau=temperature, hard=False, dim=1).squeeze()
        p = K * p
        
        # 距离矩阵处理
        dist_normalized = torch.clamp(dist / R, 0, 3)
        coverage_prob = torch.exp(-dist_normalized**2)
        
        # 每个节点被覆盖的概率
        facility_probs = p.unsqueeze(0)  # [1, N]
        uncovered_prob = torch.prod(1.0 - facility_probs * coverage_prob, dim=1)
        covered_prob = 1.0 - uncovered_prob
        
        # 主损失：最大化加权覆盖
        weighted_coverage = torch.sum(w * covered_prob)
        coverage_loss = -weighted_coverage / torch.sum(w)
        
        # 设施分散惩罚
        if torch.sum(p) > 0:
            selected_probs = p / torch.sum(p)
            facility_locations = graph.x[:, :2] if graph.x.shape[1] >= 2 else graph.x
            center_of_mass = torch.matmul(selected_probs, facility_locations)
            
            # 计算设施之间的分散程度
            pairwise_dist = torch.cdist(facility_locations, facility_locations)
            dispersion = torch.sum(
                selected_probs.unsqueeze(1) * selected_probs.unsqueeze(0) * pairwise_dist
            )
            dispersion_penalty = -beta * dispersion / (K * R)
        else:
            dispersion_penalty = 0
        
        # 辅助任务损失
        if hasattr(graph, 'coverage_mask'):
            coverage_target = graph.coverage_mask.float()
            aux_loss = F.binary_cross_entropy(coverage_pred, coverage_target)
        else:
            aux_loss = torch.tensor(0.0, device=self.device)
        
        # 设施数量约束
        facility_constraint = (p.sum() - K)**2 * 0.01
        
        # 边界鼓励（如果可用）
        edge_encouragement = 0
        if hasattr(graph, 'is_boundary'):
            edge_encouragement = -0.05 * torch.sum(p * graph.is_boundary.float())
        
        total_loss = (coverage_loss + 
                     alpha * aux_loss + 
                     dispersion_penalty + 
                     facility_constraint + 
                     edge_encouragement)
        
        return total_loss
    
    def train_on_instance(self, graph, epochs=20, K=10):
        """训练单个实例"""
        if self.model is None:
            raise ValueError("请先调用 initialize_model()")
        
        graph = graph.to(self.device)
        self.model.train()
        
        losses = []
        
        for ep in range(epochs):
            self.optimizer.zero_grad()
            
            # 前向传播
            _, scores, coverage_pred = self.model(graph)
            
            # 计算损失
            loss = self.enhanced_coverage_loss(scores, coverage_pred, graph, K)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step(ep)
            
            losses.append(loss.item())
            
            if (ep + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"    Epoch {ep+1}/{epochs} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
        return losses
    
    @torch.no_grad()
    def solve_mclp(self, graph, K, iterations=5):
        """改进的求解算法"""
        self.model.eval()
        graph = graph.to(self.device)
        
        best_coverage = 0
        best_selection = None
        best_scores = None
        
        for iter in range(iterations):
            # 多次运行获得更好的结果
            _, scores, _ = self.model(graph)
            
            # 添加随机扰动增强探索
            if iter > 0:
                noise = torch.randn_like(scores) * (0.1 * (iterations - iter) / iterations)
                perturbed_scores = scores + noise
            else:
                perturbed_scores = scores
            
            # 使用贪心+交换改进
            selected = self.greedy_with_swap(graph, perturbed_scores, K)
            
            # 计算覆盖
            coverage = self.calculate_coverage(graph, selected)
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_selection = selected
                best_scores = scores.cpu().numpy()
            
            # 打印迭代信息
            if iterations > 1:
                print(f"    迭代 {iter+1}/{iterations}: 覆盖率 = {coverage:.1f}")
        
        # 后处理优化
        if best_selection is not None:
            best_selection, best_coverage = self.post_process_optimization(
                graph, best_selection, K, best_coverage
            )
        
        return best_selection, best_coverage, best_scores
    
    def greedy_with_swap(self, graph, scores, K):
        """贪心算法带交换改进"""
        selected = []
        remaining = list(range(graph.num_nodes))
        
        # 第一阶段：贪心选择
        for _ in range(K):
            best_gain = -float('inf')
            best_node = -1
            
            for node in remaining:
                temp_selected = selected + [node]
                coverage = self.calculate_coverage(graph, temp_selected)
                
                # 综合考虑覆盖增益和节点分数
                gain = coverage + 0.2 * scores[node].item()
                
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
            
            if best_node != -1:
                selected.append(best_node)
                remaining.remove(best_node)
        
        # 第二阶段：尝试交换改进
        improved = True
        while improved:
            improved = False
            for i in range(K):
                current_coverage = self.calculate_coverage(graph, selected)
                
                for candidate in remaining:
                    new_selected = selected.copy()
                    new_selected[i] = candidate
                    new_coverage = self.calculate_coverage(graph, new_selected)
                    
                    if new_coverage > current_coverage:
                        selected = new_selected
                        remaining = [node for node in remaining if node != candidate]
                        remaining.append(selected[i])
                        improved = True
                        break
                if improved:
                    break
        
        return np.array(selected)
    
    def post_process_optimization(self, graph, selected_indices, K, current_coverage):
        """后处理优化"""
        selected_list = selected_indices.tolist()
        remaining = [i for i in range(graph.num_nodes) if i not in selected_list]
        
        # 尝试多种改进
        for _ in range(5):  # 多次尝试
            for i in range(K):
                for candidate in remaining:
                    temp_selected = selected_list.copy()
                    temp_selected[i] = candidate
                    temp_coverage = self.calculate_coverage(graph, temp_selected)
                    
                    if temp_coverage > current_coverage:
                        selected_list = temp_selected
                        remaining = [j for j in remaining if j != candidate]
                        remaining.append(selected_list[i])
                        current_coverage = temp_coverage
                        break
        
        return np.array(selected_list), current_coverage
    
    @staticmethod
    def calculate_coverage(graph, selected_indices):
        """计算覆盖率"""
        if len(selected_indices) == 0:
            return 0.0
        
        dist = graph.distance_matrix
        R = graph.coverage_radius
        
        if hasattr(graph, 'total_weights') and graph.total_weights is not None:
            w = graph.total_weights
        else:
            w = torch.ones(graph.num_nodes).to(dist.device)
        
        d = dist[:, selected_indices]
        min_dist = torch.min(d, dim=1)[0]
        covered_mask = (min_dist <= R)
        coverage = torch.sum(w[covered_mask]).item()
        
        return coverage
    
    def save_model(self, path='optimized_mclp_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        print(f"模型已保存: {path}")
    
    def load_model(self, path='optimized_mclp_model.pth'):
        """加载模型"""
        if self.model is None:
            raise ValueError("请先初始化模型")
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"模型已加载: {path}")
        else:
            print(f"模型文件不存在: {path}")