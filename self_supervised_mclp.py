# self_supervised_mclp.py (彻底修复版)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import os


# =====================================================
# 简单的GCN编码器（避免维度问题）
# =====================================================
class SimpleGCNEncoder(nn.Module):
    """
    简单但稳定的GCN编码器
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        
        # 两层GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # MCLP特征处理
        self.dist_fc = nn.Linear(1, out_channels)
        self.degree_fc = nn.Linear(1, out_channels)
        
        # 合并层
        self.merge = nn.Linear(3 * out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight, dist_feat, degree_feat):
        # GCN编码
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(self.bn1(h))
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(self.bn2(h))
        
        # 处理MCLP特征
        if dist_feat.dim() == 1:
            dist_feat = dist_feat.unsqueeze(1)
        if degree_feat.dim() == 1:
            degree_feat = degree_feat.unsqueeze(1)
            
        d_feat = F.relu(self.dist_fc(dist_feat))
        deg_feat = F.relu(self.degree_fc(degree_feat))
        
        # 确保维度一致
        assert h.shape == d_feat.shape == deg_feat.shape, \
            f"维度不匹配: h={h.shape}, d_feat={d_feat.shape}, deg_feat={deg_feat.shape}"
        
        # 合并特征
        combined = torch.cat([h, d_feat, deg_feat], dim=1)
        output = self.merge(combined)
        
        return output


# =====================================================
# 稳定的MCLP模型
# =====================================================
class StableMCLPModel(nn.Module):
    """
    稳定的MCLP模型
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super().__init__()
        
        # 编码器
        self.encoder = SimpleGCNEncoder(input_dim, hidden_dim, output_dim)
        
        # 设施选择头
        self.facility_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
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
        score = self.facility_head(emb).squeeze()
        
        return emb, score


# =====================================================
# 稳定的Wrapper
# =====================================================
class StableSelfSupervisedMCLPWrapper:
    """
    稳定的自监督MCLP包装器
    """
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        
    def initialize_model(self, input_dim, hidden_dim=64, output_dim=32):
        """初始化模型"""
        self.model = StableMCLPModel(input_dim, hidden_dim, output_dim).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        print(f"模型已初始化: 输入={input_dim}, 隐藏={hidden_dim}, 输出={output_dim}")
        
        return self.model
    
    def coverage_loss(self, scores, graph, K, temperature=0.1):
        """简单的覆盖损失"""
        dist = graph.distance_matrix.to(self.device)
        R = graph.coverage_radius
        
        # 节点权重
        if hasattr(graph, 'total_weights') and graph.total_weights is not None:
            w = graph.total_weights.to(self.device)
        else:
            w = torch.ones(graph.num_nodes).to(self.device)
        
        N = scores.shape[0]
        
        # Softmax得到概率分布
        p = F.softmax(scores / temperature, dim=0)
        
        # 归一化到K个设施
        p = K * p
        
        # 覆盖概率计算
        dist_normalized = dist / R
        coverage_prob = torch.exp(-dist_normalized**2)
        
        # 期望覆盖
        facility_probs = p.unsqueeze(0)  # [1, N]
        uncovered_prob = torch.prod(1.0 - facility_probs * coverage_prob, dim=1)
        covered_prob = 1.0 - uncovered_prob
        
        # 加权覆盖
        weighted_coverage = torch.sum(w * covered_prob)
        loss = -weighted_coverage / torch.sum(w)
        
        # 设施数量约束
        facility_constraint = (p.sum() - K)**2 * 0.01
        
        total_loss = loss + facility_constraint
        
        return total_loss
    
    def train_on_instance(self, graph, epochs=50, K=10):
        """训练单个实例"""
        if self.model is None:
            raise ValueError("请先调用 initialize_model()")
        
        graph = graph.to(self.device)
        self.model.train()
        
        losses = []
        
        for ep in range(epochs):
            self.optimizer.zero_grad()
            
            # 前向传播
            _, scores = self.model(graph)
            
            # 计算损失
            loss = self.coverage_loss(scores, graph, K)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if (ep + 1) % 10 == 0:
                print(f"    Epoch {ep+1}/{epochs} | Loss: {loss.item():.4f}")
        
        return losses
    
    @torch.no_grad()
    def solve_mclp(self, graph, K):
        """求解MCLP"""
        self.model.eval()
        graph = graph.to(self.device)
        
        _, scores = self.model(graph)
        
        # 选择Top-K
        selected = torch.topk(scores, min(K, graph.num_nodes)).indices.cpu().numpy()
        
        # 计算覆盖
        coverage = self.calculate_coverage(graph, selected)
        
        return selected, coverage, scores.cpu().numpy()
    
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
    
    def save_model(self, path='stable_mclp_model.pth'):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存: {path}")
    
    def load_model(self, path='stable_mclp_model.pth'):
        """加载模型"""
        if self.model is None:
            raise ValueError("请先初始化模型")
        
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"模型已加载: {path}")
        else:
            print(f"模型文件不存在: {path}")