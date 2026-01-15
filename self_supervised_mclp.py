import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import copy

class GCNEncoder(nn.Module):
    """GCN编码器（借鉴FLP架构）"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 节点特征增强（类似FLP）
        self.fc_dist = nn.Linear(1, out_channels)
        self.fc_degree = nn.Linear(1, out_channels)
        self.fc_merge = nn.Linear(3 * out_channels, out_channels)
        
        self.bn_dist = nn.BatchNorm1d(out_channels)
        self.bn_degree = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index, edge_weight, dist_feat, degree_feat):
        # GCN编码
        x_gcn = self.conv1(x, edge_index, edge_weight)
        x_gcn = F.relu(self.bn1(x_gcn))
        x_gcn = self.conv2(x_gcn, edge_index, edge_weight)
        x_gcn = self.bn2(x_gcn)
        x_gcn = F.relu(x_gcn)
        
        # 距离特征处理
        dist_feat = self.fc_dist(dist_feat)
        dist_feat = self.bn_dist(dist_feat)
        dist_feat = F.relu(dist_feat)
        
        # 度数特征处理
        degree_feat = self.fc_degree(degree_feat)
        degree_feat = self.bn_degree(degree_feat)
        degree_feat = F.relu(degree_feat)
        
        # 特征融合
        x_concat = torch.cat((x_gcn, dist_feat, degree_feat), dim=1)
        x_concat = self.fc_merge(x_concat)
        
        return x_concat

class MoCoMCLPModel(nn.Module):
    """基于MoCo的MCLP自监督模型（借鉴FLP）"""
    def __init__(self, dim_in, dim_hidden, dim_out, m=0.99, K=512):
        super().__init__()
        self.m = m
        self.K = K
        
        # Query network
        self.q_net = GCNEncoder(dim_in, dim_hidden, dim_out)
        
        # Key network
        self.k_net = GCNEncoder(dim_in, dim_hidden, dim_out)
        
        # 初始化key网络参数
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # 创建队列（对比学习）
        self.register_buffer("queue", torch.randn(dim_out, K))
        self.queue = F.normalize(self.queue, dim=0)
        
        # 创建权重队列（用于MCLP任务）
        self.register_buffer("queue_weights", torch.randn(K, 3))  # [热度, 交通, 需求]
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # MCLP任务头
        self.facility_head = nn.Sequential(
            nn.Linear(dim_out, dim_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim_hidden // 2, 1)
        )
        
        # 覆盖预测头
        self.coverage_head = nn.Sequential(
            nn.Linear(dim_out, dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, idx, x, edge_index, edge_weight, dist_feat, degree_feat, batch_size):
        """前向传播"""
        # 计算query嵌入
        embs_q = self.q_net(x, edge_index, edge_weight, dist_feat, degree_feat)
        embs_q = F.normalize(embs_q, dim=1)
        
        if batch_size >= x.shape[0]:
            # 返回完整嵌入用于推理
            facility_logits = self.facility_head(embs_q)
            coverage_probs = self.coverage_head(embs_q)
            return embs_q, facility_logits, coverage_probs
        
        # 提取当前batch
        q = embs_q[idx * batch_size:(idx + 1) * batch_size, :]
        
        # 计算key嵌入（动量更新）
        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, dist_feat, degree_feat)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch_size:(idx + 1) * batch_size, :]
        
        # 正样本对比
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # 负样本对比（从队列中）
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # 组合logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.07  # 温度参数
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(x.device)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        # MCLP任务输出
        facility_logits = self.facility_head(embs_q)
        coverage_probs = self.coverage_head(embs_q)
        
        return embs_q, facility_logits, coverage_probs, logits, labels
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key编码器"""
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的key
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        
        self.queue_ptr[0] = ptr
    
    def predict_facilities(self, x, edge_index, edge_weight, dist_feat, degree_feat, K):
        """预测设施位置"""
        self.eval()
        with torch.no_grad():
            embs_q, facility_logits, coverage_probs = self.forward(
                0, x, edge_index, edge_weight, dist_feat, degree_feat, x.shape[0]
            )
            
            # 结合设施logits和覆盖概率
            scores = torch.sigmoid(facility_logits).squeeze() * coverage_probs.squeeze()
            
            # 选择得分最高的K个节点
            selected_indices = torch.topk(scores, min(K, len(scores))).indices
            
        return selected_indices.cpu().numpy(), scores.cpu().numpy()

class SelfSupervisedMCLPWrapper:
    """自监督MCLP包装器"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        
    def initialize_model(self, input_dim, hidden_dim=128, output_dim=64):
        """初始化模型"""
        self.model = MoCoMCLPModel(
            dim_in=input_dim,
            dim_hidden=hidden_dim,
            dim_out=output_dim,
            m=0.99,
            K=1024
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=5e-4
        )
        
        return self.model
    
    def train_on_instance(self, graph, epochs=100, batch_size=32):
        """在单个图实例上训练"""
        if self.model is None:
            input_dim = graph.x.shape[1]
            self.initialize_model(input_dim)
        
        self.model.train()
        losses = []
        
        # 准备数据
        x = graph.x.to(self.device).float()
        edge_index = graph.edge_index.to(self.device).long()
        edge_weight = graph.edge_attr.to(self.device).float() if graph.edge_attr is not None else None
        dist_feat = graph.dist_row_sum.to(self.device).float() if hasattr(graph, 'dist_row_sum') else torch.ones(x.shape[0], 1, device=self.device)
        degree_feat = graph.degree.to(self.device).float() if hasattr(graph, 'degree') else torch.ones(x.shape[0], 1, device=self.device)
        
        n_nodes = x.shape[0]
        num_batches = max(1, n_nodes // batch_size)
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches_processed = 0
            
            # 数据增强：节点重排（类似FLP）
            perm = torch.randperm(n_nodes)
            x_aug = x[perm]
            
            # 重新计算edge_index
            if edge_index.shape[1] > 0:
                edge_index_aug = perm[edge_index]
            else:
                edge_index_aug = edge_index
            
            for batch_idx in range(num_batches):
                try:
                    # 前向传播
                    embs_q, facility_logits, coverage_probs, logits, labels = self.model(
                        batch_idx, x_aug, edge_index_aug, edge_weight,
                        dist_feat[perm], degree_feat[perm], batch_size
                    )
                    
                    # 对比损失
                    contrastive_loss = F.cross_entropy(logits, labels)
                    
                    # MCLP任务损失
                    # 1. 设施分布损失（鼓励均匀分布）
                    facility_probs = torch.sigmoid(facility_logits).squeeze()
                    facility_dist_loss = -torch.std(facility_probs)  # 鼓励多样性
                    
                    # 2. 覆盖质量损失
                    if hasattr(graph, 'distance_matrix'):
                        dist_matrix = graph.distance_matrix.to(self.device)
                        coverage_radius = getattr(graph, 'coverage_radius', dist_matrix.max() * 0.15)
                        
                        # 模拟覆盖计算
                        coverage_loss = self._compute_coverage_loss(
                            facility_probs, dist_matrix, coverage_radius
                        )
                    else:
                        coverage_loss = torch.tensor(0.0, device=self.device)
                    
                    # 总损失
                    total_loss = contrastive_loss + 0.3 * facility_dist_loss + 0.5 * coverage_loss
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches_processed += 1
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                    continue
            
            if num_batches_processed > 0:
                avg_loss = epoch_loss / num_batches_processed
                losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
        
        return losses
    
    def _compute_coverage_loss(self, facility_probs, dist_matrix, coverage_radius, K=10):
        """计算覆盖损失"""
        n = len(facility_probs)
        
        # 选择概率最高的K个节点作为候选设施
        _, candidate_indices = torch.topk(facility_probs, min(K, n))
        
        # 计算每个节点的最小距离
        dist_to_candidates = dist_matrix[:, candidate_indices]
        min_dist = torch.min(dist_to_candidates, dim=1)[0]
        
        # 计算覆盖概率（距离越近，覆盖概率越高）
        coverage_probs = torch.exp(-min_dist / coverage_radius)
        
        # 损失：鼓励高覆盖
        coverage_loss = -torch.mean(coverage_probs)
        
        return coverage_loss
    
    def solve_mclp(self, graph, K, batch_size=32):
        """求解MCLP问题"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        # 准备数据
        x = graph.x.to(self.device).float()
        edge_index = graph.edge_index.to(self.device).long()
        edge_weight = graph.edge_attr.to(self.device).float() if graph.edge_attr is not None else None
        dist_feat = graph.dist_row_sum.to(self.device).float() if hasattr(graph, 'dist_row_sum') else torch.ones(x.shape[0], 1, device=self.device)
        degree_feat = graph.degree.to(self.device).float() if hasattr(graph, 'degree') else torch.ones(x.shape[0], 1, device=self.device)
        
        # 预测设施
        selected_indices, scores = self.model.predict_facilities(
            x, edge_index, edge_weight, dist_feat, degree_feat, K
        )
        
        # 计算覆盖率
        if hasattr(graph, 'distance_matrix') and hasattr(graph, 'coverage_radius'):
            dist_matrix = graph.distance_matrix
            coverage_radius = graph.coverage_radius
            
            # 计算实际覆盖
            if len(selected_indices) > 0:
                dist_to_selected = dist_matrix[:, selected_indices]
                min_dist = torch.min(dist_to_selected, dim=1)[0]
                covered_mask = (min_dist <= coverage_radius)
                coverage = torch.sum(covered_mask.float()).item()
            else:
                coverage = 0
        else:
            coverage = len(selected_indices) * 10  # 估计值
        
        return selected_indices, coverage, scores