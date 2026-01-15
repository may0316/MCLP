import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
import numpy as np
import copy

class SelfSupervisedMCLPWrapper:
    """自监督MCLP包装器，可以无缝集成到现有框架"""
    
    def __init__(self, device='cpu', use_tourism_features=False):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.use_tourism_features = use_tourism_features
        
    def initialize_model(self, input_dim, hidden_dim=64, output_dim=32):
        """初始化模型"""
        self.model = SelfSupervisedMCLPModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=1e-4
        )
        
        return self.model
    
    def train_on_instance(self, instance, epochs=20, graph_builder=None):
        """在单个实例上进行自监督训练"""
        if self.model is None:
            # 简单估算输入维度
            if self.use_tourism_features:
                input_dim = 8  # 坐标(2) + 地形(3) + 权重(3)
            else:
                input_dim = 5  # 坐标(2) + 权重(3)
            self.initialize_model(input_dim)
        
        # 将实例转换为图
        graph = self._instance_to_graph(instance, graph_builder)
        
        # 自监督训练
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            try:
                # 创建增强视图
                views = self._create_augmented_views(graph, n_views=2)
                
                # 前向传播
                orig_emb, orig_logits = self.model(
                    graph.x, graph.edge_index, graph.edge_attr
                )
                
                # 对比损失
                contrastive_loss = 0
                for view in views:
                    view_emb, _ = self.model(view.x, view.edge_index, view.edge_attr)
                    contrastive_loss += self._compute_contrastive_loss(orig_emb, view_emb)
                contrastive_loss /= len(views)
                
                # MCLP预训练任务损失
                task_loss = self._compute_pretext_task_loss(graph, orig_emb, K=10)
                
                # 总损失
                total_loss = contrastive_loss + 0.5 * task_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                losses.append(total_loss.item())
                
                if epoch % 5 == 0:
                    print(f"    Epoch {epoch}: Loss={total_loss.item():.4f}")
                    
            except Exception as e:
                print(f"    Epoch {epoch} 失败: {e}")
                continue
        
        return losses
    
    def solve_mclp(self, instance, K, graph_builder=None):
        """使用训练好的模型求解MCLP"""
        self.model.eval()
        
        # 将实例转换为图
        graph = self._instance_to_graph(instance, graph_builder)
        
        with torch.no_grad():
            # 获取嵌入和设施概率
            embeddings, facility_logits = self.model(
                graph.x, graph.edge_index, graph.edge_attr
            )
            
            # 获取设施选择概率
            facility_probs = torch.sigmoid(facility_logits).squeeze()
            
            # 选择概率最高的K个节点
            selected_indices = torch.topk(facility_probs, min(K, len(facility_probs))).indices.cpu().numpy()
            
            # 计算覆盖率
            coverage = self._compute_coverage(instance, selected_indices)
        
        return selected_indices, coverage
    
    def _instance_to_graph(self, instance, graph_builder=None):
        """将实例转换为图数据"""
        if graph_builder is not None:
            graph = graph_builder(instance)
        else:
            # 使用create_more中的build_mclp_graph
            from create_more import build_mclp_graph
            graph = build_mclp_graph(instance)
        
        # 确保数据类型正确
        graph.x = graph.x.float().to(self.device)
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr.float().to(self.device)
        
        # 确保edge_index是long类型
        graph.edge_index = graph.edge_index.long()
        
        return graph
    
    def _create_augmented_views(self, graph, n_views=2):
        """创建增强视图 - 修复版本"""
        views = []
        
        for _ in range(n_views):
            # 克隆图
            aug_graph = graph.clone()
            
            # 1. 特征噪声
            if torch.rand(1).item() < 0.5:
                noise = torch.randn_like(aug_graph.x) * 0.1
                aug_graph.x = aug_graph.x + noise
            
            # 2. 边丢弃 - 修复：确保数据类型正确
            if torch.rand(1).item() < 0.3 and aug_graph.edge_index.shape[1] > 0:
                num_edges = aug_graph.edge_index.shape[1]
                keep_probs = torch.ones(num_edges) * 0.9
                keep_mask = torch.rand(num_edges) < keep_probs
                keep_mask = keep_mask.bool()  # 确保是bool类型
                
                if keep_mask.sum() > 0:
                    aug_graph.edge_index = aug_graph.edge_index[:, keep_mask]
                    if aug_graph.edge_attr is not None:
                        aug_graph.edge_attr = aug_graph.edge_attr[keep_mask]
            
            # 3. 节点特征掩码 - 修复：确保mask是bool类型
            if torch.rand(1).item() < 0.3:
                # 创建bool类型的掩码
                mask = torch.rand_like(aug_graph.x) > 0.8
                mask = mask.bool()  # 显式转换为bool类型
                aug_graph.x[mask] = 0
            
            views.append(aug_graph)
        
        return views
    
    def _compute_contrastive_loss(self, emb1, emb2):
        """计算对比损失"""
        # 归一化嵌入
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(emb1, emb2.T)
        
        # 温度参数
        temperature = 0.1
        
        # 对角线是正样本
        batch_size = emb1.shape[0]
        labels = torch.arange(batch_size).to(self.device).long()
        
        # InfoNCE损失
        loss = F.cross_entropy(similarity / temperature, labels)
        
        return loss
    
    def _compute_pretext_task_loss(self, graph, embeddings, K):
        """计算预训练任务损失"""
        num_nodes = embeddings.shape[0]
        
        # 生成伪标签：基于度数选择设施
        with torch.no_grad():
            # 计算节点度数
            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                # 从edge_index计算度数
                if graph.edge_index.shape[1] > 0:
                    degree = torch.zeros(num_nodes, device=self.device)
                    unique, counts = torch.unique(graph.edge_index[0], return_counts=True)
                    degree[unique] = counts.float()
                else:
                    degree = torch.ones(num_nodes, device=self.device)
            else:
                degree = torch.ones(num_nodes, device=self.device)
            
            # 选择度数最高的K个节点作为"伪设施"
            pseudo_facilities = torch.zeros(num_nodes, device=self.device)
            if num_nodes > 0:
                _, top_indices = torch.topk(degree, min(K, num_nodes))
                pseudo_facilities[top_indices] = 1
        
        # 预测哪些节点是设施
        facility_pred = self.model.facility_head(embeddings).squeeze()
        task_loss = F.binary_cross_entropy_with_logits(
            facility_pred, pseudo_facilities
        )
        
        return task_loss
    
    def _compute_coverage(self, instance, selected_indices):
        """计算覆盖率"""
        from create_more import _pairwise_euclidean
        
        points = instance['points'].to(self.device)
        
        # 计算距离矩阵
        dist_matrix = _pairwise_euclidean(points, points, self.device)
        
        # 获取覆盖半径
        coverage_radius = instance.get('coverage_radius', dist_matrix.max().item() * 0.15)
        
        # 计算覆盖
        if len(selected_indices) > 0:
            selected_indices_tensor = torch.tensor(selected_indices, device=self.device)
            dist_to_selected = dist_matrix[:, selected_indices_tensor]
            min_dist = torch.min(dist_to_selected, dim=1)[0]
            covered_mask = (min_dist <= coverage_radius)
            
            if 'total_weights' in instance and instance['total_weights'] is not None:
                total_weights = instance['total_weights'].to(self.device)
                coverage = torch.sum(total_weights[covered_mask]).item()
            else:
                coverage = torch.sum(covered_mask.float()).item()
        else:
            coverage = 0
        
        return coverage


class SelfSupervisedMCLPModel(nn.Module):
    """自监督MCLP模型"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super().__init__()
        
        # GNN编码器
        self.gnn_encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # 设施预测头
        self.facility_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        # GNN编码
        embeddings = self.gnn_encoder(x, edge_index, edge_weight)
        
        # 设施预测
        facility_logits = self.facility_head(embeddings)
        
        return embeddings, facility_logits


class GNNEncoder(nn.Module):
    """GNN编码器"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        # 确保edge_index是long类型
        if edge_index is not None:
            edge_index = edge_index.long()
        
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        
        return x