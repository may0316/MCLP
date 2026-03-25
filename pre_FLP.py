# pre_FLP.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import os
import pickle
import sys
from config import MCLPConfig

# 设备设置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 获取配置
# 可以从命令行参数获取配置
if len(sys.argv) > 1:
    # 用法: python pre_FLP.py graph_size p radius
    graph_size = int(sys.argv[1])
    p = int(sys.argv[2])
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    config = MCLPConfig(graph_size=graph_size, p=p, radius=radius)
else:
    # 使用默认配置
    config = MCLPConfig()

print(f"使用配置:\n{config}")

class GCN_MCLP(torch.nn.Module):
    """用于MCLP的GCN编码器，考虑覆盖半径和需求权重"""
    def __init__(self, in_channels, hidden_channels, out_channels, n_nodes, max_demand=100):
        super(GCN_MCLP, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.be = nn.BatchNorm1d(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bd = nn.BatchNorm1d(out_channels)

        # 距离编码（覆盖半径内的距离信息）
        self.fc_distance = nn.Linear(1, out_channels)
        # 度编码（覆盖的需求权重之和）
        self.fc_degree = nn.Linear(1, out_channels)  # 改为Linear以适应连续的需求权重
        # 融合层
        self.fc_fusion = nn.Linear(3 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, distance_features, degree_features):
        """
        Args:
            x: 节点特征 (n_nodes, in_channels)
            edge_index: 边索引
            edge_weight: 边权重（距离）
            distance_features: 距离特征 (n_nodes, 1) - 节点到最近设施的距离相关信息
            degree_features: 度特征 (n_nodes, 1) - 覆盖的需求权重之和
        """
        # GCN编码
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn(x)
        x = F.relu(x)

        # 距离编码
        distance_enc = self.fc_distance(distance_features)
        distance_enc = self.be(distance_enc)
        distance_enc = F.relu(distance_enc)

        # 度编码（覆盖需求权重）
        degree_enc = self.fc_degree(degree_features.float())
        degree_enc = self.bd(degree_enc)
        degree_enc = F.relu(degree_enc)

        # 特征融合
        x_concat = torch.cat((x, distance_enc, degree_enc), dim=1)
        x_concat = self.fc_fusion(x_concat)
        
        return x_concat


class MocoModel_MCLP(torch.nn.Module):
    """MoCo自监督学习框架用于MCLP"""
    def __init__(self, dim_in, dim_hidden, dim_out, n_nodes, m=0.99, K=256, max_demand=100):
        super().__init__()
        self.m = m
        self.K = K

        # Query network
        self.q_net = GCN_MCLP(dim_in, dim_hidden, dim_out, n_nodes, max_demand)

        # Key network
        self.k_net = GCN_MCLP(dim_in, dim_hidden, dim_out, n_nodes, max_demand)

        # 初始化key网络
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("queue", torch.randn(dim_out, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, idx, x, edge_index, edge_weight, distance_features, 
                degree_features, batch_size):
        """
        前向传播，返回查询嵌入
        """
        # 计算查询嵌入
        embs_q = self.q_net(x, edge_index, edge_weight, distance_features, degree_features)
        embs_q = F.normalize(embs_q, dim=1)
        
        if batch_size == x.shape[0]:
            return embs_q
            
        q = embs_q[idx * batch_size:(idx + 1) * batch_size, :]

        # 计算键嵌入
        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, distance_features, degree_features)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch_size:(idx + 1) * batch_size, :]

        # 正样本对
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # 负样本对
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.07

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        # 更新队列
        self._dequeue_and_enqueue(k)

        return embs_q, logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key encoder"""
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


def compute_coverage_graph(points, radius, demand_weights, device):
    """
    根据覆盖半径构建图
    Args:
        points: 节点坐标 (n_nodes, 2)
        radius: 覆盖半径
        demand_weights: 需求权重 (n_nodes,)
        device: 设备
    Returns:
        edge_index: 边索引
        edge_weight: 边权重（距离）
        covered_demand: 每个节点覆盖的需求权重之和（度特征）(n_nodes, 1)
        dist_matrix: 距离矩阵
    """
    n_nodes = points.shape[0]
    
    # 计算距离矩阵
    diff = points.unsqueeze(1) - points.unsqueeze(0)
    dist_matrix = torch.norm(diff, dim=2)
    
    # 在覆盖半径内建立边（包括自环）
    mask = dist_matrix <= radius
    edge_index = mask.nonzero(as_tuple=False).t()
    
    # 边权重为距离
    edge_weight = dist_matrix[edge_index[0], edge_index[1]].unsqueeze(1)
    
    # 计算每个节点覆盖的需求权重之和
    # 对于每个节点i，计算所有能被i覆盖的节点的需求权重之和
    covered_demand = (mask.float() * demand_weights.unsqueeze(0)).sum(dim=1).unsqueeze(1)
    
    return edge_index, edge_weight, covered_demand, dist_matrix


def compute_distance_features(points, selected_facilities, device):
    """计算每个节点到最近设施的距离"""
    if selected_facilities is None or len(selected_facilities) == 0:
        return torch.ones(points.shape[0], 1).to(device)
    
    n_nodes = points.shape[0]
    diff = points.unsqueeze(1) - points[selected_facilities].unsqueeze(0)
    dist_to_facilities = torch.norm(diff, dim=2)
    min_dist, _ = dist_to_facilities.min(dim=1)
    
    return min_dist.unsqueeze(1)


def train_epoch(model, optimizer, criterion, graph_data, batch_size, device):
    """训练一个epoch"""
    model.train()
    loss_list = []
    
    n_nodes = graph_data['x'].shape[0]
    num_batches = n_nodes // batch_size
    
    if num_batches == 0:
        return 0
    
    for i in range(num_batches):
        _, logits, labels = model(
            i, 
            graph_data['x'], 
            graph_data['edge_index'], 
            graph_data['edge_weight'],
            graph_data['distance_features'],
            graph_data['degree_features'],
            batch_size
        )
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
    
    return np.mean(loss_list) if loss_list else 0


def pretrain_mclp_model(config):
    """
    预训练MCLP模型
    """
    # 加载数据
    print(f"加载数据: {config.data_path}")
    
    if not os.path.exists(config.data_path):
        print(f"数据文件不存在: {config.data_path}")
        print("请先运行: python sponet_gen_data.py")
        return None
    
    with open(config.data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"数据集大小: {len(dataset)} 个实例")
    
    # 初始化模型
    model = MocoModel_MCLP(2, config.hidden_dim, config.out_dim, config.graph_size,
                           m=config.momentum, K=config.queue_size).to(device)
    optimizer = torch.optim.Adam(model.q_net.parameters(), lr=config.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs_pretrain):
        epoch_losses = []
        
        for idx, instance in enumerate(dataset):
            points = instance['loc'].to(device)
            radius = instance['radius']
            p = instance['p']
            demand_weights = instance['demand'].to(device)
            
            # 构建覆盖图
            edge_index, edge_weight, degree_features, dist_matrix = compute_coverage_graph(
                points, radius, demand_weights, device
            )
            
            # 节点特征（坐标）
            x = points
            
            # 距离特征（预训练时使用随机选择）
            n_facilities = min(p, points.shape[0])
            random_indices = torch.randperm(points.shape[0])[:n_facilities]
            distance_features = compute_distance_features(points, random_indices, device)
            
            graph_data = {
                'x': x,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'distance_features': distance_features,
                'degree_features': degree_features
            }
            
            loss = train_epoch(model, optimizer, criterion, graph_data, config.batch_size, device)
            if loss > 0:
                epoch_losses.append(loss)
            
            if idx % 100 == 0:
                print(f"  Instance {idx}/{len(dataset)}, Loss: {loss:.4f}")
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f'Epoch {epoch}/{config.num_epochs_pretrain}, Avg Loss: {avg_loss:.4f}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), config.pretrained_path)
                print(f'  模型保存到 {config.pretrained_path}')
        else:
            print(f'Epoch {epoch}/{config.num_epochs_pretrain}, No valid losses')
    
    return model


if __name__ == "__main__":
    print("开始预训练...")
    model = pretrain_mclp_model(config)
    if model is not None:
        print("预训练完成！")
    else:
        print("预训练失败！")