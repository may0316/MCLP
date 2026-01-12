import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import os
import create_more

n_nodes = 500
device = torch.device('cpu')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_nodes):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.be = nn.BatchNorm1d(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bd = nn.BatchNorm1d(out_channels)

        # 用于处理边特征和度特征
        self.fc_0 = nn.Linear(1, out_channels)
        self.fc_1 = nn.Embedding(500, out_channels)
        self.fc_2 = nn.Linear(3 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, edges, degree):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
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

        # Query network
        self.q_net = GCN(dim_in, dim_hidden, dim_out, n_nodes)

        # Key network
        self.k_net = GCN(dim_in, dim_hidden, dim_out, n_nodes)

        # 初始化key网络参数与query网络相同
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # 不通过梯度更新

        # 创建队列
        self.register_buffer("queue", torch.randn(dim_out, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_w", torch.rand(n_nodes, K))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, idx, x, edge_index, edge_weight, edge, degree, batch, dist, perm):
        # 计算query嵌入
        embs_q = self.q_net(x, edge_index, edge_weight, edge, degree)
        embs_q = F.normalize(embs_q, dim=1)
        q = embs_q[idx * batch:(idx + 1) * batch, :]  # n * c

        w = dist[idx * batch:(idx + 1) * batch, :]
        p = perm[idx * batch:(idx + 1) * batch]

        weight = self.queue_w.clone().detach()[p]
        weight = weight / dist.max() * 16 - 6
        weight = torch.sigmoid(weight)

        # 计算key嵌入
        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, edge, degree)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch:(idx + 1) * batch, :]  # n * c

        # 正样本对
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # 负样本对
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_neg = l_neg * weight

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.01

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        # 更新队列
        self._dequeue_and_enqueue(k, w)

        return embs_q, logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key编码器"""
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, weight):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # 简化处理

        # 替换ptr位置的key（出队和入队）
        self.queue[:, ptr: ptr + batch_size] = keys.T
        self.queue_w[:, ptr: ptr + batch_size] = weight.T
        ptr = (ptr + batch_size) % self.K  # 移动指针

        self.queue_ptr[0] = ptr


import torch.optim as optim

# 加载训练数据 - 使用向后兼容的函数
print("正在加载预训练数据...")
train_dataset = create_more.get_random_data(n_nodes, 2, 0, device)
print(f"预训练数据集加载完成，包含 {len(train_dataset)} 个实例")

# 或者使用预生成的pretrain_dataset.pkl（如果已存在）
# try:
#     train_dataset = create_more.load_dataset('pretrain_dataset.pkl')
#     print(f"从文件加载预训练数据集，包含 {len(train_dataset)} 个实例")
# except FileNotFoundError:
#     print("未找到pretrain_dataset.pkl，使用随机生成的数据")
#     train_dataset = create_more.get_random_data(n_nodes, 2, 0, device)

model = MocoModel(2, 128, 64, n_nodes).to(device)
optimizer = optim.Adam(model.q_net.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss().to(device)


def train(x, edge_index, edge_weight, dist_row, degree, batch_size, n_nodes, dist, perm):
    """训练函数"""
    loss_list = []
    model.train()
    num_batches = (n_nodes // batch_size)  # 舍去最后的余数
    
    for i in range(num_batches):
        embs, logits, labels = model(i, x, edge_index, edge_weight, 
                                     dist_row, degree, batch_size, dist, perm)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss)
    
    loss_avg = sum(loss_list) / len(loss_list)
    return loss_avg.item()


# 预处理数据
for index, (_, points) in enumerate(train_dataset):
    graph, dist_all = create_more.build_graph_from_points(points, None, True, 'euclidean')
    dist0 = dist_all

# 训练循环
loss_fin = 100
for epoch in range(200):
    for index, (_, points) in enumerate(train_dataset):
        n_nodes = points.shape[0]
        graph, dist_all = create_more.build_graph_from_points(points, None, True, 'euclidean')
        diameter = dist_all.max()
        
        # 计算度矩阵
        adj_matrix = to_dense_adj(graph.edge_index)
        adj_matrix = torch.squeeze(adj_matrix)
        degree = torch.sum(adj_matrix, dim=0).to(device)
        degree = degree.long()
        
        perm = torch.arange(n_nodes)
        loss = train(graph.x, graph.edge_index, graph.edge_attr, 
                    graph.dist_row, degree, 16, n_nodes, dist0, perm)
        
        print(f'Epoch {epoch}, index {index}, loss {loss:.4f}')
        
        # 数据增强：随机重排节点
        perm = torch.randperm(n_nodes)
        dist0 = dist_all[perm]
        graph.x = graph.x[perm]
        graph.edge_index[0] = perm[graph.edge_index[0]]
        graph.edge_index[1] = perm[graph.edge_index[1]]
        
        # 保存最佳模型
        if epoch > 100 and loss_fin > loss:
            loss_fin = loss
            torch.save(model.state_dict(), 'pre_mclp.pth')
            print(f'Model saved with loss {loss:.4f}')

print("Pre-training completed!")
print(f"最佳模型已保存到 pre_mclp.pth (loss: {loss_fin:.4f})")