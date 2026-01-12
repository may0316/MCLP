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

        #1
        self.fc_0 = nn.Linear(1,out_channels)
        self.fc_1 = nn.Embedding(500, out_channels)
        self.fc_2 = nn.Linear(3*out_channels, out_channels)

    # 1
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

        x_concat = torch.cat((x,edges,degree), dim=1)
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

        # Register a momentum version of the key encoder
        # self.k_net_momentum = GCN(dim_in, dim_hidden, dim_out)
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False    # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim_out, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_w", torch.rand(n_nodes, K))
        # self.queue_w = nn.functional.normalize(self.queue_w, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, idx, x, edge_index, edge_weight, edge, degree, batch, dist, perm):
        # Compute query embeddings
        embs_q = self.q_net(x, edge_index, edge_weight, edge, degree)
        embs_q = F.normalize(embs_q, dim=1)
        # q = F.normalize(q, dim=1)
        q = embs_q[idx*batch:(idx+1)*batch,:] # n * c

        w = dist[idx*batch:(idx+1)*batch,:]
        p = perm[idx*batch:(idx+1)*batch]

        weight = self.queue_w.clone().detach()[p]
        weight = weight/diameter*16-6
        # weight = torch.mean(weight,dim=1) * 20 - 10
        weight = torch.sigmoid(weight)
        # Compute key embeddings
        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, edge, degree)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch:(idx + 1) * batch, :]  # n * c

        # Positive pairs
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # l_pos = l_pos * 0.004

        # Negative pairs
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_neg = l_neg * weight

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.01


        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, w)

        return embs_q, logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, weight):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        self.queue_w[:, ptr: ptr + batch_size] = weight.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


import torch.optim as optim
train_dataset = create_more.get_random_data(n_nodes, 2, 0, device)
# train_dataset = create_more.get_starbucks_data(device)

model = MocoModel(2, 128, 64, n_nodes).to(device)
optimizer = optim.Adam(model.q_net.parameters(),lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss().to(device)

def train(x, edge_index, edge_weight, dist_row, degree, batch_size,n_nodes, dist, perm):
    loss_list = []
    model.train()
    num_batches = (n_nodes // batch_size) # 舍去最后的余数
    for i in range(num_batches):
        embs, logits, labels = model(i, x, edge_index, edge_weight, dist_row, degree, batch_size,dist,perm)
        loss = criterion(logits, labels)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss)
    loss_avg = sum(loss_list) / len(loss_list)
    return loss_avg.item()

for index, (_, points) in enumerate(train_dataset):
    graph, dist_all = create_more.build_graph_from_points(points, None, True, 'euclidean')
    dist0 = dist_all

loss_fin = 100
for epoch in range(200):
    for index, (_, points) in enumerate(train_dataset):
        n_nodes = points.shape[0]
        graph, dist_all = create_more.build_graph_from_points(points, None, True, 'euclidean')
        diameter = dist_all.max()
        adj_matrix = to_dense_adj(graph.edge_index)
        adj_matrix = torch.squeeze(adj_matrix)
        degree = torch.sum(adj_matrix, dim=0).to(device)
        degree = degree.long()
        perm = torch.arange(n_nodes)
        loss = train(graph.x, graph.edge_index, graph.edge_attr, graph.dist_row, degree,16, n_nodes, dist0, perm)
        print('Epoch {}, index {}, loss {:.4f}'.format(epoch, index, loss))
        # 用新的节点编号对x和edge_index进行重新编号
        perm = torch.randperm(n_nodes)
        dist0 = dist_all[perm]
        graph.x = graph.x[perm]
        graph.edge_index[0] = perm[graph.edge_index[0]]
        graph.edge_index[1] = perm[graph.edge_index[1]]
        if(epoch > 100 and loss_fin>loss):
            loss_fin = loss
            torch.save(model.state_dict(), 'pre.pth')