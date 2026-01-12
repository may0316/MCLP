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

        self.fc_0 = nn.Linear(1,out_channels)
        self.fc_1 = nn.Embedding(500, out_channels)
        self.fc_2 = nn.Linear(3*out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, edges, degree):
        x = self.conv1(x, edge_index,edge_weight)
        x = F.relu(x)
        x = F.dropout(x, 0)
        x = self.conv2(x, edge_index,edge_weight)
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
        self.register_buffer("queue_w", torch.randn(n_nodes, K))
        self.queue_w = nn.functional.normalize(self.queue_w, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, idx, x, edge_index, edge_weight, edge, degree, batch):
        # Compute query embeddings
        embs_q = self.q_net(x, edge_index, edge_weight, edge, degree)
        embs_q = F.normalize(embs_q, dim=1)
        # q = F.normalize(q, dim=1)
        if (batch == x.shape[0]):
            return embs_q
        q = embs_q[idx*batch:(idx+1)*batch,:] # n * c

        # Compute key embeddings
        with torch.no_grad():
            self._momentum_update_key_encoder()
            embs_k = self.k_net(x, edge_index, edge_weight, edge, degree)
            embs_k = F.normalize(embs_k, dim=1)
            k = embs_k[idx * batch:(idx + 1) * batch, :]  # n * c

        # Positive pairs
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative pairs
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.07

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return embs_q, logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_net.parameters(), self.k_net.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


import torch.optim as optim

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
 # 调用函数加载保存的dataset数据
train_dataset = create_more.load_dataset('dataset_800.pkl')
# train_dataset = create_more.get_starbucks_data(device)
model = MocoModel(2, 128, 64, n_nodes).to(device)
model.load_state_dict(torch.load('pre.pth'))
K = 50

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_hidden1), nn.ReLU(), nn.Linear(dim_hidden1, dim_out)
        # )

    def forward(self, embs):
        mlp_embs = self.fc(embs)
        return mlp_embs

class MLP1(torch.nn.Module):
    def __init__(self, model, dim_in, dim_hidden, dim_hidden1, dim_out):
        super().__init__()
        self.pretrained_model = model
        for param in self.pretrained_model.parameters():
            param.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_hidden1), nn.ReLU(), nn.Linear(dim_hidden1, dim_out)
        # )

    def forward(self, idx, x, edge_index, edge_weight, edge, degree, batch):
        embs = self.pretrained_model(0, x, edge_index, edge_weight, edge, degree, x.shape[0])
        mlp_embs = self.fc(embs)
        return mlp_embs

def rounding(x):
    '''
    Fast pipage rounding implementation for uniform matroid
    '''
    i = 0
    j = 1
    x = x.clone()
    for t in range(len(x)-1):
        if x[i] == 0 and x[j] == 0:
            i = max((i,j)) + 1
        elif x[i] + x[j] < 1:
            if np.random.rand() < x[i]/(x[i] + x[j]):
                x[i] = x[i] + x[j]
                x[j] = 0
                j = max((i,j)) + 1
            else:
                x[j] = x[i] + x[j]
                x[i] = 0
                i = max((i,j)) + 1
        else:
            if np.random.rand() < (1 - x[j])/(2 - x[i] - x[j]):
                x[j] = x[i] + x[j] - 1
                x[i] = 1
                i = max((i,j)) + 1

            else:
                x[i] = x[i] + x[j] - 1
                x[j] = 1
                j = max((i,j)) + 1
    return x



for index, (_, points) in enumerate(train_dataset):
    start = time.time()
    model_mlp = MLP(64, 32, K).to(device)
    # model_mlp1 = MLP1(model, 64, 32, 16, K).to(device)
    optimizer = optim.Adam(model_mlp.parameters(),
                           lr=0.001, weight_decay=5e-4)
    graph, dist_all = create_more.build_graph_from_points(points, None, True, 'euclidean')
    diameter = dist_all.max()
    adj_matrix = to_dense_adj(graph.edge_index)
    adj_matrix = torch.squeeze(adj_matrix)
    degree = torch.sum(adj_matrix, dim=0).to(device)
    degree = degree.long()

    embs = model(0, graph.x, graph.edge_index, graph.edge_attr, graph.dist_row, degree, graph.x.shape[0])
    embs = embs.detach()
    for t in range(2000):
        dist = model_mlp(embs)
        # dist = model_mlp1(0, graph.x, graph.edge_index, graph.edge_attr, graph.dist_row, degree, graph.x.shape[0])
        x_best = torch.softmax(dist * 100, 0).sum(dim=1).to(device)
        x_best = 2 * (torch.sigmoid(4 * x_best) - 0.5)
        if x_best.sum() > K:
            x_best = K * x_best / x_best.sum()

        dist_0, order = torch.sort(dist_all, dim=1)
        dmax_vec = diameter * torch.ones(dist_0.shape[0], 1).to(device)
        off_one = torch.cat((dist_0[:, 1:], dmax_vec), dim=1).to(device)
        m = dist_0 - off_one.to(device)

        x_sort = x_best[order].to(device)
        probs = 1 - torch.cumprod(1 - x_sort, dim=1).to(device)
        vals = diameter + (m * probs).sum(dim=1).to(device)
        loss = torch.sum(vals).to(device)
        # print(t, "-",loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    testvals = []
    for _ in range(10):
        if x_best.sum() != K:
            x_best = K * x_best / x_best.sum()
        y = rounding(x_best)
        # sorted_indices = torch.argsort(x_best, descending=True)
        # marked_indices = sorted_indices[:K]
        #
        # y = torch.zeros_like(x_best)
        # y[marked_indices] = 1
        # y = y.detach().cpu().numpy()
        # data = points.cpu().numpy()  # 转换为NumPy数组
        # colors = ['blue' if val == 0 else 'red' for val in y]
        # plt.scatter(data[:, 0], data[:, 1], c=colors)
        # plt.title('Random Data Visualization')
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        # plt.show()
        # indices = [index for index, value in enumerate(y) if value == 1]
        # output_tensor = dist_all[:, indices]
        # min_indices = torch.argmin(output_tensor, dim=1)
        # min_indices = min_indices.cpu().numpy()
        # colors = np.random.rand(K, 3)  # 生成随机的RGB颜色
        # plt.figure(figsize=(8, 8))
        # for index in range(K):
        #     color_points = data[min_indices == index]
        #     plt.scatter(color_points[:, 0], color_points[:, 1], s=6, color=colors[index])
        # for index in range(K):
        #     color_points = data[y == 1]
        #     plt.scatter(color_points[index, 0], color_points[index, 1], color=colors[index])
        # plt.title('Points with Different Colors')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.legend()
        # plt.show()
        dist_0, order = torch.sort(dist_all, dim=1)
        dmax_vec = diameter * torch.ones(dist_0.shape[0], 1).to(device)
        off_one = torch.cat((dist_0[:, 1:], dmax_vec), dim=1).to(device)
        m = dist_0 - off_one.to(device)

        x_sort = y[order].to(device)
        probs = 1 - torch.cumprod(1 - x_sort, dim=1).to(device)
        vals = diameter + (m * probs).sum(dim=1).to(device)

        testvals.append(torch.sum(vals).item())
    end = time.time()
    print("time:", end - start)
    print(index, testvals[np.argmin(testvals)])