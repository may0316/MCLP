import numpy as np
import torch
import torch_geometric as pyg
import time

def build_graph_from_points(points, dist=None, return_dist=False, distance_metric='euclidean'):
    if dist is None:
        dist_func = _get_distance_func(distance_metric)
        dist = dist_func(points, points, points.device)
    # norm_dist = dist * 1.414 / dist.max()
    edge_indices = torch.nonzero(dist <= 0.05*1.414, as_tuple=False).transpose(0, 1)
    edge_attrs = dist[torch.nonzero(dist <= 0.05*1.414, as_tuple=True)]
    edge_attrs = 1/(edge_attrs+1e-3)
    dist_row = torch.sum(dist, dim=1)
    dist_row = dist_row.unsqueeze(1)
    g = pyg.data.Data(x=points, edge_index=edge_indices, edge_attr=edge_attrs, dist_row=dist_row)
    if return_dist:
        return g, dist
    else:
        return g

def _get_distance_func(distance):
    if distance == 'euclidean':
        return _pairwise_euclidean
    elif distance == 'cosine':
        return _pairwise_cosine
    elif distance == 'manhattan':
        return _pairwise_manhattan
    else:
        raise NotImplementedError


def _pairwise_euclidean(data1, data2, device=torch.device('cpu')):
    """Compute pairwise Euclidean distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    dis = torch.sqrt(dis)
    return dis


def _pairwise_manhattan(data1, data2, device=torch.device('cpu')):
    """Compute pairwise Manhattan distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    dis = torch.abs(A - B)
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    return dis


def _pairwise_cosine(data1, data2, device=torch.device('cpu')):
    """Compute pairwise cosine distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze(-1)
    return cosine_dis

def get_random_data(num_data, dim, seed, device):
    torch.random.manual_seed(seed)
    dataset = [(f'rand{_}', torch.rand(num_data, dim, device=device)) for _ in range(1)]
    return dataset

def get_starbucks_data(device):
    dataset = []
    areas = ['london', 'newyork', 'shanghai', 'seoul']
    for area in areas:
        with open(f'data/starbucks/{area}.csv', encoding='utf-8-sig') as f:
            locations = []
            for l in f.readlines():
                l_str = l.strip().split(',')
                if l_str[0] == 'latitude' and l_str[1] == 'longitude':
                    continue
                n1, n2 = float(l_str[0]) / 365 * 400, float(l_str[1]) / 365 * 400  # real-world coordinates: x100km
                locations.append((n1, n2))
        dataset.append((area, torch.tensor(locations, device=device)))
    return dataset

device = torch.device('cpu')
# train_dataset = get_starbucks_data(device)
# for index, (_, points) in enumerate(train_dataset):
#     graph, dist = build_graph_from_points(points, None, True, 'euclidean')

import pickle

def save_dataset(dataset, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)

def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset


# 调用函数保存dataset数据到文件
dataset = get_random_data(500, 2, 10, device)
save_dataset(dataset, 'dataset_800.pkl')





