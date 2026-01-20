import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np


# =====================================================
# MCLP-aware GCN Encoder
# =====================================================
class GCNEncoder(nn.Module):
    """
    MCLP-aware GCN Encoder

    输入：
    - x            : 节点属性（坐标 + 文旅特征 + 权重）
    - dist_feat    : soft coverage potential
    - degree_feat  : 覆盖半径内的需求权重和

    输出：
    - 节点嵌入（facility suitability embedding）
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.fc_dist = nn.Linear(1, out_channels)
        self.fc_degree = nn.Linear(1, out_channels)

        self.merge = nn.Linear(3 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, dist_feat, degree_feat):
        # --- GCN ---
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(self.bn1(h))
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(self.bn2(h))

        # --- MCLP-aware encodings ---
        d_feat = F.relu(self.fc_dist(dist_feat))
        deg_feat = F.relu(self.fc_degree(degree_feat))

        # --- 融合 ---
        h = torch.cat([h, d_feat, deg_feat], dim=1)
        h = self.merge(h)

        return h


# =====================================================
# Self-supervised MCLP Model
# =====================================================
class MCLPSelfSupervisedModel(nn.Module):
    """
    自监督 MCLP 模型

    输出 score_i ≈ 选点 i 作为设施的“覆盖势能”
    """
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()

        self.encoder = GCNEncoder(dim_in, dim_hidden, dim_out)

        self.facility_head = nn.Sequential(
            nn.Linear(dim_out, dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, 1)
        )

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_attr.squeeze()
        dist_feat = graph.dist_row_sum
        degree_feat = graph.degree

        emb = self.encoder(
            x, edge_index, edge_weight,
            dist_feat, degree_feat
        )

        score = self.facility_head(emb).squeeze()
        return emb, score


# =====================================================
# Wrapper
# =====================================================
class SelfSupervisedMCLPWrapper:
    """
    自监督 MCLP 训练 + 求解包装器
    """
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None

    # -------------------------------------------------
    def initialize_model(self, input_dim, hidden_dim=128, output_dim=64):
        self.model = MCLPSelfSupervisedModel(
            input_dim, hidden_dim, output_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=5e-4
        )

        return self.model

    # -------------------------------------------------
    def coverage_loss(self, scores, graph, K, temperature=0.1):
        """
        连续可导的 MCLP 覆盖损失（Soft Top-K）

        scores      : [N]
        返回        : scalar loss
        """
        dist = graph.distance_matrix.to(self.device)   # [N, N]
        R = graph.coverage_radius
        w = graph.total_weights.to(self.device)         # [N]

        # -------------------------------------------------
        # 1. soft facility selection (≈ top-K)
        # -------------------------------------------------
        # 将 score 映射为 [0,1]，并控制“接近 K 个被选中”
        p = torch.sigmoid(scores / temperature)         # [N]

        # 归一化到“期望 K 个设施”
        p = K * p / (p.sum() + 1e-6)                     # Σ p ≈ K

        # -------------------------------------------------
        # 2. soft coverage
        # -------------------------------------------------
        # coverage_ij = p_i * exp(-d_ij / R)
        cover_ij = p.unsqueeze(0) * torch.exp(-dist / R)

        # 每个需求点被“至少一个设施”覆盖的强度
        cover_j = 1.0 - torch.exp(-cover_ij.sum(dim=1))

        # -------------------------------------------------
        # 3. 加权最大覆盖目标
        # -------------------------------------------------
        weighted_cover = w * cover_j
        loss = -weighted_cover.mean()

        return loss

    def train_on_instance(self, graph, epochs=100, K=10, batch_size=None, **kwargs):
        """
        单实例（整图）自监督训练
        """
        if self.model is None:
            self.initialize_model(graph.x.shape[1])

        graph = graph.to(self.device)
        self.model.train()

        losses = []

        for ep in range(epochs):
            # forward
            _, scores = self.model(graph)

            # === 关键：使用 soft / 可导的 MCLP loss ===
            loss = self.coverage_loss(scores, graph, K)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())

            if ep % 10 == 0:
                print(f"Epoch {ep:03d} | Loss = {loss.item():.4f}")

        return losses

    # -------------------------------------------------
    @torch.no_grad()
    def solve_mclp(self, graph, K):
        """
        用训练好的模型求解 MCLP
        """
        self.model.eval()
        graph = graph.to(self.device)

        _, scores = self.model(graph)
        selected = torch.topk(scores, K).indices.cpu().numpy()

        # 计算真实覆盖
        dist = graph.distance_matrix
        R = graph.coverage_radius

        d = dist[:, selected]
        covered = (d.min(dim=1)[0] <= R).float()
        coverage = covered.sum().item()

        return selected, coverage, scores.cpu().numpy()
