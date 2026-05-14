import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1) -> None:
        super(SelfAttention, self).__init__()

        self.input_dim = input_dim
        self.attn_dim = hidden_dim
        self.output_dim = input_dim

        self.w_qs = nn.Linear(self.input_dim, self.attn_dim, bias=False)
        self.w_ks = nn.Linear(self.input_dim, self.attn_dim, bias=False)
        self.w_vs = nn.Linear(self.input_dim, self.attn_dim, bias=False)

        self.fc = nn.Linear(self.attn_dim, self.output_dim, bias=False)
        self.layer_norm1 = nn.LayerNorm(self.input_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.output_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLPBlock(self.output_dim, hidden_dim, dropout)

    def forward(self, feat: Tensor, mask=None):
        residual = feat

        query_feat = self.w_qs(feat)
        key_feat = self.w_ks(feat).transpose(1, 2)
        value_feat = self.w_vs(feat)

        attn = torch.bmm(query_feat, key_feat) / math.sqrt(self.attn_dim)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attn = attn.masked_fill(mask, -1e12)
        attn = F.softmax(attn, dim=-1)
        feat = torch.bmm(attn, value_feat)

        feat = self.layer_norm1(self.dropout(self.fc(feat)) + residual)
        feat = self.layer_norm2(self.mlp(feat) + feat)

        return feat


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_head, dropout=0.1) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self.input_dim = input_dim
        self.attn_dim = hidden_dim // n_head
        self.n_head = n_head
        self.output_dim = input_dim

        self.w_qs = nn.Linear(self.input_dim, self.n_head * self.attn_dim, bias=False)
        self.w_ks = nn.Linear(self.input_dim, self.n_head * self.attn_dim, bias=False)
        self.w_vs = nn.Linear(self.input_dim, self.n_head * self.attn_dim, bias=False)

        self.fc = nn.Linear(self.n_head * self.attn_dim, self.output_dim, bias=False)

        self.layer_norm1 = nn.LayerNorm(self.input_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.output_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLPBlock(self.output_dim, hidden_dim, dropout)

    def forward(self, feat: Tensor, mask=None):
        residual = feat
        B, N, _ = feat.shape

        query_feat = (
            self.w_qs(feat)
            .view(B, N, self.n_head, self.attn_dim)
            .transpose(1, 2)
            .reshape(B * self.n_head, N, self.attn_dim)
        )
        key_feat = (
            self.w_ks(feat)
            .view(B, N, self.n_head, self.attn_dim)
            .transpose(1, 2)
            .reshape(B * self.n_head, N, self.attn_dim)
            .transpose(1, 2)
        )
        value_feat = (
            self.w_vs(feat)
            .view(B, N, self.n_head, self.attn_dim)
            .transpose(1, 2)
            .reshape(B * self.n_head, N, self.attn_dim)
        )
        attn = torch.bmm(query_feat, key_feat) / math.sqrt(self.attn_dim)

        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(self.n_head, 1, 1)
            attn = attn.masked_fill(mask, -1e12)
        attn = F.softmax(attn, dim=-1)
        feat = (
            torch.bmm(attn, value_feat)
            .reshape(B, self.n_head, N, self.attn_dim)
            .transpose(1, 2)
            .reshape(B, N, -1)
        )

        feat = self.layer_norm1(self.dropout(self.fc(feat)) + residual)
        feat = self.layer_norm2(self.mlp(feat) + feat)

        return feat


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        )


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):

        super(MLPBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.block(x)


# code from https://github.com/Curt-Park/rainbow-is-all-you-need
class NoisyLinear(nn.Module):
    """
    Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.25,
    ):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


# Code adapted from https://github.com/SilenKZYoung/CuboidAbstractionViaSeg/blob/main/network.py
class Feature_extract(nn.Module):
    def __init__(self, tet_idim, emb_dims=1024, low_dim_idx=0, k=20):
        super(Feature_extract, self).__init__()
        self.tet_idim = tet_idim
        self.emb_dims = emb_dims
        self.k = k
        self.low_dim_idx = low_dim_idx

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm1d(self.emb_dims)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.tet_idim * 2, 64, kernel_size=1, bias=False),
            self.bn1_1,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn1_2,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2_1,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2_2,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, self.emb_dims, kernel_size=1, bias=False),
            self.bn3_1,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]

        return idx

    def get_graph_feature(self, x, k=20, idx=None, dim9=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if dim9 == False:
                idx = self.knn(x, k=k)
            else:
                idx = self.knn(x[:, 6:], k=k)

        idx_base = (
            torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

    def forward(self, pc):
        # batch_size = pc.size(0)
        x = pc.transpose(2, 1)
        idx = self.knn(x, k=self.k)
        x = self.get_graph_feature(
            x, k=self.k, idx=idx if self.low_dim_idx == 1 else None
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = self.get_graph_feature(
            x1, k=self.k, idx=idx if self.low_dim_idx == 1 else None
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x_per = torch.cat((x1, x2), dim=1)  # (batch_size, 64*2, num_points)
        x_global = self.conv3(
            x_per
        )  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_global = x_global.max(dim=-1, keepdim=True)[
            0
        ]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        return x_global
