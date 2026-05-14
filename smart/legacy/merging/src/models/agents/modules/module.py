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
            mask = mask.repeat(self.n_head, 1, 1)
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
