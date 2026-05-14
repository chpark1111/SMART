import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basenet import BaseNet
from .modules.module import MultiHeadSelfAttention, NoisyLinear, SelfAttention


class AttenNet(BaseNet):
    def __init__(
        self,
        tet_idim: int,
        part_idim: int,
        num_init_part: int,
        n_head: int,
        only_nearby: bool,
        duel: bool,
        noisy: bool,
        sample_part: int,
    ):
        super(AttenNet, self).__init__()

        self.tet_idim = tet_idim
        self.part_idim = part_idim - sample_part * 3
        self.num_init_part = num_init_part
        self.attn_dim = 512
        self.n_head = n_head
        self.only_nearby = only_nearby
        self.duel = duel
        self.noisy = noisy
        self.sample_part = sample_part
        self.transformer_layers = 12
        self.exit_value = nn.Parameter(torch.tensor(-1.0), requires_grad=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tet_conv1 = nn.Sequential(
            nn.Conv1d(tet_idim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.tet_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.tet_conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
        )

        self.part_conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.part_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.part_conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
        )

        self.fc1 = nn.Linear(1024 + 1024 + self.part_idim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.ln1 = nn.LayerNorm(1024, eps=1e-6)
        self.ln2 = nn.LayerNorm(512, eps=1e-6)
        self.ln3 = nn.LayerNorm(256, eps=1e-6)

        transformer = [
            MultiHeadSelfAttention(256, self.attn_dim, n_head)
            for i in range(self.transformer_layers)
        ]
        # transformer = [
        #    SelfAttention(256, self.attn_dim) for i in range(self.transformer_layers)
        # ]
        self.transformer = nn.ModuleList(transformer)

        if self.noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        if self.duel:
            self.value_layer = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                linear(256, 1),
            )
            self.advantage_layer = nn.Sequential(
                linear(256, self.num_init_part),
            )
        else:
            self.fc_last = linear(256, self.num_init_part)

        self.relu = nn.ReLU()

    def forward(
        self, obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        :param tet_x: B x N x [(x, y, z) x 4 + vol]
        :param part_x: B x N_P x [sample_pts, vol_portion]
        :param part_mask: B x N_P x 1 or B x N_P x N_P (only_nearby)

        :ret q_value: B x N_P x N_P
        """
        tet_x, part_x, part_mask = self.normalize_feature(obs)
        part_pts, part_feat = (
            part_x[:, :, : self.sample_part * 3],
            part_x[:, :, self.sample_part * 3 :],
        )

        if not self.only_nearby:
            part_mask_t = part_mask.transpose(1, 2)
            part_mask = torch.bmm(part_mask, part_mask_t)

        part_mask = torch.logical_not(part_mask)

        tet_x = tet_x.transpose(1, 2)
        tet_x = self.tet_conv1(tet_x)
        tet_x = self.tet_conv2(tet_x)
        tet_x = self.tet_conv3(tet_x)

        tet_feat = torch.max(tet_x, dim=2, keepdim=True)[0].transpose(1, 2).contiguous()
        # B, 1, 1024 = tet_feat.shape
        tet_feat = tet_feat.expand(-1, self.num_init_part, -1)

        part_pts = (
            part_pts.reshape(-1, self.sample_part * 3)
            .reshape(-1, self.sample_part, 3)
            .transpose(1, 2)
        )

        part_pts = self.part_conv1(part_pts)
        part_pts = self.part_conv2(part_pts)
        part_pts = self.part_conv3(part_pts)

        part_pts = (
            torch.max(part_pts, dim=2, keepdim=True)[0]
            .transpose(1, 2)
            .contiguous()
            .squeeze(1)
            .reshape(-1, self.num_init_part, 1024)
        )
        # B, N_P, 1024 = part_pts.shape
        part_feat = torch.concat((part_pts, part_feat), dim=-1)
        # B, N_P, 1024 + self.part_idim = part_feat.shape

        feat = torch.concat((tet_feat, part_feat), dim=-1)

        feat = feat.view(-1, 1024 + 1024 + self.part_idim)

        feat = self.relu(self.ln1(self.fc1(feat)))
        feat = self.relu(self.ln2(self.fc2(feat)))
        feat = self.relu(self.ln3(self.fc3(feat)))

        feat = feat.view(-1, self.num_init_part, 256)
        for layer in self.transformer:
            feat = layer(feat, part_mask)

        if self.duel:
            value = self.value_layer(feat)

            adv = self.advantage_layer(feat)
            adv_t = adv.transpose(1, 2)
            adv_value = (adv + adv_t) / 2

            q_value = value + adv_value - torch.mean(adv_value, dim=-1, keepdim=True)
        else:
            feat = self.fc_last(feat)
            feat_t = feat.transpose(1, 2)
            q_value = (feat + feat_t) / 2

        mask = (
            torch.eye(self.num_init_part)
            .to(self.device)
            .repeat(q_value.shape[0], 1, 1)
            .bool()
        )
        q_value[mask] = 0
        # part_mask.shape = B, N_P, N_P
        q_value[part_mask] = torch.finfo(torch.float32).min

        return q_value

    def reset_noise(self):
        if self.duel:
            # self.value_layer[0].reset_noise()
            # self.advantage_layer[0].reset_noise()
            self.value_layer[2].reset_noise()
            self.advantage_layer[0].reset_noise()
        else:
            self.fc_last.reset_noise()
