import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basenet import BaseNet


class MLPQNet(BaseNet):
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
        super(MLPQNet, self).__init__()

        # n_head is not used in mlp
        self.tet_idim = tet_idim
        self.part_idim = part_idim - sample_part * 3
        self.num_init_part = num_init_part
        self.only_nearby = only_nearby
        self.sample_part = sample_part

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
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
        )

        self.fc1 = nn.Linear(1024 + 256 + self.part_idim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)

        self.ln1 = nn.LayerNorm(1024, eps=1e-6)
        self.ln2 = nn.LayerNorm(512, eps=1e-6)
        self.ln3 = nn.LayerNorm(256, eps=1e-6)
        self.ln4 = nn.LayerNorm(64, eps=1e-6)

        self.final_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.relu = nn.ReLU()

    def forward(
        self, obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        :param tet_x: B x N x [(x, y, z) x 4 + vol + num_init_part]
        :param part_x: B x N_P x [sample_pts, vol_portion + (x, y, z) x 2]
        :param part_mask: B x N_P x 1 or B x N_P x N_P (only_nearby)

        :ret q_value: B x N_P x N_P
        """
        raise NotImplementedError("MLPNet is deprecated")
        tet_x, part_x, part_mask = self.normalize_feature(obs)
        part_pts, part_feat = (
            part_x[:, :, : self.sample_part * 3],
            part_x[:, :, self.sample_part * 3 :],
        )

        tet_x = tet_x.transpose(1, 2)
        part_pts = part_pts.reshape(-1, self.sample_part, 3).transpose(1, 2)
        # part_x-sample_pts [:args.num_sample]
        tet_x = self.tet_conv1(tet_x)
        tet_x = self.tet_conv2(tet_x)
        tet_x = self.tet_conv3(tet_x)

        tet_feat = torch.max(tet_x, dim=2, keepdim=True)[0].transpose(1, 2)
        # B, 1, 1024 = tet_feat.shape
        tet_feat = tet_feat.expand(-1, self.num_init_part, -1)

        part_pts = self.part_conv1(part_pts)
        part_pts = self.part_conv2(part_pts)
        part_pts = self.part_conv3(part_pts)

        part_pts = (
            torch.max(part_pts, dim=2, keepdim=True)[0]
            .transpose(1, 2)
            .contiguous()
            .reshape(part_x.shape[0], -1, 256)
        )
        # B, N_P, 256 = part_pts.shape

        part_feat = torch.concat((part_pts, part_feat), dim=-1)
        # B, N_P, 256 + self.part_idim = part_feat.shape

        feat = torch.concat((tet_feat, part_feat), dim=-1)

        feat = feat.view(-1, 1024 + 256 + self.part_idim)

        feat = self.relu(self.ln1(self.fc1(feat)))
        feat = self.relu(self.ln2(self.fc2(feat)))
        feat = self.relu(self.ln3(self.fc3(feat)))
        feat = self.relu(self.ln4(self.fc4(feat)))

        feat = feat.view(-1, self.num_init_part, 64)
        # B, N_P, 64

        f_feat = feat.unsqueeze(2).expand(-1, -1, self.num_init_part, -1)
        s_feat = feat.unsqueeze(1).expand(-1, self.num_init_part, -1, -1)
        feat = torch.concat((f_feat, s_feat), dim=-1)
        q_value = self.final_mlp(feat).squeeze(-1)
        # q_value.shape = B, N_P, N_P

        mask = (
            torch.eye(self.num_init_part)
            .to(self.device)
            .repeat(q_value.shape[0], 1, 1)
            .bool()
        )
        q_value[mask] = 0

        if not self.only_nearby:
            part_mask_t = part_mask.transpose(1, 2)
            part_mask = torch.bmm(part_mask, part_mask_t)

        part_mask = torch.logical_not(part_mask)
        # part_mask.shape = B, N_P, N_P
        q_value[part_mask] = torch.finfo(torch.float32).min

        return q_value
