import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basenet import BaseNet
from .modules.module import Feature_extract, MultiHeadSelfAttention, NoisyLinear


class AttenNet(BaseNet):
    def __init__(
        self,
        tet_idim: int,
        bbox_idim: int,
        max_num_bbox: int,
        max_num_actions: int,
        max_step: int,
        n_head: int,
        duel: bool,
        noisy: bool,
        edge_conv: bool,
    ):
        super(AttenNet, self).__init__()

        self.tet_idim = tet_idim
        self.bbox_idim = bbox_idim
        self.max_num_bbox = max_num_bbox
        self.max_num_actions = max_num_actions
        self.max_step = max_step

        self.attn_dim = 512
        self.n_head = n_head
        self.duel = duel
        self.noisy = noisy
        self.edge_conv = edge_conv
        self.transformer_layers = 12

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.edge_conv:
            self.encoder = Feature_extract(self.tet_idim, emb_dims=256)
        else:
            self.tet_conv1 = nn.Sequential(
                nn.Linear(tet_idim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
            )
            self.tet_conv2 = nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
            self.tet_conv3 = nn.Sequential(
                nn.Linear(128, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
            )

        self.bbox_conv1 = nn.Sequential(
            nn.Linear(bbox_idim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.bbox_conv2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.bbox_conv3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        self.step_fc = nn.Linear(max_step, 256)

        self.fc1 = nn.Linear(1024 + 256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)

        transformer = [
            MultiHeadSelfAttention(256, self.attn_dim, n_head)
            for i in range(self.transformer_layers)
        ]
        self.transformer = nn.ModuleList(transformer)

        if self.noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        if self.duel:
            self.value_layer = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                linear(128, 1),
            )
            self.advantage_layer = nn.Sequential(
                linear(256, max_num_actions // max_num_bbox),
            )
        else:
            self.fc_last = linear(256, max_num_actions // max_num_bbox)

        # if self.duel:
        #     self.value_layer[0].weight.data.normal_(0, 0.1)
        #     self.value_layer[2].weight.data.normal_(0, 0.1)

        #     self.advantage_layer[0].weight.data.normal_(0, 0.1)
        # else:
        #     self.fc_last.weight.data.normal_(0, 0.1)

        self.relu = nn.ReLU()

    def forward(
        self, obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        N = self.max_points
        N_B = self.max_bboxs

        Per tetrahedral info: N x [(x, y, z) x 4, volume] (4 points of tetrahedral mesh)
        Per bounding box info: N_B x [x_min, y_min, z_min, x_max, y_max, z_max]
        Step vector: max_steps
        Action mask: num_actions

        :ret q_value: B x num_actions
        """
        tet_x, bbox_x, step_x, action_mask = self.normalize_feature(obs)
        action_mask = action_mask.bool()

        mask_idxs = self.max_num_bbox - torch.sum(action_mask, dim=-1) / (
            self.max_num_actions // self.max_num_bbox
        )
        bbox_mask = (
            torch.arange(self.max_num_bbox)
            .unsqueeze(0)
            .expand(tet_x.shape[0], self.max_num_bbox)
            .to(mask_idxs.device)
        )
        bbox_mask = bbox_mask >= mask_idxs[:, None]

        if self.edge_conv:
            tet_feat = self.encoder(tet_x)
            tet_feat = tet_feat.transpose(1, 2)

        else:
            tet_x = self.tet_conv1(tet_x)
            tet_x = self.tet_conv2(tet_x)
            tet_x = self.tet_conv3(tet_x)

            tet_feat = torch.max(tet_x, dim=1, keepdim=True)[0]
        # B, 1, 1024 = tet_feat.shape
        tet_feat = tet_feat.expand(-1, self.max_num_bbox, -1)

        bbox_x = self.bbox_conv1(bbox_x)
        bbox_x = self.bbox_conv2(bbox_x)
        bbox_feat = self.bbox_conv3(bbox_x)

        step_feat = (
            self.relu(self.step_fc(step_x)).unsqueeze(1).expand(-1, self.max_num_bbox, -1)
        )

        # feat = torch.concat((tet_feat, bbox_feat, step_feat), dim=-1).view(
        #     -1, 1024 + 256 + 256
        # )
        feat = torch.concat((tet_feat, bbox_feat), dim=-1).view(-1, 1024 + 256)

        feat = self.relu(self.ln1(self.fc1(feat)))
        feat = self.relu(self.ln2(self.fc2(feat)))
        feat = self.relu(self.ln3(self.fc3(feat)))

        feat = feat.view(-1, self.max_num_bbox, 256)
        for layer in self.transformer:
            feat = layer(feat, bbox_mask)

        if self.duel:
            value = self.value_layer(feat)
            adv_value = self.advantage_layer(feat)

            q_value = value + adv_value - torch.mean(adv_value, dim=-1, keepdim=True)
            q_value = q_value.view(q_value.shape[0], -1)
        else:
            q_value = self.fc_last(feat)

            q_value = q_value.view(q_value.shape[0], -1)

        q_value[action_mask] = torch.finfo(torch.float32).min

        # print("q_value", q_value.shape)
        # print("max_value", torch.max(q_value, dim=-1)[0])
        # print("max_action", torch.max(q_value, dim=-1)[1])
        return q_value

    def reset_noise(self):
        if self.duel:
            self.value_layer[2].reset_noise()
            self.advantage_layer[0].reset_noise()
        else:
            self.fc_last.reset_noise()
