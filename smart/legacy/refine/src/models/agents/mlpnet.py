import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basenet import BaseNet
from .modules.module import NoisyLinear


class MLPQNet(BaseNet):
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
        super(MLPQNet, self).__init__()

        # n_head is not used in mlp
        self.tet_idim = tet_idim
        self.bbox_idim = bbox_idim
        self.max_num_bbox = max_num_bbox
        self.max_num_actions = max_num_actions
        self.max_step = max_step

        self.duel = duel
        self.noisy = noisy

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tet_conv1 = nn.Sequential(
            nn.Linear(tet_idim, 64),
            nn.ReLU(),
        )
        self.tet_conv2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.tet_conv3 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
        )

        self.box_fc1 = nn.Linear(max_num_bbox * self.bbox_idim, 256)
        self.box_fc2 = nn.Linear(256, 512)

        self.step_fc = nn.Linear(max_step, 256)

        self.fc1 = nn.Linear(1024 + 512 + 256, 1024 + 512 + 256)
        self.fc2 = nn.Linear(1024 + 512 + 256, 2048)

        if self.noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear

        if self.duel:
            self.value_fc = linear(2048, 1)
            self.adv_fc = linear(2048, max_num_actions)
        else:
            self.action_fc = linear(2048, max_num_actions)

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

        :ret q_value: num_actions
        """

        tet_x, bbox_x, step_x, action_mask = self.normalize_feature(obs)
        action_mask = action_mask.bool()

        tet_x = self.tet_conv1(tet_x)
        tet_x = self.tet_conv2(tet_x)
        tet_x = self.tet_conv3(tet_x)

        tet_feat = torch.max(tet_x, dim=1)[0]
        # B, 1024 = tet_feat.shape

        bbox_x = bbox_x.reshape(bbox_x.shape[0], -1)

        bbox_x = self.relu(self.box_fc1(bbox_x))
        bbox_feat = self.relu(self.box_fc2(bbox_x))

        step_feat = self.relu(self.step_fc(step_x))

        feat = torch.concat((tet_feat, bbox_feat, step_feat), dim=-1)
        # feat = torch.concat((tet_feat, bbox_feat), dim=-1)

        feat = self.relu(self.fc1(feat))
        feat = self.relu(self.fc2(feat))

        if self.duel:
            value = self.value_fc(feat)
            adv_value = self.adv_fc(feat)

            q_value = value + adv_value - torch.mean(adv_value, dim=-1, keepdim=True)
        else:
            q_value = self.action_fc(feat)

        q_value[action_mask] = torch.finfo(torch.float32).min
        print(q_value.shape)
        print("q_value", q_value)
        print("max_value", torch.max(q_value, dim=-1)[0])
        print("max_action", torch.max(q_value, dim=-1)[1])
        return q_value

    def reset_noise(self):
        if self.duel:
            self.value_fc.reset_noise()
            self.adv_fc.reset_noise()
        else:
            self.action_fc.reset_noise()
