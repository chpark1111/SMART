from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .agents.attnnet import AttenNet
from .agents.mlpnet import MLPQNet


class TetMeshPolicy(nn.Module):
    def __init__(
        self,
        tet_idim: int,
        bbox_idim: int,
        max_num_bbox: int,
        max_num_actions: int,
        max_step: int,
        learning_rate: float,
        agent: str,
        n_head: int,
        duel: bool,
        noisy: bool,
        edge_conv: bool,
    ):
        super(TetMeshPolicy, self).__init__()

        self.max_num_bbox = max_num_bbox
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.agent = agent

        if agent == "attn":
            net = AttenNet
        elif agent == "mlp":
            net = MLPQNet
        else:
            assert 0, "Invalid agent"

        self.q_net = nn.DataParallel(
            net(
                tet_idim,
                bbox_idim,
                max_num_bbox,
                max_num_actions,
                max_step,
                n_head,
                duel,
                noisy,
                edge_conv,
            ).to(self.device),
            output_device=0,
        )
        self.q_net_target = nn.DataParallel(
            net(
                tet_idim,
                bbox_idim,
                max_num_bbox,
                max_num_actions,
                max_step,
                n_head,
                duel,
                noisy,
                edge_conv,
            ).to(self.device),
            output_device=0,
        )
        self.hard_update()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), learning_rate, weight_decay=0.01
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def predict(self, observation: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):

        obs = self.obs2ten(observation)
        with torch.no_grad():
            q_values = self.q_net(obs).squeeze()

            actions = int(q_values.argmax())

        return actions

    def set_training_mode(self, train: bool):
        if train:
            self.q_net.train()
            self.q_net_target.train()
        else:
            self.q_net.eval()
            self.q_net_target.eval()

    def hard_update(self):
        for target, source in zip(
            self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target.data.copy_(source.data)

    def obs2ten(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Numpy to torch tensor
        """
        return (
            torch.as_tensor(obs[0]).float().to(self.device).unsqueeze(0),
            torch.as_tensor(obs[1]).float().to(self.device).unsqueeze(0),
            torch.as_tensor(obs[2]).float().to(self.device).unsqueeze(0),
            torch.as_tensor(obs[3]).float().to(self.device).unsqueeze(0),
        )
