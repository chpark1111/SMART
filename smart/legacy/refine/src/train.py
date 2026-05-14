import os

import numpy as np
import torch

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.bbox_environment import MeshBBoxEnv
from src.environment.multi_bbox_environment import MultiMeshBBoxEnv
from src.models.dqn import DQN
from src.models.policies import TetMeshPolicy


def train(args) -> None:
    dataset = STM_DataLoader(False, args)

    env = MultiMeshBBoxEnv(args, dataset)
    # args.learning_start = int(args.buffer_size * 0.4)

    dqn = DQN(
        TetMeshPolicy,
        env,
        learning_rate=args.lr,
        learning_rate_final=args.lr_final,
        learning_rate_fraction=args.lr_fraction,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_start,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.grad_step,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exp,
        exploration_final_eps=args.exp_final,
        exploration_fraction=args.exp_fraction,
        max_grad_norm=args.max_grad_norm,
        agent=args.agent,
        ddqn=args.ddqn,
        duel=args.duel,
        noisy=args.noisy,
        per=args.per,
        n_step=args.n_step,
        alpha=args.alpha,
        beta=args.beta,
        n_head=args.n_head,
        edge_conv=args.edge_conv,
        log_path=args.log_path,
        tag=args.tag,
        seed=args.seed,
        debug=args.debug,
    )
    # dqn.imit_learn(240000)
    dqn.learn(300000)
