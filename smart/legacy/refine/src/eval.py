import argparse
import json
import os
import time

import numpy as np
import torch

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.multi_bbox_environment import MultiMeshBBoxEnv
from src.models.dqn import DQN
from src.models.policies import TetMeshPolicy


def eval(parser, exp_path, pt_id, debug, final_k=0):
    with open(
        os.path.join(os.path.join(exp_path, "hyperparameter"), "args.json"), "r"
    ) as json_file:
        config = json.load(json_file)

    parser.set_defaults(**config)
    args = parser.parse_args()

    args.debug = debug
    args.final_k = final_k

    dataset = STM_DataLoader(False, args)
    env = MultiMeshBBoxEnv(args, dataset)

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
        eval_freq=args.eval_freq,
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
        log_path=args.log_path,
        only_nearby=args.only_nearby,
        tag=args.tag,
        seed=args.seed,
        debug=args.debug,
    )

    dqn.load_params(os.path.join(exp_path, "checkpoint"), pt_id)
    print("loaded %d checkpoint" % (pt_id))

    if debug:
        dqn.debug_test()
    else:
        dqn.eval()
