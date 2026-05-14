import argparse
from argparse import ArgumentParser

import datetime

def get_parser() -> ArgumentParser:

    """parse input arguments"""
    parser = argparse.ArgumentParser(description="Unsup3DMeshSeg-Agent")

    parser.add_argument(
        "--debug", action="store_true", default=False, help="when debuging"
    )
    parser.add_argument(
        "--data_gen_eps",
        type=float,
        default=-10000000,
        help="Data generation epsilon bound",
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="",
        choices=["train", "dataset", "eval", "greedy", "var3dseg ", ""],
        help="running type: choose from train/eval/greedy",
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="bsp",
        choices=["bsp", "bbox", "coacd", "fps", "random ", ""],
        help="choose initialization type",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="shape category",
    )
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--worker", type=int, default=8, help="number of workers to use")
    parser.add_argument(
        "--seed", type=int, default=7777, help="seeds for number pseudo random generators"
    )
    parser.add_argument(
        "--path_to_pt_file",
        type=str,
        default="",
        help="directory to checkpoint file to load agent",
    )
    parser.add_argument(
        "--load_idx",
        type=int,
        default=0,
        help="agent checkpoint load idx",
    )
    parser.add_argument(
        "--path_to_msh_file",
        type=str,
        default="",
        help="data directory",
    )
    parser.add_argument(
        "--path_to_bbox_file",
        type=str,
        default="",
        help="bbox directory",
    )
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=[],
        help="meshes to use in training",
    )
    parser.add_argument(
        "--all",
        default=False,
        action="store_true",
        help="use all meshes",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[""],
        help="meshes to exclude in training/testing",
    )
    parser.add_argument(
        "--print_off",
        default=False,
        action="store_true",
        help="mute merging process",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/",
        help="directory to save the training log (tensorboard)",
    )
    parser.add_argument(
        "--mesh_txt",
        type=str,
        default="",
        help="text file of meshes to use in training",
    )
    now = datetime.datetime.now()
    
    parser.add_argument(
        "--result_path",
        type=str,
        default="./logs/exp",
        # default=f"./logs/chanho_test/result",
        help="directory to save the training log (tensorboard)",
    )
    parser.add_argument(
        "--data_batch_size",
        type=int,
        default=1,
        help="number of data batch to use in training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch to use in training"
    )
    parser.add_argument(
        "--only_nearby",
        default=False,
        action="store_true",
        help="only allow nearby merging",
    )

    parser.add_argument("--lr", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument(
        "--lr_final", type=float, default=1e-8, help="final learning rate"
    )
    parser.add_argument(
        "--lr_fraction",
        type=float,
        default=0.4,
        help="fraction of timestep when learning rate becomes final learing rate",
    )
    parser.add_argument(
        "--learning_start",
        type=int,
        default=300,
        help="number of samples to collect before agent actually learns",
    )

    parser.add_argument("--tau", type=float, default=1.0, help="soft update coefficient")
    parser.add_argument("--gamma", type=float, default=1.0, help="the discount factor")

    parser.add_argument(
        "--train_freq",
        type=int,
        default=4,
        help="number of trajectories to collect per each update",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=500,
        help="frequency of testing the current policy",
    )

    parser.add_argument(
        "--grad_step",
        type=int,
        default=1,
        help="number of updates to perform per iteration",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=5000,
        help="update the target network every target_update_interval environment steps",
    )

    parser.add_argument(
        "--exp",
        type=float,
        default=1.0,
        help="initial value of random action probability",
    )
    parser.add_argument(
        "--exp_final",
        type=float,
        default=0.05,
        help="final value of random action probability",
    )
    parser.add_argument(
        "--exp_fraction",
        type=float,
        default=0.3,
        help="fraction of entire training period over which the exploration rate is reduced",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10,
        help="The maximum value for the gradient clipping",
    )

    parser.add_argument(
        "--final_k",
        type=int,
        default=0,
        help="Number of final partitions to make",
    )
    parser.add_argument(
        "--merge_eps",
        type=float,
        default=0,
        help="Number of final partitions to make",
    )
    parser.add_argument(
        "--tilted",
        default=False,
        action="store_true",
        help="Whether to use titled bounding box",
    )
    parser.add_argument(
        "--fast_merge",
        default=False,
        action="store_true",
        help="Whether to use fast init merging",
    )
    parser.add_argument(
        "--mov",
        default=False,
        action="store_true",
        help="Whether to use MOV metric in the reward",
    )
    parser.add_argument(
        "--mov_alpha",
        type=float,
        default=1.0,
        help="Weight of reward to give at MOV",
    )
    parser.add_argument(
        "--tov",
        default=False,
        action="store_true",
        help="Whether to use TOV metric in the reward",
    )
    parser.add_argument(
        "--tov_beta",
        type=float,
        default=1.0,
        help="Weight of reward to give at TOV",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="attn",
        help="Type of agent to use",
    )
    parser.add_argument(
        "--ddqn",
        default=False,
        action="store_true",
        help="Whether to use double DQN learning update",
    )
    parser.add_argument(
        "--duel",
        default=False,
        action="store_true",
        help="Whether to use dueling network",
    )
    parser.add_argument(
        "--noisy",
        default=False,
        action="store_true",
        help="Whether to use noisy Linear or not",
    )

    parser.add_argument(
        "--buffer_size", type=int, default=2**14, help="capacity of a buffer"
    )
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n in N-step learning",
    )
    parser.add_argument(
        "--per",
        default=False,
        action="store_true",
        help="Whether to use prioritized experience replay",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Alpha value of PER",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.6,
        help="Beta value of PER",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=1,
        help="Number of heads to use in the transformer self attention layer",
    )
    parser.add_argument(
        "--sample_part",
        type=int,
        default=128,
        help="number of points to sample at a partition",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag for experiments",
    )

    return parser
