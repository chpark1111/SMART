import argparse
import multiprocessing
import os
import resource
import shutil
import signal
import subprocess

import tqdm

MAX_VIRTUAL_MEMORY = 30 * 1024 * 1024 * 1024  # 30 GB


def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY))


def parse_args():

    """parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Run multiprocessing greedy merging to shapenet tetmesh with bsp presegmentation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Directory of input triangular meshes",
    )
    parser.add_argument(
        "--path_to_bbox",
        type=str,
        default="/",
        help="Path to bounding boxes",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        choices=["cubseg", "ha", "vp", ""],
        help="Directory of input triangular meshes",
    )
    parser.add_argument(
        "--mcts_iter",
        type=int,
        default=10000,
        help="number of trajectories to collect per each update",
    )
    parser.add_argument(
        "--exp_w",
        type=float,
        default=0.001,
        help="number of trajectories to collect per each update",
    )
    parser.add_argument(
        "--bbox_init",
        type=str,
        default="bbox_direct",
        choices=["bsp_preseg", "grd_merged", "bbox_direct", "random"],
        help="selecting which initialization to use for the bounding boxes",
    )
    parser.add_argument(
        "--num_worker", type=int, default=32, help="Number of workers to multiprocess"
    )
    parser.add_argument(
        "--merge_eps", type=float, default=0.0, help="Merging eps in greedy merging step"
    )
    parser.add_argument(
        "--action_unit",
        type=float,
        default=2e-2,
        help="action unit to modify bounding boxes",
    )

    parser.add_argument(
        "--mask_prun",
        default=False,
        action="store_true",
        help="whether to use mask pruning in mcts search",
    )
    parser.add_argument(
        "--grdexp",
        default=False,
        action="store_true",
        help="whether to use greedy expansion",
    )
    parser.add_argument(
        "--pns",
        default=False,
        action="store_true",
        help="whether to use prioritized node selection",
    )
    parser.add_argument(
        "--skip_rate",
        type=float,
        default=0.7,
        help="Skip rate when using prioritized node selection",
    )

    parser.add_argument(
        "--num_action_scale", type=int, default=1, help="number of action scale"
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=100,
        help="number of max action steps to perform with greedy",
    )
    parser.add_argument(
        "--cover_penalty",
        type=float,
        default=100.0,
        help="penalty to give if not covered",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="category",
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="bsp",
        choices=["bsp", "coacd", "fps", "random ", ""],
        help="choose initialization type",
    )

    args = parser.parse_args()
    return args


def func(fn):
    if args.init_type != "bsp":
        result_path = os.path.join("./logs/", "%s/mcts_all/exp" % (args.init_type))
    else:
        result_path = os.path.join("./logs/", "mcts_all/exp")

    p = subprocess.Popen(
        "python3 run.py --run_type mcts %s%s%s--skip_rate %g --exp_w %g --bbox_init %s --path_to_bbox %s --tilted --result_path %s --print_off --meshes %s --path_to_msh_file %s --action_unit %g --num_action_scale %d --max_step %d --cover_penalty %d --mcts_iter %d"
        % (
            ("--mask_prun " if args.mask_prun else ""),
            ("--grdexp " if args.grdexp else ""),
            ("--pns " if args.pns else ""),
            args.skip_rate,
            args.exp_w,
            args.bbox_init,
            args.path_to_bbox,
            result_path,
            fn[:10],
            args.data_path,
            args.action_unit,
            args.num_action_scale,
            args.max_step,
            args.cover_penalty,
            args.mcts_iter,
        ),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=limit_virtual_memory,
    )
    try:
        ret = p.wait(timeout=3600 * 10)
        return ret

    except subprocess.TimeoutExpired:
        p.terminate()
        p.kill()
        return 1


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.data_path):
        assert 0, "Data path does not exist"

    filenames = []
    split_path = "/home/chpark1111/docker/geometry2/research/Evaluation/timing"
    if os.path.exists(os.path.join(split_path, "%s.txt" % (args.category))):
        with open(os.path.join(split_path, "%s.txt" % (args.category)), "r") as f:
            nw = f.readline()
            while nw:
                filenames.append(nw.strip("\n"))
                nw = f.readline()
    else:
        filenames = os.listdir(args.data_path)

    with multiprocessing.Pool(args.num_worker) as pool:
        results = list(
            tqdm.tqdm(pool.imap_unordered(func, filenames), total=len(filenames))
        )

    print(
        "Total processed: %d, Success: %d, Failed: %d"
        % (len(filenames), results.count(0), len(filenames) - results.count(0))
    )
    try:
        os.kill(-os.getpid(), signal.SIGINT)
    except ProcessLookupError:
        print("Exited normally")
