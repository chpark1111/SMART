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
        "--bbox_init",
        type=str,
        default="grd_merged",
        choices=["bsp_preseg", "grd_merged", "bbox_direct", "random"],
        help="selecting which initialization to use for the bounding boxes",
    )
    parser.add_argument(
        "--path_to_bbox",
        type=str,
        default="",
        help="Path to bounding boxes",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        choices=["cubseg", "ha", "vp", "mcts", ""],
        help="Directory of input triangular meshes",
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
        default=1e-2,
        help="action unit to modify bounding boxes",
    )
    parser.add_argument(
        "--num_action_scale", type=int, default=1, help="number of action scale"
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=2000,
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
    tetmesh = os.path.join(os.path.join(args.data_path, fn), "tetra.msh")
    tetra_surf = os.path.join(os.path.join(args.data_path, fn), "tetra.msh__sf.obj")
    if args.baseline == "" and args.bbox_init == "grd_merged":
        grd_text = os.path.join(
            os.path.join(args.data_path, fn),
            "greedy_segment0_%smgeps%g.txt"
            % ("coacd_" if args.init_type == "coacd" else "", args.merge_eps),
        )
    if args.init_type != "bsp":
        result_path = os.path.join("./logs/", "%s/post_greedy" % (args.init_type))
    else:
        result_path = os.path.join("./logs/", "greedy")

    if not os.path.exists(os.path.join(args.data_path, fn)):
        return 1
    if (
        not os.path.exists(tetmesh)
        or not os.path.exists(tetra_surf)
        or (
            args.baseline == ""
            and (args.bbox_init == "grd_merged" and not os.path.exists(grd_text))
        )
    ):
        return 1

    if args.baseline == "":
        p = subprocess.Popen(
            "python3 run.py --run_type greedy --tilted --path_to_bbox %s --result_path %s --print_off --bbox_init %s --meshes %s --path_to_msh_file %s --merge_eps %g --action_unit %g --num_action_scale %d --max_step %d --cover_penalty %d --init_type %s"
            % (
                args.path_to_bbox,
                result_path,
                args.bbox_init,
                fn[:10],
                args.data_path,
                args.merge_eps,
                args.action_unit,
                args.num_action_scale,
                args.max_step,
                args.cover_penalty,
                args.init_type,
            ),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=limit_virtual_memory,
        )
    else:
        p = subprocess.Popen(
            "python3 run.py --run_type greedy --bbox_init bbox_direct --path_to_bbox %s --baseline %s --tilted --result_path %s --print_off --meshes %s --path_to_msh_file %s --action_unit %g --num_action_scale %d --max_step %d --cover_penalty %d --init_type %s"
            % (
                args.path_to_bbox,
                args.baseline,
                result_path,
                fn[:10],
                args.data_path,
                args.action_unit,
                args.num_action_scale,
                args.max_step,
                args.cover_penalty,
                args.init_type,
            ),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=limit_virtual_memory,
        )
    try:
        ret = p.wait(timeout=3600 * 3)
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
