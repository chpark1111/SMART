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
        default="/home/chpark1111/docker/geometry2/research/shapenet",
        help="Directory of input triangular meshes",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "--path_to_bbox_file",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "--data_gen_eps",
        type=float,
        default=-1000000000.0,
        help="Data generation epsilon bound",
    )
    parser.add_argument(
        "--num_worker", type=int, default=16, help="Number of workers to multiprocess"
    )
    parser.add_argument(
        "--merge_eps", type=float, default=0.0, help="Merging eps in greedy merging step"
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
        help="Whether to use fast merge",
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
        choices=["bsp", "coacd", "fps", "random", "bbox", ""],
        help="choose initialization type",
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="greedy",
        choices=["train", "dataset", "eval", "greedy", "var3dseg ", ""],
    )

    args = parser.parse_args()
    return args


def func(fn):
    tetmesh = os.path.join(os.path.join(args.data_path, fn), "tetra.msh")
    tetra_surf = os.path.join(os.path.join(args.data_path, fn), "tetra.msh__sf.obj")

    if not os.path.exists(os.path.join(args.data_path, fn)):
        return 1
    if not os.path.exists(tetmesh) or not os.path.exists(tetra_surf):
        return 1

    p = subprocess.Popen(
        "python3 run.py --data_gen_eps %f --category %s --result_path %s --path_to_bbox_file %s --run_type %s%s%s --print_off --meshes %s --path_to_msh_file %s --merge_eps %g --init_type %s"
        % (
            args.data_gen_eps,
            args.category,
            args.result_path,
            args.path_to_bbox_file,
            args.run_type,
            " --tilted" if args.tilted else "",
            " --fast_merge" if args.fast_merge else "",
            fn[:10],
            args.data_path,
            args.merge_eps,
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
    split_path = ""# "/home/chpark1111/docker/geometry2/research/Evaluation/timing"
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
