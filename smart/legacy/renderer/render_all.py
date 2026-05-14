import argparse
import multiprocessing
import os
import shutil
import signal
import subprocess

import tqdm


def parse_args():

    """parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Run multiprocessing rendering to target bounding boxes"
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="",
        help="Directory of input object files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.environ.get("SMART_RENDER_OUTPUT", "runs/render"),
        help="Directory of result rendering files will be stored",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="",
        choices=["merge", "refine", "cubseg", "mesh", "ha", "vp", "mcts", "smart"],
        help="data input type",
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="Number of gpu to use for rendering"
    )
    parser.add_argument(
        "--num_worker", type=int, default=8, help="Number of workers to multiprocess"
    )
    parser.add_argument(
        "--joint_mesh",
        default=False,
        action="store_true",
        help="whether to jointly render the meshes",
    )

    args = parser.parse_args()
    return args


def func(fn):
    if args.input_type == "merge":
        input_path = os.path.join(os.path.join(args.data_path, fn), "bboxs")
    elif args.input_type == "refine":
        bbox_update_path = os.path.join(args.data_path, fn)
        steps = [
            (0 if file.endswith(".txt") else int(file[11:]))
            for file in os.listdir(bbox_update_path)
        ]
        steps.sort()
        best_step = steps[-1]

        input_path = os.path.join(bbox_update_path, "bboxs_steps%d" % (best_step))
    elif args.input_type == "mcts":
        bbox_update_path = os.path.join(os.path.join(args.data_path, fn), "result")

        updates = [int(file[7:]) for file in os.listdir(bbox_update_path)]
        updates.sort()

        assert len(updates), "rl does not have any results"

        best_update = updates[-1]

        bbox_update_path = os.path.join(
            os.path.join(bbox_update_path, "updated%d" % (best_update)), fn
        )

        steps = [int(file[11:]) for file in os.listdir(bbox_update_path)]
        steps.sort()
        best_step = steps[-1]

        input_path = os.path.join(bbox_update_path, "bboxs_steps%d" % (best_step))
    elif args.input_type == "cubseg":
        res_name = "cube_masked"
        input_path = os.path.join(os.path.join(args.data_path, fn), res_name)
    elif args.input_type == "ha":
        input_path = os.path.join(args.data_path, fn)
    elif args.input_type == "mesh":
        input_path = os.path.join(args.data_path, fn)
    elif args.input_type == "smart":
        input_path = os.path.join(args.data_path, fn)
    else:
        return 1

    if os.path.exists(
        os.path.join(
            os.path.join(os.path.join(args.save_path, args.category), fn),
            ("jointmesh_" if args.joint_mesh else "") + args.exp_name + ".png",
        )
    ):
        return 0

    if args.input_type == "mesh":
        p = subprocess.Popen(
            "CUDA_VISIBLE_DEVICES=%s blender --background boxes.blend --python blender_mesh_teaser.py -- %s %s"
            % (
                args.gpu,
                input_path,
                os.path.join(
                    os.path.join(os.path.join(args.save_path, args.category), fn),
                    args.exp_name + ".png",
                ),
            ),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        p = subprocess.Popen(
            "CUDA_VISIBLE_DEVICES=%s blender --background boxes.blend --python render_teaser.py -- %s %s %d %s %d"
            % (
                args.gpu,
                input_path,
                os.path.join(
                    os.path.join(os.path.join(args.save_path, args.category), fn),
                    ("jointmesh_" if args.joint_mesh else "") + args.exp_name + ".png",
                ),
                int(args.joint_mesh),
                fn,
                1 if args.input_type == "ha" else 0,
            ),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        return p.wait(timeout=3600 * 3)

    except subprocess.TimeoutExpired:
        p.terminate()
        p.kill()
        return 1


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.exp_path):
        assert 0, "Data path does not exist"

    args.exp_name = str(args.exp_path).split("/")[-1]
    categories = [
        "airplane",
        "chair",
        "table",
        "real0.004",
        "real0.02",
        "real0.08",
        "riffle",
        "cabinet",
        "couch",
        "lamp",
        "display",
        "bench",
        "omni3d",
        "objaverse",
    ]
    args.category = None
    for name in categories:
        if args.input_type in [
            "merge",
        ]:
            if name in str(args.exp_path).split("_"):
                args.category = name
        elif args.input_type in ["refine", "mcts", "mesh"]:
            if name in str(args.exp_name).split("_"):
                args.category = name
        else:
            if name in str(args.exp_path).split("/"):
                args.category = name

    assert args.category is not None, "Category not selected"

    if args.input_type == "merge":
        args.data_path = os.path.join(args.exp_path, "result")
    elif args.input_type == "refine":
        args.result_path = os.path.join(args.exp_path, "result")

        updates = [int(file[7:]) for file in os.listdir(args.result_path)]
        updates.sort()

        assert len(updates), "rl does not have any results"

        best_update = updates[-1]

        args.data_path = os.path.join(args.result_path, "updated%d" % (best_update))
    elif args.input_type == "mcts":
        args.data_path = args.exp_path
    elif args.input_type == "cubseg":
        args.exp_name = "CuboidAbstractionViaSeg"

        args.data_path = os.path.join(args.exp_path, "infer")
    elif args.input_type == "ha":
        args.exp_name = "cuboid_abstraction"

        args.data_path = os.path.join(args.exp_path, "infer")
    elif args.input_type == "mesh":
        args.exp_name = "tetmesh"
        args.data_path = args.exp_path
    else:
        assert 0, "Invalid input type selected"

    filenames = []
    split_path = os.environ.get("SMART_RENDER_SPLIT_PATH", "")
    if os.path.exists(os.path.join(split_path, "%s.txt" % (args.category))):
        with open(os.path.join(split_path, "%s.txt" % (args.category)), "r") as f:
            nw = f.readline()
            while nw:
                filenames.append(nw.strip("\n"))
                nw = f.readline()
    else:
        filenames = os.listdir(args.data_path)

    with multiprocessing.Pool(processes=args.num_worker) as pool:
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
