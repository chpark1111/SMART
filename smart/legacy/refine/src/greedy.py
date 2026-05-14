import os
import time
import math

import numpy as np
import pymesh

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.bbox_environment import MeshBBoxEnv
from src.utils.utils import calculate_reward

try:
    import smart.rust as smart_rust
except ImportError:
    smart_rust = None


def _dataset_name(path):
    return os.path.basename(os.path.normpath(path))


def greedy(args):

    dataset = STM_DataLoader(False, args)

    avg_reward = []
    avg_covered = []
    avg_num_bbox = []
    avg_improved_occ = []
    avg_improved_mov = []
    avg_improved_bvs = []
    avg_improved_tov = []
    avg_improved_iou = []
    for batch in dataset:

        vertices, faces, voxels, name = batch
        env = MeshBBoxEnv(vertices, faces, voxels, args, name[0])

        env.exp_name = _dataset_name(
            args.path_to_msh_file
        ) + "_%s_%stilted%d_maxstep%d_covpen%d_acscale%d_acunit%.5g_mgeps%.5g_timing" % (
            args.bbox_init,
            ((args.baseline + "_") if args.baseline != "" else args.baseline),
            args.tilted,
            args.max_step,
            args.cover_penalty,
            args.num_action_scale,
            args.action_unit,
            args.merge_eps,
        )

        if not getattr(args, "skip_initial_render", False):
            env.render()

        st = time.time()
        skip_summary_metrics = getattr(args, "skip_summary_metrics", False)
        if skip_summary_metrics:
            init_occ = init_mov = init_bvs = init_tov = init_covered = init_iou = math.nan
        else:
            (
                init_occ,
                init_mov,
                init_bvs,
                init_tov,
                init_covered,
                init_iou,
            ) = env.current_state_summary()
        init_bbox = env.num_bbox

        backend = getattr(args, "greedy_backend", "auto")
        use_rust_greedy = (
            backend in {"auto", "rust", "rust_stateful"}
            and smart_rust is not None
            and smart_rust.using_rust()
            and hasattr(smart_rust, "run_greedy_refine_callbacks")
        )
        if use_rust_greedy:
            try:
                rewards, cnt = smart_rust.run_greedy_refine_callbacks(args, env)
            except Exception:
                if backend in {"rust", "rust_stateful"}:
                    raise
                rewards, cnt = _run_python_greedy_refine(args, env)
        else:
            if backend in {"rust", "rust_stateful"}:
                raise RuntimeError("greedy_backend=rust requested but smart._rust is unavailable")
            rewards, cnt = _run_python_greedy_refine(args, env)

        path_to_result = os.path.join(
            os.path.join(os.path.join(args.result_path, env.exp_name), "result/updated0"),
            name[0],
        )
        os.makedirs(path_to_result, exist_ok=True)
        with open(
            os.path.join(
                path_to_result,
                "time.txt",
            ),
            "w",
        ) as f:
            f.write(str(init_bbox))
            f.write("\n")
            f.write(str(env.num_valid_bboxs()))
            f.write("\n")
            f.write(str(time.time() - st))

        if skip_summary_metrics:
            occ = mov = bvs = tov = covered = iou = math.nan
            print(
                "Shape %s, Reward: %g, Number of bbox (%d -> %d), Summary metrics skipped, Elapsed time: %g"
                % (
                    name[0][:10],
                    calculate_reward(rewards, args.gamma),
                    init_bbox,
                    env.num_valid_bboxs(),
                    time.time() - st,
                )
            )
        else:
            occ, mov, bvs, tov, covered, iou = env.current_state_summary()
            print(
                "Shape %s, Reward: %g, Number of bbox (%d -> %d), Final occupancy (%g -> %g), MOV (%g -> %g), BVS (%g -> %g), TOV (%g -> %g), Covered (%g -> %g), IoU (%g -> %g), Elapsed time: %g"
                % (
                    name[0][:10],
                    calculate_reward(rewards, args.gamma),
                    init_bbox,
                    env.num_valid_bboxs(),
                    init_occ,
                    occ,
                    init_mov,
                    mov,
                    init_bvs,
                    bvs,
                    init_tov,
                    tov,
                    init_covered,
                    covered,
                    init_iou,
                    iou,
                    time.time() - st,
                )
            )
        env.render()

        avg_reward.append(calculate_reward(rewards, args.gamma))
        avg_covered.append(covered)
        avg_num_bbox.append(env.num_valid_bboxs())
        avg_improved_occ.append(occ - init_occ)
        avg_improved_mov.append(mov - init_mov)
        avg_improved_bvs.append(bvs - init_bvs)
        avg_improved_tov.append(tov - init_tov)
        avg_improved_iou.append(iou - init_iou)

    total = len(avg_covered)
    avg_reward = np.mean(avg_reward)
    avg_covered = np.mean(avg_covered)
    avg_num_bbox = np.mean(avg_num_bbox)
    avg_improved_occ = np.mean(avg_improved_occ)
    avg_improved_mov = np.mean(avg_improved_mov)
    avg_improved_bvs = np.mean(avg_improved_bvs)
    avg_improved_tov = np.mean(avg_improved_tov)
    avg_improved_iou = np.mean(avg_improved_iou)

    print(
        "Total %d meshes, avg reward %g, avgnumber of bbox %g, avgimp occupancy %g, avgimp MOV %g, avgimp BVS %g, avgimp TOV %g, avg Covered %g, avgimp IoU %g"
        % (
            total,
            avg_reward,
            avg_num_bbox,
            avg_improved_occ,
            avg_improved_mov,
            avg_improved_bvs,
            avg_improved_tov,
            avg_covered,
            avg_improved_iou,
        )
    )


def _run_python_greedy_refine(args, env):
    env.reset()
    cnt = 0
    done = 0
    rewards = []

    while not done:
        ac, rw = env.greedy_sample(True)

        if rw <= 0:
            break
        r, obs, done = env.step(ac, apply=1)
        if not args.print_off:
            print(ac, r)
        rewards.append(r)
        cnt += 1

    return rewards, cnt
