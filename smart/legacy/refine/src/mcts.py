import os
import pdb
import random
import time

import numpy as np
import pymesh

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.bbox_environment import MeshBBoxEnv
from src.models.tree_search import MCTSTreeSearch
from src.utils.utils import calculate_reward

try:
    import smart.rust as smart_rust
except ImportError:
    smart_rust = None


def _dataset_name(path):
    return os.path.basename(os.path.normpath(path))


def _bbox_output_mtime(path):
    latest = os.path.getmtime(path)
    for filename in os.listdir(path):
        child = os.path.join(path, filename)
        try:
            latest = max(latest, os.path.getmtime(child))
        except FileNotFoundError:
            pass
    return latest


def _has_bbox_output(path_to_result, mesh_name, since=None):
    result_root = os.path.join(path_to_result, "result")
    if not os.path.isdir(result_root):
        return False
    for update_name in os.listdir(result_root):
        if not update_name.startswith("updated"):
            continue
        mesh_root = os.path.join(result_root, update_name, mesh_name)
        if not os.path.isdir(mesh_root):
            continue
        for bbox_name in os.listdir(mesh_root):
            bbox_root = os.path.join(mesh_root, bbox_name)
            if bbox_name.startswith("bboxs") and os.path.isdir(bbox_root):
                if since is not None and _bbox_output_mtime(bbox_root) < since:
                    continue
                for filename in os.listdir(bbox_root):
                    if filename.endswith(".obj") and filename.startswith("bbox"):
                        return True
    return False


def mcts(args):

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = STM_DataLoader(False, args)

    for batch in dataset:

        vertices, faces, voxels, name = batch
        env = MeshBBoxEnv(vertices, faces, voxels, args, name[0])

        env.exp_name = _dataset_name(
            args.path_to_msh_file
        ) + "_timing_mcts%d_expW%g%s%s%s%s_%s_tilted%d_maxstep%d_covpen%d_acunit%.5ggamma%g/%s" % (
            args.mcts_iter,
            args.exp_w,
            ("_prun" if args.mask_prun else ""),
            ("_grdexp" if args.grdexp else ""),
            ("_pns%g" % (args.skip_rate) if args.pns else ""),
            ("_tt" if args.transposition_table else ""),
            args.bbox_init,
            args.tilted,
            args.max_step,
            args.cover_penalty,
            args.action_unit,
            args.gamma,
            env.name,
        )
        exp_tag = str(getattr(args, "mcts_exp_tag", "") or "").strip()
        if exp_tag:
            env.exp_name = env.exp_name.rstrip("/") + "_%s" % exp_tag

        path_to_result = os.path.join(args.result_path, env.exp_name)
        os.makedirs(path_to_result, exist_ok=True)
        rust_stats_path = os.path.join(path_to_result, "rust_stats.json")
        try:
            os.remove(rust_stats_path)
        except FileNotFoundError:
            pass

        if not getattr(args, "skip_initial_render", False):
            env.render()

        st = time.time()
        backend = getattr(args, "mcts_backend", "auto")
        reward_backend = str(getattr(args, "reward_backend", "manifold"))
        rust_stats = None
        # The current Rust MCTS runner still calls back into the Python env for
        # exact Manifold rewards. On larger exact-Manifold sweeps that callback
        # path is slower than the legacy Python tree, so keep it opt-in until
        # the rollout state is moved deeper into Rust.
        auto_rust_mcts = backend == "auto" and reward_backend not in {
            "manifold",
            "manifold_stateful",
            "manifold_bridge",
        }
        use_rust_mcts = (
            (backend in {"rust", "rust_stateful"} or auto_rust_mcts)
            and smart_rust is not None
            and smart_rust.using_rust()
            and hasattr(smart_rust, "run_mcts_callbacks")
        )
        if use_rust_mcts:
            try:
                rust_stats = smart_rust.run_mcts_callbacks(args, env, args.mcts_iter)
                if (
                    getattr(args, "transposition_table", False)
                    or getattr(args, "action_prior_weight", 0.0)
                ) and not getattr(args, "print_off", False):
                    print("MCTS rust stats: %s" % rust_stats)
            except Exception:
                if backend in {"rust", "rust_stateful"}:
                    raise
                tree = MCTSTreeSearch(args, env)
                tree.run_mcts(args.mcts_iter)
        else:
            if backend in {"rust", "rust_stateful"}:
                raise RuntimeError("mcts_backend=rust requested but smart._rust is unavailable")
            tree = MCTSTreeSearch(args, env)
            tree.run_mcts(args.mcts_iter)

        if not _has_bbox_output(path_to_result, env.name, since=st):
            env.reset()
            env.render(num_update=0)
        if rust_stats is None and reward_backend == "manifold_stateful":
            rust_stats = {
                "mcts_runner": "python_tree",
                "mcts_backend": str(backend),
                "reward_backend": reward_backend,
                "iterations_requested": float(args.mcts_iter),
                "stateful_union_cache": float(
                    bool(getattr(args, "stateful_union_cache", True))
                ),
            }
        if rust_stats is not None:
            try:
                for key, value in env._manifold_stateful_cache_stats().items():
                    rust_stats["manifold_state_%s" % key] = float(value)
            except Exception:
                pass
            try:
                for key, value in env._candidate_prefilter_report().items():
                    rust_stats["candidate_prefilter_%s" % key] = float(value)
            except Exception:
                pass
            with open(rust_stats_path, "w") as f:
                import json

                json.dump(rust_stats, f, indent=2, sort_keys=True)
        with open(
            os.path.join(
                path_to_result,
                "time.txt",
            ),
            "w",
        ) as f:
            f.write(str(time.time() - st))
