import os
import pdb
import random
import time

import numpy as np
import smart.pymesh_compat as pymesh
from smart.action_prior import load_action_prior

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.bbox_environment import MeshBBoxEnv
from src.models.tree_search import MCTSTreeSearch
from src.utils.utils import calculate_reward

try:
    import smart.native as smart_native
except ImportError:
    smart_native = None


NATIVE_BACKENDS = {"cpp", "cpp_stateful", "cpp_native", "native", "native_stateful"}


def _native_action_prior_logits(args, env):
    path = str(getattr(args, "action_prior_path", "") or "")
    if not path:
        return []
    try:
        prior = load_action_prior(
            path,
            inference_device=str(getattr(args, "action_prior_device", "json") or "json"),
        )
        context = (
            env.action_prior_context()
            if hasattr(env, "action_prior_context")
            else {}
        )
        context = dict(context)
        context.update(
            {
                "mcts_iter": 0,
                "mcts_not_updated": 0,
                "mcts_best_reward": 0.0,
                "mcts_escape_active": False,
                "mcts_escape_policy": bool(getattr(args, "escape_policy", False)),
                "mcts_action_prior_top_k": int(getattr(args, "action_prior_top_k", 0) or 0),
            }
        )
        num_actions = int(env.num_bbox) * (6 * int(env.num_action_scale) + 1)
        return [
            float(value)
            for value in prior.action_logits_for(
                range(num_actions),
                num_action_scale=int(env.num_action_scale),
                context=context,
            )
        ]
    except Exception:
        return []


def _native_action_value_logits(args, env):
    path = str(getattr(args, "action_prior_path", "") or "")
    if not path:
        return []
    try:
        prior = load_action_prior(
            path,
            inference_device=str(getattr(args, "action_prior_device", "json") or "json"),
        )
        if not hasattr(prior, "action_values_for"):
            return []
        context = (
            env.action_prior_context()
            if hasattr(env, "action_prior_context")
            else {}
        )
        context = dict(context)
        context.update(
            {
                "mcts_iter": 0,
                "mcts_not_updated": 0,
                "mcts_best_reward": 0.0,
                "mcts_escape_active": False,
                "mcts_escape_policy": bool(getattr(args, "escape_policy", False)),
                "mcts_action_prior_top_k": int(getattr(args, "action_prior_top_k", 0) or 0),
            }
        )
        num_actions = int(env.num_bbox) * (6 * int(env.num_action_scale) + 1)
        return [
            float(value)
            for value in prior.action_values_for(
                range(num_actions),
                num_action_scale=int(env.num_action_scale),
                context=context,
            )
        ]
    except Exception:
        return []


def _native_action_prior_value_logits(args, env, *, need_prior, need_value):
    if not need_prior and not need_value:
        return [], []
    path = str(getattr(args, "action_prior_path", "") or "")
    if not path:
        return [], []
    try:
        prior = load_action_prior(
            path,
            inference_device=str(getattr(args, "action_prior_device", "json") or "json"),
        )
        context = (
            env.action_prior_context()
            if hasattr(env, "action_prior_context")
            else {}
        )
        context = dict(context)
        context.update(
            {
                "mcts_iter": 0,
                "mcts_not_updated": 0,
                "mcts_best_reward": 0.0,
                "mcts_escape_active": False,
                "mcts_escape_policy": bool(getattr(args, "escape_policy", False)),
                "mcts_action_prior_top_k": int(getattr(args, "action_prior_top_k", 0) or 0),
            }
        )
        num_actions = int(env.num_bbox) * (6 * int(env.num_action_scale) + 1)
        if hasattr(prior, "action_logits_values_for"):
            logits, values = prior.action_logits_values_for(
                range(num_actions),
                num_action_scale=int(env.num_action_scale),
                context=context,
            )
        else:
            logits = prior.action_logits_for(
                range(num_actions),
                num_action_scale=int(env.num_action_scale),
                context=context,
            )
            values = (
                prior.action_values_for(
                    range(num_actions),
                    num_action_scale=int(env.num_action_scale),
                    context=context,
                )
                if hasattr(prior, "action_values_for")
                else []
            )
        return (
            [float(value) for value in logits] if need_prior else [],
            [float(value) for value in values] if need_value else [],
        )
    except Exception:
        return [], []


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
        native_stats_path = os.path.join(path_to_result, "native_stats.json")
        try:
            os.remove(native_stats_path)
        except FileNotFoundError:
            pass

        if not getattr(args, "skip_initial_render", False):
            env.render()

        st = time.time()
        backend = getattr(args, "mcts_backend", "auto")
        reward_backend = str(getattr(args, "reward_backend", "manifold"))
        puct_prior_weight = float(getattr(args, "puct_prior_weight", 0.0) or 0.0)
        action_value_weight = float(getattr(args, "action_value_weight", 0.0) or 0.0)
        action_prior_weight = float(getattr(args, "action_prior_weight", 0.0) or 0.0)
        action_prior_top_k = int(getattr(args, "action_prior_top_k", 0) or 0)
        native_stats = None
        native_action_prior_logits = []
        native_action_value_logits = []
        native_action_prior_logits, native_action_value_logits = _native_action_prior_value_logits(
            args,
            env,
            need_prior=action_prior_weight != 0.0 or puct_prior_weight != 0.0,
            need_value=action_value_weight != 0.0,
        )
        action_prior_select = str(getattr(args, "action_prior_select", "legacy") or "legacy").lower()
        escape_policy = bool(getattr(args, "escape_policy", False))
        del escape_policy
        uses_static_prior = (
            action_prior_weight != 0.0
            or puct_prior_weight != 0.0
            or action_value_weight != 0.0
        )
        has_required_prior = (
            action_prior_weight == 0.0
            and puct_prior_weight == 0.0
        ) or bool(native_action_prior_logits)
        has_required_value = action_value_weight == 0.0 or bool(native_action_value_logits)
        native_supports_prior = (
            not uses_static_prior
            or (
                has_required_prior
                and has_required_value
                and action_prior_select in {"legacy", "best", "softmax"}
            )
        )
        if backend == "cpp_native":
            engine = env.make_native_smart_engine()
            native_prior_weight = (
                action_prior_weight if action_prior_weight != 0.0 else puct_prior_weight
            )
            result = engine.run_mcts(
                int(args.mcts_iter),
                int(args.max_step),
                float(args.cover_penalty),
                float(env.pen_rate),
                float(args.exp_w),
                float(args.gamma),
                int(getattr(args, "seed", 7777)),
                native_action_prior_logits,
                native_action_value_logits,
                float(native_prior_weight),
                float(action_value_weight),
                bool(getattr(args, "transposition_table", False)),
                int(getattr(args, "transposition_table_size", 8192)),
            )
            env.sync_native_smart_engine(engine, len(result["actions"]))
            if (
                getattr(args, "skip_render_partition", False)
                and hasattr(engine, "export_bbox_dir")
            ):
                env.export_native_engine_bbox_dir(engine, num_update=0)
            else:
                env.render(num_update=0)
            native_stats = dict(engine.stats())
            native_stats.update(
                {
                    "backend": "cpp_native",
                    "iterations_run": int(result["iterations_run"]),
                    "node_count": int(result["node_count"]),
                    "best_reward": float(result["best_reward"]),
                    "actions": list(result["actions"]),
                    "rewards": list(result["rewards"]),
                }
            )
        else:
            # The native MCTS callback runner is supplied by smart._cpp. It still
            # calls the Python env for unsupported learned
            # prior/value paths, so those stay on the Python tree runner.
            auto_native_mcts = backend == "auto" and reward_backend not in {
                "manifold",
                "manifold_stateful",
                "manifold_bridge",
            }
            use_native_mcts = (
                (backend in NATIVE_BACKENDS or auto_native_mcts)
                and smart_native is not None
                and getattr(smart_native, "native_core_available", lambda: False)()
                and hasattr(smart_native, "run_mcts_callbacks")
                and native_supports_prior
            )
            if use_native_mcts:
                try:
                    native_stats = smart_native.run_mcts_callbacks(
                        args,
                        env,
                        args.mcts_iter,
                        native_action_prior_logits,
                        native_action_value_logits,
                    )
                    if (
                        getattr(args, "transposition_table", False)
                        or getattr(args, "action_prior_weight", 0.0)
                    ) and not getattr(args, "print_off", False):
                        print("MCTS native stats: %s" % native_stats)
                except Exception:
                    if backend in NATIVE_BACKENDS:
                        raise
                    tree = MCTSTreeSearch(args, env)
                    tree.run_mcts(args.mcts_iter)
            else:
                if (
                    backend in NATIVE_BACKENDS
                    and puct_prior_weight == 0.0
                    and action_prior_weight == 0.0
                    and action_value_weight == 0.0
                    and action_prior_top_k <= 0
                ):
                    raise RuntimeError(
                        "mcts_backend=%s requested but no native MCTS callback runner is available"
                        % backend
                    )
                if backend in NATIVE_BACKENDS and not getattr(args, "print_off", False):
                    print("MCTS learned prior fell back to python_tree; native callback runner supports static prior/value/top-k/PUCT/escape when logits are available")
                tree = MCTSTreeSearch(args, env)
                tree.run_mcts(args.mcts_iter)

        if not _has_bbox_output(path_to_result, env.name, since=st):
            env.reset()
            env.render(num_update=0)
        if native_stats is None and reward_backend == "manifold_stateful":
            native_stats = {
                "mcts_runner": "python_tree",
                "mcts_backend": str(backend),
                "reward_backend": reward_backend,
                "iterations_requested": float(args.mcts_iter),
                "stateful_union_cache": float(
                    bool(getattr(args, "stateful_union_cache", True))
                ),
            }
        if native_stats is not None:
            try:
                for key, value in env._manifold_stateful_cache_stats().items():
                    native_stats["manifold_state_%s" % key] = float(value)
            except Exception:
                pass
            try:
                for key, value in env._candidate_prefilter_report().items():
                    native_stats["candidate_prefilter_%s" % key] = float(value)
            except Exception:
                pass
            with open(native_stats_path, "w") as f:
                import json

                json.dump(native_stats, f, indent=2, sort_keys=True)
        with open(
            os.path.join(
                path_to_result,
                "time.txt",
            ),
            "w",
        ) as f:
            f.write(str(time.time() - st))
