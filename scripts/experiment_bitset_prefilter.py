from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_REFINE = REPO_ROOT / "smart" / "legacy" / "refine"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LEGACY_REFINE))

os.environ.setdefault(
    "SMART_MANIFOLD_PYTHON",
    str(REPO_ROOT / "smart" / "vendor" / "manifold" / "build" / "bindings" / "python"),
)
sys.path.insert(0, os.environ["SMART_MANIFOLD_PYTHON"])

from smart.pipeline.config import load_config
from smart.pipeline.stages import latest_exp_dir_for_bbox, stage_root, tetra_root
from smart import rust as smart_rust
from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.bbox_environment import MeshBBoxEnv
from configs.args import get_parser as get_legacy_parser


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a voxel-centroid/bitset-style coverage proxy against exact "
            "SMART Manifold action rewards."
        )
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument("--category", default="airplane")
    parser.add_argument("--mesh", default="1f5537f4747ec847622c69c3abc6f80")
    parser.add_argument("--stage", choices=["refine", "mcts"], default="mcts")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--output", default="runs/bench_exact/bitset_prefilter_probe.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    category = _find_category(cfg, args.category)
    env = _build_env(cfg, category, args.mesh, args.stage)

    started_proxy = time.perf_counter()
    proxy_records = _score_proxy_actions(env)
    proxy_time = time.perf_counter() - started_proxy

    started_exact = time.perf_counter()
    exact_records = _score_exact_actions(env)
    exact_time = time.perf_counter() - started_exact

    result = _summarize(env, proxy_records, exact_records, args.top_k)
    result.update(
        {
            "config": args.config,
            "category": args.category,
            "mesh": args.mesh,
            "stage": args.stage,
            "num_actions": len(exact_records),
            "proxy_time_sec": proxy_time,
            "exact_time_sec": exact_time,
            "proxy_speedup_vs_exact_scoring": (
                exact_time / proxy_time if proxy_time > 0 else None
            ),
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _find_category(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for category in cfg.get("categories", []):
        if category.get("name") == name:
            return category
    raise ValueError(f"category not found in config: {name}")


def _build_env(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    stage: str,
) -> MeshBBoxEnv:
    refine_cfg = cfg.get("refine", {})
    mcts_cfg = cfg.get("mcts", {})
    merge_cfg = cfg.get("merge", {})
    stage_cfg = refine_cfg if stage == "refine" else mcts_cfg
    bbox_init = "grd_merged" if stage == "refine" else "bbox_direct"
    path_to_bbox = ""
    if stage == "mcts":
        refine_exp = latest_exp_dir_for_bbox(stage_root(cfg, "refine", category), mesh_id)
        if refine_exp is None:
            raise RuntimeError("mcts bitset probe requires existing refine bbox output")
        path_to_bbox = str(refine_exp)

    legacy_args = get_legacy_parser().parse_args(
        [
            "--run_type",
            "greedy" if stage == "refine" else "mcts",
            "--result_path",
            str(stage_root(cfg, stage, category)),
            "--path_to_msh_file",
            str(tetra_root(cfg, category)),
            "--path_to_bbox",
            path_to_bbox,
            "--bbox_init",
            bbox_init,
            "--init_type",
            str(merge_cfg.get("init_type", "coacd")),
            "--merge_eps",
            str(merge_cfg.get("merge_eps", 0.02)),
            "--max_step",
            str(stage_cfg.get("max_step", 150 if stage == "mcts" else 2000)),
            "--cover_penalty",
            str(stage_cfg.get("cover_penalty", 100)),
            "--score_cache_size",
            str(stage_cfg.get("score_cache_size", 4096)),
            "--reward_backend",
            "manifold",
            "--action_unit",
            str(stage_cfg.get("action_unit", 0.02 if stage == "mcts" else 0.01)),
            "--num_action_scale",
            str(stage_cfg.get("num_action_scale", 1)),
            "--worker",
            "0",
            "--data_batch_size",
            "1",
            "--print_off",
            "--meshes",
            mesh_id[:10],
            "--skip_initial_render",
            "--skip_render_partition",
            "--skip_summary_metrics",
        ]
    )
    if merge_cfg.get("tilted", True):
        legacy_args.tilted = True
    if merge_cfg.get("fast_merge", True):
        legacy_args.fast_merge = True

    dataset = STM_DataLoader(False, legacy_args)
    for batch in dataset:
        vertices, faces, voxels, name = batch
        return MeshBBoxEnv(vertices, faces, voxels, legacy_args, name[0])
    raise RuntimeError("selected mesh was not loaded")


def _score_exact_actions(env: MeshBBoxEnv) -> list[dict[str, float | int]]:
    records = []
    for action in _axis_actions(env):
        reward = float(env.step(action, apply=0))
        records.append({"action": int(action), "exact_reward": reward})
    return records


def _score_proxy_actions(env: MeshBBoxEnv) -> list[dict[str, float | int]]:
    bounds = [list(bbox.box) for bbox in env.bbox_list]
    rotations = [np.asarray(bbox.rot, dtype=float).reshape(3, 3) for bbox in env.bbox_list]
    if hasattr(smart_rust, "centroid_proxy_axis_rewards"):
        return [
            {"action": int(action), "proxy_reward": float(reward)}
            for action, reward in smart_rust.centroid_proxy_axis_rewards(
                env.centroid.tolist(),
                np.asarray(env.volume, dtype=float).tolist(),
                bounds,
                [rotation.reshape(-1).tolist() for rotation in rotations],
                int(env.num_action_scale),
                float(env.action_unit),
                float(env.volume_sum),
                float(env.last_bbox_score),
                float(env.args.cover_penalty),
                float(env.pen_rate),
            )
        ]

    base_masks = [
        _centroid_mask(env.centroid, bounds[idx], rotations[idx])
        if env.bbox_list[idx].valid_bbox()
        else np.zeros(len(env.centroid), dtype=bool)
        for idx in range(env.num_bbox)
    ]
    volumes = np.asarray(env.volume, dtype=float)
    base_total_volume = sum(_box_volume(box) for box in bounds if _box_valid(box))
    base_box_volumes = [_box_volume(box) if _box_valid(box) else 0.0 for box in bounds]

    records = []
    for action in _axis_actions(env):
        bbox_idx, coord_idx, scale_idx = env._decode_action(action)
        candidate = list(bounds[bbox_idx])
        candidate[coord_idx] += env.action_scale[scale_idx] * env.action_unit
        if not _box_valid(candidate):
            proxy_reward = -float("inf")
        else:
            candidate_mask = _centroid_mask(env.centroid, candidate, rotations[bbox_idx])
            union = candidate_mask.copy()
            for idx, mask in enumerate(base_masks):
                if idx != bbox_idx:
                    union |= mask
            covered = float(volumes[union].sum() / env.volume_sum)
            new_total_volume = (
                base_total_volume - base_box_volumes[bbox_idx] + _box_volume(candidate)
            )
            bvs = new_total_volume / env.volume_sum
            proxy_score = -abs(bvs - 1.0) - (1.0 - covered) * env.pen_rate * env.args.cover_penalty
            proxy_reward = proxy_score - env.last_bbox_score
        records.append({"action": int(action), "proxy_reward": float(proxy_reward)})
    return records


def _axis_actions(env: MeshBBoxEnv):
    for bbox_idx in range(env.num_bbox):
        for local in range(6 * env.num_action_scale):
            yield bbox_idx * env._actions_per_bbox + local


def _centroid_mask(centroids: np.ndarray, box: list[float], rotation: np.ndarray) -> np.ndarray:
    pts = centroids @ rotation.T
    return (
        (pts[:, 0] >= box[0])
        & (pts[:, 0] <= box[3])
        & (pts[:, 1] >= box[1])
        & (pts[:, 1] <= box[4])
        & (pts[:, 2] >= box[2])
        & (pts[:, 2] <= box[5])
    )


def _summarize(
    env: MeshBBoxEnv,
    proxy_records: list[dict[str, float | int]],
    exact_records: list[dict[str, float | int]],
    top_k: list[int],
) -> dict[str, Any]:
    proxy_by_action = {int(row["action"]): float(row["proxy_reward"]) for row in proxy_records}
    exact_by_action = {int(row["action"]): float(row["exact_reward"]) for row in exact_records}
    exact_sorted = sorted(exact_by_action, key=lambda action: exact_by_action[action], reverse=True)
    proxy_sorted = sorted(proxy_by_action, key=lambda action: proxy_by_action[action], reverse=True)
    best_exact_action = exact_sorted[0]
    best_proxy_action = proxy_sorted[0]
    exact_values = np.asarray([exact_by_action[action] for action in exact_sorted], dtype=float)
    proxy_values_for_exact_order = np.asarray(
        [proxy_by_action[action] for action in exact_sorted], dtype=float
    )
    finite = np.isfinite(proxy_values_for_exact_order) & np.isfinite(exact_values)
    pearson = None
    if finite.sum() >= 2:
        pearson = float(np.corrcoef(exact_values[finite], proxy_values_for_exact_order[finite])[0, 1])

    topk = {}
    for k in top_k:
        selected = proxy_sorted[: min(int(k), len(proxy_sorted))]
        best_in_selected = max(selected, key=lambda action: exact_by_action[action])
        topk[str(k)] = {
            "contains_exact_best": best_exact_action in selected,
            "best_exact_reward_in_proxy_topk": exact_by_action[best_in_selected],
            "reward_gap_vs_global_exact": exact_by_action[best_exact_action]
            - exact_by_action[best_in_selected],
        }

    return {
        "num_bbox": env.num_bbox,
        "last_bbox_score": float(env.last_bbox_score),
        "best_exact_action": best_exact_action,
        "best_exact_reward": exact_by_action[best_exact_action],
        "best_proxy_action": best_proxy_action,
        "best_proxy_action_exact_reward": exact_by_action[best_proxy_action],
        "best_proxy_reward": proxy_by_action[best_proxy_action],
        "exact_best_proxy_rank": proxy_sorted.index(best_exact_action) + 1,
        "proxy_best_exact_rank": exact_sorted.index(best_proxy_action) + 1,
        "pearson_exact_vs_proxy": pearson,
        "topk": topk,
    }


def _box_valid(box: list[float]) -> bool:
    return box[0] < box[3] and box[1] < box[4] and box[2] < box[5]


def _box_volume(box: list[float]) -> float:
    if not _box_valid(box):
        return 0.0
    return float((box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2]))


if __name__ == "__main__":
    raise SystemExit(main())
