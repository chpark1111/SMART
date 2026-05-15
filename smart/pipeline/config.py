from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "demo",
    "workspace": "runs/demo",
    "data_root": "data",
    "python": None,
    "gpu": "0",
    "categories": [],
    "stages": {
        "normalize": True,
        "tetra": True,
        "preseg": True,
        "merge": True,
        "refine": True,
        "mcts": True,
        "local_refine": False,
        "render": True,
    },
    "tools": {
        "manifoldplus_bin": "external/mesh2tet/ManifoldPlus/build/manifold",
        "ftetwild_bin": "external/mesh2tet/fTetWild/build/FloatTetwild_bin",
        "blender_bin": None,
        "mesh2tet_external_root": "external/mesh2tet",
        "rust_target": None,
    },
    "normalization": {
        "enabled": True,
        "mode": "bbox_diagonal",
        "target": 1.0,
        "center": "bbox",
        "source_filename": "model.obj",
    },
    "tetra": {
        "num_worker": 1,
        "manifold_timeout_sec": 1200,
        "ftetwild_timeout_sec": 3600,
        "ftetwild_threads": 8,
        "ftetwild_level": 2,
        "validate": True,
        "require_single_component": False,
        "min_tetra_count": 20,
        "min_surface_faces": 20,
        "retry": {
            "enabled": True,
            "epsilon_scale": 2.0,
            "edge_length_scale": 2.0,
            "coarsen": True,
            "extra_attempts": [
                {
                    "name": "coarse_retry",
                    "epsilon_scale": 4.0,
                    "edge_length_scale": 3.0,
                    "coarsen": True,
                    "timeout_sec": 300,
                },
                {
                    "name": "robust_wn_retry",
                    "epsilon_scale": 4.0,
                    "edge_length_scale": 3.0,
                    "coarsen": True,
                    "use_floodfill": False,
                    "use_general_wn": True,
                    "manifold_surface": False,
                    "timeout_sec": 300,
                }
            ],
        },
    },
    "preseg": {
        "type": "coacd",
        "timeout_sec": 1800,
        "coacd": {
            "threshold": 0.05,
            "max_convex_hull": 64,
            "preprocess_mode": "auto",
            "preprocess_resolution": 50,
            "resolution": 2000,
            "mcts_nodes": 20,
            "mcts_iterations": 150,
            "mcts_max_depth": 3,
            "pca": False,
            "merge": True,
            "decimate": True,
            "seed": 7777,
        },
    },
    "merge": {
        "init_type": "coacd",
        "tilted": True,
        "merge_eps": 0.02,
        "fast_merge": True,
        "only_nearby": False,
        "final_k": 0,
        "data_gen_eps": -1000000000.0,
        "timeout_sec": 10800,
        "worker": 0,
    },
    "refine": {
        "backend": "auto",
        "bbox_init": "grd_merged",
        "action_unit": 0.01,
        "num_action_scale": 1,
        "max_step": 2000,
        "cover_penalty": 100,
        "score_cache_size": 4096,
        "candidate_backend": "exact",
        "candidate_top_k": 8,
        "manifold_volume_method": "mesh",
        "stateful_union_cache": True,
        "stateful_cache_capacity": 65536,
        "stateful_unscored_apply": False,
        "render_initial": False,
        "render_partition": False,
        "summary_metrics": False,
        "timeout_sec": 10800,
        "worker": 0,
    },
    "mcts": {
        "backend": "auto",
        "bbox_init": "bbox_direct",
        "action_unit": 0.02,
        "max_step": 150,
        "cover_penalty": 100,
        "mcts_iter": 3000,
        "exp_w": 0.001,
        "grdexp": True,
        "pns": True,
        "skip_rate": 0.9,
        "transposition_table": False,
        "transposition_table_size": 8192,
        "allow_search_order_changes": False,
        "action_prior_path": "",
        "action_prior_weight": 0.0,
        "puct_prior_weight": 0.0,
        "action_value_weight": 0.0,
        "candidate_trace_path": "",
        "candidate_trace_top_k": 0,
        "score_cache_size": 4096,
        "candidate_backend": "exact",
        "candidate_top_k": 8,
        "manifold_volume_method": "mesh",
        "stateful_union_cache": True,
        "stateful_cache_capacity": 65536,
        "stateful_unscored_apply": False,
        "render_initial": False,
        "render_partition": False,
        "summary_metrics": False,
        "timeout_sec": 21600,
        "accept_timeout_output": True,
        "worker": 0,
    },
    "local_refine": {
        "input_stage": "mcts",
        "backend": "auto",
        "bbox_init": "bbox_direct",
        "action_unit": 0.005,
        "num_action_scale": 1,
        "max_step": 300,
        "cover_penalty": 100,
        "score_cache_size": 8192,
        "candidate_backend": "exact",
        "candidate_top_k": 8,
        "reward_backend": "manifold_stateful",
        "manifold_volume_method": "mesh",
        "stateful_union_cache": True,
        "stateful_cache_capacity": 65536,
        "stateful_unscored_apply": False,
        "render_initial": False,
        "render_partition": False,
        "summary_metrics": False,
        "timeout_sec": 10800,
        "worker": 0,
    },
    "render": {
        "backend": "blender",
        "input_stage": "mcts",
        "joint_mesh": False,
        "fallback": True,
        "num_worker": 1,
        "timeout_sec": 10800,
        "gpu": "0",
        "transparent": True,
        "camera": {
            "airplane": {
                "ortho_scale": 1.12,
                "shift_x": 0.10,
                "shift_y": -0.38,
                "rotation": [-0.25, 0.58, -0.08],
            },
            "chair": {
                "ortho_scale": 1.20,
                "shift_x": 0.10,
                "shift_y": -0.28,
            },
            "table": {
                "ortho_scale": 1.55,
                "shift_x": 0.02,
                "shift_y": -0.25,
            },
        },
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        cwd_path = Path.cwd() / config_path
        repo_config_path = REPO_ROOT / config_path
        package_config_path = Path(__file__).resolve().parents[1] / "configs" / config_path.name
        if cwd_path.exists():
            config_path = cwd_path
        elif repo_config_path.exists():
            config_path = repo_config_path
        else:
            config_path = package_config_path
    text = config_path.read_text(encoding="utf-8")
    loaded = _load_mapping(text, config_path)
    cfg = deep_update(DEFAULT_CONFIG, loaded)
    cfg["_config_path"] = str(config_path)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def _load_mapping(text: str, path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            return json.loads(text)
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return loaded


def repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def workspace_path(cfg: dict[str, Any], *parts: str) -> Path:
    workspace = repo_path(cfg["workspace"])
    assert workspace is not None
    return workspace.joinpath(*parts)


def category_mesh_root(category: dict[str, Any]) -> Path:
    root = category.get("mesh_root")
    if root is None:
        raise KeyError(f"Category {category.get('name', '<unknown>')} lacks mesh_root")
    resolved = repo_path(root)
    assert resolved is not None
    return resolved


def category_tetra_params(cfg: dict[str, Any], category: dict[str, Any]) -> tuple[float, float]:
    stage = deep_update(cfg.get("tetra", {}), category.get("tetra", {}))
    return float(stage["epsilon"]), float(stage["edge_length"])


def enabled_stages(cfg: dict[str, Any]) -> set[str]:
    stages = cfg.get("stages", {})
    return {name for name, enabled in stages.items() if enabled}
