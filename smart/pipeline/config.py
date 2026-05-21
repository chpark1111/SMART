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
    "engine": "cpp_native",
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
        "coacd_external_root": "external/CoACD",
        "coacd_bin": "external/CoACD/python/package/bin/coacd",
        "coacd_build": False,
        "coacd_install_python": True,
        "build_cpp_with_tools": True,
        "smart_cpp_native_bin": "build/smart-cpp-native",
    },
    "native_pipeline": {
        "timeout_sec": None,
    },
    "normalization": {
        "enabled": True,
        "backend": "cpp_native_executable",
        "native_executable_required": False,
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
        "input_repair": {
            "enabled": True,
            "basic_cleanup": True,
            "fix_normals": True,
            "fill_holes": False,
            "keep_largest_component": False,
            "fallback_variants": [
                {
                    "name": "fill_holes",
                    "fill_holes": True,
                    "keep_largest_component": False,
                },
                {
                    "name": "largest_component_fill_holes",
                    "enabled": False,
                    "fill_holes": True,
                    "keep_largest_component": True,
                },
            ],
        },
        "retry": {
            "enabled": True,
            "fine_retry": {
                "enabled": True,
                "epsilon_scale": 0.5,
                "edge_length_scale": 0.5,
                "coarsen": False,
                "timeout_sec": 600,
            },
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
        "backend": "auto",
        "type": "coacd",
        "timeout_sec": 1800,
        "write_partition_metadata": True,
        "partition_metadata_backend": "cpp_native",
        "partition_metadata_required": False,
        "coacd_cli_required": False,
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
        "backend": "legacy_python",
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
        "path_to_bbox": "",
        "action_unit": 0.01,
        "num_action_scale": 1,
        "max_step": 2000,
        "cover_penalty": 100,
        "score_cache_size": 4096,
        "candidate_backend": "exact",
        "candidate_top_k": 8,
        "candidate_require_exact_fallback": True,
        "candidate_pruned_categories": "",
        "candidate_pruned_max_aspect_mean": 0.0,
        "candidate_pruned_min_fill_ratio": 0.0,
        "candidate_bypass_on_exact_fallback": False,
        "manifold_volume_method": "mesh",
        "stateful_union_cache": True,
        "stateful_cache_capacity": 65536,
        "strict_legacy_bbox_params": True,
        "stateful_unscored_apply": False,
        "native_axis_rollout_step": False,
        "native_axis_rollout_segment": False,
        "render_initial": False,
        "render_partition": False,
        "summary_metrics": False,
        "candidate_trace_path": "",
        "candidate_trace_top_k": 0,
        "forced_first_action": -1,
        "forced_action_sequence": "",
        "forced_first_action_min_reward": 0.0,
        "timeout_sec": 10800,
        "worker": 0,
    },
    "mcts": {
        "backend": "auto",
        "bbox_init": "bbox_direct",
        "path_to_bbox": "",
        "action_unit": 0.02,
        "max_step": 150,
        "cover_penalty": 100,
        "mcts_iter": 3000,
        "seed": 7777,
        "exp_w": 0.001,
        "grdexp": True,
        "pns": True,
        "skip_rate": 0.9,
        "transposition_table": False,
        "transposition_table_size": 8192,
        "cpp_rng": False,
        "cpp_rng_seed": 7777,
        "allow_search_order_changes": False,
        "action_prior_path": "",
        "action_prior_device": "json",
        "action_prior_weight": 0.0,
        "puct_prior_weight": 0.0,
        "action_value_weight": 0.0,
        "action_prior_top_k": 0,
        "action_prior_select": "legacy",
        "action_prior_select_temperature": 1.0,
        "action_prior_keep_upper": True,
        "escape_policy": False,
        "escape_after_no_update": 20,
        "escape_action_top_k": 0,
        "escape_probability": 0.5,
        "candidate_trace_path": "",
        "candidate_trace_top_k": 0,
        "candidate_trace_node_top_k": 0,
        "score_cache_size": 4096,
        "candidate_backend": "exact",
        "candidate_top_k": 8,
        "candidate_require_exact_fallback": True,
        "candidate_pruned_categories": "",
        "candidate_pruned_max_aspect_mean": 0.0,
        "candidate_pruned_min_fill_ratio": 0.0,
        "candidate_bypass_on_exact_fallback": False,
        "manifold_volume_method": "mesh",
        "stateful_union_cache": True,
        "stateful_cache_capacity": 65536,
        "strict_legacy_bbox_params": True,
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
        "candidate_require_exact_fallback": True,
        "candidate_pruned_categories": "",
        "candidate_pruned_max_aspect_mean": 0.0,
        "candidate_pruned_min_fill_ratio": 0.0,
        "candidate_bypass_on_exact_fallback": False,
        "allow_search_order_changes": False,
        "action_prior_path": "",
        "action_prior_device": "json",
        "action_prior_weight": 0.0,
        "action_value_weight": 0.0,
        "action_prior_top_k": 0,
        "action_prior_keep_upper": True,
        "reward_backend": "manifold_stateful",
        "manifold_volume_method": "mesh",
        "stateful_union_cache": True,
        "stateful_cache_capacity": 65536,
        "stateful_unscored_apply": False,
        "render_initial": False,
        "render_partition": False,
        "summary_metrics": False,
        "candidate_trace_path": "",
        "candidate_trace_top_k": 0,
        "forced_first_action": -1,
        "forced_action_sequence": "",
        "forced_first_action_min_reward": 0.0,
        "timeout_sec": 10800,
        "worker": 0,
    },
    "local_refine_gate": {
        "enabled": False,
        "gate_path": "",
        "gate_threshold": 0.5,
        "input_stage": "mcts_guarded",
        "stage": "local_refine_guarded",
        "from_input_manifest": False,
        "categories": "",
        "per_category_limit": None,
        "max_step": 100,
        "action_unit": 0.005,
        "covered_tolerance": 0.0,
        "metric_tolerance": 1e-6,
        "selection_mode": "improved",
        "selection_objective": "legacy",
        "reuse_local_refine": False,
        "output": "runs/bench_exact/quality_guarded_local_refine.json",
    },
    "render": {
        "backend": "fallback",
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
    _apply_engine_defaults(cfg, loaded)
    cfg["_config_path"] = str(config_path)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def _apply_engine_defaults(cfg: dict[str, Any], loaded: dict[str, Any]) -> None:
    if cfg.get("engine") != "cpp_native":
        return
    merge_loaded = loaded.get("merge", {}) if isinstance(loaded.get("merge"), dict) else {}
    refine_loaded = loaded.get("refine", {}) if isinstance(loaded.get("refine"), dict) else {}
    mcts_loaded = loaded.get("mcts", {}) if isinstance(loaded.get("mcts"), dict) else {}
    preseg_loaded = loaded.get("preseg", {}) if isinstance(loaded.get("preseg"), dict) else {}
    normalization_loaded = loaded.get("normalization", {}) if isinstance(loaded.get("normalization"), dict) else {}
    if "backend" not in merge_loaded:
        cfg["merge"]["backend"] = "cpp_native"
    if "backend" not in refine_loaded:
        cfg["refine"]["backend"] = "cpp_native"
    if "backend" not in mcts_loaded:
        cfg["mcts"]["backend"] = "cpp_native"
    if cfg["merge"].get("backend") == "cpp_native" and "direct_file_runner_required" not in merge_loaded:
        cfg["merge"]["direct_file_runner_required"] = True
    if cfg["refine"].get("backend") == "cpp_native" and "direct_file_runner_required" not in refine_loaded:
        cfg["refine"]["direct_file_runner_required"] = True
    if cfg["mcts"].get("backend") == "cpp_native" and "direct_file_runner_required" not in mcts_loaded:
        cfg["mcts"]["direct_file_runner_required"] = True
    if cfg["normalization"].get("backend") == "cpp_native_executable" and "native_executable_required" not in normalization_loaded:
        cfg["normalization"]["native_executable_required"] = True
    if str(cfg["preseg"].get("type", "coacd")) == "coacd":
        if "backend" not in preseg_loaded:
            cfg["preseg"]["backend"] = "coacd_cli"
        if "partition_metadata_backend" not in preseg_loaded:
            cfg["preseg"]["partition_metadata_backend"] = "cpp_native"
        if "partition_metadata_required" not in preseg_loaded:
            cfg["preseg"]["partition_metadata_required"] = True
    if "reward_backend" not in refine_loaded:
        cfg["refine"]["reward_backend"] = "manifold_stateful"
    if "reward_backend" not in mcts_loaded:
        cfg["mcts"]["reward_backend"] = "manifold_stateful"


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
    workspace_value = Path(str(cfg["workspace"])).expanduser()
    if workspace_value.is_absolute():
        workspace = workspace_value
    else:
        cwd_candidate = Path.cwd() / workspace_value
        repo_candidate = REPO_ROOT / workspace_value
        workspace = repo_candidate if repo_candidate.exists() else cwd_candidate
    return workspace.joinpath(*parts)


def category_mesh_root(category: dict[str, Any]) -> Path:
    root = category.get("mesh_root")
    if root is None:
        raise KeyError(f"Category {category.get('name', '<unknown>')} lacks mesh_root")
    root_path = Path(str(root)).expanduser()
    if root_path.is_absolute():
        return root_path
    cwd_candidate = Path.cwd() / root_path
    if cwd_candidate.exists():
        return cwd_candidate
    resolved = repo_path(root)
    assert resolved is not None
    return resolved


def category_tetra_params(cfg: dict[str, Any], category: dict[str, Any]) -> tuple[float, float]:
    stage = deep_update(cfg.get("tetra", {}), category.get("tetra", {}))
    return float(stage["epsilon"]), float(stage["edge_length"])


def enabled_stages(cfg: dict[str, Any]) -> set[str]:
    stages = cfg.get("stages", {})
    return {name for name, enabled in stages.items() if enabled}
