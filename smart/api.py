from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .pipeline.config import REPO_ROOT, load_config as _load_config, workspace_path
from .pipeline.stages import data_status, run_pipeline as _run_pipeline
from .pipeline.tools import diagnose_environment


ASSET_ALIASES: dict[str, dict[str, str]] = {
    "gates": {
        "rich": "local_refine_gate_combined_manifest52_mctsguarded44_rich.json",
        "combined_rich": "local_refine_gate_combined_manifest52_mctsguarded44_rich.json",
        "combined_expanded": "local_refine_gate_combined_manifest52_mctsguarded44_expanded.json",
        "expanded": "local_refine_gate_combined_manifest52_mctsguarded44_expanded.json",
        "cat20": "local_refine_gate_mcts_guarded_cat20_covtol001.json",
        "manifest52": "local_refine_gate_manifest52.json",
    },
    "priors": {
        "rich_v2": "category_general_candidate_pg_agent_rich_v2.json",
        "candidate_pg_rich_v2": "category_general_candidate_pg_agent_rich_v2.json",
        "pg_rich_v2": "category_general_candidate_pg_agent_rich_v2.json",
        "policy_value": "category_general_policy_value_agent_prior.json",
        "pv": "category_general_policy_value_agent_prior.json",
        "offline_rl": "category_general_all_available_offline_rl_mlp_prior.json",
        "global_offline_rl": "category_general_all_available_offline_rl_mlp_prior.json",
        "category_dispatch": "category_dispatch_offline_rl_mlp_prior.json",
        "per_category": "category_dispatch_offline_rl_mlp_prior.json",
        "local_refine_value": "local_refine_policy_value_final_return_cat10.json",
        "local_refine_cat10": "local_refine_policy_value_final_return_cat10.json",
        "airplane": "airplane_offline_rl_mlp_prior.json",
        "chair": "chair_offline_rl_mlp_prior.json",
        "table": "table_offline_rl_mlp_prior.json",
    },
    "skills": {
        "macro_v1": "macro_skill_knowledge_base_v1.json",
        "macro_skill_v1": "macro_skill_knowledge_base_v1.json",
        "macro_memory_v1": "macro_memory_policy_v1.json",
        "macro_budget_quality_v1": "macro_budget_quality_rule_v1.json",
        "macro_quality_gate_ridge_v1": "macro_quality_gate_ridge_v1.json",
    },
}


def load_config(
    config: str | Path | None = "configs/demo.yaml",
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a SMART config and apply optional nested dictionary overrides."""
    cfg = _load_config(config)
    if overrides:
        _deep_update_in_place(cfg, overrides)
    return cfg


def load(config: str | Path | None = "configs/demo.yaml", *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Alias for :func:`load_config`."""

    return load_config(config, overrides=overrides)


def run_pipeline(
    config: str | Path | dict[str, Any] | None = "configs/demo.yaml",
    *,
    stage: str | None = None,
    category: str | None = None,
    meshes: Iterable[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run SMART pipeline stages and return manifest-style dictionaries.

    `stage=None` runs the enabled stages in paper order. Use `stage="tetra"`,
    `category="airplane"`, or `meshes=[...]` for narrower runs.
    """
    cfg = _coerce_config(config, overrides=overrides)
    records = _run_pipeline(
        cfg,
        only_stage=stage,
        category_name=category,
        meshes=list(meshes) if meshes is not None else None,
        dry_run=dry_run,
        force=force,
    )
    return [asdict(record) for record in records]


def run(
    config: str | Path | dict[str, Any] | None = "configs/demo.yaml",
    *,
    stage: str | None = None,
    category: str | None = None,
    meshes: Iterable[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Alias for :func:`run_pipeline`, kept for short README examples."""

    return run_pipeline(
        config,
        stage=stage,
        category=category,
        meshes=meshes,
        dry_run=dry_run,
        force=force,
        overrides=overrides,
    )


def run_native_pipeline(
    *,
    input_mesh: str | Path,
    work_dir: str | Path,
    manifoldplus_bin: str | Path,
    ftetwild_bin: str | Path,
    coacd_bin: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one mesh through the compiled `smart-cpp-native run-pipeline` path.

    Python is only the package API here; the executable performs normalization,
    Mesh2Tet orchestration, CoACD CLI splitting/partitioning, merge, refine,
    and MCTS.
    """
    from .native_runner import run_pipeline_from_files

    return run_pipeline_from_files(
        input_mesh=input_mesh,
        work_dir=work_dir,
        manifoldplus_bin=manifoldplus_bin,
        ftetwild_bin=ftetwild_bin,
        coacd_bin=coacd_bin,
        **kwargs,
    )


def run_macro_skill_controller(engine: Any, *, category: str, **kwargs: Any) -> dict[str, Any]:
    """Run the packaged exact-validated macro-skill controller on an engine.

    This is an opt-in research/production bridge.  The learned/knowledge-based
    controller proposes variable-length skills, but every accepted state update
    is still validated by the native exact SMART reward backend.
    """
    from .macro_skills import run_builtin_macro_skill_controller

    return run_builtin_macro_skill_controller(engine, category=category, **kwargs)


def run_macro_skill_controller_from_files(
    *,
    msh_path: str | Path,
    bbox_metadata_path: str | Path,
    category: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load prepared SMART tetra/bbox files and run the macro-skill controller."""
    from .macro_skills import run_builtin_macro_skill_controller_from_files

    return run_builtin_macro_skill_controller_from_files(
        msh_path=msh_path,
        bbox_metadata_path=bbox_metadata_path,
        category=category,
        **kwargs,
    )


def evaluate(
    config: str | Path | dict[str, Any] | None = "configs/demo.yaml",
    *,
    stage: str = "mcts",
    category: str | None = None,
    meshes: Iterable[str] | None = None,
    chamfer_points: int = 2048,
    output_path: str | Path | None = None,
    from_manifest: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate SMART bbox outputs with the paper metrics."""
    from .evaluation import evaluate_config

    cfg = _coerce_config(config, overrides=overrides)
    return evaluate_config(
        cfg,
        stage=stage,
        category_name=category,
        meshes=list(meshes) if meshes is not None else None,
        chamfer_points=chamfer_points,
        output_path=output_path,
        from_manifest=from_manifest,
    )


def doctor(
    config: str | Path | dict[str, Any] | None = "configs/demo.yaml",
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return runtime/tool availability checks for the configured pipeline."""
    return diagnose_environment(_coerce_config(config, overrides=overrides))


def cpp_native_available() -> bool:
    """Return true when the package-facing C++ SMART engine is importable."""
    try:
        from . import cpp

        return bool(cpp.native_core_available() and cpp.NativeSmartEngine is not None)
    except Exception:
        return False


def check_data(
    config: str | Path | dict[str, Any] | None = "configs/demo.yaml",
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return mesh counts and normalization sanity checks for configured data."""
    return data_status(_coerce_config(config, overrides=overrides))


def workspace(
    config: str | Path | dict[str, Any] | None = "configs/demo.yaml",
    *parts: str,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Resolve the configured SMART workspace path."""
    return workspace_path(_coerce_config(config, overrides=overrides), *parts)


def config_profiles() -> list[dict[str, str]]:
    """List SMART YAML config profiles visible to the active installation."""
    root_dir = REPO_ROOT / "configs"
    package_dir = Path(__file__).resolve().parent / "configs"
    names = {path.name for path in root_dir.glob("*.yaml")} | {path.name for path in package_dir.glob("*.yaml")}
    profiles: list[dict[str, str]] = []
    for name in sorted(names):
        root_path = root_dir / name
        package_path = package_dir / name
        profiles.append(
            {
                "name": name,
                "root_path": str(root_path) if root_path.exists() else "",
                "packaged_path": str(package_path) if package_path.exists() else "",
            }
        )
    return profiles


def asset_profiles(kind: str | None = None) -> list[dict[str, Any]]:
    """List optional SMART JSON model assets when present in an installation."""
    assets_dir = Path(__file__).resolve().parent / "assets"
    kinds = [kind] if kind else ["gates", "priors", "skills"]
    profiles: list[dict[str, Any]] = []
    for asset_kind in kinds:
        kind_name = _normalize_asset_kind(asset_kind)
        kind_dir = assets_dir / kind_name
        for path in sorted(kind_dir.glob("*.json")):
            profiles.append(_asset_profile(kind_name, path))
    return profiles


def asset_path(kind: str, name: str) -> Path:
    """Resolve an optional SMART model asset by kind and filename or alias."""
    kind_name = _normalize_asset_kind(kind)
    aliases = ASSET_ALIASES.get(kind_name, {})
    filename = aliases.get(name, name)
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    path = Path(__file__).resolve().parent / "assets" / kind_name / filename
    if not path.exists():
        available = ", ".join(item["name"] for item in asset_profiles(kind_name))
        raise FileNotFoundError(f"unknown SMART asset {kind_name}/{name}; available: {available}")
    return path


def _coerce_config(
    config: str | Path | dict[str, Any] | None,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(config, dict):
        cfg = copy.deepcopy(config)
    else:
        cfg = load_config(config)
    if overrides:
        _deep_update_in_place(cfg, overrides)
    return cfg


def _deep_update_in_place(target: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update_in_place(target[key], value)
        else:
            target[key] = value


def _normalize_asset_kind(kind: str) -> str:
    normalized = str(kind).strip().lower().replace("\\", "/").strip("/")
    if normalized in {"gate", "gates"}:
        return "gates"
    if normalized in {"prior", "priors"}:
        return "priors"
    if normalized in {"skill", "skills", "macro", "macros"}:
        return "skills"
    raise ValueError(f"unknown SMART asset kind: {kind!r}")


def _asset_profile(kind: str, path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    aliases = sorted(alias for alias, filename in ASSET_ALIASES.get(kind, {}).items() if filename == path.name)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "kind": kind,
        "name": path.name,
        "path": str(path),
        "aliases": aliases,
        "policy_type": str(payload.get("policy_type") or payload.get("type") or ""),
        "feature_set": str(payload.get("feature_set") or metadata.get("feature_set") or ""),
        "model_type": str(metadata.get("model_type") or payload.get("model_type") or ""),
    }
