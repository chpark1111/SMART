from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .evaluation import evaluate_config
from .pipeline.config import load_config, workspace_path
from .pipeline.stages import data_status, run_pipeline as _run_pipeline
from .pipeline.tools import diagnose_environment


def load(config: str | Path | None = "configs/demo.yaml", *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a SMART config and apply optional nested dictionary overrides."""
    cfg = load_config(config)
    if overrides:
        _deep_update_in_place(cfg, overrides)
    return cfg


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


def evaluate(
    config: str | Path | dict[str, Any] | None = "configs/demo.yaml",
    *,
    stage: str = "mcts",
    category: str | None = None,
    meshes: Iterable[str] | None = None,
    chamfer_points: int = 2048,
    output_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate SMART bbox outputs with the paper metrics."""
    cfg = _coerce_config(config, overrides=overrides)
    return evaluate_config(
        cfg,
        stage=stage,
        category_name=category,
        meshes=list(meshes) if meshes is not None else None,
        chamfer_points=chamfer_points,
        output_path=output_path,
    )


def doctor(
    config: str | Path | dict[str, Any] | None = "configs/demo.yaml",
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return runtime/tool availability checks for the configured pipeline."""
    return diagnose_environment(_coerce_config(config, overrides=overrides))


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
