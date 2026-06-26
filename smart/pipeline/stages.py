from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .config import (
    REPO_ROOT,
    category_mesh_root,
    category_tetra_params,
    deep_update,
    enabled_stages,
    repo_path,
    workspace_path,
)
from .manifest import ManifestWriter, StageRecord
from .runner import find_executable, run_command
from ..native_executable import native_executable_path, run_native_command

try:
    import smart.native as smart_native
except ImportError:  # pragma: no cover - source-tree fallback
    smart_native = None


STAGE_ORDER = [
    "normalize",
    "tetra",
    "preseg",
    "merge",
    "refine",
    "mcts",
    "macro_skill",
    "local_refine",
    "render",
]

_ACTION_PRIOR_DEVICE_CACHE: dict[str, str] = {}
NATIVE_SEARCH_BACKENDS = {
    "cpp",
    "cpp_stateful",
    "cpp_native",
    "native",
    "native_stateful",
}
NATIVE_SEARCH_BACKEND_LABEL = "cpp/cpp_stateful/cpp_native/native/native_stateful"

_NATIVE_PREPROCESSING_CACHE_REQUIRED = (
    "normalized/model.obj",
    "tetra/model_manifold.obj",
    "tetra/tetra.msh",
    "tetra/tetra.msh__sf.obj",
    "tetra/coacd_partitions.json",
)
_NATIVE_PREPROCESSING_CACHE_OPTIONAL_FILES = (
    "tetra/log.txt",
    "tetra/log.txt_.csv",
    "tetra/tetra.msh__cutting.stl",
    "tetra/tetra.msh__simplify.off",
    "native_pipeline_stats.json",
)
_NATIVE_PREPROCESSING_CACHE_OPTIONAL_DIRS = ("coacd",)


def _manifest_mode(cfg: dict[str, Any]) -> str:
    manifest_cfg = cfg.get("manifest", {})
    if isinstance(manifest_cfg, dict):
        return str(manifest_cfg.get("mode", "latest")).lower()
    return str(manifest_cfg or "latest").lower()


def _resolve_action_prior_device(device: Any) -> str:
    requested = str(device or "json")
    if requested.lower() != "auto":
        return requested
    if requested in _ACTION_PRIOR_DEVICE_CACHE:
        return _ACTION_PRIOR_DEVICE_CACHE[requested]
    resolved = "json"
    try:
        import torch  # type: ignore

        if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
            resolved = "mps"
        elif torch.cuda.is_available():
            resolved = "cuda"
    except Exception:
        resolved = "json"
    _ACTION_PRIOR_DEVICE_CACHE[requested] = resolved
    return resolved


def _command_failure_summary(tool: str, result: Any) -> str:
    if getattr(result, "timed_out", False):
        return f"{tool} timed out after {result.elapsed_sec:.1f}s"
    if result.returncode < 0:
        signal = -int(result.returncode)
        signal_names = {9: "SIGKILL", 11: "SIGSEGV", 15: "SIGTERM"}
        name = signal_names.get(signal, f"signal {signal}")
        if signal == 11:
            return f"{tool} crashed with {name}; likely invalid or degenerate mesh input"
        if signal in {9, 15}:
            return f"{tool} was killed by {name}; likely external stop or timeout wrapper"
        return f"{tool} stopped by {name}"
    if result.returncode == 124:
        return f"{tool} timed out or was killed by timeout wrapper rc=124"
    if result.returncode == 127:
        return f"{tool} executable not found rc=127"
    return f"{tool} failed rc={result.returncode}"


def _command_failure_class(result: Any) -> str:
    if getattr(result, "timed_out", False) or result.returncode == 124:
        return "command_timeout"
    if result.returncode < 0:
        signal = -int(result.returncode)
        if signal == 11:
            return "command_crash"
        return "command_killed"
    return "command_failure"


def _validation_failure_class(error: str) -> str:
    lowered = error.lower()
    if "not watertight" in lowered:
        return "validation_open_surface"
    if "tetra element count below minimum" in lowered:
        return "validation_low_tetra_count"
    if "surface face count below minimum" in lowered:
        return "validation_low_surface_faces"
    if "multiple connected components" in lowered:
        return "validation_disconnected"
    return "validation_failure"


def run_pipeline(
    cfg: dict[str, Any],
    *,
    only_stage: str | None = None,
    category_name: str | None = None,
    meshes: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> list[StageRecord]:
    workspace = workspace_path(cfg)
    workspace.mkdir(parents=True, exist_ok=True)
    writer = ManifestWriter(workspace / "manifests")
    records: list[StageRecord] = []
    selected_stages = [only_stage] if only_stage else STAGE_ORDER
    enabled = enabled_stages(cfg)

    for category in cfg.get("categories", []):
        if category_name and category["name"] != category_name:
            continue
        mesh_ids = list_mesh_ids(category, explicit=meshes)
        for stage in selected_stages:
            if stage not in enabled and only_stage is None:
                continue
            for mesh_id in mesh_ids:
                record = run_stage(
                    cfg,
                    category,
                    mesh_id,
                    stage=stage,
                    dry_run=dry_run,
                    force=force,
                )
                writer.append(record)
                records.append(record)

    if _manifest_mode(cfg) in {"append", "cumulative"}:
        writer.write_summary_from_files()
    else:
        writer.write_stage_records(records)
        writer.write_summary(records)
    return records


def run_native_pipelines(
    cfg: dict[str, Any],
    *,
    category_name: str | None = None,
    meshes: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> list[StageRecord]:
    """Run the full one-mesh SMART path through smart-cpp-native.

    This keeps Python at the package/batch layer: each mesh is handed to the
    C++ executable, which drives normalization, Mesh2Tet orchestration,
    CoACD splitting/partitioning, merge, refine, and MCTS.
    """

    workspace = workspace_path(cfg)
    workspace.mkdir(parents=True, exist_ok=True)
    writer = ManifestWriter(workspace / "manifests")
    records: list[StageRecord] = []

    for category in cfg.get("categories", []):
        if category_name and category["name"] != category_name:
            continue
        for mesh_id in list_mesh_ids(category, explicit=meshes):
            record = run_native_pipeline_mesh(
                cfg,
                category,
                mesh_id,
                dry_run=dry_run,
                force=force,
            )
            writer.append(record)
            records.append(record)

    if _manifest_mode(cfg) in {"append", "cumulative"}:
        writer.write_summary_from_files()
    else:
        writer.write_stage_records(records)
        writer.write_summary(records)
    return records


def run_native_pipeline_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage = "native_pipeline"
    source_mesh = category_mesh_root(category) / mesh_id / "model.obj"
    work_dir = workspace_path(cfg, "native_pipeline", category["name"], mesh_id)
    expected = work_dir / "mcts_bboxs_steps0" / "bbox0.obj"
    if expected.exists() and not force:
        return _base_record(
            cfg,
            category,
            mesh_id,
            stage,
            started,
            status="skipped",
            output_path=work_dir / "mcts_bboxs_steps0",
            metadata={"reason": "existing_output", "backend": "smart-cpp-native"},
        )
    if not source_mesh.exists():
        return _base_record(
            cfg,
            category,
            mesh_id,
            stage,
            started,
            status="skipped",
            output_path=work_dir,
            error=f"missing source mesh: {source_mesh}",
        )

    tetra_cfg = deep_update(cfg.get("tetra", {}), category.get("tetra", {}))
    preseg_cfg = deep_update(cfg.get("preseg", {}), category.get("preseg", {}))
    merge_cfg = deep_update(cfg.get("merge", {}), category.get("merge", {}))
    refine_cfg = deep_update(cfg.get("refine", {}), category.get("refine", {}))
    mcts_cfg = deep_update(cfg.get("mcts", {}), category.get("mcts", {}))
    norm_cfg = cfg.get("normalization", {})
    coacd_cfg = dict(preseg_cfg.get("coacd", {}))

    manifold_bin = _mesh2tet_tool(
        cfg,
        "manifoldplus_bin",
        "SMART_MANIFOLDPLUS_BIN",
        "manifold",
    )
    ftetwild_bin = _mesh2tet_tool(
        cfg,
        "ftetwild_bin",
        "SMART_FTETWILD_BIN",
        "FloatTetwild_bin",
    )
    coacd_bin = _resolve_coacd_cli(cfg, preseg_cfg)
    skip_manifoldplus = bool(tetra_cfg.get("skip_manifoldplus", False))
    missing = [
        name
        for name, value in [
            ("ManifoldPlus", manifold_bin if not skip_manifoldplus else "skipped"),
            ("fTetWild", ftetwild_bin),
            ("CoACD", coacd_bin),
        ]
        if not value
    ]
    if missing and not dry_run:
        return _base_record(
            cfg,
            category,
            mesh_id,
            stage,
            started,
            status="blocked",
            output_path=work_dir,
            error="Missing native pipeline tool(s): " + ", ".join(missing),
            metadata={"backend": "smart-cpp-native"},
        )

    try:
        from smart import native_runner
    except Exception as exc:  # noqa: BLE001
        return _base_record(
            cfg,
            category,
            mesh_id,
            stage,
            started,
            status="blocked",
            output_path=work_dir,
            error=f"failed to import smart.native_runner: {exc}",
        )

    native_bin = native_executable_path(cfg.get("tools", {}).get("smart_cpp_native_bin"))
    if native_bin is None and not dry_run:
        return _base_record(
            cfg,
            category,
            mesh_id,
            stage,
            started,
            status="blocked",
            output_path=work_dir,
            error="smart-cpp-native executable is not available; run `smart build-cpp`",
            metadata={"backend": "smart-cpp-native"},
        )

    kwargs = _native_pipeline_kwargs(
        cfg,
        source_mesh=source_mesh,
        work_dir=work_dir,
        manifoldplus_bin=manifold_bin or "$SMART_MANIFOLDPLUS_BIN",
        ftetwild_bin=ftetwild_bin or "$SMART_FTETWILD_BIN",
        coacd_bin=coacd_bin or "$SMART_COACD_BIN",
        mesh_id=mesh_id,
        tetra_cfg=tetra_cfg,
        preseg_cfg=preseg_cfg,
        coacd_cfg=coacd_cfg,
        merge_cfg=merge_cfg,
        refine_cfg=refine_cfg,
        mcts_cfg=mcts_cfg,
        norm_cfg=norm_cfg,
    )
    manifold_depth_attempts = _native_manifold_depth_attempts(
        str(category["name"]), tetra_cfg
    )
    if manifold_depth_attempts:
        kwargs["manifold_depth"] = manifold_depth_attempts[0]
    preprocessing_cache = (
        {"enabled": False, "hit": False, "reason": "disabled_for_manifold_depth_attempts"}
        if manifold_depth_attempts
        else _native_preprocessing_cache_prepare(
        cfg,
        source_mesh=source_mesh,
        work_dir=work_dir,
        kwargs=kwargs,
        manifoldplus_bin=manifold_bin or "$SMART_MANIFOLDPLUS_BIN",
        ftetwild_bin=ftetwild_bin or "$SMART_FTETWILD_BIN",
        coacd_bin=coacd_bin or "$SMART_COACD_BIN",
        dry_run=dry_run,
        )
    )
    if preprocessing_cache.get("hit"):
        kwargs["reuse_preprocessing"] = True
    args = native_runner.pipeline_args_from_files(**kwargs)
    command = [str(native_bin or "smart-cpp-native"), *args]
    if dry_run:
        metadata: dict[str, Any] = {"backend": "smart-cpp-native", "combined": True}
        if manifold_depth_attempts:
            metadata["manifold_depth_attempts"] = manifold_depth_attempts
        if preprocessing_cache.get("enabled"):
            metadata["preprocessing_cache"] = preprocessing_cache
        return _base_record(
            cfg,
            category,
            mesh_id,
            stage,
            started,
            status="dry_run",
            output_path=work_dir,
            command=command,
            metadata=metadata,
        )

    timeout = float(cfg.get("native_pipeline", {}).get("timeout_sec", 0.0) or 0.0) or None
    attempts_metadata: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    last_error: str | None = None
    attempt_depths = manifold_depth_attempts or [int(kwargs.get("manifold_depth", 0) or 0)]
    for attempt_index, depth in enumerate(attempt_depths):
        attempt_kwargs = dict(kwargs)
        attempt_kwargs["manifold_depth"] = int(depth)
        if attempt_index > 0 or manifold_depth_attempts:
            _clear_native_pipeline_attempt_outputs(work_dir)
        attempt_started = time.time()
        try:
            result = native_runner.run_pipeline_from_files(
                **attempt_kwargs,
                timeout=timeout,
            )
            attempts_metadata.append(
                {
                    "manifold_depth": int(depth),
                    "status": str(result.get("status", "success")),
                    "elapsed_sec_wall": time.time() - attempt_started,
                }
            )
            kwargs = attempt_kwargs
            command = list(result.get("command") or command)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            attempts_metadata.append(
                {
                    "manifold_depth": int(depth),
                    "status": "failed",
                    "elapsed_sec_wall": time.time() - attempt_started,
                    "error": last_error,
                }
            )
            result = None

    if result is None:
        return _base_record(
            cfg,
            category,
            mesh_id,
            stage,
            started,
            status="failed",
            output_path=work_dir,
            command=command,
            error=last_error or "native pipeline failed",
            metadata={
                "backend": "smart-cpp-native",
                "combined": True,
                "manifold_depth_attempts": attempts_metadata,
            },
        )

    metadata = dict(result.get("metadata") or {})
    if attempts_metadata:
        metadata["manifold_depth_attempts"] = attempts_metadata
    if preprocessing_cache.get("enabled"):
        saved = _native_preprocessing_cache_save(
            cache_dir=Path(str(preprocessing_cache["path"])),
            work_dir=work_dir,
            metadata=preprocessing_cache,
        )
        preprocessing_cache["saved"] = saved
        metadata["preprocessing_cache"] = preprocessing_cache

    return _base_record(
        cfg,
        category,
        mesh_id,
        stage,
        started,
        status=str(result.get("status", "success")),
        output_path=result.get("output_path"),
        command=list(result.get("command") or command),
        metadata=metadata,
    )


def _native_pipeline_kwargs(
    cfg: dict[str, Any],
    *,
    source_mesh: Path,
    work_dir: Path,
    manifoldplus_bin: str | Path,
    ftetwild_bin: str | Path,
    coacd_bin: str | Path,
    mesh_id: str,
    tetra_cfg: dict[str, Any],
    preseg_cfg: dict[str, Any],
    coacd_cfg: dict[str, Any],
    merge_cfg: dict[str, Any],
    refine_cfg: dict[str, Any],
    mcts_cfg: dict[str, Any],
    norm_cfg: dict[str, Any],
) -> dict[str, Any]:
    retry = tetra_cfg.get("retry", {}) if isinstance(tetra_cfg.get("retry"), dict) else {}
    ftetwild_threads = tetra_cfg.get("ftetwild_threads")
    if ftetwild_threads is not None and not _executable_supports_option(
        str(ftetwild_bin),
        "--max-threads",
    ):
        ftetwild_threads = None
    return {
        "input_mesh": source_mesh,
        "work_dir": work_dir,
        "manifoldplus_bin": manifoldplus_bin,
        "ftetwild_bin": ftetwild_bin,
        "coacd_bin": coacd_bin,
        "mesh_id": mesh_id,
        "epsilon": float(tetra_cfg.get("epsilon", 0.002)),
        "edge_length": float(tetra_cfg.get("edge_length", 0.1)),
        "merge_eps": float(merge_cfg.get("merge_eps", 0.02)),
        "refine_max_step": int(refine_cfg.get("max_step", 2000)),
        "mcts_iter": int(mcts_cfg.get("mcts_iter", 3000)),
        "mcts_max_step": int(mcts_cfg.get("max_step", 150)),
        "normalize_mode": str(norm_cfg.get("mode", "bbox_diagonal")),
        "normalize_target": float(norm_cfg.get("target", 1.0)),
        "normalize_center": str(norm_cfg.get("center", "bbox")),
        "cover_penalty": float(mcts_cfg.get("cover_penalty", refine_cfg.get("cover_penalty", 100))),
        "refine_action_unit": float(refine_cfg.get("action_unit", 0.01)),
        "mcts_action_unit": float(mcts_cfg.get("action_unit", 0.02)),
        "num_action_scale": max(1, int(refine_cfg.get("num_action_scale", 1))) * 2,
        "manifold_timeout_sec": float(tetra_cfg.get("manifold_timeout_sec", 600)),
        "manifold_depth": int(tetra_cfg.get("manifold_depth", 0) or 0),
        "skip_manifoldplus": bool(tetra_cfg.get("skip_manifoldplus", False)),
        "ftetwild_timeout_sec": float(tetra_cfg.get("ftetwild_timeout_sec", 1200)),
        "ftetwild_threads": ftetwild_threads,
        "ftetwild_level": int(tetra_cfg.get("ftetwild_level", 2)),
        "retry_epsilon_scale": float(retry.get("epsilon_scale", 2.0)),
        "retry_edge_length_scale": float(retry.get("edge_length_scale", 2.0)),
        "coacd_timeout_sec": float(preseg_cfg.get("timeout_sec", 1200)),
        "coacd_threshold": float(coacd_cfg.get("threshold", 0.05)),
        "coacd_max_convex_hull": int(coacd_cfg.get("max_convex_hull", 64)),
        "coacd_preprocess_mode": str(coacd_cfg.get("preprocess_mode", "auto")),
        "coacd_preprocess_resolution": int(coacd_cfg.get("preprocess_resolution", 50)),
        "coacd_resolution": int(coacd_cfg.get("resolution", 2000)),
        "coacd_mcts_nodes": int(coacd_cfg.get("mcts_nodes", 20)),
        "coacd_mcts_iterations": int(coacd_cfg.get("mcts_iterations", 150)),
        "coacd_mcts_max_depth": int(coacd_cfg.get("mcts_max_depth", 3)),
        "coacd_seed": int(coacd_cfg.get("seed", mcts_cfg.get("seed", 7777))),
        "coacd_pca": bool(coacd_cfg.get("pca", False)),
        "coacd_merge": bool(coacd_cfg.get("merge", True)),
        "coacd_decimate": bool(coacd_cfg.get("decimate", True)),
        "merge_tilted": bool(merge_cfg.get("tilted", True)),
        "merge_only_nearby": bool(merge_cfg.get("only_nearby", False)),
        "final_k": int(merge_cfg.get("final_k", 0)),
        "exp_w": float(mcts_cfg.get("exp_w", 0.001)),
        "gamma": float(mcts_cfg.get("gamma", 1.0)),
        "cache_capacity": int(
            mcts_cfg.get("stateful_cache_capacity", refine_cfg.get("stateful_cache_capacity", 65536))
        ),
        "volume_method": str(mcts_cfg.get("manifold_volume_method", "mesh")),
        "stateful_union_cache": bool(mcts_cfg.get("stateful_union_cache", True)),
        "transposition_table": bool(mcts_cfg.get("transposition_table", False)),
        "reuse_existing": bool(cfg.get("native_pipeline", {}).get("reuse_existing", False)),
        "reuse_preprocessing": bool(cfg.get("native_pipeline", {}).get("reuse_preprocessing", False)),
        "seed": int(mcts_cfg.get("seed", 0)),
    }


def _native_manifold_depth_attempts(
    category_name: str, tetra_cfg: dict[str, Any]
) -> list[int]:
    if bool(tetra_cfg.get("skip_manifoldplus", False)):
        return []
    raw_by_category = tetra_cfg.get("manifold_depth_candidates_by_category", {})
    raw: Any = None
    if isinstance(raw_by_category, dict):
        raw = raw_by_category.get(category_name)
    if raw is None:
        raw = tetra_cfg.get("manifold_depth_candidates", [])
    attempts = _parse_depth_attempts(raw)
    if not attempts:
        return []
    fallback = int(tetra_cfg.get("manifold_depth_fallback", 8) or 8)
    if fallback > 0 and fallback not in attempts:
        attempts.append(fallback)
    return attempts


def _parse_depth_attempts(raw: Any) -> list[int]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [raw]
    attempts: list[int] = []
    for value in values:
        if value is None or value == "":
            continue
        depth = int(value)
        if depth <= 0:
            continue
        if depth not in attempts:
            attempts.append(depth)
    return attempts


def _clear_native_pipeline_attempt_outputs(work_dir: Path) -> None:
    for rel in (
        "tetra",
        "coacd",
        "bsp_parts",
        "merge",
        "refine_bboxs_steps0",
        "mcts_bboxs_steps0",
    ):
        path = work_dir / rel
        if path.exists():
            shutil.rmtree(path)


def _native_preprocessing_cache_prepare(
    cfg: dict[str, Any],
    *,
    source_mesh: Path,
    work_dir: Path,
    kwargs: dict[str, Any],
    manifoldplus_bin: str | Path,
    ftetwild_bin: str | Path,
    coacd_bin: str | Path,
    dry_run: bool,
) -> dict[str, Any]:
    native_cfg = cfg.get("native_pipeline", {})
    if not isinstance(native_cfg, dict) or not native_cfg.get("preprocessing_cache", False):
        return {"enabled": False}
    cache_root_value = native_cfg.get("preprocessing_cache_root", "runs/.smart_preprocessing_cache")
    cache_root = repo_path(cache_root_value)
    if cache_root is None:
        cache_root = Path("runs/.smart_preprocessing_cache")
    key = _native_preprocessing_cache_key(
        source_mesh=source_mesh,
        kwargs=kwargs,
        manifoldplus_bin=manifoldplus_bin,
        ftetwild_bin=ftetwild_bin,
        coacd_bin=coacd_bin,
    )
    cache_dir = cache_root / key
    status: dict[str, Any] = {
        "enabled": True,
        "hit": False,
        "saved": False,
        "key": key,
        "path": str(cache_dir),
    }
    if dry_run:
        return status
    try:
        status["hit"] = _native_preprocessing_cache_restore(cache_dir, work_dir)
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"restore failed: {exc}"
        status["hit"] = False
    return status


def _native_preprocessing_cache_key(
    *,
    source_mesh: Path,
    kwargs: dict[str, Any],
    manifoldplus_bin: str | Path,
    ftetwild_bin: str | Path,
    coacd_bin: str | Path,
) -> str:
    preproc_keys = (
        "epsilon",
        "edge_length",
        "normalize_mode",
        "normalize_target",
        "normalize_center",
        "manifold_depth",
        "skip_manifoldplus",
        "ftetwild_level",
        "retry_epsilon_scale",
        "retry_edge_length_scale",
        "coacd_threshold",
        "coacd_max_convex_hull",
        "coacd_preprocess_mode",
        "coacd_preprocess_resolution",
        "coacd_resolution",
        "coacd_mcts_nodes",
        "coacd_mcts_iterations",
        "coacd_mcts_max_depth",
        "coacd_seed",
        "coacd_pca",
        "coacd_merge",
        "coacd_decimate",
    )
    payload = {
        "version": 1,
        "source_mesh_sha256": _file_digest(source_mesh),
        "preprocessing": {key: kwargs.get(key) for key in preproc_keys},
        "tools": {
            "manifoldplus": None
            if bool(kwargs.get("skip_manifoldplus", False))
            else _tool_signature(manifoldplus_bin),
            "ftetwild": _tool_signature(ftetwild_bin),
            "coacd": _tool_signature(coacd_bin),
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _file_digest(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_signature(path: str | Path) -> dict[str, Any]:
    candidate = repo_path(path)
    if candidate is None:
        candidate = Path(str(path)).expanduser()
    try:
        stat = candidate.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(candidate),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _native_preprocessing_cache_restore(cache_dir: Path, work_dir: Path) -> bool:
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        return False
    if not all((cache_dir / rel).exists() for rel in _NATIVE_PREPROCESSING_CACHE_REQUIRED):
        return False
    for rel in _NATIVE_PREPROCESSING_CACHE_REQUIRED:
        _copy_cache_file(cache_dir / rel, work_dir / rel)
    for rel in _NATIVE_PREPROCESSING_CACHE_OPTIONAL_FILES:
        src = cache_dir / rel
        if src.exists():
            _copy_cache_file(src, work_dir / rel)
    for rel in _NATIVE_PREPROCESSING_CACHE_OPTIONAL_DIRS:
        src = cache_dir / rel
        if src.exists():
            shutil.copytree(src, work_dir / rel, dirs_exist_ok=True)
    return True


def _native_preprocessing_cache_save(
    *,
    cache_dir: Path,
    work_dir: Path,
    metadata: dict[str, Any],
) -> bool:
    if cache_dir.exists():
        return False
    missing = [
        rel
        for rel in _NATIVE_PREPROCESSING_CACHE_REQUIRED
        if not (work_dir / rel).exists()
    ]
    if missing:
        metadata["save_error"] = "missing required preprocessing outputs: " + ", ".join(missing)
        return False
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp.{os.getpid()}.{time.time_ns()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        for rel in _NATIVE_PREPROCESSING_CACHE_REQUIRED:
            _copy_cache_file(work_dir / rel, tmp_dir / rel)
        for rel in _NATIVE_PREPROCESSING_CACHE_OPTIONAL_FILES:
            src = work_dir / rel
            if src.exists():
                _copy_cache_file(src, tmp_dir / rel)
        for rel in _NATIVE_PREPROCESSING_CACHE_OPTIONAL_DIRS:
            src = work_dir / rel
            if src.exists():
                shutil.copytree(src, tmp_dir / rel, dirs_exist_ok=True)
        (tmp_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "required": list(_NATIVE_PREPROCESSING_CACHE_REQUIRED),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            tmp_dir.rename(cache_dir)
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        metadata["save_error"] = str(exc)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False


def _copy_cache_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_stage(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    stage: str,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    if stage == "tetra":
        return run_tetra_mesh(cfg, category, mesh_id, dry_run=dry_run, force=force)
    if stage == "normalize":
        return run_normalize_mesh(cfg, category, mesh_id, dry_run=dry_run, force=force)
    if stage == "preseg":
        return run_preseg_mesh(cfg, category, mesh_id, dry_run=dry_run, force=force)
    if stage == "merge":
        return run_merge_mesh(cfg, category, mesh_id, dry_run=dry_run, force=force)
    if stage == "refine":
        return run_refine_mesh(cfg, category, mesh_id, dry_run=dry_run, force=force)
    if stage == "mcts":
        return run_mcts_mesh(cfg, category, mesh_id, dry_run=dry_run, force=force)
    if stage == "macro_skill":
        return run_macro_skill_mesh(
            cfg, category, mesh_id, dry_run=dry_run, force=force
        )
    if stage == "local_refine":
        return run_local_refine_mesh(
            cfg, category, mesh_id, dry_run=dry_run, force=force
        )
    if stage == "render":
        return run_render_mesh(cfg, category, mesh_id, dry_run=dry_run, force=force)
    raise ValueError(f"Unknown stage: {stage}")


def list_mesh_ids(category: dict[str, Any], explicit: list[str] | None = None) -> list[str]:
    root = category_mesh_root(category)
    if explicit:
        return [mesh for mesh in explicit if (root / mesh / "model.obj").exists()]
    configured = category.get("meshes")
    if configured:
        return [str(mesh) for mesh in configured if (root / str(mesh) / "model.obj").exists()]
    if not root.exists():
        return []
    mesh_ids = sorted(path.name for path in root.iterdir() if (path / "model.obj").exists())
    limit = category.get("limit")
    if limit:
        mesh_ids = mesh_ids[: int(limit)]
    return mesh_ids


def tetra_dataset_name(cfg: dict[str, Any], category: dict[str, Any], epsilon: float, edge_length: float) -> str:
    norm = normalization_suffix(cfg) if normalization_enabled(cfg) else "raw"
    return f"{category_mesh_root(category).name}_{norm}_e{epsilon:g}_l{edge_length:g}"


def tetra_root(cfg: dict[str, Any], category: dict[str, Any]) -> Path:
    epsilon, edge_length = category_tetra_params(cfg, category)
    return workspace_path(cfg, "tetra", tetra_dataset_name(cfg, category, epsilon, edge_length))


def mesh_tetra_dir(cfg: dict[str, Any], category: dict[str, Any], mesh_id: str) -> Path:
    return tetra_root(cfg, category) / mesh_id


def normalized_mesh_dir(cfg: dict[str, Any], category: dict[str, Any], mesh_id: str) -> Path:
    return workspace_path(cfg, "normalized", category["name"], mesh_id)


def normalized_mesh_path(cfg: dict[str, Any], category: dict[str, Any], mesh_id: str) -> Path:
    return normalized_mesh_dir(cfg, category, mesh_id) / str(cfg.get("normalization", {}).get("source_filename", "model.obj"))


def stage_root(cfg: dict[str, Any], stage: str, category: dict[str, Any]) -> Path:
    return workspace_path(cfg, stage, category["name"])


def normalization_enabled(cfg: dict[str, Any]) -> bool:
    stages = cfg.get("stages", {})
    return bool(stages.get("normalize", False) and cfg.get("normalization", {}).get("enabled", True))


def normalization_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_cfg = cfg.get("normalization", {})
    return {
        "mode": str(stage_cfg.get("mode", "bbox_diagonal")),
        "target": float(stage_cfg.get("target", 1.0)),
        "center": str(stage_cfg.get("center", "bbox")),
        "source_filename": str(stage_cfg.get("source_filename", "model.obj")),
    }


def normalization_suffix(cfg: dict[str, Any]) -> str:
    signature = normalization_signature(cfg)
    mode = signature["mode"]
    target = f"{signature['target']:g}".replace(".", "p")
    if mode == "bbox_diagonal":
        return f"norm-bboxdiag{target}"
    if mode == "unit_sphere":
        return f"norm-sphere{target}"
    if mode == "unit_bbox":
        return f"norm-bbox{target}"
    return f"norm-{mode}{target}"


def _base_record(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    stage: str,
    started: float,
    *,
    status: str,
    output_path: str | Path | None = None,
    log_path: str | Path | None = None,
    command: list[str] | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StageRecord:
    return StageRecord.now(
        stage=stage,
        category=category["name"],
        mesh_id=mesh_id,
        status=status,
        started_at=started,
        output_path=output_path,
        log_path=log_path,
        command=command,
        error=error,
        metadata=metadata,
    )


def run_normalize_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage_cfg = cfg.get("normalization", {})
    source_name = str(stage_cfg.get("source_filename", "model.obj"))
    source_mesh = category_mesh_root(category) / mesh_id / source_name
    out_dir = normalized_mesh_dir(cfg, category, mesh_id)
    output = out_dir / source_name
    metadata_path = out_dir / "normalization.json"
    signature = normalization_signature(cfg)

    if output.exists() and metadata_path.exists() and not force:
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if existing.get("signature") == signature:
            return _base_record(
                cfg,
                category,
                mesh_id,
                "normalize",
                started,
                status="skipped",
                output_path=output,
                metadata={"reason": "existing_output", **existing.get("after", {})},
            )

    if dry_run:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "normalize",
            started,
            status="dry_run",
            output_path=output,
            metadata=signature,
        )
    if not source_mesh.exists():
        return _base_record(
            cfg,
            category,
            mesh_id,
            "normalize",
            started,
            status="failed",
            output_path=output,
            error=f"missing source mesh: {source_mesh}",
        )

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        native_exec = _normalize_obj_file_executable(cfg, source_mesh, output, signature, mesh_id)
        command = list(native_exec.get("command") or []) if native_exec else None
        log_path = Path(native_exec["log_path"]) if native_exec and native_exec.get("log_path") else None
        if native_exec is not None:
            stats = dict(native_exec["stats"])
        else:
            native_stats = _normalize_obj_file_native(source_mesh, output, signature)
            command = None
            log_path = None
            if native_stats is not None:
                stats = native_stats
            else:
                if stage_cfg.get("native_executable_required", False):
                    raise RuntimeError("smart-cpp-native normalize executable is required but unavailable")
                obj_lines, vertices = _read_obj_vertices(source_mesh)
                normalized, stats = _normalize_vertices(vertices, signature)
                _write_normalized_obj(obj_lines, normalized, output)
                stats["backend"] = "python_obj"
        metadata = {"signature": signature, **stats, "source": str(source_mesh), "output": str(output)}
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return _base_record(
            cfg,
            category,
            mesh_id,
            "normalize",
            started,
            status="failed",
            output_path=output,
            error=str(exc),
        )

    return _base_record(
        cfg,
        category,
        mesh_id,
        "normalize",
        started,
        status="success",
        output_path=output,
        log_path=log_path,
        command=command,
        metadata={"scale": stats["scale"], **stats["after"]},
    )


def _normalize_obj_file_executable(
    cfg: dict[str, Any],
    source: Path,
    output: Path,
    signature: dict[str, Any],
    mesh_id: str,
) -> dict[str, Any] | None:
    stage_cfg = cfg.get("normalization", {})
    if str(stage_cfg.get("backend", "cpp_native_executable")) != "cpp_native_executable":
        return None
    tools = cfg.get("tools", {})
    explicit = tools.get("smart_cpp_native_bin") if isinstance(tools, dict) else None
    binary_path = native_executable_path(explicit)
    if binary_path is None:
        return None
    binary = str(binary_path)
    command = [
        binary,
        "normalize",
        "--input",
        str(source),
        "--output",
        str(output),
        "--mode",
        str(signature["mode"]),
        "--center",
        str(signature["center"]),
        "--target",
        str(float(signature["target"])),
    ]
    log_path = workspace_path(cfg, "logs", "normalize", f"{mesh_id}.log")
    result = run_command(command, timeout=600, log_path=log_path)
    if not result.ok:
        if stage_cfg.get("native_executable_required", False):
            raise RuntimeError(f"smart-cpp-native normalize failed; see {log_path}")
        return None
    payload = json.loads(result.stdout or "{}")
    if payload.get("status") != "success":
        if stage_cfg.get("native_executable_required", False):
            raise RuntimeError(f"smart-cpp-native normalize returned non-success; see {log_path}")
        return None
    payload["backend"] = "cpp_native_executable"
    return {"stats": payload, "command": command, "log_path": str(log_path)}


def _normalize_obj_file_native(
    source: Path,
    output: Path,
    signature: dict[str, Any],
) -> dict[str, Any] | None:
    if smart_native is None:
        return None
    try:
        if not smart_native.native_core_available() or not hasattr(smart_native, "native_normalize_obj_file"):
            return None
        stats = smart_native.native_normalize_obj_file(
            str(source),
            str(output),
            mode=str(signature["mode"]),
            center=str(signature["center"]),
            target=float(signature["target"]),
        )
        payload = dict(stats)
        payload["backend"] = "cpp_native_obj"
        return payload
    except Exception:
        return None


def tetra_source_mesh(cfg: dict[str, Any], category: dict[str, Any], mesh_id: str) -> Path:
    if normalization_enabled(cfg):
        return normalized_mesh_path(cfg, category, mesh_id)
    source_name = str(cfg.get("normalization", {}).get("source_filename", "model.obj"))
    return category_mesh_root(category) / mesh_id / source_name


def _prepare_tetra_input_mesh(source: Path, output: Path, stage_cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    repair_cfg = stage_cfg.get("input_repair", {})
    if not repair_cfg.get("enabled", False):
        return source, {"enabled": False}
    metadata: dict[str, Any] = {"enabled": True, "used": False, "source": str(source)}
    try:
        import trimesh  # type: ignore
        import trimesh.repair  # type: ignore
    except ModuleNotFoundError:
        metadata["skipped_reason"] = "trimesh is not installed"
        return source, metadata
    try:
        mesh = trimesh.load(str(source), force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            metadata["skipped_reason"] = f"unsupported mesh object: {type(mesh).__name__}"
            return source, metadata
        keep_largest_component = bool(repair_cfg.get("keep_largest_component", False))
        components = _split_mesh_components(trimesh, mesh) if keep_largest_component else []
        metadata["before"] = {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "components": int(len(components)) if keep_largest_component and len(mesh.faces) else None,
        }
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            metadata["skipped_reason"] = "empty mesh"
            return source, metadata
        if keep_largest_component:
            if components:
                mesh = max(components, key=lambda component: len(component.faces))
        if repair_cfg.get("basic_cleanup", True):
            if hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
                try:
                    mesh.update_faces(mesh.unique_faces())
                except Exception:  # noqa: BLE001
                    pass
            else:
                _call_optional_mesh_method(mesh, "remove_duplicate_faces")
            if hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
                try:
                    mesh.update_faces(mesh.nondegenerate_faces())
                except Exception:  # noqa: BLE001
                    pass
            else:
                _call_optional_mesh_method(mesh, "remove_degenerate_faces")
            _call_optional_mesh_method(mesh, "remove_infinite_values")
            _call_optional_mesh_method(mesh, "remove_unreferenced_vertices")
            _call_optional_mesh_method(mesh, "merge_vertices")
        if repair_cfg.get("fill_holes", False):
            trimesh.repair.fill_holes(mesh)
        if repair_cfg.get("fix_normals", True):
            trimesh.repair.fix_normals(mesh)
        components = _split_mesh_components(trimesh, mesh) if keep_largest_component else []
        metadata["after"] = {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "components": int(len(components)) if keep_largest_component and len(mesh.faces) else None,
        }
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            metadata["skipped_reason"] = "repair produced empty mesh"
            return source, metadata
        output.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output), file_type="obj")
        if not output.exists() or output.stat().st_size == 0:
            metadata["skipped_reason"] = "repair export failed"
            return source, metadata
        metadata["used"] = True
        metadata["path"] = str(output)
        return output, metadata
    except Exception as exc:  # noqa: BLE001
        metadata["error"] = str(exc)
        return source, metadata


def _tetra_input_candidates(
    source: Path,
    log_dir: Path,
    stage_cfg: dict[str, Any],
    *,
    active_failure_classes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repair_cfg = stage_cfg.get("input_repair", {})
    primary_path, primary_metadata = _prepare_tetra_input_mesh(
        source,
        log_dir / "input_repaired.obj",
        stage_cfg,
    )
    candidates: list[dict[str, Any]] = [
        {
            "name": "primary",
            "path": primary_path,
            "repair": primary_metadata,
        }
    ]
    repair_records: list[dict[str, Any]] = [primary_metadata]
    seen_paths = {primary_path.resolve() if primary_path.exists() else primary_path}

    if not isinstance(repair_cfg, dict) or not repair_cfg.get("enabled", False):
        return candidates, repair_records

    for index, variant in enumerate(repair_cfg.get("fallback_variants", []) or []):
        if not isinstance(variant, dict):
            continue
        if variant.get("enabled", True) is False:
            continue
        triggers = {str(item) for item in variant.get("triggers", []) or []}
        if active_failure_classes is not None and triggers and not (triggers & active_failure_classes):
            continue
        name = str(variant.get("name", f"repair_fallback_{index + 1}"))
        variant_repair_cfg = dict(repair_cfg)
        variant_repair_cfg.update(variant)
        variant_repair_cfg.pop("fallback_variants", None)
        variant_cfg = deep_update(stage_cfg, {"input_repair": variant_repair_cfg})
        variant_path, variant_metadata = _prepare_tetra_input_mesh(
            source,
            log_dir / f"input_repaired_{name}.obj",
            variant_cfg,
        )
        variant_metadata["variant"] = name
        repair_records.append(variant_metadata)
        if not variant_metadata.get("used", False):
            continue
        resolved = variant_path.resolve() if variant_path.exists() else variant_path
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidates.append(
            {
                "name": name,
                "path": variant_path,
                "repair": variant_metadata,
                "triggers": sorted(triggers),
            }
        )

    return candidates, repair_records


def _append_tetra_repair_candidates(
    input_candidates: list[dict[str, Any]],
    input_repair_metadata: list[dict[str, Any]],
    source_mesh: Path,
    log_dir: Path,
    stage_cfg: dict[str, Any],
    active_failure_classes: set[str],
    queued_names: set[str],
) -> list[str]:
    if not active_failure_classes:
        return []
    candidates, repair_records = _tetra_input_candidates(
        source_mesh,
        log_dir,
        stage_cfg,
        active_failure_classes=active_failure_classes,
    )
    for record in repair_records:
        variant = str(record.get("variant") or "")
        if variant and not any(existing.get("variant") == variant for existing in input_repair_metadata):
            input_repair_metadata.append(record)

    added: list[str] = []
    for candidate in candidates:
        name = str(candidate.get("name", "primary"))
        if name == "primary" or name in queued_names:
            continue
        input_candidates.append(candidate)
        queued_names.add(name)
        added.append(name)
    return added


def _call_optional_mesh_method(mesh: Any, name: str) -> None:
    method = getattr(mesh, name, None)
    if not callable(method):
        return
    try:
        method()
    except Exception:  # noqa: BLE001
        return


def _split_mesh_components(trimesh_module: Any, mesh: Any) -> list[Any]:
    try:
        return list(trimesh_module.graph.split(mesh, only_watertight=False))
    except TypeError:
        return list(trimesh_module.graph.split(mesh))
    except Exception:  # noqa: BLE001
        return []


def _quick_obj_counts(path: Path) -> dict[str, int]:
    vertices = 0
    faces = 0
    if not path.exists():
        return {"vertices": 0, "faces": 0, "size_bytes": 0}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    return {"vertices": vertices, "faces": faces, "size_bytes": int(path.stat().st_size)}


def _read_obj_vertices(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    vertices: list[tuple[float, float, float]] = []
    for line in lines:
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"malformed vertex line in {path}: {line.strip()}")
        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError(f"no vertices found in OBJ: {path}")
    return lines, vertices


def _normalize_vertices(
    vertices: list[tuple[float, float, float]],
    signature: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    if smart_native is not None:
        try:
            if smart_native.native_core_available():
                return smart_native.native_normalize_vertices(
                    vertices,
                    mode=str(signature["mode"]),
                    center=str(signature["center"]),
                    target=float(signature["target"]),
                )
        except Exception:
            pass

    before = _vertex_stats(vertices)
    center_mode = signature["center"]
    if center_mode == "bbox":
        center = tuple((before["bbox_min"][i] + before["bbox_max"][i]) / 2.0 for i in range(3))
    elif center_mode == "mean":
        count = float(len(vertices))
        center = tuple(sum(vertex[i] for vertex in vertices) / count for i in range(3))
    else:
        raise ValueError(f"unsupported normalization center: {center_mode}")

    mode = signature["mode"]
    if mode == "bbox_diagonal":
        denominator = before["bbox_diagonal"]
    elif mode == "unit_bbox":
        denominator = max(before["bbox_extent"])
    elif mode == "unit_sphere":
        denominator = max(_distance(vertex, center) for vertex in vertices)
    else:
        raise ValueError(f"unsupported normalization mode: {mode}")
    if denominator <= 0:
        raise ValueError("degenerate mesh scale")

    scale = float(signature["target"]) / denominator
    normalized = [
        (
            (vertex[0] - center[0]) * scale,
            (vertex[1] - center[1]) * scale,
            (vertex[2] - center[2]) * scale,
        )
        for vertex in vertices
    ]
    after = _vertex_stats(normalized)
    return normalized, {"before": before, "after": after, "center": list(center), "scale": scale}


def _write_normalized_obj(lines: list[str], vertices: list[tuple[float, float, float]], output: Path) -> None:
    vertex_index = 0
    out_lines: list[str] = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            x, y, z = vertices[vertex_index]
            extras = parts[4:]
            suffix = (" " + " ".join(extras)) if extras else ""
            newline = "\n" if line.endswith("\n") else ""
            out_lines.append(f"v {x:.9g} {y:.9g} {z:.9g}{suffix}{newline}")
            vertex_index += 1
        else:
            out_lines.append(line)
    output.write_text("".join(out_lines), encoding="utf-8")


def _vertex_stats(vertices: list[tuple[float, float, float]]) -> dict[str, Any]:
    xs, ys, zs = zip(*vertices)
    bbox_min = [min(xs), min(ys), min(zs)]
    bbox_max = [max(xs), max(ys), max(zs)]
    extent = [bbox_max[i] - bbox_min[i] for i in range(3)]
    bbox_center = [(bbox_min[i] + bbox_max[i]) / 2.0 for i in range(3)]
    return {
        "vertex_count": len(vertices),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_extent": extent,
        "bbox_diagonal": (extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2) ** 0.5,
        "bbox_center": bbox_center,
        "sphere_radius": max(_distance(vertex, bbox_center) for vertex in vertices),
    }


def _distance(vertex: tuple[float, float, float], center: Iterable[float]) -> float:
    cx, cy, cz = center
    return ((vertex[0] - cx) ** 2 + (vertex[1] - cy) ** 2 + (vertex[2] - cz) ** 2) ** 0.5


def run_tetra_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage_cfg = deep_update(cfg.get("tetra", {}), category.get("tetra", {}))
    source_mesh = tetra_source_mesh(cfg, category, mesh_id)
    out_dir = mesh_tetra_dir(cfg, category, mesh_id)
    tetmesh = out_dir / "tetra.msh"
    surface = out_dir / "tetra.msh__sf.obj"
    log_dir = workspace_path(cfg, "logs", "tetra", category["name"], mesh_id)
    epsilon = float(stage_cfg["epsilon"])
    edge_length = float(stage_cfg["edge_length"])

    if tetmesh.exists() and surface.exists() and not force:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "tetra",
            started,
            status="skipped",
            output_path=out_dir,
            metadata={"reason": "existing_output"},
        )

    manifold_bin = _mesh2tet_tool(cfg, "manifoldplus_bin", "SMART_MANIFOLDPLUS_BIN", "manifold")
    ftetwild_bin = _mesh2tet_tool(cfg, "ftetwild_bin", "SMART_FTETWILD_BIN", "FloatTetwild_bin")
    if dry_run:
        command = [
            manifold_bin or "$SMART_MANIFOLDPLUS_BIN",
            "--input",
            str(source_mesh),
            "--output",
            str(out_dir / "model_manifold.obj"),
        ]
        return _base_record(
            cfg,
            category,
            mesh_id,
            "tetra",
            started,
            status="dry_run",
            output_path=out_dir,
            command=command,
            metadata={"epsilon": epsilon, "edge_length": edge_length},
        )
    if not manifold_bin or not ftetwild_bin:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "tetra",
            started,
            status="blocked",
            output_path=out_dir,
            error="Missing ManifoldPlus or fTetWild binary. Run `smart build-tools` or set SMART_MANIFOLDPLUS_BIN/SMART_FTETWILD_BIN.",
        )
    if not source_mesh.exists():
        return _base_record(
            cfg,
            category,
            mesh_id,
            "tetra",
            started,
            status="skipped",
            output_path=out_dir,
            error=f"missing source mesh: {source_mesh}",
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    input_candidates: list[dict[str, Any]] = [{"name": "source", "path": source_mesh, "repair": {"enabled": False}}]
    input_repair_metadata: list[dict[str, Any]] = []
    if not dry_run:
        repair_cfg = stage_cfg.get("input_repair", {})
        auto_repair = bool(
            isinstance(repair_cfg, dict)
            and repair_cfg.get("enabled", False)
            and repair_cfg.get("auto_retry_by_failure", True)
        )
        input_candidates, input_repair_metadata = _tetra_input_candidates(
            source_mesh,
            log_dir,
            stage_cfg,
            active_failure_classes=set() if auto_repair else None,
        )

    attempts: list[dict[str, Any]] = [
        {
            "epsilon": epsilon,
            "edge_length": edge_length,
            "coarsen": bool(stage_cfg.get("coarsen", False)),
            "name": "primary",
        }
    ]
    retry = stage_cfg.get("retry", {})
    if retry.get("enabled", True):
        fine_retry = retry.get("fine_retry", {})
        if fine_retry.get("enabled", True):
            attempts.append(
                {
                    "epsilon": epsilon * float(fine_retry.get("epsilon_scale", 0.5)),
                    "edge_length": edge_length * float(fine_retry.get("edge_length_scale", 0.5)),
                    "coarsen": bool(fine_retry.get("coarsen", False)),
                    "timeout_sec": fine_retry.get("timeout_sec"),
                    "name": str(fine_retry.get("name", "fine_retry")),
                }
            )
        attempts.append(
            {
                "epsilon": epsilon * float(retry.get("epsilon_scale", 2.0)),
                "edge_length": edge_length * float(retry.get("edge_length_scale", 2.0)),
                "coarsen": bool(retry.get("coarsen", True)),
                "timeout_sec": retry.get("timeout_sec"),
                "name": "retry",
            }
        )
        for idx, extra in enumerate(retry.get("extra_attempts", []) or []):
            attempt = dict(extra)
            attempt["epsilon"] = epsilon * float(extra.get("epsilon_scale", 4.0))
            attempt["edge_length"] = edge_length * float(extra.get("edge_length_scale", 3.0))
            attempt["coarsen"] = bool(extra.get("coarsen", True))
            attempt["name"] = str(extra.get("name", f"extra_retry_{idx + 1}"))
            attempts.append(attempt)

    errors: list[str] = []
    attempt_metadata: list[dict[str, Any]] = []
    last_command: list[str] | None = None
    last_log: Path | None = None
    queued_input_variants = {str(candidate.get("name", "primary")) for candidate in input_candidates}
    observed_failure_classes: set[str] = set()
    repair_cfg = stage_cfg.get("input_repair", {})
    auto_repair = bool(
        isinstance(repair_cfg, dict)
        and repair_cfg.get("enabled", False)
        and repair_cfg.get("auto_retry_by_failure", True)
    )
    immediate_repair_failures = {
        str(item) for item in repair_cfg.get("immediate_retry_failures", []) or []
    } if isinstance(repair_cfg, dict) else set()
    input_index = 0
    while input_index < len(input_candidates):
        input_candidate = input_candidates[input_index]
        input_index += 1
        prepared_source_mesh = Path(input_candidate["path"])
        input_variant = str(input_candidate.get("name", "primary"))
        for attempt in attempts:
            eps = float(attempt["epsilon"])
            length = float(attempt["edge_length"])
            coarsen = bool(attempt.get("coarsen", False))
            base_attempt_name = str(attempt["name"])
            attempt_name = base_attempt_name if input_variant == "primary" else f"{input_variant}_{base_attempt_name}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            manmesh = out_dir / "model_manifold.obj"
            manifold_log = log_dir / f"{attempt_name}_manifoldplus.log"
            ftetwild_log = log_dir / f"{attempt_name}_ftetwild.log"
            manifold_cmd = [manifold_bin, "--input", str(prepared_source_mesh), "--output", str(manmesh)]
            result = run_command(
                manifold_cmd,
                timeout=stage_cfg.get("manifold_timeout_sec"),
                log_path=manifold_log,
                dry_run=dry_run,
            )
            last_command = manifold_cmd
            last_log = manifold_log
            if not result.ok:
                failure = _command_failure_summary("ManifoldPlus", result)
                failure_class = _command_failure_class(result)
                observed_failure_classes.add(failure_class)
                added_repair_variants = (
                    _append_tetra_repair_candidates(
                        input_candidates,
                        input_repair_metadata,
                        source_mesh,
                        log_dir,
                        stage_cfg,
                        observed_failure_classes,
                        queued_input_variants,
                    )
                    if auto_repair
                    else []
                )
                errors.append(f"{attempt_name}: {failure}")
                attempt_metadata.append(
                    {
                        "attempt": attempt_name,
                        "input_variant": input_variant,
                        "input_mesh": str(prepared_source_mesh),
                        "tool": "ManifoldPlus",
                        "epsilon": eps,
                        "edge_length": length,
                        "coarsen": coarsen,
                        "returncode": result.returncode,
                        "elapsed_sec": result.elapsed_sec,
                        "timed_out": result.timed_out,
                        "failure": failure,
                        "failure_class": failure_class,
                        "queued_repair_variants": added_repair_variants,
                    }
                )
                if added_repair_variants and failure_class in immediate_repair_failures:
                    break
                continue

            manifold_counts: dict[str, int] = {}
            max_manifold_faces = int(stage_cfg.get("max_manifold_faces_for_ftetwild", 0) or 0)
            if max_manifold_faces > 0 and not bool(attempt.get("allow_large_manifold_surface", False)):
                manifold_counts = _quick_obj_counts(manmesh)
                if int(manifold_counts.get("faces", 0)) > max_manifold_faces:
                    failure_class = "repair_surface_too_large"
                    observed_failure_classes.add(failure_class)
                    added_repair_variants = (
                        _append_tetra_repair_candidates(
                            input_candidates,
                            input_repair_metadata,
                            source_mesh,
                            log_dir,
                            stage_cfg,
                            observed_failure_classes,
                            queued_input_variants,
                        )
                        if auto_repair
                        else []
                    )
                    failure = (
                        "ManifoldPlus repaired surface too large for fTetWild: "
                        f"{manifold_counts.get('faces', 0)} > {max_manifold_faces}"
                    )
                    errors.append(f"{attempt_name}: {failure}")
                    attempt_metadata.append(
                        {
                            "attempt": attempt_name,
                            "input_variant": input_variant,
                            "input_mesh": str(prepared_source_mesh),
                            "tool": "ManifoldPlus",
                            "epsilon": eps,
                            "edge_length": length,
                            "coarsen": coarsen,
                            "failure": failure,
                            "failure_class": failure_class,
                            "manifold_counts": manifold_counts,
                            "max_manifold_faces_for_ftetwild": max_manifold_faces,
                            "queued_repair_variants": added_repair_variants,
                        }
                    )
                    break

            ftetwild_cmd = [
                ftetwild_bin,
                "--input",
                str(manmesh),
                "--output",
                str(tetmesh),
                "-q",
                "-l",
                str(length),
                "-e",
                str(eps),
                "--log",
                str(out_dir / "log.txt"),
                "--level",
                str(int(stage_cfg.get("ftetwild_level", 2))),
                "--no-binary",
            ]
            if attempt.get("use_floodfill", stage_cfg.get("use_floodfill", True)):
                ftetwild_cmd.append("--use-floodfill")
            if attempt.get("use_general_wn", stage_cfg.get("use_general_wn", False)):
                ftetwild_cmd.append("--use-general-wn")
            if attempt.get("use_input_for_wn", stage_cfg.get("use_input_for_wn", False)):
                ftetwild_cmd.append("--use-input-for-wn")
            if attempt.get("manifold_surface", stage_cfg.get("manifold_surface", True)):
                ftetwild_cmd.append("--manifold-surface")
            if attempt.get("skip_simplify", stage_cfg.get("skip_simplify", False)):
                ftetwild_cmd.append("--skip-simplify")
            if stage_cfg.get("ftetwild_threads") is not None and _executable_supports_option(ftetwild_bin, "--max-threads"):
                ftetwild_cmd.extend(["--max-threads", str(int(stage_cfg.get("ftetwild_threads", 8)))])
            if coarsen:
                ftetwild_cmd.append("--coarsen")
            result = run_command(
                ftetwild_cmd,
                timeout=attempt.get("timeout_sec", stage_cfg.get("ftetwild_timeout_sec")),
                log_path=ftetwild_log,
                dry_run=dry_run,
            )
            last_command = ftetwild_cmd
            last_log = ftetwild_log
            if not result.ok:
                failure = _command_failure_summary("fTetWild", result)
                failure_class = _command_failure_class(result)
                observed_failure_classes.add(failure_class)
                added_repair_variants = (
                    _append_tetra_repair_candidates(
                        input_candidates,
                        input_repair_metadata,
                        source_mesh,
                        log_dir,
                        stage_cfg,
                        observed_failure_classes,
                        queued_input_variants,
                    )
                    if auto_repair
                    else []
                )
                errors.append(f"{attempt_name}: {failure}")
                attempt_metadata.append(
                    {
                        "attempt": attempt_name,
                        "input_variant": input_variant,
                        "input_mesh": str(prepared_source_mesh),
                        "tool": "fTetWild",
                        "epsilon": eps,
                        "edge_length": length,
                        "coarsen": coarsen,
                        "returncode": result.returncode,
                        "elapsed_sec": result.elapsed_sec,
                        "timed_out": result.timed_out,
                        "timeout_sec": attempt.get("timeout_sec", stage_cfg.get("ftetwild_timeout_sec")),
                        "failure": failure,
                        "failure_class": failure_class,
                        "queued_repair_variants": added_repair_variants,
                    }
                )
                if added_repair_variants and failure_class in immediate_repair_failures:
                    break
                continue
            if dry_run:
                return _base_record(
                    cfg,
                    category,
                    mesh_id,
                    "tetra",
                    started,
                    status="dry_run",
                    output_path=out_dir,
                    log_path=last_log,
                    command=last_command,
                )
            validation_metadata: dict[str, Any] = {}
            validation_error = None
            if stage_cfg.get("validate", True):
                validation_error, validation_metadata = inspect_tetra_output(
                    out_dir,
                    require_single_component=bool(stage_cfg.get("require_single_component", False)),
                    min_tetra_count=int(stage_cfg.get("min_tetra_count", 20)),
                    min_surface_faces=int(stage_cfg.get("min_surface_faces", 20)),
                )
            if validation_error:
                failure_class = _validation_failure_class(validation_error)
                observed_failure_classes.add(failure_class)
                added_repair_variants = (
                    _append_tetra_repair_candidates(
                        input_candidates,
                        input_repair_metadata,
                        source_mesh,
                        log_dir,
                        stage_cfg,
                        observed_failure_classes,
                        queued_input_variants,
                    )
                    if auto_repair
                    else []
                )
                errors.append(f"{attempt_name}: validation failed: {validation_error}")
                attempt_metadata.append(
                    {
                        "attempt": attempt_name,
                        "input_variant": input_variant,
                        "input_mesh": str(prepared_source_mesh),
                        "tool": "validation",
                        "epsilon": eps,
                        "edge_length": length,
                        "coarsen": coarsen,
                        "failure": validation_error,
                        "failure_class": failure_class,
                        "queued_repair_variants": added_repair_variants,
                        "metadata": validation_metadata,
                    }
                )
                if added_repair_variants and failure_class in immediate_repair_failures:
                    break
                continue
            metadata = {
                "epsilon": eps,
                "edge_length": length,
                "coarsen": coarsen,
                "attempt": attempt_name,
                "input_variant": input_variant,
                "input_mesh": str(prepared_source_mesh),
                "previous_attempts": attempt_metadata,
            }
            if input_repair_metadata:
                metadata["input_repair"] = input_repair_metadata
            if validation_metadata:
                metadata["validation"] = validation_metadata
            return _base_record(
                cfg,
                category,
                mesh_id,
                "tetra",
                started,
                status="success",
                output_path=out_dir,
                log_path=last_log,
                command=last_command,
                metadata=metadata,
            )

    shutil.rmtree(out_dir, ignore_errors=True)
    return _base_record(
        cfg,
        category,
        mesh_id,
        "tetra",
        started,
        status="failed",
        output_path=out_dir,
        log_path=last_log,
        command=last_command,
        error="; ".join(errors) if errors else "unknown tetrahedralization failure",
        metadata={"attempts": attempt_metadata, "input_repair": input_repair_metadata},
    )


def _mesh2tet_tool(cfg: dict[str, Any], config_key: str, env_key: str, fallback_name: str) -> str | None:
    explicit = cfg.get("tools", {}).get(config_key)
    if explicit:
        path = repo_path(explicit)
        if path is not None and path.exists():
            return str(path)
        if path is None and Path(str(explicit)).exists():
            return str(explicit)
    return find_executable(None, env_key, fallback_name)


_OPTION_SUPPORT_CACHE: dict[tuple[str, str], bool] = {}


def _executable_supports_option(executable: str, option: str) -> bool:
    key = (executable, option)
    if key in _OPTION_SUPPORT_CACHE:
        return _OPTION_SUPPORT_CACHE[key]
    try:
        completed = subprocess.run(
            [executable, "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        help_text = f"{completed.stdout}\n{completed.stderr}"
        supported = option in help_text
    except Exception:  # noqa: BLE001
        supported = False
    _OPTION_SUPPORT_CACHE[key] = supported
    return supported


def validate_tetra_output(out_dir: Path, *, require_single_component: bool = True) -> str | None:
    error, _ = inspect_tetra_output(out_dir, require_single_component=require_single_component)
    return error


def _inspect_ascii_msh_counts(path: Path) -> dict[str, int | None]:
    counts: dict[str, int | None] = {"nodes": None, "elements": None}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        return counts
    for idx, line in enumerate(lines):
        if line.strip() == "$Nodes" and idx + 1 < len(lines):
            try:
                counts["nodes"] = int(lines[idx + 1].strip())
            except ValueError:
                pass
        if line.strip() == "$Elements" and idx + 1 < len(lines):
            try:
                counts["elements"] = int(lines[idx + 1].strip())
            except ValueError:
                pass
    return counts


def inspect_tetra_output(
    out_dir: Path,
    *,
    require_single_component: bool = False,
    min_tetra_count: int = 20,
    min_surface_faces: int = 20,
) -> tuple[str | None, dict[str, Any]]:
    tetmesh = out_dir / "tetra.msh"
    surface = out_dir / "tetra.msh__sf.obj"
    metadata: dict[str, Any] = {
        "require_single_component": require_single_component,
        "min_tetra_count": min_tetra_count,
        "min_surface_faces": min_surface_faces,
        "tetra_exists": tetmesh.exists(),
        "surface_exists": surface.exists(),
    }
    if not tetmesh.exists():
        return "missing tetra.msh", metadata
    if not surface.exists():
        return "missing tetra.msh__sf.obj", metadata
    msh_counts = _inspect_ascii_msh_counts(tetmesh)
    metadata.update(
        {
            "tetra_nodes": msh_counts.get("nodes"),
            "tetra_elements": msh_counts.get("elements"),
        }
    )
    if msh_counts.get("elements") is not None and int(msh_counts["elements"] or 0) < min_tetra_count:
        return f"tetra element count below minimum: {msh_counts['elements']} < {min_tetra_count}", metadata
    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError:
        return None, metadata
    try:
        mesh = trimesh.load(surface, file_type="obj", process=False)
        components = _split_mesh_components(trimesh, mesh)
        metadata.update(
            {
                "surface_vertices": int(len(mesh.vertices)),
                "surface_faces": int(len(mesh.faces)),
                "surface_watertight": bool(mesh.is_watertight),
                "surface_component_count": int(len(components)),
                "largest_surface_component_faces": int(max((len(component.faces) for component in components), default=0)),
            }
        )
        if len(mesh.faces) < min_surface_faces:
            return f"surface face count below minimum: {len(mesh.faces)} < {min_surface_faces}", metadata
        if not mesh.is_watertight:
            return "surface is not watertight", metadata
        if require_single_component and len(components) > 1:
            return "surface has multiple connected components", metadata
    except Exception as exc:  # noqa: BLE001
        return str(exc), metadata
    return None, metadata


def _write_coacd_partition_metadata(
    tet_dir: Path,
    mesh_id: str,
    *,
    force: bool = False,
    require_native: bool = False,
) -> dict[str, Any]:
    """Cache CoACD part-to-tet assignments for the direct C++ merge runner."""
    metadata_path = tet_dir / "coacd_partitions.json"
    if metadata_path.exists() and not force:
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return {
                "partition_metadata_path": str(metadata_path),
                "partition_count": len(data.get("partitions", [])),
                "partition_metadata_cached": True,
            }
        except Exception:
            pass

    native_error = None
    try:
        from smart import native_runner

        if native_runner.cpp_native_file_runner_available():
            native_result = native_runner.run_coacd_partition_from_files(
                msh_path=tet_dir / "tetra.msh",
                coacd_dir=tet_dir / "coacd",
                output_path=metadata_path,
                mesh_id=mesh_id,
            )
            if native_result.get("status") == "success" and metadata_path.exists():
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                return {
                    "partition_metadata_path": str(metadata_path),
                    "partition_count": len(data.get("partitions", [])),
                    "partition_metadata_cached": False,
                    "partition_metadata_backend": "smart-cpp-native",
                    "partition_metadata_native": dict(native_result.get("metadata") or {}),
                }
    except Exception as exc:  # noqa: BLE001
        native_error = str(exc)

    if require_native:
        detail = f": {native_error}" if native_error else ""
        raise RuntimeError(
            "smart-cpp-native partition-coacd is required for cpp_native preseg"
            + detail
        )

    manifold_python = os.environ.get("SMART_MANIFOLD_PYTHON")
    manifold_candidates = []
    if manifold_python:
        manifold_candidates.append(Path(manifold_python).expanduser())
    manifold_candidates.extend(
        [
            REPO_ROOT / "smart" / "pymanifold_runtime",
            REPO_ROOT / "smart" / "vendor" / "manifold" / "build" / "bindings" / "python",
        ]
    )
    for candidate in manifold_candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break

    from smart.legacy.merging.src.utils.preseg import presegmentation
    import smart.pymesh_compat as pymesh

    tetmsh = pymesh.load_mesh(tet_dir / "tetra.msh")
    tetmsh.enable_connectivity()
    partitions = presegmentation(str(tet_dir), tetmsh, "coacd", "", mesh_id)
    clean_partitions = [
        sorted({int(value) for value in partition})
        for partition in partitions
        if partition
    ]
    payload = {
        "schema_version": 1,
        "source": "smart.pipeline.preseg.coacd_partition_metadata",
        "init_type": "coacd",
        "mesh_id": mesh_id,
        "part_obj_count": len(list((tet_dir / "coacd").glob("*.obj"))),
        "tet_count": int(len(tetmsh.voxels)),
        "partitions": clean_partitions,
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "partition_metadata_path": str(metadata_path),
        "partition_count": len(clean_partitions),
        "partition_metadata_cached": False,
        "partition_metadata_backend": "legacy_python",
        "partition_metadata_native_error": native_error,
    }


def _resolve_coacd_cli(cfg: dict[str, Any], stage_cfg: dict[str, Any]) -> str | None:
    candidates: list[str | Path | None] = [
        os.environ.get("SMART_COACD_BIN"),
        stage_cfg.get("coacd_bin"),
        cfg.get("tools", {}).get("coacd_bin"),
        "external/CoACD/python/package/bin/coacd",
        "external/CoACD/build/main",
    ]
    for configured in candidates:
        if not configured:
            continue
        path = repo_path(configured)
        if path is not None and path.exists():
            return str(path)
        if Path(str(configured)).is_absolute() and Path(str(configured)).exists():
            return str(configured)
    return shutil.which("coacd")


def _command_for_script_or_executable(path: str) -> list[str]:
    candidate = Path(path)
    if os.access(candidate, os.X_OK):
        return [str(candidate)]
    try:
        with candidate.open("rb") as handle:
            header = handle.read(128)
    except OSError:
        return [str(candidate)]
    if header.startswith(b"#!") and b"python" in header.lower():
        return [sys.executable, str(candidate)]
    return [str(candidate)]


def _run_coacd_cli_preseg(
    cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    surface: Path,
    out_dir: Path,
    *,
    dry_run: bool,
) -> tuple[bool, dict[str, Any], str | None]:
    coacd_bin = _resolve_coacd_cli(cfg, stage_cfg)
    if not coacd_bin:
        return False, {}, "CoACD C++ CLI not found"
    native_bin = native_executable_path()
    if native_bin is None:
        return False, {}, "smart-cpp-native is required to split CoACD CLI output"

    combined = out_dir / "coacd_parts.obj"
    kwargs = dict(stage_cfg.get("coacd", {}))
    command = [
        *_command_for_script_or_executable(coacd_bin),
        "-i",
        str(surface),
        "-o",
        str(combined),
        "-t",
        str(float(kwargs.get("threshold", 0.05))),
        "-c",
        str(int(kwargs.get("max_convex_hull", 64))),
        "-pm",
        str(kwargs.get("preprocess_mode", "auto")),
        "-pr",
        str(int(kwargs.get("preprocess_resolution", 50))),
        "-r",
        str(int(kwargs.get("resolution", 2000))),
        "-mn",
        str(int(kwargs.get("mcts_nodes", 20))),
        "-mi",
        str(int(kwargs.get("mcts_iterations", 150))),
        "-md",
        str(int(kwargs.get("mcts_max_depth", 3))),
        "--seed",
        str(int(kwargs.get("seed", 7777))),
    ]
    if bool(kwargs.get("pca", False)):
        command.append("--pca")
    if not bool(kwargs.get("merge", True)):
        command.append("-nm")
    if bool(kwargs.get("decimate", True)):
        command.append("-d")
    if dry_run:
        return True, {"coacd_cli_command": command, "splitter": str(native_bin)}, None

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "coacd_cli.log"
    result = run_command(
        command,
        timeout=float(stage_cfg.get("timeout_sec", 1800)),
        log_path=log_path,
    )
    if not result.ok or not combined.exists():
        error = _command_failure_summary("CoACD CLI", result)
        if result.stderr:
            error += f": {result.stderr.strip()[:500]}"
        return False, {"coacd_cli_command": command, "coacd_cli_log": str(log_path)}, error

    split = run_native_command(
        [
            "split-obj-parts",
            "--input",
            str(combined),
            "--output_dir",
            str(out_dir),
            "--prefix",
            "part",
        ]
    )
    if split.returncode != 0:
        return (
            False,
            {"coacd_cli_command": command, "coacd_cli_log": str(log_path)},
            "smart-cpp-native split-obj-parts failed: "
            + (split.stderr.strip() or split.stdout.strip()),
        )
    parts = sorted(path for path in out_dir.glob("part_*.obj"))
    return (
        True,
        {
            "backend": "coacd_cli",
            "coacd_cli": coacd_bin,
            "coacd_cli_command": command,
            "coacd_cli_log": str(log_path),
            "combined_obj": str(combined),
            "splitter": str(native_bin),
            "split_stdout": split.stdout.strip(),
            "parts": len(parts),
        },
        None,
    )


def run_preseg_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage_cfg = cfg.get("preseg", {})
    tet_dir = mesh_tetra_dir(cfg, category, mesh_id)
    surface = tet_dir / "tetra.msh__sf.obj"
    out_dir = tet_dir / "coacd"
    if out_dir.exists() and list(out_dir.glob("*.obj")) and not force:
        return _base_record(cfg, category, mesh_id, "preseg", started, status="skipped", output_path=out_dir)
    if not surface.exists():
        return _base_record(cfg, category, mesh_id, "preseg", started, status="skipped", error="missing tetra surface", output_path=out_dir)
    if stage_cfg.get("type", "coacd") != "coacd":
        return _base_record(cfg, category, mesh_id, "preseg", started, status="blocked", error="only CoACD presegmentation is wired in the official pipeline")
    if dry_run:
        return _base_record(cfg, category, mesh_id, "preseg", started, status="dry_run", output_path=out_dir)

    backend = str(stage_cfg.get("backend", "auto"))
    metadata: dict[str, Any] = {}
    cli_attempted = backend in {"auto", "coacd_cli", "cpp_native"}
    if cli_attempted:
        ok, cli_metadata, cli_error = _run_coacd_cli_preseg(
            cfg,
            stage_cfg,
            surface,
            out_dir,
            dry_run=dry_run,
        )
        metadata.update(cli_metadata)
        if ok:
            if stage_cfg.get("write_partition_metadata", True):
                try:
                    require_native_partition = (
                        str(stage_cfg.get("partition_metadata_backend", "cpp_native")) == "cpp_native"
                        and bool(stage_cfg.get("partition_metadata_required", False))
                    )
                    metadata.update(
                        _write_coacd_partition_metadata(
                            tet_dir,
                            mesh_id,
                            force=force,
                            require_native=require_native_partition,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    if bool(stage_cfg.get("partition_metadata_required", False)):
                        return _base_record(
                            cfg,
                            category,
                            mesh_id,
                            "preseg",
                            started,
                            status="failed",
                            output_path=out_dir,
                            error=str(exc),
                            metadata=metadata,
                        )
                    metadata["partition_metadata_error"] = str(exc)
            return _base_record(
                cfg,
                category,
                mesh_id,
                "preseg",
                started,
                status="success",
                output_path=out_dir,
                metadata=metadata,
            )
        if backend in {"coacd_cli", "cpp_native"} and bool(stage_cfg.get("coacd_cli_required", False)):
            return _base_record(
                cfg,
                category,
                mesh_id,
                "preseg",
                started,
                status="failed",
                output_path=out_dir,
                error=cli_error or "CoACD C++ CLI failed",
                metadata=metadata,
            )
        if cli_error:
            metadata["coacd_cli_error"] = cli_error

    try:
        import coacd  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        return _base_record(cfg, category, mesh_id, "preseg", started, status="blocked", output_path=out_dir, error=f"missing dependency: {exc.name}")

    try:
        used_native_obj_io = False
        try:
            if (
                smart_native is not None
                and smart_native.native_core_available()
                and hasattr(smart_native, "native_load_obj_mesh")
            ):
                vertices, faces = smart_native.native_load_obj_mesh(str(surface))
                coacd_mesh = coacd.Mesh(
                    np.asarray(vertices, dtype=np.float64),
                    np.asarray(faces, dtype=np.int32),
                )
                used_native_obj_io = True
            else:
                raise RuntimeError("native OBJ IO unavailable")
        except Exception:
            import trimesh  # type: ignore

            mesh = trimesh.load(surface, file_type="obj", process=False)
            coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        kwargs = dict(stage_cfg.get("coacd", {}))
        try:
            parts = coacd.run_coacd(coacd_mesh, **kwargs)
        except TypeError:
            parts = coacd.run_coacd(coacd_mesh)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, part in enumerate(parts):
            vertices, faces = part
            output_part = out_dir / f"part_{index:04d}.obj"
            if used_native_obj_io and hasattr(smart_native, "native_save_obj_mesh"):
                smart_native.native_save_obj_mesh(str(output_part), vertices, faces)
            else:
                import trimesh  # type: ignore

                part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                part_mesh.export(output_part)
    except Exception as exc:  # noqa: BLE001
        return _base_record(cfg, category, mesh_id, "preseg", started, status="failed", output_path=out_dir, error=str(exc))

    metadata.update(
        {
            "backend": "coacd_python",
            "parts": len(list(out_dir.glob("*.obj"))),
            "obj_io_backend": "cpp_native" if used_native_obj_io else "trimesh",
        }
    )
    if stage_cfg.get("write_partition_metadata", True):
        try:
            require_native_partition = (
                str(stage_cfg.get("partition_metadata_backend", "cpp_native")) == "cpp_native"
                and bool(stage_cfg.get("partition_metadata_required", False))
            )
            metadata.update(
                _write_coacd_partition_metadata(
                    tet_dir,
                    mesh_id,
                    force=force,
                    require_native=require_native_partition,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if bool(stage_cfg.get("partition_metadata_required", False)):
                return _base_record(
                    cfg,
                    category,
                    mesh_id,
                    "preseg",
                    started,
                    status="failed",
                    output_path=out_dir,
                    error=str(exc),
                    metadata=metadata,
                )
            metadata["partition_metadata_error"] = str(exc)

    return _base_record(
        cfg,
        category,
        mesh_id,
        "preseg",
        started,
        status="success",
        output_path=out_dir,
        metadata=metadata,
    )


def run_merge_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage_cfg = cfg.get("merge", {})
    tet_dir = mesh_tetra_dir(cfg, category, mesh_id)
    expected = greedy_segment_path(tet_dir, stage_cfg)
    if expected.exists() and not force:
        return _base_record(cfg, category, mesh_id, "merge", started, status="skipped", output_path=expected)
    if not (tet_dir / "tetra.msh").exists():
        return _base_record(cfg, category, mesh_id, "merge", started, status="skipped", error="missing tetra mesh", output_path=expected)
    if stage_cfg.get("init_type", "coacd") == "coacd" and not list((tet_dir / "coacd").glob("*.obj")):
        return _base_record(cfg, category, mesh_id, "merge", started, status="skipped", error="missing CoACD parts", output_path=expected)

    direct_result = None
    try:
        direct_result = _run_cpp_native_merge_file_runner(
            cfg,
            category,
            mesh_id,
            stage_cfg,
            expected,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        if stage_cfg.get("direct_file_runner_required", False):
            return _base_record(
                cfg,
                category,
                mesh_id,
                "merge",
                started,
                status="failed",
                output_path=expected,
                error=f"cpp_native merge file runner failed: {exc}",
            )
    if direct_result is None and _cpp_native_search_direct_enabled(stage_cfg) and stage_cfg.get(
        "direct_file_runner_required", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "merge",
            started,
            status="failed",
            output_path=expected,
            error=(
                "cpp_native merge file runner required but unavailable. "
                "Check smart-cpp-native build and CoACD partition metadata."
            ),
        )
    if direct_result is not None:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "merge",
            started,
            status=str(direct_result["status"]),
            output_path=direct_result.get("output_path"),
            command=list(direct_result.get("command") or []),
            metadata=dict(direct_result.get("metadata") or {}),
        )

    result_root = stage_root(cfg, "merge", category)
    log_path = workspace_path(cfg, "logs", "merge", category["name"], f"{mesh_id}.log")
    command = [
        _python(cfg),
        "run.py",
        "--run_type",
        "greedy",
        "--category",
        category["name"],
        "--result_path",
        str(result_root),
        "--path_to_msh_file",
        str(tetra_root(cfg, category)),
        "--path_to_bbox_file",
        str(stage_cfg.get("path_to_bbox_file", "")),
        "--data_gen_eps",
        str(stage_cfg.get("data_gen_eps", -1000000000.0)),
        "--merge_eps",
        str(stage_cfg.get("merge_eps", 0.02)),
        "--init_type",
        str(stage_cfg.get("init_type", "coacd")),
        "--final_k",
        str(stage_cfg.get("final_k", 0)),
        "--merge_backend",
        _legacy_merge_backend_name(stage_cfg.get("backend", "legacy_python")),
        "--worker",
        str(stage_cfg.get("worker", 0)),
        "--data_batch_size",
        "1",
        "--print_off",
        "--meshes",
        mesh_id[:10],
    ]
    if stage_cfg.get("tilted", True):
        command.append("--tilted")
    if stage_cfg.get("fast_merge", True):
        command.append("--fast_merge")
    if stage_cfg.get("only_nearby", False):
        command.append("--only_nearby")
    if stage_cfg.get("cpp_native_allow_tilted_axis", False):
        command.append("--cpp_native_merge_allow_tilted_axis")
    result = run_command(
        command,
        cwd=REPO_ROOT / "smart" / "legacy" / "merging",
        timeout=stage_cfg.get("timeout_sec"),
        env=_legacy_env(),
        log_path=log_path,
        dry_run=dry_run,
    )
    status = "dry_run" if dry_run else "success" if result.ok and expected.exists() else "failed"
    error = None if status in {"success", "dry_run"} else f"{_command_failure_summary('merge', result)}; expected {expected}"
    return _base_record(
        cfg,
        category,
        mesh_id,
        "merge",
        started,
        status=status,
        output_path=expected,
        log_path=log_path,
        command=command,
        error=error,
        metadata={"timeout_sec": stage_cfg.get("timeout_sec"), "elapsed_sec": result.elapsed_sec},
    )


def greedy_segment_path(tet_dir: Path, stage_cfg: dict[str, Any]) -> Path:
    final_k = int(stage_cfg.get("final_k", 0))
    init_type = stage_cfg.get("init_type", "coacd")
    init_suffix = f"_{init_type}" if init_type != "bsp" else ""
    merge_eps = float(stage_cfg.get("merge_eps", 0.02))
    fast_suffix = "_fm" if stage_cfg.get("fast_merge", True) else ""
    return tet_dir / f"greedy_segment{final_k}{init_suffix}_mgeps{merge_eps:g}{fast_suffix}.txt"


def _cpp_native_search_direct_enabled(stage_cfg: dict[str, Any]) -> bool:
    return str(stage_cfg.get("backend", "auto")) in {"cpp_native", "cpp_native_executable"} and bool(
        stage_cfg.get("direct_file_runner", True)
    )


def _legacy_merge_backend_name(backend: Any) -> str:
    backend_name = str(backend)
    if backend_name == "cpp_native_executable":
        return "cpp_native"
    return backend_name


def _run_cpp_native_merge_file_runner(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    stage_cfg: dict[str, Any],
    output_segment: Path,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    if not _cpp_native_search_direct_enabled(stage_cfg):
        return None
    try:
        from smart import native_runner
    except Exception:
        return None
    if not native_runner.cpp_native_file_runner_available():
        return None
    tet_dir = mesh_tetra_dir(cfg, category, mesh_id)
    configured = stage_cfg.get("partition_metadata_path")
    if configured:
        metadata_path = repo_path(configured)
    else:
        metadata_path = native_runner.find_partition_metadata(
            tet_dir, str(stage_cfg.get("init_type", "coacd"))
        )
    if metadata_path is None or not metadata_path.exists():
        return None
    return native_runner.run_merge_from_partitions_file(
        msh_path=tet_dir / "tetra.msh",
        partition_metadata_path=metadata_path,
        output_segment_path=output_segment,
        category=category["name"],
        merge_eps=float(stage_cfg.get("merge_eps", 0.02)),
        final_k=int(stage_cfg.get("final_k", 0)),
        tilted=bool(stage_cfg.get("tilted", True)),
        only_nearby=bool(stage_cfg.get("only_nearby", False)),
        dry_run=dry_run,
    )


def _cpp_native_refine_exp_name(
    cfg: dict[str, Any],
    category: dict[str, Any],
    stage_cfg: dict[str, Any],
    merge_cfg: dict[str, Any],
) -> str:
    name = (
        f"{tetra_root(cfg, category).name}_{stage_cfg.get('bbox_init', 'grd_merged')}"
        f"_cppnative_maxstep{int(stage_cfg.get('max_step', 2000))}"
        f"_covpen{int(stage_cfg.get('cover_penalty', 100))}"
        f"_acscale{int(stage_cfg.get('num_action_scale', 1))}"
        f"_acunit{float(stage_cfg.get('action_unit', 0.01)):.5g}"
        f"_mgeps{float(merge_cfg.get('merge_eps', 0.02)):.5g}_timing"
    )
    router_cfg = _refine_learned_router_config(stage_cfg)
    if router_cfg["enabled"]:
        name += f"_deepset_{router_cfg['profile']}"
    exp_tag = str(stage_cfg.get("exp_tag", "") or "").strip()
    return f"{name}_{exp_tag}" if exp_tag else name


def _refine_learned_router_config(stage_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = stage_cfg.get("learned_router", False)
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        policy = str(raw.get("policy", stage_cfg.get("learned_router_policy", "default")))
        profile = str(raw.get("profile", stage_cfg.get("learned_router_profile", "auto")))
        overrides_raw = raw.get("overrides", stage_cfg.get("learned_router_overrides", {}))
    else:
        enabled = bool(raw)
        policy = str(stage_cfg.get("learned_router_policy", "default"))
        profile = str(stage_cfg.get("learned_router_profile", "auto"))
        overrides_raw = stage_cfg.get("learned_router_overrides", {})
    overrides = dict(overrides_raw) if isinstance(overrides_raw, dict) else {}
    return {
        "enabled": enabled,
        "policy": policy,
        "profile": profile,
        "overrides": overrides,
    }


def _cpp_native_mcts_exp_name(
    cfg: dict[str, Any],
    category: dict[str, Any],
    stage_cfg: dict[str, Any],
) -> str:
    name = (
        f"{tetra_root(cfg, category).name}_{stage_cfg.get('bbox_init', 'bbox_direct')}"
        f"_cppnative_mcts{int(stage_cfg.get('mcts_iter', 3000))}"
        f"_maxstep{int(stage_cfg.get('max_step', 150))}"
        f"_covpen{int(stage_cfg.get('cover_penalty', 100))}"
        f"_acunit{float(stage_cfg.get('action_unit', 0.02)):.5g}_timing"
    )
    learned_prior_cfg = _mcts_learned_prior_config(stage_cfg)
    if learned_prior_cfg["enabled"]:
        name += f"_deepsetmcts_{learned_prior_cfg['mode']}"
    exp_tag = str(stage_cfg.get("exp_tag", "") or "").strip()
    return f"{name}_{exp_tag}" if exp_tag else name


def _mcts_learned_prior_config(stage_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = stage_cfg.get("learned_prior", False)
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        policy = str(raw.get("policy", stage_cfg.get("learned_prior_policy", "default")))
        mode = str(raw.get("mode", stage_cfg.get("learned_prior_mode", "guarded")))
        overrides_raw = raw.get("overrides", stage_cfg.get("learned_prior_overrides", {}))
        num_iter = raw.get("num_iter", stage_cfg.get("learned_prior_num_iter", None))
        max_step = raw.get("max_step", stage_cfg.get("learned_prior_max_step", None))
        transposition_table = raw.get(
            "transposition_table",
            stage_cfg.get("learned_prior_transposition_table", True),
        )
    else:
        enabled = bool(raw)
        policy = str(stage_cfg.get("learned_prior_policy", "default"))
        mode = str(stage_cfg.get("learned_prior_mode", "guarded"))
        overrides_raw = stage_cfg.get("learned_prior_overrides", {})
        num_iter = stage_cfg.get("learned_prior_num_iter", None)
        max_step = stage_cfg.get("learned_prior_max_step", None)
        transposition_table = stage_cfg.get("learned_prior_transposition_table", True)
    overrides = dict(overrides_raw) if isinstance(overrides_raw, dict) else {}
    return {
        "enabled": enabled,
        "policy": policy,
        "mode": mode,
        "num_iter": None if num_iter is None else int(num_iter),
        "max_step": None if max_step is None else int(max_step),
        "transposition_table": bool(transposition_table),
        "overrides": overrides,
    }


def _run_cpp_native_refine_file_runner(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    stage_cfg: dict[str, Any],
    merge_cfg: dict[str, Any],
    proposal_root: Path,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    if not _cpp_native_search_direct_enabled(stage_cfg):
        return None
    try:
        from smart import native_runner
    except Exception:
        return None
    learned_router_cfg = _refine_learned_router_config(stage_cfg)
    if bool(learned_router_cfg["enabled"]):
        if not dry_run and not native_runner.cpp_native_deepset_router_available():
            return None
    elif not native_runner.cpp_native_file_runner_available():
        return None
    bbox_init = str(stage_cfg.get("bbox_init", "grd_merged"))
    if bbox_init == "grd_merged":
        segment_path = greedy_segment_path(mesh_tetra_dir(cfg, category, mesh_id), merge_cfg)
        metadata_path = Path(str(segment_path) + ".bbox_params.json")
        strict_legacy_params = bool(stage_cfg.get("strict_legacy_bbox_params", True))
        if strict_legacy_params and not dry_run:
            metadata_path = native_runner.write_legacy_grd_bbox_params_from_segment(
                segment_path,
                mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh",
                tilted=bool(merge_cfg.get("tilted", True)),
            )
    elif bbox_init == "bbox_direct":
        metadata_path = native_runner.find_bbox_params_metadata(proposal_root, mesh_id)
        if metadata_path is None:
            return None
    else:
        return None
    if not metadata_path.exists():
        return None
    return native_runner.run_refine_from_files(
        msh_path=mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh",
        bbox_metadata_path=metadata_path,
        output_root=stage_root(cfg, "refine", category),
        exp_name=_cpp_native_refine_exp_name(cfg, category, stage_cfg, merge_cfg),
        mesh_id=mesh_id,
        category=category["name"],
        max_step=int(stage_cfg.get("max_step", 2000)),
        cover_penalty=float(stage_cfg.get("cover_penalty", 100)),
        action_unit=float(stage_cfg.get("action_unit", 0.01)),
        num_action_scale=native_runner.effective_native_num_action_scale(
            stage_cfg.get("num_action_scale", 1)
        ),
        stateful_union_cache=bool(stage_cfg.get("stateful_union_cache", True)),
        cache_capacity=int(stage_cfg.get("stateful_cache_capacity", 65536)),
        volume_method=str(stage_cfg.get("manifold_volume_method", "mesh")),
        native_recenter=bool(stage_cfg.get("native_recenter", False))
        or (bbox_init == "grd_merged" and bool(stage_cfg.get("strict_legacy_bbox_params", True))),
        learned_router=bool(learned_router_cfg["enabled"]),
        learned_router_profile=str(learned_router_cfg["profile"]),
        learned_router_policy=str(learned_router_cfg["policy"]),
        learned_router_overrides=dict(learned_router_cfg["overrides"]),
        dry_run=dry_run,
    )


def _run_cpp_native_mcts_file_runner(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    stage_cfg: dict[str, Any],
    bbox_input_root: Path,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    if not _cpp_native_search_direct_enabled(stage_cfg):
        return None
    learned_prior_cfg = _mcts_learned_prior_config(stage_cfg)
    uses_static_prior = (
        float(stage_cfg.get("action_prior_weight", 0.0) or 0.0) != 0.0
        or float(stage_cfg.get("puct_prior_weight", 0.0) or 0.0) != 0.0
        or float(stage_cfg.get("action_value_weight", 0.0) or 0.0) != 0.0
    )
    if uses_static_prior and not stage_cfg.get("action_prior_path") and not learned_prior_cfg["enabled"]:
        return None
    if (
        int(stage_cfg.get("action_prior_top_k", 0) or 0) > 0
        and not uses_static_prior
        and not learned_prior_cfg["enabled"]
    ):
        return None
    if str(stage_cfg.get("action_prior_select", "legacy") or "legacy") != "legacy":
        return None
    try:
        from smart import native_runner
    except Exception:
        return None
    if learned_prior_cfg["enabled"]:
        if not dry_run and not native_runner.cpp_native_deepset_mcts_prior_available():
            return None
    elif not native_runner.cpp_native_file_runner_available():
        return None
    metadata_path = native_runner.find_bbox_params_metadata(bbox_input_root, mesh_id)
    if metadata_path is None or not metadata_path.exists():
        return None
    if learned_prior_cfg["enabled"]:
        return native_runner.run_deepset_prior_mcts_from_files(
            msh_path=mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh",
            bbox_metadata_path=metadata_path,
            output_root=stage_root(cfg, "mcts", category),
            exp_name=_cpp_native_mcts_exp_name(cfg, category, stage_cfg),
            mesh_id=mesh_id,
            category=category["name"],
            mode=str(learned_prior_cfg["mode"]),
            policy=str(learned_prior_cfg["policy"]),
            num_iter=learned_prior_cfg["num_iter"],
            max_step=learned_prior_cfg["max_step"],
            cover_penalty=float(stage_cfg.get("cover_penalty", 100)),
            action_unit=float(stage_cfg.get("action_unit", 0.02)),
            num_action_scale=native_runner.effective_native_num_action_scale(
                stage_cfg.get("num_action_scale", 1)
            ),
            exp_weight=float(stage_cfg.get("exp_w", 0.001)),
            gamma=float(stage_cfg.get("gamma", 1.0)),
            seed=int(stage_cfg.get("seed", stage_cfg.get("cpp_rng_seed", 7777))),
            transposition_table=bool(learned_prior_cfg["transposition_table"]),
            transposition_table_size=int(stage_cfg.get("transposition_table_size", 8192)),
            stateful_union_cache=bool(stage_cfg.get("stateful_union_cache", True)),
            cache_capacity=int(stage_cfg.get("stateful_cache_capacity", 65536)),
            volume_method=str(stage_cfg.get("manifold_volume_method", "mesh")),
            overrides=dict(learned_prior_cfg["overrides"]),
            dry_run=dry_run,
        )
    return native_runner.run_mcts_from_files(
        msh_path=mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh",
        bbox_metadata_path=metadata_path,
        output_root=stage_root(cfg, "mcts", category),
        exp_name=_cpp_native_mcts_exp_name(cfg, category, stage_cfg),
        mesh_id=mesh_id,
        category=category["name"],
        mcts_iter=int(stage_cfg.get("mcts_iter", 3000)),
        max_step=int(stage_cfg.get("max_step", 150)),
        cover_penalty=float(stage_cfg.get("cover_penalty", 100)),
        action_unit=float(stage_cfg.get("action_unit", 0.02)),
        num_action_scale=native_runner.effective_native_num_action_scale(
            stage_cfg.get("num_action_scale", 1)
        ),
        exp_weight=float(stage_cfg.get("exp_w", 0.001)),
        gamma=float(stage_cfg.get("gamma", 1.0)),
        seed=int(stage_cfg.get("seed", stage_cfg.get("cpp_rng_seed", 7777))),
        transposition_table=bool(stage_cfg.get("transposition_table", False)),
        transposition_table_size=int(stage_cfg.get("transposition_table_size", 8192)),
        stateful_union_cache=bool(stage_cfg.get("stateful_union_cache", True)),
        cache_capacity=int(stage_cfg.get("stateful_cache_capacity", 65536)),
        volume_method=str(stage_cfg.get("manifold_volume_method", "mesh")),
        action_prior_path=repo_path(stage_cfg["action_prior_path"])
        if stage_cfg.get("action_prior_path")
        else None,
        action_prior_device=_resolve_action_prior_device(
            stage_cfg.get("action_prior_device", "json")
        ),
        action_prior_weight=float(stage_cfg.get("action_prior_weight", 0.0) or 0.0),
        puct_prior_weight=float(stage_cfg.get("puct_prior_weight", 0.0) or 0.0),
        action_value_weight=float(stage_cfg.get("action_value_weight", 0.0) or 0.0),
        action_prior_top_k=int(stage_cfg.get("action_prior_top_k", 0) or 0),
        native_recenter=bool(stage_cfg.get("native_recenter", False)),
        dry_run=dry_run,
    )


def _run_cpp_native_refine_mcts_file_runner(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    stage_cfg: dict[str, Any],
    merge_cfg: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    if not bool(stage_cfg.get("combined_refine", False)):
        return None
    if not _cpp_native_search_direct_enabled(stage_cfg):
        return None
    uses_static_prior = (
        float(stage_cfg.get("action_prior_weight", 0.0) or 0.0) != 0.0
        or float(stage_cfg.get("puct_prior_weight", 0.0) or 0.0) != 0.0
        or float(stage_cfg.get("action_value_weight", 0.0) or 0.0) != 0.0
    )
    if uses_static_prior and not stage_cfg.get("action_prior_path"):
        return None
    if int(stage_cfg.get("action_prior_top_k", 0) or 0) > 0 and not uses_static_prior:
        return None
    if str(stage_cfg.get("action_prior_select", "legacy") or "legacy") != "legacy":
        return None
    try:
        from smart import native_runner
    except Exception:
        return None
    if not native_runner.cpp_native_file_runner_available() and not dry_run:
        return None

    refine_cfg = cfg.get("refine", {})
    bbox_init = str(
        stage_cfg.get(
            "combined_refine_bbox_init",
            refine_cfg.get("bbox_init", "grd_merged"),
        )
    )
    if bbox_init == "grd_merged":
        segment_path = greedy_segment_path(mesh_tetra_dir(cfg, category, mesh_id), merge_cfg)
        metadata_path = Path(str(segment_path) + ".bbox_params.json")
        strict_legacy_params = bool(
            stage_cfg.get(
                "strict_legacy_bbox_params",
                refine_cfg.get("strict_legacy_bbox_params", True),
            )
        )
        if strict_legacy_params and not dry_run:
            metadata_path = native_runner.write_legacy_grd_bbox_params_from_segment(
                segment_path,
                mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh",
                tilted=bool(merge_cfg.get("tilted", True)),
            )
    elif bbox_init == "bbox_direct":
        proposal_root = repo_path(stage_cfg.get("combined_refine_path_to_bbox", ""))
        if proposal_root is None:
            proposal_root = repo_path(refine_cfg.get("path_to_bbox", ""))
        if proposal_root is None:
            return None
        metadata_path = native_runner.find_bbox_params_metadata(proposal_root, mesh_id)
        if metadata_path is None:
            return None
    else:
        return None
    if not metadata_path.exists() and not dry_run:
        return None

    refine_exp = _cpp_native_refine_exp_name(cfg, category, refine_cfg, merge_cfg)
    mcts_exp = _cpp_native_mcts_exp_name(cfg, category, stage_cfg)
    refine_mesh_root = stage_root(cfg, "refine", category) / refine_exp / "result" / "updated0" / mesh_id
    mcts_mesh_root = stage_root(cfg, "mcts", category) / mcts_exp / "result" / "updated0" / mesh_id
    refine_bbox_dir = refine_mesh_root / "bboxs_steps0"
    mcts_bbox_dir = mcts_mesh_root / "bboxs_steps0"

    result = native_runner.run_refine_mcts_from_files(
        msh_path=mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh",
        bbox_metadata_path=metadata_path,
        refine_output_dir=refine_bbox_dir,
        mcts_output_dir=mcts_bbox_dir,
        category=category["name"],
        mesh_id=mesh_id,
        refine_max_step=int(refine_cfg.get("max_step", 2000)),
        mcts_iter=int(stage_cfg.get("mcts_iter", 3000)),
        mcts_max_step=int(stage_cfg.get("max_step", 150)),
        cover_penalty=float(stage_cfg.get("cover_penalty", refine_cfg.get("cover_penalty", 100))),
        refine_action_unit=float(refine_cfg.get("action_unit", 0.01)),
        mcts_action_unit=float(stage_cfg.get("action_unit", 0.02)),
        num_action_scale=native_runner.effective_native_num_action_scale(
            stage_cfg.get("num_action_scale", refine_cfg.get("num_action_scale", 1))
        ),
        exp_weight=float(stage_cfg.get("exp_w", 0.001)),
        gamma=float(stage_cfg.get("gamma", 1.0)),
        seed=int(stage_cfg.get("seed", stage_cfg.get("cpp_rng_seed", 7777))),
        transposition_table=bool(stage_cfg.get("transposition_table", False)),
        transposition_table_size=int(stage_cfg.get("transposition_table_size", 8192)),
        stateful_union_cache=bool(stage_cfg.get("stateful_union_cache", refine_cfg.get("stateful_union_cache", True))),
        cache_capacity=int(stage_cfg.get("stateful_cache_capacity", refine_cfg.get("stateful_cache_capacity", 65536))),
        volume_method=str(stage_cfg.get("manifold_volume_method", refine_cfg.get("manifold_volume_method", "mesh"))),
        action_prior_path=repo_path(stage_cfg["action_prior_path"])
        if stage_cfg.get("action_prior_path")
        else None,
        action_prior_device=_resolve_action_prior_device(
            stage_cfg.get("action_prior_device", "json")
        ),
        action_prior_weight=float(stage_cfg.get("action_prior_weight", 0.0) or 0.0),
        puct_prior_weight=float(stage_cfg.get("puct_prior_weight", 0.0) or 0.0),
        action_value_weight=float(stage_cfg.get("action_value_weight", 0.0) or 0.0),
        action_prior_top_k=int(stage_cfg.get("action_prior_top_k", 0) or 0),
        native_recenter=bool(stage_cfg.get("native_recenter", refine_cfg.get("native_recenter", False)))
        or (bbox_init == "grd_merged" and strict_legacy_params),
        dry_run=dry_run,
    )
    if not dry_run:
        metadata = dict(result.get("metadata") or {})
        elapsed = float(metadata.get("elapsed_sec_wall", 0.0) or 0.0)
        refine_mesh_root.mkdir(parents=True, exist_ok=True)
        mcts_mesh_root.mkdir(parents=True, exist_ok=True)
        exported = len(list(mcts_bbox_dir.glob("bbox*.obj")))
        refine_exported = len(list(refine_bbox_dir.glob("bbox*.obj")))
        refine_mesh_root.joinpath("time.txt").write_text(
            f"{refine_exported}\n{refine_exported}\n{elapsed}\n",
            encoding="utf-8",
        )
        mcts_mesh_root.joinpath("time.txt").write_text(
            f"{refine_exported}\n{exported}\n{elapsed}\n",
            encoding="utf-8",
        )
        mcts_mesh_root.joinpath("native_stats.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def run_refine_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage_cfg = cfg.get("refine", {})
    merge_cfg = cfg.get("merge", {})
    existing = latest_bbox_dir(stage_root(cfg, "refine", category), mesh_id)
    if existing and not force:
        return _base_record(cfg, category, mesh_id, "refine", started, status="skipped", output_path=existing)
    bbox_init = str(stage_cfg.get("bbox_init", "grd_merged"))
    proposal_bbox_root = str(stage_cfg.get("path_to_bbox", "") or "")
    if bbox_init == "grd_merged" and not greedy_segment_path(mesh_tetra_dir(cfg, category, mesh_id), merge_cfg).exists():
        return _base_record(cfg, category, mesh_id, "refine", started, status="skipped", error="missing greedy merged segment")
    if bbox_init == "bbox_direct" and proposal_bbox_root:
        proposal_root = repo_path(proposal_bbox_root)
        if not (proposal_root / "result").exists():
            return _base_record(
                cfg,
                category,
                mesh_id,
                "refine",
                started,
                status="skipped",
                error=f"missing bbox_direct proposal result directory: {proposal_root / 'result'}",
            )
    else:
        proposal_root = Path("")

    direct_result = None
    try:
        direct_result = _run_cpp_native_refine_file_runner(
            cfg,
            category,
            mesh_id,
            stage_cfg,
            merge_cfg,
            proposal_root,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        if stage_cfg.get("direct_file_runner_required", False):
            return _base_record(
                cfg,
                category,
                mesh_id,
                "refine",
                started,
                status="failed",
                error=f"cpp_native file runner failed: {exc}",
            )
    if direct_result is None and _cpp_native_search_direct_enabled(stage_cfg) and stage_cfg.get(
        "direct_file_runner_required", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "refine",
            started,
            status="failed",
            error=(
                "cpp_native refine file runner required but unavailable. "
                "Check smart-cpp-native build and bbox_params.json from merge/proposal."
            ),
        )
    if direct_result is not None:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "refine",
            started,
            status=str(direct_result["status"]),
            output_path=direct_result.get("output_path"),
            command=list(direct_result.get("command") or []),
            metadata=dict(direct_result.get("metadata") or {}),
        )

    result_root = stage_root(cfg, "refine", category)
    log_path = workspace_path(cfg, "logs", "refine", category["name"], f"{mesh_id}.log")
    command = [
        _python(cfg),
        "run.py",
        "--run_type",
        "greedy",
        "--result_path",
        str(result_root),
        "--path_to_msh_file",
        str(tetra_root(cfg, category)),
        "--category",
        category["name"],
        "--path_to_bbox",
        str(proposal_root) if proposal_bbox_root else "",
        "--bbox_init",
        bbox_init,
        "--init_type",
        str(merge_cfg.get("init_type", "coacd")),
        "--merge_eps",
        str(merge_cfg.get("merge_eps", 0.02)),
        "--max_step",
        str(stage_cfg.get("max_step", 2000)),
        "--greedy_backend",
        str(stage_cfg.get("backend", "auto")),
        "--cover_penalty",
        str(stage_cfg.get("cover_penalty", 100)),
        "--score_cache_size",
        str(stage_cfg.get("score_cache_size", 4096)),
        "--candidate_backend",
        str(stage_cfg.get("candidate_backend", "exact")),
        "--candidate_top_k",
        str(stage_cfg.get("candidate_top_k", 8)),
        "--candidate_pruned_max_aspect_mean",
        str(stage_cfg.get("candidate_pruned_max_aspect_mean", 0.0)),
        "--candidate_pruned_min_fill_ratio",
        str(stage_cfg.get("candidate_pruned_min_fill_ratio", 0.0)),
        "--reward_backend",
        str(stage_cfg.get("reward_backend", "manifold")),
        "--manifold_volume_method",
        str(stage_cfg.get("manifold_volume_method", "mesh")),
        "--stateful_cache_capacity",
        str(stage_cfg.get("stateful_cache_capacity", 65536)),
        "--tet_clipping_max_boxes",
        str(stage_cfg.get("tet_clipping_max_boxes", 12)),
        "--action_unit",
        str(stage_cfg.get("action_unit", 0.01)),
        "--num_action_scale",
        str(stage_cfg.get("num_action_scale", 1)),
        "--worker",
        str(stage_cfg.get("worker", 0)),
        "--data_batch_size",
        "1",
        "--print_off",
        "--meshes",
        mesh_id[:10],
    ]
    if not stage_cfg.get("render_initial", False):
        command.append("--skip_initial_render")
    if not stage_cfg.get("render_partition", False):
        command.append("--skip_render_partition")
    if not stage_cfg.get("summary_metrics", False):
        command.append("--skip_summary_metrics")
    if not stage_cfg.get("stateful_union_cache", True):
        command.append("--no-stateful_union_cache")
    if not stage_cfg.get("candidate_require_exact_fallback", True):
        command.append("--no-candidate_require_exact_fallback")
    if stage_cfg.get("candidate_bypass_on_exact_fallback", False):
        command.append("--candidate_bypass_on_exact_fallback")
    if stage_cfg.get("candidate_pruned_categories"):
        command.extend(
            ["--candidate_pruned_categories", str(stage_cfg.get("candidate_pruned_categories", ""))]
        )
    if not stage_cfg.get("cache_initial_bbox_state", True):
        command.append("--no-cache_initial_bbox_state")
    if stage_cfg.get("stateful_unscored_apply", False):
        command.append("--stateful_unscored_apply")
    if stage_cfg.get("fused_rollout_step", False):
        command.append("--mcts_fused_rollout_step")
    if stage_cfg.get("native_axis_rollout_step", False):
        command.append("--mcts_native_axis_rollout_step")
    if stage_cfg.get("native_axis_rollout_segment", False):
        command.append("--mcts_native_axis_rollout_segment")
    if stage_cfg.get("trace_actions_path"):
        command.extend(["--trace_actions_path", str(repo_path(stage_cfg["trace_actions_path"]))])
    if stage_cfg.get("candidate_trace_path"):
        command.extend(
            [
                "--candidate_trace_path",
                str(repo_path(stage_cfg["candidate_trace_path"])),
                "--candidate_trace_top_k",
                str(stage_cfg.get("candidate_trace_top_k", 0)),
                "--candidate_trace_node_top_k",
                str(stage_cfg.get("candidate_trace_node_top_k", 0)),
            ]
        )
    if stage_cfg.get("action_prior_path"):
        command.extend(["--action_prior_path", str(repo_path(stage_cfg["action_prior_path"]))])
        command.extend(["--action_prior_device", _resolve_action_prior_device(stage_cfg.get("action_prior_device", "json"))])
        command.extend(["--action_prior_weight", str(stage_cfg.get("action_prior_weight", 0.0))])
        command.extend(["--action_value_weight", str(stage_cfg.get("action_value_weight", 0.0))])
        command.extend(["--action_prior_top_k", str(stage_cfg.get("action_prior_top_k", 0))])
    if stage_cfg.get("exp_tag"):
        command.extend(["--mcts_exp_tag", str(stage_cfg["exp_tag"])])
    if merge_cfg.get("tilted", True):
        command.append("--tilted")
    if merge_cfg.get("fast_merge", True):
        command.append("--fast_merge")
    result = run_command(
        command,
        cwd=REPO_ROOT / "smart" / "legacy" / "refine",
        timeout=stage_cfg.get("timeout_sec"),
        env=_legacy_env(),
        log_path=log_path,
        dry_run=dry_run,
    )
    output = latest_bbox_dir(result_root, mesh_id, since=started)
    status = "dry_run" if dry_run else "success" if result.ok and output else "failed"
    error = None if status in {"success", "dry_run"} else _command_failure_summary("refine", result)
    return _base_record(
        cfg,
        category,
        mesh_id,
        "refine",
        started,
        status=status,
        output_path=output,
        log_path=log_path,
        command=command,
        error=error,
        metadata={"timeout_sec": stage_cfg.get("timeout_sec"), "elapsed_sec": result.elapsed_sec},
    )


def run_mcts_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage_cfg = cfg.get("mcts", {})
    merge_cfg = cfg.get("merge", {})
    mcts_backend = str(stage_cfg.get("backend", "auto"))
    if mcts_backend in (NATIVE_SEARCH_BACKENDS - {"cpp_native"}) and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                f"mcts.backend={NATIVE_SEARCH_BACKEND_LABEL} can change MCTS search order. "
                "Use backend=auto for paper-compatible exact runs, or set "
                "mcts.allow_search_order_changes=true for experimental runs."
            ),
        )
    if stage_cfg.get("transposition_table", False) and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                "mcts.transposition_table changes search order. Keep it disabled "
                "for exact legacy metric compatibility, or set "
                "mcts.allow_search_order_changes=true for experimental runs."
            ),
        )
    if stage_cfg.get("cpp_rng", False) and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                "mcts.cpp_rng changes the MCTS random sequence. Keep it disabled "
                "for paper-compatible runs, or set "
                "mcts.allow_search_order_changes=true for experimental runs."
            ),
        )
    if float(stage_cfg.get("action_prior_weight", 0.0) or 0.0) != 0.0 and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                "mcts.action_prior_weight changes MCTS search order. Keep it at "
                "0 for paper-compatible runs, or set "
                "mcts.allow_search_order_changes=true for research runs."
            ),
        )
    if float(stage_cfg.get("puct_prior_weight", 0.0) or 0.0) != 0.0 and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                "mcts.puct_prior_weight changes MCTS search order. Keep it at "
                "0 for paper-compatible runs, or set "
                "mcts.allow_search_order_changes=true for research runs."
            ),
        )
    if float(stage_cfg.get("action_value_weight", 0.0) or 0.0) != 0.0 and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                "mcts.action_value_weight changes MCTS search order. Keep it at "
                "0 for paper-compatible runs, or set "
                "mcts.allow_search_order_changes=true for research runs."
            ),
        )
    if int(stage_cfg.get("action_prior_top_k", 0) or 0) > 0 and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                "mcts.action_prior_top_k prunes the MCTS search tree. Keep it "
                "0 for paper-compatible runs, or set "
                "mcts.allow_search_order_changes=true for research runs."
            ),
        )
    if str(stage_cfg.get("action_prior_select", "legacy") or "legacy") != "legacy" and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                "mcts.action_prior_select changes MCTS untried-action expansion "
                "order. Keep it at legacy for paper-compatible runs, or set "
                "mcts.allow_search_order_changes=true for research runs."
            ),
        )
    if stage_cfg.get("escape_policy", False) and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="blocked",
            error=(
                "mcts.escape_policy changes MCTS search order to explore "
                "local-minimum escape branches. Keep it disabled for "
                "paper-compatible runs, or set "
                "mcts.allow_search_order_changes=true for research runs."
            ),
        )
    existing = latest_bbox_dir(stage_root(cfg, "mcts", category), mesh_id)
    if existing and not force:
        return _base_record(cfg, category, mesh_id, "mcts", started, status="skipped", output_path=existing)
    combined_result = None
    try:
        combined_result = _run_cpp_native_refine_mcts_file_runner(
            cfg,
            category,
            mesh_id,
            stage_cfg,
            merge_cfg,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        if stage_cfg.get("direct_file_runner_required", False):
            return _base_record(
                cfg,
                category,
                mesh_id,
                "mcts",
                started,
                status="failed",
                error=f"cpp_native refine-mcts file runner failed: {exc}",
            )
    if (
        combined_result is None
        and bool(stage_cfg.get("combined_refine", False))
        and _cpp_native_search_direct_enabled(stage_cfg)
        and stage_cfg.get("direct_file_runner_required", False)
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="failed",
            error=(
                "cpp_native refine-mcts file runner required but unavailable. "
                "Check smart-cpp-native build and merged bbox_params.json."
            ),
        )
    if combined_result is not None:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status=str(combined_result["status"]),
            output_path=combined_result.get("output_path"),
            command=list(combined_result.get("command") or []),
            metadata=dict(combined_result.get("metadata") or {}),
        )
    configured_bbox_root = str(stage_cfg.get("path_to_bbox", "") or "")
    if configured_bbox_root:
        bbox_input_root = repo_path(configured_bbox_root)
        if not (bbox_input_root / "result").exists():
            return _base_record(
                cfg,
                category,
                mesh_id,
                "mcts",
                started,
                status="skipped",
                error=f"missing bbox_direct proposal result directory: {bbox_input_root / 'result'}",
            )
    else:
        refine_exp = latest_exp_dir_for_bbox(stage_root(cfg, "refine", category), mesh_id)
        if refine_exp is None:
            return _base_record(cfg, category, mesh_id, "mcts", started, status="skipped", error="missing refine bbox output")
        bbox_input_root = refine_exp

    direct_result = None
    try:
        direct_result = _run_cpp_native_mcts_file_runner(
            cfg,
            category,
            mesh_id,
            stage_cfg,
            bbox_input_root,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        if stage_cfg.get("direct_file_runner_required", False):
            return _base_record(
                cfg,
                category,
                mesh_id,
                "mcts",
                started,
                status="failed",
                error=f"cpp_native file runner failed: {exc}",
            )
    if direct_result is None and _cpp_native_search_direct_enabled(stage_cfg) and stage_cfg.get(
        "direct_file_runner_required", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status="failed",
            error=(
                "cpp_native mcts file runner required but unavailable. "
                "Check smart-cpp-native build and bbox_params.json from refine/proposal."
            ),
        )
    if direct_result is not None:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "mcts",
            started,
            status=str(direct_result["status"]),
            output_path=direct_result.get("output_path"),
            command=list(direct_result.get("command") or []),
            metadata=dict(direct_result.get("metadata") or {}),
        )

    result_root = stage_root(cfg, "mcts", category)
    log_path = workspace_path(cfg, "logs", "mcts", category["name"], f"{mesh_id}.log")
    command = [
        _python(cfg),
        "run.py",
        "--run_type",
        "mcts",
        "--result_path",
        str(result_root),
        "--path_to_msh_file",
        str(tetra_root(cfg, category)),
        "--category",
        category["name"],
        "--path_to_bbox",
        str(bbox_input_root),
        "--bbox_init",
        str(stage_cfg.get("bbox_init", "bbox_direct")),
        "--init_type",
        str(merge_cfg.get("init_type", "coacd")),
        "--merge_eps",
        str(merge_cfg.get("merge_eps", 0.02)),
        "--max_step",
        str(stage_cfg.get("max_step", 150)),
        "--cover_penalty",
        str(stage_cfg.get("cover_penalty", 100)),
        "--score_cache_size",
        str(stage_cfg.get("score_cache_size", 4096)),
        "--candidate_backend",
        str(stage_cfg.get("candidate_backend", "exact")),
        "--candidate_top_k",
        str(stage_cfg.get("candidate_top_k", 8)),
        "--candidate_pruned_max_aspect_mean",
        str(stage_cfg.get("candidate_pruned_max_aspect_mean", 0.0)),
        "--candidate_pruned_min_fill_ratio",
        str(stage_cfg.get("candidate_pruned_min_fill_ratio", 0.0)),
        "--reward_backend",
        str(stage_cfg.get("reward_backend", "manifold")),
        "--manifold_volume_method",
        str(stage_cfg.get("manifold_volume_method", "mesh")),
        "--stateful_cache_capacity",
        str(stage_cfg.get("stateful_cache_capacity", 65536)),
        "--tet_clipping_max_boxes",
        str(stage_cfg.get("tet_clipping_max_boxes", 12)),
        "--action_unit",
        str(stage_cfg.get("action_unit", 0.02)),
        "--mcts_iter",
        str(stage_cfg.get("mcts_iter", 3000)),
        "--seed",
        str(stage_cfg.get("seed", 7777)),
        "--mcts_no_reward_stop_after",
        str(stage_cfg.get("no_reward_stop_after", 101)),
        "--mcts_backend",
        str(stage_cfg.get("backend", "auto")),
        "--exp_w",
        str(stage_cfg.get("exp_w", 0.001)),
        "--skip_rate",
        str(stage_cfg.get("skip_rate", 0.9)),
        "--log_path",
        str(stage_cfg.get("tensorboard_log_path", "")),
        "--worker",
        str(stage_cfg.get("worker", 0)),
        "--data_batch_size",
        "1",
        "--print_off",
        "--meshes",
        mesh_id[:10],
    ]
    if not stage_cfg.get("render_initial", False):
        command.append("--skip_initial_render")
    if not stage_cfg.get("render_partition", False):
        command.append("--skip_render_partition")
    if not stage_cfg.get("summary_metrics", False):
        command.append("--skip_summary_metrics")
    if not stage_cfg.get("stateful_union_cache", True):
        command.append("--no-stateful_union_cache")
    if not stage_cfg.get("candidate_require_exact_fallback", True):
        command.append("--no-candidate_require_exact_fallback")
    if stage_cfg.get("candidate_bypass_on_exact_fallback", False):
        command.append("--candidate_bypass_on_exact_fallback")
    if stage_cfg.get("candidate_pruned_categories"):
        command.extend(
            ["--candidate_pruned_categories", str(stage_cfg.get("candidate_pruned_categories", ""))]
        )
    if not stage_cfg.get("cache_initial_bbox_state", True):
        command.append("--no-cache_initial_bbox_state")
    if stage_cfg.get("stateful_unscored_apply", False):
        command.append("--stateful_unscored_apply")
    if stage_cfg.get("fused_rollout_step", False):
        command.append("--mcts_fused_rollout_step")
    if stage_cfg.get("native_axis_rollout_step", False):
        command.append("--mcts_native_axis_rollout_step")
    if stage_cfg.get("native_axis_rollout_segment", False):
        command.append("--mcts_native_axis_rollout_segment")
    if stage_cfg.get("cpp_rng", False):
        command.extend(
            [
                "--mcts_cpp_rng",
                "--mcts_cpp_rng_seed",
                str(stage_cfg.get("cpp_rng_seed", stage_cfg.get("seed", 7777))),
            ]
        )
    if stage_cfg.get("trace_actions_path"):
        command.extend(["--trace_actions_path", str(repo_path(stage_cfg["trace_actions_path"]))])
    if stage_cfg.get("candidate_trace_path"):
        command.extend(
            [
                "--candidate_trace_path",
                str(repo_path(stage_cfg["candidate_trace_path"])),
                "--candidate_trace_top_k",
                str(stage_cfg.get("candidate_trace_top_k", 0)),
                "--candidate_trace_node_top_k",
                str(stage_cfg.get("candidate_trace_node_top_k", 0)),
            ]
        )
    if stage_cfg.get("action_prior_path"):
        command.extend(["--action_prior_path", str(repo_path(stage_cfg["action_prior_path"]))])
        command.extend(["--action_prior_device", _resolve_action_prior_device(stage_cfg.get("action_prior_device", "json"))])
        command.extend(["--action_prior_weight", str(stage_cfg.get("action_prior_weight", 0.0))])
        command.extend(["--puct_prior_weight", str(stage_cfg.get("puct_prior_weight", 0.0))])
        command.extend(["--action_value_weight", str(stage_cfg.get("action_value_weight", 0.0))])
        command.extend(["--action_prior_top_k", str(stage_cfg.get("action_prior_top_k", 0))])
        command.extend(["--action_prior_select", str(stage_cfg.get("action_prior_select", "legacy"))])
        command.extend(
            [
                "--action_prior_select_temperature",
                str(stage_cfg.get("action_prior_select_temperature", 1.0)),
            ]
        )
        if not stage_cfg.get("action_prior_keep_upper", True):
            command.append("--no-action_prior_keep_upper")
    if stage_cfg.get("escape_policy", False):
        command.append("--escape_policy")
        command.extend(["--escape_after_no_update", str(stage_cfg.get("escape_after_no_update", 20))])
        command.extend(["--escape_action_top_k", str(stage_cfg.get("escape_action_top_k", 0))])
        command.extend(["--escape_probability", str(stage_cfg.get("escape_probability", 0.5))])
    if stage_cfg.get("exp_tag"):
        command.extend(["--mcts_exp_tag", str(stage_cfg["exp_tag"])])
    if merge_cfg.get("tilted", True):
        command.append("--tilted")
    if merge_cfg.get("fast_merge", True):
        command.append("--fast_merge")
    if stage_cfg.get("grdexp", True):
        command.append("--grdexp")
    if stage_cfg.get("pns", True):
        command.append("--pns")
    if stage_cfg.get("transposition_table", False):
        command.extend(
            [
                "--transposition_table",
                "--transposition_table_size",
                str(stage_cfg.get("transposition_table_size", 8192)),
            ]
        )
    result = run_command(
        command,
        cwd=REPO_ROOT / "smart" / "legacy" / "refine",
        timeout=stage_cfg.get("timeout_sec"),
        env=_legacy_env(),
        log_path=log_path,
        dry_run=dry_run,
    )
    output = latest_bbox_dir(result_root, mesh_id, since=started)
    timeout_output = bool(
        not dry_run
        and result.timed_out
        and output
        and stage_cfg.get("accept_timeout_output", True)
    )
    status = (
        "dry_run"
        if dry_run
        else "success"
        if result.ok and output
        else "success_timeout_output"
        if timeout_output
        else "failed"
    )
    error = None if status in {"success", "dry_run", "success_timeout_output"} else _command_failure_summary("mcts", result)
    return _base_record(
        cfg,
        category,
        mesh_id,
        "mcts",
        started,
        status=status,
        output_path=output,
        log_path=log_path,
        command=command,
        error=error,
        metadata={
            "timeout_sec": stage_cfg.get("timeout_sec"),
            "elapsed_sec": result.elapsed_sec,
            "timeout_output_accepted": timeout_output,
        },
    )


def _macro_skill_exp_name(
    cfg: dict[str, Any],
    category: dict[str, Any],
    stage_cfg: dict[str, Any],
) -> str:
    preset = str(stage_cfg.get("quality_preset", "balanced") or "balanced")
    repeat = str(stage_cfg.get("repeat_mode", "guarded_variable") or "guarded_variable")
    input_stage = str(stage_cfg.get("input_stage", "mcts") or "mcts")
    name = (
        f"{tetra_root(cfg, category).name}_{input_stage}_macro_skill"
        f"_{preset}_{repeat}"
        f"_top{int(stage_cfg.get('top_k', 5) or 5)}"
        f"_steps{int(stage_cfg.get('max_steps', 16) or 16)}"
    )
    exact_budget = stage_cfg.get("exact_budget")
    if exact_budget is not None:
        name += f"_budget{int(exact_budget)}"
    memory_pool = int(stage_cfg.get("macro_memory_pool_size", 5) or 5)
    if memory_pool != 5:
        name += f"_mempool{memory_pool}"
    planner_cfg = stage_cfg.get("planner", {}) if isinstance(stage_cfg.get("planner", {}), dict) else {}
    if planner_cfg.get("enabled", False):
        schedule = "-".join(str(item) for item in planner_cfg.get("profile_schedule", ["balanced"]) if str(item))
        name += f"_planner_r{int(planner_cfg.get('max_rounds', 3) or 3)}"
        if schedule:
            name += f"_{schedule}"
    if not stage_cfg.get("native_executor", True):
        name += "_pythonexec"
    exp_tag = str(stage_cfg.get("exp_tag", "") or "").strip()
    return f"{name}_{exp_tag}" if exp_tag else name


def _macro_skill_stage_command(
    *,
    msh_path: Path,
    metadata_path: Path,
    category_name: str,
    stage_cfg: dict[str, Any],
    result_json: Path,
    bbox_out: Path,
) -> list[str]:
    command = [
        "smart",
        "macro-skill",
        "--msh",
        str(msh_path),
        "--bbox-metadata",
        str(metadata_path),
        "--category",
        category_name,
        "--cover-penalty",
        str(stage_cfg.get("cover_penalty", 100)),
        "--pen-rate",
        str(stage_cfg.get("pen_rate", 1.0)),
        "--num-action-scale",
        str(stage_cfg.get("num_action_scale", 2)),
        "--action-unit",
        str(stage_cfg.get("action_unit", 0.01)),
        "--candidate-count",
        str(stage_cfg.get("candidate_count", 256)),
        "--top-k",
        str(stage_cfg.get("top_k", 5)),
        "--macro-memory-pool-size",
        str(stage_cfg.get("macro_memory_pool_size", 5)),
        "--max-steps",
        str(stage_cfg.get("max_steps", 16)),
        "--quality-preset",
        str(stage_cfg.get("quality_preset", "balanced")),
        "--repeat-mode",
        str(stage_cfg.get("repeat_mode", "guarded_variable")),
        "--volume-method",
        str(stage_cfg.get("volume_method", "mesh")),
        "--cache-capacity",
        str(stage_cfg.get("stateful_cache_capacity", 65536)),
        "--output",
        str(result_json),
        "--output-bbox-dir",
        str(bbox_out),
        "--json",
    ]
    planner_cfg = stage_cfg.get("planner", {}) if isinstance(stage_cfg.get("planner", {}), dict) else {}
    if planner_cfg.get("enabled", False):
        command.append("--planner")
        command.extend(["--planner-max-rounds", str(planner_cfg.get("max_rounds", 3))])
        schedule = [str(item) for item in planner_cfg.get("profile_schedule", ["balanced"]) if str(item)]
        if schedule:
            command.append("--planner-profile-schedule")
            command.extend(schedule)
        command.extend(["--planner-min-round-delta", str(planner_cfg.get("min_round_delta", 1.0e-12))])
        if not planner_cfg.get("stop_after_noop", True):
            command.append("--no-planner-stop-after-noop")
    if stage_cfg.get("exact_budget") is not None:
        command.extend(["--exact-budget", str(stage_cfg["exact_budget"])])
    if not stage_cfg.get("stateful_union_cache", True):
        command.append("--no-stateful-union-cache")
    if not stage_cfg.get("native_executor", True):
        command.append("--no-native-executor")
    if stage_cfg.get("target_aware_execution", False):
        command.append("--target-aware-execution")
    return command


def run_macro_skill_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    """Run the packaged exact-validated macro-skill controller as a stage.

    The stage is an opt-in post-MCTS/post-refine polish pass.  It exports a bbox
    directory even when no skill is accepted, because the controller restores the
    original exact state before returning.  Downstream render/local-refine stages
    can therefore consume ``macro_skill`` without special fallback logic.
    """

    started = time.time()
    stage_cfg = cfg.get("macro_skill", {})
    existing = latest_bbox_dir(stage_root(cfg, "macro_skill", category), mesh_id)
    if existing and not force:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "macro_skill",
            started,
            status="skipped",
            output_path=existing,
            metadata={"safe_noop_fallback": True},
        )

    input_stage = str(stage_cfg.get("input_stage", "mcts") or "mcts")
    input_root = stage_root(cfg, input_stage, category)
    input_exp = latest_exp_dir_for_bbox(input_root, mesh_id)

    try:
        from smart import native_runner
    except Exception as exc:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "macro_skill",
            started,
            status="blocked",
            error=f"native runner unavailable: {exc}",
            metadata={"input_stage": input_stage, "safe_noop_fallback": True},
        )

    metadata_path = native_runner.find_bbox_params_metadata(input_exp, mesh_id) if input_exp is not None else None
    if metadata_path is None:
        native_bbox = native_pipeline_bbox_dir(cfg, category, mesh_id, input_stage)
        if native_bbox is not None:
            metadata_path = native_bbox / "bbox_params.json"
    if metadata_path is None:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "macro_skill",
            started,
            status="skipped",
            error=f"missing {input_stage} bbox output",
            metadata={"input_stage": input_stage, "safe_noop_fallback": True},
        )
    if metadata_path is None or not metadata_path.exists():
        return _base_record(
            cfg,
            category,
            mesh_id,
            "macro_skill",
            started,
            status="skipped",
            error=f"missing bbox_params.json for {input_stage} output",
            metadata={"input_stage": input_stage, "safe_noop_fallback": True},
        )

    msh_path = mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh"
    if not msh_path.exists():
        return _base_record(
            cfg,
            category,
            mesh_id,
            "macro_skill",
            started,
            status="skipped",
            error="missing tetra.msh",
            metadata={"input_stage": input_stage, "safe_noop_fallback": True},
        )

    result_root = stage_root(cfg, "macro_skill", category)
    exp_name = _macro_skill_exp_name(cfg, category, stage_cfg)
    mesh_root = result_root / exp_name / "result" / "updated0" / mesh_id
    bbox_out = mesh_root / "bboxs_steps0"
    result_json = mesh_root / "macro_skill_result.json"
    command = _macro_skill_stage_command(
        msh_path=msh_path,
        metadata_path=metadata_path,
        category_name=category["name"],
        stage_cfg=stage_cfg,
        result_json=result_json,
        bbox_out=bbox_out,
    )

    metadata: dict[str, Any] = {
        "input_stage": input_stage,
        "bbox_metadata_path": str(metadata_path),
        "msh_path": str(msh_path),
        "safe_noop_fallback": True,
        "quality_preset": str(stage_cfg.get("quality_preset", "balanced")),
        "repeat_mode": str(stage_cfg.get("repeat_mode", "guarded_variable")),
        "native_executor": bool(stage_cfg.get("native_executor", True)),
        "exact_validator": "native_smart_manifold",
    }
    planner_cfg = stage_cfg.get("planner", {}) if isinstance(stage_cfg.get("planner", {}), dict) else {}
    planner_enabled = bool(planner_cfg.get("enabled", False))
    if planner_enabled:
        metadata.update(
            {
                "planner_enabled": True,
                "planner_max_rounds": int(planner_cfg.get("max_rounds", 3) or 3),
                "planner_profile_schedule": [
                    str(item) for item in planner_cfg.get("profile_schedule", ["balanced"]) if str(item)
                ],
            }
        )
    if dry_run:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "macro_skill",
            started,
            status="dry_run",
            output_path=bbox_out,
            command=command,
            metadata=metadata,
        )

    try:
        if planner_enabled:
            from smart.api import run_macro_skill_planner_from_files as _run_macro_skill_from_files
        else:
            from smart.api import run_macro_skill_controller_from_files as _run_macro_skill_from_files

        macro_kwargs: dict[str, Any] = {
            "msh_path": msh_path,
            "bbox_metadata_path": metadata_path,
            "category": category["name"],
            "cover_penalty": float(stage_cfg.get("cover_penalty", 100)),
            "pen_rate": float(stage_cfg.get("pen_rate", 1.0)),
            "num_action_scale": int(stage_cfg.get("num_action_scale", 2)),
            "action_unit": float(stage_cfg.get("action_unit", 0.01)),
            "candidate_count": int(stage_cfg.get("candidate_count", 256)),
            "top_k": int(stage_cfg.get("top_k", 5)),
            "exact_budget": stage_cfg.get("exact_budget"),
            "max_steps": int(stage_cfg.get("max_steps", 16)),
            "quality_preset": str(stage_cfg.get("quality_preset", "balanced")),
            "repeat_mode": str(stage_cfg.get("repeat_mode", "guarded_variable")),
            "target_aware_execution": bool(stage_cfg.get("target_aware_execution", False)),
            "native_executor": bool(stage_cfg.get("native_executor", True)),
            "volume_method": str(stage_cfg.get("volume_method", "mesh")),
            "stateful_union_cache": bool(stage_cfg.get("stateful_union_cache", True)),
            "cache_capacity": int(stage_cfg.get("stateful_cache_capacity", 65536)),
        }
        if planner_enabled:
            macro_kwargs.update(
                {
                    "max_rounds": int(planner_cfg.get("max_rounds", 3) or 3),
                    "profile_schedule": tuple(
                        str(item) for item in planner_cfg.get("profile_schedule", ["balanced"]) if str(item)
                    ),
                    "min_round_delta": float(planner_cfg.get("min_round_delta", 1.0e-12)),
                    "stop_after_noop": bool(planner_cfg.get("stop_after_noop", True)),
                }
            )
        result = _run_macro_skill_from_files(**macro_kwargs)
        engine = result.pop("engine", None)
        if engine is None or not hasattr(engine, "export_bbox_dir"):
            raise RuntimeError("macro-skill controller returned no exportable native engine")
        bbox_out.mkdir(parents=True, exist_ok=True)
        exported = int(engine.export_bbox_dir(str(bbox_out)))
        result["exported_bbox_count"] = exported
        result["output_bbox_dir"] = str(bbox_out)
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        metadata.update(
            {
                "accepted": bool(result.get("accepted", False)),
                "accepted_non_worse": bool(result.get("accepted_non_worse", False)),
                "score_delta": float(result.get("score_delta", 0.0) or 0.0),
                "attempt_count": int(result.get("attempt_count", 0) or 0),
                "exact_budget": int(result.get("exact_budget", 0) or 0),
                "exported_bbox_count": exported,
                "result_json": str(result_json),
                "deployment_status": str(result.get("deployment_status", "")),
                "planner_enabled": planner_enabled,
            }
        )
        if planner_enabled:
            metadata["accepted_rounds"] = int(result.get("accepted_rounds", 0) or 0)
            metadata["round_count"] = int(result.get("round_count", 0) or 0)
        status = "success" if exported > 0 else "failed"
        error = None if status == "success" else "macro-skill exported no bbox objects"
    except Exception as exc:
        status = "failed"
        error = f"macro-skill failed: {exc}"

    return _base_record(
        cfg,
        category,
        mesh_id,
        "macro_skill",
        started,
        status=status,
        output_path=bbox_out if status == "success" else None,
        command=command,
        error=error,
        metadata=metadata,
    )


def run_local_refine_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage_cfg = cfg.get("local_refine", {})
    merge_cfg = cfg.get("merge", {})
    if float(stage_cfg.get("action_prior_weight", 0.0) or 0.0) != 0.0 and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "local_refine",
            started,
            status="blocked",
            error=(
                "local_refine.action_prior_weight changes greedy local-search order. "
                "Keep it at 0 for paper-compatible runs, or set "
                "local_refine.allow_search_order_changes=true for research runs."
            ),
        )
    if float(stage_cfg.get("action_value_weight", 0.0) or 0.0) != 0.0 and not stage_cfg.get(
        "allow_search_order_changes", False
    ):
        return _base_record(
            cfg,
            category,
            mesh_id,
            "local_refine",
            started,
            status="blocked",
            error=(
                "local_refine.action_value_weight changes greedy local-search order. "
                "Keep it at 0 for paper-compatible runs, or set "
                "local_refine.allow_search_order_changes=true for research runs."
            ),
        )
    existing = latest_bbox_dir(stage_root(cfg, "local_refine", category), mesh_id)
    if existing and not force:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "local_refine",
            started,
            status="skipped",
            output_path=existing,
        )

    input_stage = str(stage_cfg.get("input_stage", "mcts"))
    input_exp = latest_exp_dir_for_bbox(stage_root(cfg, input_stage, category), mesh_id)
    if input_exp is None:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "local_refine",
            started,
            status="skipped",
            error=f"missing {input_stage} bbox output",
        )

    result_root = stage_root(cfg, "local_refine", category)
    log_path = workspace_path(cfg, "logs", "local_refine", category["name"], f"{mesh_id}.log")
    command = [
        _python(cfg),
        "run.py",
        "--run_type",
        "greedy",
        "--result_path",
        str(result_root),
        "--path_to_msh_file",
        str(tetra_root(cfg, category)),
        "--category",
        category["name"],
        "--path_to_bbox",
        str(input_exp),
        "--bbox_init",
        str(stage_cfg.get("bbox_init", "bbox_direct")),
        "--init_type",
        str(merge_cfg.get("init_type", "coacd")),
        "--merge_eps",
        str(merge_cfg.get("merge_eps", 0.02)),
        "--max_step",
        str(stage_cfg.get("max_step", 300)),
        "--greedy_backend",
        str(stage_cfg.get("backend", "auto")),
        "--cover_penalty",
        str(stage_cfg.get("cover_penalty", 100)),
        "--score_cache_size",
        str(stage_cfg.get("score_cache_size", 8192)),
        "--candidate_backend",
        str(stage_cfg.get("candidate_backend", "exact")),
        "--candidate_top_k",
        str(stage_cfg.get("candidate_top_k", 8)),
        "--candidate_pruned_max_aspect_mean",
        str(stage_cfg.get("candidate_pruned_max_aspect_mean", 0.0)),
        "--candidate_pruned_min_fill_ratio",
        str(stage_cfg.get("candidate_pruned_min_fill_ratio", 0.0)),
        "--reward_backend",
        str(stage_cfg.get("reward_backend", "manifold_stateful")),
        "--manifold_volume_method",
        str(stage_cfg.get("manifold_volume_method", "mesh")),
        "--stateful_cache_capacity",
        str(stage_cfg.get("stateful_cache_capacity", 65536)),
        "--tet_clipping_max_boxes",
        str(stage_cfg.get("tet_clipping_max_boxes", 12)),
        "--action_unit",
        str(stage_cfg.get("action_unit", 0.005)),
        "--num_action_scale",
        str(stage_cfg.get("num_action_scale", 1)),
        "--worker",
        str(stage_cfg.get("worker", 0)),
        "--data_batch_size",
        "1",
        "--print_off",
        "--meshes",
        mesh_id[:10],
    ]
    if not stage_cfg.get("render_initial", False):
        command.append("--skip_initial_render")
    if not stage_cfg.get("render_partition", False):
        command.append("--skip_render_partition")
    if not stage_cfg.get("summary_metrics", False):
        command.append("--skip_summary_metrics")
    if not stage_cfg.get("stateful_union_cache", True):
        command.append("--no-stateful_union_cache")
    if not stage_cfg.get("candidate_require_exact_fallback", True):
        command.append("--no-candidate_require_exact_fallback")
    if stage_cfg.get("candidate_bypass_on_exact_fallback", False):
        command.append("--candidate_bypass_on_exact_fallback")
    if stage_cfg.get("candidate_pruned_categories"):
        command.extend(
            ["--candidate_pruned_categories", str(stage_cfg.get("candidate_pruned_categories", ""))]
        )
    if not stage_cfg.get("cache_initial_bbox_state", True):
        command.append("--no-cache_initial_bbox_state")
    if stage_cfg.get("stateful_unscored_apply", False):
        command.append("--stateful_unscored_apply")
    if str(stage_cfg.get("forced_action_sequence", "") or ""):
        command.extend(["--forced_action_sequence", str(stage_cfg.get("forced_action_sequence", ""))])
        command.extend(
            [
                "--forced_first_action_min_reward",
                str(stage_cfg.get("forced_first_action_min_reward", 0.0)),
            ]
        )
    elif int(stage_cfg.get("forced_first_action", -1) or -1) >= 0:
        command.extend(["--forced_first_action", str(stage_cfg.get("forced_first_action", -1))])
        command.extend(
            [
                "--forced_first_action_min_reward",
                str(stage_cfg.get("forced_first_action_min_reward", 0.0)),
            ]
        )
    if stage_cfg.get("trace_actions_path"):
        command.extend(["--trace_actions_path", str(repo_path(stage_cfg["trace_actions_path"]))])
    if stage_cfg.get("candidate_trace_path"):
        command.extend(
            [
                "--candidate_trace_path",
                str(repo_path(stage_cfg["candidate_trace_path"])),
                "--candidate_trace_top_k",
                str(stage_cfg.get("candidate_trace_top_k", 0)),
            ]
        )
    if stage_cfg.get("action_prior_path"):
        command.extend(["--action_prior_path", str(repo_path(stage_cfg["action_prior_path"]))])
        command.extend(["--action_prior_device", _resolve_action_prior_device(stage_cfg.get("action_prior_device", "json"))])
        command.extend(["--action_prior_weight", str(stage_cfg.get("action_prior_weight", 0.0))])
        command.extend(["--action_value_weight", str(stage_cfg.get("action_value_weight", 0.0))])
        command.extend(["--action_prior_top_k", str(stage_cfg.get("action_prior_top_k", 0))])
        if not stage_cfg.get("action_prior_keep_upper", True):
            command.append("--no-action_prior_keep_upper")
    if stage_cfg.get("exp_tag"):
        command.extend(["--mcts_exp_tag", str(stage_cfg["exp_tag"])])
    if merge_cfg.get("tilted", True):
        command.append("--tilted")
    if merge_cfg.get("fast_merge", True):
        command.append("--fast_merge")
    result = run_command(
        command,
        cwd=REPO_ROOT / "smart" / "legacy" / "refine",
        timeout=stage_cfg.get("timeout_sec"),
        env=_legacy_env(),
        log_path=log_path,
        dry_run=dry_run,
    )
    output = latest_bbox_dir(result_root, mesh_id, since=started)
    status = "dry_run" if dry_run else "success" if result.ok and output else "failed"
    error = None if status in {"success", "dry_run"} else _command_failure_summary("local_refine", result)
    return _base_record(
        cfg,
        category,
        mesh_id,
        "local_refine",
        started,
        status=status,
        output_path=output,
        log_path=log_path,
        command=command,
        error=error,
        metadata={"timeout_sec": stage_cfg.get("timeout_sec"), "elapsed_sec": result.elapsed_sec},
    )


def run_render_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> StageRecord:
    started = time.time()
    stage_cfg = cfg.get("render", {})
    variants, explicit_variants = _render_variants(stage_cfg)
    outputs = _render_outputs(cfg, category, mesh_id, variants, explicit_variants=explicit_variants)
    primary_output = next(iter(outputs.values()))
    if all(output.exists() for output in outputs.values()) and not force:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "render",
            started,
            status="skipped",
            output_path=primary_output,
            metadata={"outputs": {name: str(path) for name, path in outputs.items()}},
        )
    input_stage = stage_cfg.get("input_stage", "mcts")
    bbox_dir = bbox_dir_for_render(cfg, category, mesh_id, input_stage)
    if bbox_dir is None and input_stage == "mcts":
        bbox_dir = bbox_dir_for_render(cfg, category, mesh_id, "refine")
    if bbox_dir is None:
        return _base_record(cfg, category, mesh_id, "render", started, status="skipped", error=f"missing bbox output for {input_stage}", output_path=primary_output)

    backend = str(stage_cfg.get("backend", "fallback")).lower()
    if backend not in {"fallback", "preview", "software", "paper_fallback", "auto", "blender"}:
        return _base_record(cfg, category, mesh_id, "render", started, status="failed", error=f"unknown render backend: {backend}", output_path=primary_output)
    if backend in {"fallback", "preview", "software", "paper_fallback"}:
        if dry_run:
            return _base_record(
                cfg,
                category,
                mesh_id,
                "render",
                started,
                status="dry_run",
                output_path=primary_output,
                metadata={"renderer": "software_preview", "outputs": {name: str(path) for name, path in outputs.items()}},
            )
        errors: dict[str, str] = {}
        for name, joint_mesh in variants:
            fallback_error = fallback_render_scene(cfg, category, mesh_id, bbox_dir, outputs[name], joint_mesh=joint_mesh)
            if fallback_error is not None:
                errors[name] = fallback_error
        if not errors:
            return _base_record(
                cfg,
                category,
                mesh_id,
                "render",
                started,
                status="success",
                output_path=primary_output,
                metadata={"renderer": "software_preview", "outputs": {name: str(path) for name, path in outputs.items()}},
            )
        return _base_record(
            cfg,
            category,
            mesh_id,
            "render",
            started,
            status="failed",
            output_path=primary_output,
            error=f"fallback render failed: {errors}",
            metadata={"outputs": {name: str(path) for name, path in outputs.items()}},
        )

    blender = _blender(cfg)
    if blender is None:
        if stage_cfg.get("fallback", True) and not dry_run:
            errors: dict[str, str] = {}
            for name, joint_mesh in variants:
                fallback_error = fallback_render_scene(cfg, category, mesh_id, bbox_dir, outputs[name], joint_mesh=joint_mesh)
                if fallback_error is not None:
                    errors[name] = fallback_error
            if not errors:
                return _base_record(
                    cfg,
                    category,
                    mesh_id,
                    "render",
                    started,
                    status="success",
                    output_path=primary_output,
                    metadata={"renderer": "software_preview", "reason": "missing_blender", "outputs": {name: str(path) for name, path in outputs.items()}},
                )
        return _base_record(cfg, category, mesh_id, "render", started, status="blocked", error="missing blender executable", output_path=primary_output)
    env = os.environ.copy()
    env["CRASHREPORTER_DISABLE"] = str(stage_cfg.get("disable_crash_reporter", "1"))
    env["CUDA_VISIBLE_DEVICES"] = str(stage_cfg.get("gpu", cfg.get("gpu", "0")))
    env["SMART_RENDER_MESH_ROOT"] = str(tetra_root(cfg, category))
    env["SMART_RENDER_CATEGORY"] = category["name"]
    env["SMART_RENDER_TRANSPARENT"] = "1" if stage_cfg.get("transparent", True) else "0"
    _set_render_camera_env(env, stage_cfg.get("camera", {}), category["name"])

    successes: dict[str, str] = {}
    errors: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    log_paths: dict[str, str] = {}
    for name, joint_mesh in variants:
        output = outputs[name]
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = workspace_path(cfg, "logs", "render", category["name"], f"{mesh_id}__{name}.log")
        command = [
            blender,
            "--disable-crash-handler",
            "--background",
            "boxes.blend",
            "--python",
            "render_teaser.py",
            "--",
            str(bbox_dir),
            str(output),
            "1" if joint_mesh else "0",
            mesh_id,
            "0",
        ]
        commands[name] = command
        log_paths[name] = str(log_path)
        result = run_command(
            command,
            cwd=REPO_ROOT / "smart" / "legacy" / "renderer",
            timeout=stage_cfg.get("timeout_sec"),
            env=env,
            log_path=log_path,
            dry_run=dry_run,
        )
        if dry_run:
            continue
        if result.ok and output.exists():
            successes[name] = str(output)
            continue
        if stage_cfg.get("fallback", True):
            fallback_error = fallback_render_scene(cfg, category, mesh_id, bbox_dir, output, joint_mesh=joint_mesh)
            if fallback_error is None:
                successes[name] = str(output)
                continue
            errors[name] = f"render failed rc={result.returncode}; fallback failed: {fallback_error}"
        else:
            errors[name] = f"render failed rc={result.returncode}"

    if dry_run:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "render",
            started,
            status="dry_run",
            output_path=primary_output,
            log_path=next(iter(log_paths.values()), None),
            command=next(iter(commands.values()), None),
            metadata={
                "renderer": "blender",
                "outputs": {name: str(path) for name, path in outputs.items()},
                "commands": commands,
                "logs": log_paths,
            },
        )
    if not errors:
        return _base_record(
            cfg,
            category,
            mesh_id,
            "render",
            started,
            status="success",
            output_path=primary_output,
            log_path=next(iter(log_paths.values()), None),
            command=next(iter(commands.values()), None),
            metadata={"renderer": "blender", "outputs": successes, "logs": log_paths},
        )
    return _base_record(
        cfg,
        category,
        mesh_id,
        "render",
        started,
        status="failed",
        output_path=primary_output,
        log_path=next(iter(log_paths.values()), None),
        command=next(iter(commands.values()), None),
        error=str(errors),
        metadata={"renderer": "blender", "outputs": successes, "errors": errors, "logs": log_paths},
    )


def _render_variants(stage_cfg: dict[str, Any]) -> tuple[list[tuple[str, bool]], bool]:
    raw = stage_cfg.get("variants")
    explicit = raw is not None
    if raw is None:
        return [("with_mesh" if stage_cfg.get("joint_mesh", False) else "boxes_only", bool(stage_cfg.get("joint_mesh", False)))], False
    if isinstance(raw, str):
        raw_values = [raw]
    else:
        raw_values = list(raw)
    variants: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for item in raw_values:
        key = str(item).lower().replace("-", "_")
        if key in {"with_mesh", "mesh", "mesh_overlay", "joint_mesh", "overlay"}:
            name, joint_mesh = "with_mesh", True
        elif key in {"boxes_only", "box_only", "bbox_only", "no_mesh", "without_mesh"}:
            name, joint_mesh = "boxes_only", False
        else:
            raise ValueError(f"unknown render variant: {item}")
        if name not in seen:
            variants.append((name, joint_mesh))
            seen.add(name)
    if not variants:
        raise ValueError("render.variants must contain at least one variant")
    return variants, explicit


def _render_outputs(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    variants: list[tuple[str, bool]],
    *,
    explicit_variants: bool,
) -> dict[str, Path]:
    root = stage_root(cfg, "render", category) / mesh_id
    if not explicit_variants and len(variants) == 1:
        return {variants[0][0]: root / f"{mesh_id}.png"}
    return {name: root / f"{mesh_id}__{name}.png" for name, _ in variants}


def _set_render_camera_env(env: dict[str, str], camera_cfg: dict[str, Any], category_name: str) -> None:
    if not isinstance(camera_cfg, dict):
        return

    def apply(prefix: str, values: dict[str, Any]) -> None:
        env_prefix = f"SMART_RENDER_{prefix}_" if prefix else "SMART_RENDER_"
        if "ortho_scale" in values:
            env[f"{env_prefix}ORTHO_SCALE"] = str(values["ortho_scale"])
        if "shift_x" in values:
            env[f"{env_prefix}SHIFT_X"] = str(values["shift_x"])
        if "shift_y" in values:
            env[f"{env_prefix}SHIFT_Y"] = str(values["shift_y"])
        if "rotation" in values:
            env[f"{env_prefix}ROTATION"] = ",".join(str(item) for item in values["rotation"])

    default_cfg = camera_cfg.get("default", {})
    if isinstance(default_cfg, dict):
        apply("", default_cfg)

    category_cfg = camera_cfg.get(category_name, {})
    if isinstance(category_cfg, dict):
        apply(category_name.upper(), category_cfg)


def fallback_render_scene(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    bbox_dir: Path,
    output: Path,
    *,
    joint_mesh: bool | None = None,
) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(workspace_path(cfg, "matplotlib")))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import trimesh  # type: ignore
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    except Exception as exc:  # noqa: BLE001
        return f"missing fallback dependency: {exc}"

    try:
        stage_cfg = cfg.get("render", {})
        coord_rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        mesh_path = mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh__sf.obj"
        mesh_vertices = None
        mesh_faces = None
        norm_center = None
        norm_diag = None
        if mesh_path.exists():
            mesh_vertices, mesh_faces = _load_render_obj(mesh_path, trimesh)
            if mesh_vertices is not None and len(mesh_vertices):
                norm_center = mesh_vertices.mean(axis=0)
                extent = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
                norm_diag = float(np.linalg.norm(extent))

        mesh_candidates = sorted(path for path in bbox_dir.glob("*.obj") if not path.name.startswith("bbox"))
        bbox_candidates = sorted(bbox_dir.glob("bbox*.obj"))
        if not mesh_candidates and not bbox_candidates:
            return f"no OBJ files in {bbox_dir}"

        fig = plt.figure(figsize=(7, 7), dpi=int(stage_cfg.get("dpi", 220)))
        axis = fig.add_subplot(111, projection="3d")
        axis.set_proj_type("ortho")
        render_items: list[dict[str, Any]] = []

        draw_mesh = bool(stage_cfg.get("joint_mesh", True)) if joint_mesh is None else joint_mesh
        if draw_mesh and mesh_vertices is not None and mesh_faces is not None:
            vertices = _paper_render_transform(mesh_vertices, coord_rot, norm_center, norm_diag)
            render_items.append(
                {
                    "kind": "mesh",
                    "vertices": vertices,
                    "faces": mesh_faces,
                    "color": (0.95, 0.95, 0.95, float(stage_cfg.get("mesh_alpha", 0.62))),
                    "edge_color": (0.08, 0.08, 0.08, 0.16),
                    "linewidth": float(stage_cfg.get("mesh_linewidth", 0.08)),
                }
            )

        cmap = plt.get_cmap(str(stage_cfg.get("colormap", "turbo")))
        for index, bbox_path in enumerate(bbox_candidates):
            vertices, faces = _load_render_obj(bbox_path, trimesh)
            if vertices is None or faces is None:
                continue
            vertices = _paper_render_transform(vertices, coord_rot, norm_center, norm_diag)
            color = cmap(index / max(len(bbox_candidates), 1))
            render_items.append(
                {
                    "kind": "bbox",
                    "vertices": vertices,
                    "faces": faces,
                    "color": (color[0], color[1], color[2], float(stage_cfg.get("bbox_alpha", 0.38))),
                    "edge_color": (0.02, 0.02, 0.02, float(stage_cfg.get("bbox_edge_alpha", 0.72))),
                    "linewidth": float(stage_cfg.get("bbox_linewidth", 0.55)),
                }
            )

        if not render_items:
            return "no vertices to render"

        if category["name"] == "airplane":
            rotation = _euler_xyz(np, -0.2, 0.3, 0.4)
            for item in render_items:
                item["vertices"] = item["vertices"] @ rotation.T

        min_z = min(float(item["vertices"][:, 2].min()) for item in render_items if len(item["vertices"]))
        for item in render_items:
            item["vertices"][:, 2] -= min_z

        points = np.concatenate([item["vertices"] for item in render_items], axis=0)
        for item in render_items:
            vertices = item["vertices"]
            faces = item["faces"]
            if len(faces):
                collection = Poly3DCollection(
                    vertices[faces],
                    facecolor=item["color"],
                    edgecolor=item["edge_color"],
                    linewidths=item["linewidth"],
                )
                axis.add_collection3d(collection)
            if item["kind"] == "bbox":
                lines = [(vertices[a], vertices[b]) for a, b in _box_edges(vertices, faces)]
                if lines:
                    axis.add_collection3d(
                        Line3DCollection(
                            lines,
                            colors=[item["edge_color"]],
                            linewidths=float(stage_cfg.get("bbox_wire_linewidth", 1.1)),
                        )
                    )

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float((maxs - mins).max()) / 2.0, 1e-6) * float(stage_cfg.get("camera_padding", 1.18))
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        if category["name"] == "airplane":
            _set_view(
                axis,
                elev=float(stage_cfg.get("airplane_elev", 18)),
                azim=float(stage_cfg.get("airplane_azim", -42)),
                roll=float(stage_cfg.get("airplane_roll", 0)),
            )
        else:
            _set_view(
                axis,
                elev=float(stage_cfg.get("elev", 18)),
                azim=float(stage_cfg.get("azim", -42)),
                roll=float(stage_cfg.get("roll", 0)),
            )
        axis.set_axis_off()
        axis.set_box_aspect((1, 1, 1))
        transparent = bool(stage_cfg.get("transparent", True))
        fig.patch.set_alpha(0.0 if transparent else 1.0)
        axis.patch.set_alpha(0.0 if transparent else 1.0)
        if not transparent:
            fig.patch.set_facecolor("white")
            axis.set_facecolor("white")
        fig.subplots_adjust(0, 0, 1, 1)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", pad_inches=0, transparent=transparent)
        plt.close(fig)
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def _load_render_obj(path: Path, trimesh_module: Any) -> tuple[Any, Any]:
    import numpy as np

    mesh = trimesh_module.load(path, file_type="obj", process=False)
    if isinstance(mesh, trimesh_module.Scene):
        mesh = trimesh_module.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int)


def _paper_render_transform(vertices: Any, coord_rot: Any, center: Any, diagonal: float | None) -> Any:
    import numpy as np

    transformed = np.asarray(vertices, dtype=float).copy()
    if center is not None and diagonal and diagonal > 0:
        transformed -= center
        transformed /= diagonal
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = transformed @ coord_rot.T
    if not np.isfinite(transformed).all():
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    return transformed


def _box_edges(vertices: Any, faces: Any) -> list[tuple[int, int]]:
    import numpy as np

    if len(vertices) == 8:
        pairs: list[tuple[float, int, int]] = []
        for i in range(8):
            for j in range(i + 1, 8):
                pairs.append((float(np.linalg.norm(vertices[i] - vertices[j])), i, j))
        pairs.sort(key=lambda item: item[0])
        return [(i, j) for _, i, j in pairs[:12]]
    edges = set()
    for face in faces:
        ids = [int(idx) for idx in face]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edges.add(tuple(sorted((a, b))))
    return sorted(edges)


def _euler_xyz(np_module: Any, x: float, y: float, z: float) -> Any:
    cx, sx = np_module.cos(x), np_module.sin(x)
    cy, sy = np_module.cos(y), np_module.sin(y)
    cz, sz = np_module.cos(z), np_module.sin(z)
    rx = np_module.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np_module.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np_module.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _set_view(axis: Any, *, elev: float, azim: float, roll: float) -> None:
    try:
        axis.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        axis.view_init(elev=elev, azim=azim)


def _bbox_dir_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    for child in path.iterdir():
        try:
            latest = max(latest, child.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def latest_bbox_dir(root: Path, mesh_id: str, *, since: float | None = None) -> Path | None:
    candidates = [path for path in root.glob(f"**/result/updated*/{mesh_id}/bboxs_steps*") if path.is_dir()]
    candidates.extend(path for path in root.glob(f"**/result/{mesh_id}/bboxs*") if path.is_dir())
    if since is not None:
        candidates = [path for path in candidates if _bbox_dir_mtime(path) >= since]
    if not candidates:
        return None
    return max(candidates, key=_bbox_dir_mtime)


def latest_manifest_bbox_dir(cfg: dict[str, Any], category: dict[str, Any], mesh_id: str, stage: str) -> Path | None:
    manifest = workspace_path(cfg, "manifests", f"{stage}.jsonl")
    if not manifest.exists():
        return None

    selected: tuple[float, Path] | None = None
    with manifest.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("stage") != stage:
                continue
            if record.get("category") != category["name"] or record.get("mesh_id") != mesh_id:
                continue
            if record.get("status") != "success" or not record.get("output_path"):
                continue
            path = Path(str(record["output_path"]))
            if not path.exists():
                continue
            finished_at = float(record.get("finished_at") or 0.0)
            if selected is None or finished_at >= selected[0]:
                selected = (finished_at, path)
    return selected[1] if selected is not None else None


def latest_exp_dir_for_bbox(root: Path, mesh_id: str) -> Path | None:
    bbox = latest_bbox_dir(root, mesh_id)
    if bbox is None:
        return None
    parts = bbox.parts
    if "result" not in parts:
        return None
    result_index = parts.index("result")
    return Path(*parts[:result_index])


def native_pipeline_bbox_dir(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    input_stage: str,
) -> Path | None:
    """Return bbox metadata exported by the C++ native one-shot pipeline.

    The legacy Python stages write bbox directories under
    ``<stage>/<category>/<exp>/result/...``.  The C++ native pipeline writes the
    same bbox artifacts directly under ``native_pipeline/<category>/<mesh>``.
    Downstream stages should be able to consume either layout.
    """

    stage_dir_by_input = {
        "refine": "refine_bboxs_steps0",
        "mcts": "mcts_bboxs_steps0",
        "mcts_guarded": "mcts_bboxs_steps0",
    }
    stage_dir = stage_dir_by_input.get(str(input_stage))
    if stage_dir is None:
        return None
    candidate = workspace_path(cfg, "native_pipeline", category["name"], mesh_id, stage_dir)
    if candidate.is_dir() and (candidate / "bbox_params.json").exists():
        return candidate
    return None


def bbox_dir_for_render(cfg: dict[str, Any], category: dict[str, Any], mesh_id: str, input_stage: str) -> Path | None:
    manifest_bbox = latest_manifest_bbox_dir(cfg, category, mesh_id, input_stage)
    if manifest_bbox is not None:
        return manifest_bbox
    if input_stage in {
        "mcts",
        "mcts_guarded",
        "refine",
        "macro_skill",
        "local_refine",
        "local_refine_guarded",
    }:
        found = latest_bbox_dir(stage_root(cfg, input_stage, category), mesh_id)
        if found is not None:
            return found
        return native_pipeline_bbox_dir(cfg, category, mesh_id, input_stage)
    if input_stage == "merge":
        return latest_bbox_dir(stage_root(cfg, "merge", category), mesh_id)
    generic_stage_root = stage_root(cfg, input_stage, category)
    if generic_stage_root.exists():
        found = latest_bbox_dir(generic_stage_root, mesh_id)
        if found is not None:
            return found
    candidate = Path(input_stage).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if candidate.is_dir() and any(candidate.glob("bbox*.obj")):
        return candidate
    return latest_bbox_dir(candidate, mesh_id)


def _python(cfg: dict[str, Any]) -> str:
    return str(cfg.get("python") or sys.executable)


def _legacy_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    packaged_manifold_python = REPO_ROOT / "smart" / "pymanifold_runtime"
    source_manifold_python = REPO_ROOT / "smart" / "vendor" / "manifold" / "build" / "bindings" / "python"
    default_manifold_python = (
        packaged_manifold_python
        if any(packaged_manifold_python.glob("pymanifold*"))
        else source_manifold_python
    )
    env.setdefault(
        "SMART_MANIFOLD_PYTHON",
        str(default_manifold_python),
    )
    return env


def _blender(cfg: dict[str, Any]) -> str | None:
    configured = cfg.get("tools", {}).get("blender_bin")
    if configured:
        path = repo_path(configured)
        return str(path) if path is not None else str(configured)
    blender = find_executable(None, "SMART_BLENDER_BIN", "blender")
    if blender:
        return blender
    mac_app = Path("/Applications/Blender.app/Contents/MacOS/blender")
    if mac_app.exists():
        return str(mac_app)
    return None


def data_status(cfg: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for category in cfg.get("categories", []):
        root = category_mesh_root(category)
        mesh_ids = list_mesh_ids(category)
        status[category["name"]] = {
            "mesh_root": str(root),
            "model_obj_count": len(mesh_ids),
            "sample_bbox_diagonal": _sample_bbox_diagonal(root / mesh_ids[0] / "model.obj") if mesh_ids else None,
        }
    data_root = repo_path(str(cfg.get("data_root", "data")))
    manifest = data_root / "shapenet_samples_manifest.json" if data_root else None
    if manifest and manifest.exists():
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        by_category: dict[str, int] = {}
        for row in rows:
            category = row.get("category", "unknown")
            by_category[category] = by_category.get(category, 0) + 1
        status["manifest"] = {"rows": len(rows), "by_category": by_category}
    return status


def _sample_bbox_diagonal(obj_path: Path) -> float | None:
    vertices: list[tuple[float, float, float]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append((float(x), float(y), float(z)))
    if not vertices:
        return None
    xs, ys, zs = zip(*vertices)
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2) ** 0.5


def iter_stage_records(records: Iterable[StageRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        summary.setdefault(record.stage, {})
        summary[record.stage][record.status] = summary[record.stage].get(record.status, 0) + 1
    return summary
