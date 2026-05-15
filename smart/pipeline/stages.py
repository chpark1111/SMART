from __future__ import annotations

import json
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


STAGE_ORDER = [
    "normalize",
    "tetra",
    "preseg",
    "merge",
    "refine",
    "mcts",
    "local_refine",
    "render",
]


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

    writer.write_summary(records)
    return records


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
        obj_lines, vertices = _read_obj_vertices(source_mesh)
        normalized, stats = _normalize_vertices(vertices, signature)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_normalized_obj(obj_lines, normalized, output)
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
        metadata={"scale": stats["scale"], **stats["after"]},
    )


def tetra_source_mesh(cfg: dict[str, Any], category: dict[str, Any], mesh_id: str) -> Path:
    if normalization_enabled(cfg):
        return normalized_mesh_path(cfg, category, mesh_id)
    source_name = str(cfg.get("normalization", {}).get("source_filename", "model.obj"))
    return category_mesh_root(category) / mesh_id / source_name


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
        attempts.append(
            {
                "epsilon": epsilon * float(retry.get("epsilon_scale", 2.0)),
                "edge_length": edge_length * float(retry.get("edge_length_scale", 2.0)),
                "coarsen": bool(retry.get("coarsen", True)),
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
    for attempt in attempts:
        eps = float(attempt["epsilon"])
        length = float(attempt["edge_length"])
        coarsen = bool(attempt.get("coarsen", False))
        attempt_name = str(attempt["name"])
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manmesh = out_dir / "model_manifold.obj"
        manifold_log = log_dir / f"{attempt_name}_manifoldplus.log"
        ftetwild_log = log_dir / f"{attempt_name}_ftetwild.log"
        manifold_cmd = [manifold_bin, "--input", str(source_mesh), "--output", str(manmesh)]
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
            errors.append(f"{attempt_name}: {failure}")
            attempt_metadata.append(
                {
                    "attempt": attempt_name,
                    "tool": "ManifoldPlus",
                    "epsilon": eps,
                    "edge_length": length,
                    "coarsen": coarsen,
                    "returncode": result.returncode,
                    "elapsed_sec": result.elapsed_sec,
                    "timed_out": result.timed_out,
                    "failure": failure,
                }
            )
            continue

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
            errors.append(f"{attempt_name}: {failure}")
            attempt_metadata.append(
                {
                    "attempt": attempt_name,
                    "tool": "fTetWild",
                    "epsilon": eps,
                    "edge_length": length,
                    "coarsen": coarsen,
                    "returncode": result.returncode,
                    "elapsed_sec": result.elapsed_sec,
                    "timed_out": result.timed_out,
                    "timeout_sec": attempt.get("timeout_sec", stage_cfg.get("ftetwild_timeout_sec")),
                    "failure": failure,
                }
            )
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
            errors.append(f"{attempt_name}: validation failed: {validation_error}")
            attempt_metadata.append(
                {
                    "attempt": attempt_name,
                    "tool": "validation",
                    "epsilon": eps,
                    "edge_length": length,
                    "coarsen": coarsen,
                    "failure": validation_error,
                    "metadata": validation_metadata,
                }
            )
            continue
        metadata = {
            "epsilon": eps,
            "edge_length": length,
            "coarsen": coarsen,
            "attempt": attempt_name,
            "previous_attempts": attempt_metadata,
        }
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
        metadata={"attempts": attempt_metadata},
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
        components = trimesh.graph.split(mesh)
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

    try:
        import coacd  # type: ignore
        import trimesh  # type: ignore
    except ModuleNotFoundError as exc:
        return _base_record(cfg, category, mesh_id, "preseg", started, status="blocked", output_path=out_dir, error=f"missing dependency: {exc.name}")

    try:
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
            part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            part_mesh.export(out_dir / f"part_{index:04d}.obj")
    except Exception as exc:  # noqa: BLE001
        return _base_record(cfg, category, mesh_id, "preseg", started, status="failed", output_path=out_dir, error=str(exc))

    return _base_record(
        cfg,
        category,
        mesh_id,
        "preseg",
        started,
        status="success",
        output_path=out_dir,
        metadata={"parts": len(list(out_dir.glob("*.obj")))},
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
    if not greedy_segment_path(mesh_tetra_dir(cfg, category, mesh_id), merge_cfg).exists():
        return _base_record(cfg, category, mesh_id, "refine", started, status="skipped", error="missing greedy merged segment")

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
        "",
        "--bbox_init",
        str(stage_cfg.get("bbox_init", "grd_merged")),
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
    if not stage_cfg.get("cache_initial_bbox_state", True):
        command.append("--no-cache_initial_bbox_state")
    if stage_cfg.get("stateful_unscored_apply", False):
        command.append("--stateful_unscored_apply")
    if stage_cfg.get("fused_rollout_step", False):
        command.append("--mcts_fused_rollout_step")
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
        command.extend(["--action_prior_weight", str(stage_cfg.get("action_prior_weight", 0.0))])
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
    if mcts_backend in {"rust", "rust_stateful"} and not stage_cfg.get(
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
                "mcts.backend=rust/rust_stateful can change MCTS search order. "
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
    existing = latest_bbox_dir(stage_root(cfg, "mcts", category), mesh_id)
    if existing and not force:
        return _base_record(cfg, category, mesh_id, "mcts", started, status="skipped", output_path=existing)
    refine_exp = latest_exp_dir_for_bbox(stage_root(cfg, "refine", category), mesh_id)
    if refine_exp is None:
        return _base_record(cfg, category, mesh_id, "mcts", started, status="skipped", error="missing refine bbox output")

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
        str(refine_exp),
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
    if not stage_cfg.get("cache_initial_bbox_state", True):
        command.append("--no-cache_initial_bbox_state")
    if stage_cfg.get("stateful_unscored_apply", False):
        command.append("--stateful_unscored_apply")
    if stage_cfg.get("fused_rollout_step", False):
        command.append("--mcts_fused_rollout_step")
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
        command.extend(["--action_prior_weight", str(stage_cfg.get("action_prior_weight", 0.0))])
        command.extend(["--puct_prior_weight", str(stage_cfg.get("puct_prior_weight", 0.0))])
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
    if not stage_cfg.get("cache_initial_bbox_state", True):
        command.append("--no-cache_initial_bbox_state")
    if stage_cfg.get("stateful_unscored_apply", False):
        command.append("--stateful_unscored_apply")
    if stage_cfg.get("trace_actions_path"):
        command.extend(["--trace_actions_path", str(repo_path(stage_cfg["trace_actions_path"]))])
    if stage_cfg.get("action_prior_path"):
        command.extend(["--action_prior_path", str(repo_path(stage_cfg["action_prior_path"]))])
        command.extend(["--action_prior_weight", str(stage_cfg.get("action_prior_weight", 0.0))])
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
        if backend == "auto" and stage_cfg.get("fallback", True):
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


def bbox_dir_for_render(cfg: dict[str, Any], category: dict[str, Any], mesh_id: str, input_stage: str) -> Path | None:
    manifest_bbox = latest_manifest_bbox_dir(cfg, category, mesh_id, input_stage)
    if manifest_bbox is not None:
        return manifest_bbox
    if input_stage in {"mcts", "mcts_guarded", "refine", "local_refine", "local_refine_guarded"}:
        return latest_bbox_dir(stage_root(cfg, input_stage, category), mesh_id)
    if input_stage == "merge":
        return latest_bbox_dir(stage_root(cfg, "merge", category), mesh_id)
    candidate = Path(input_stage).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return latest_bbox_dir(candidate, mesh_id)


def _python(cfg: dict[str, Any]) -> str:
    return str(cfg.get("python") or sys.executable)


def _legacy_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault(
        "SMART_MANIFOLD_PYTHON",
        str(REPO_ROOT / "smart" / "vendor" / "manifold" / "build" / "bindings" / "python"),
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
