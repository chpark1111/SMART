from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_ON_DISK = SCRIPT_DIR.parent
for candidate in (REPO_ROOT_ON_DISK, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import smart.rust as smart_rust
from compare_tet_clipping_manifold import (  # noqa: E402
    _bbox_volume_diagnostics,
    rust_tet_clipping_metrics,
    tet_clipping_metrics,
)
from smart.evaluation import EvaluationMetrics, _load_bbox_meshes, _load_mesh  # noqa: E402
from smart.pipeline.config import REPO_ROOT, load_config  # noqa: E402
from smart.pipeline.stages import (  # noqa: E402
    bbox_dir_for_render,
    list_mesh_ids,
    mesh_tetra_dir,
    normalized_mesh_path,
)


METRIC_KEYS = ("BVS", "MOV", "TOV", "Covered", "vIoU")


@dataclass
class ParityRecord:
    category: str
    mesh_id: str
    stage: str
    status: str
    elapsed_sec: float
    num_boxes: int = 0
    num_tets: int = 0
    num_tets_used: int = 0
    clipping_backend: str | None = None
    max_abs_diff: float | None = None
    passed: bool | None = None
    mesh_path: str | None = None
    surface_path: str | None = None
    tetra_path: str | None = None
    bbox_path: str | None = None
    manifold: dict[str, float] | None = None
    tet_clipping: dict[str, float] | None = None
    abs_diff: dict[str, float] | None = None
    max_bbox_mesh_hull_abs_diff: float | None = None
    error: str | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep existing SMART outputs and compare current Manifold metrics against "
            "exact tetrahedron-box clipping metrics."
        )
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument(
        "--stage",
        action="append",
        choices=["merge", "refine", "mcts"],
        help="Stage to check. Repeat to check multiple stages. Default: mcts.",
    )
    parser.add_argument("--category", action="append", help="Category name. Repeatable.")
    parser.add_argument("--mesh", action="append", help="Mesh id. Repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum attempted records.")
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=12,
        help="Refuse exact inclusion-exclusion above this many boxes.",
    )
    parser.add_argument(
        "--max-tets",
        type=int,
        default=0,
        help="Debug only: limit tetrahedra for Python backend. Default 0 uses all tets.",
    )
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "python", "rust"], default="auto")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    started = time.time()
    cfg = load_config(args.config)
    stages = args.stage or ["mcts"]
    records: list[ParityRecord] = []
    attempted = 0
    mesh_filter = set(args.mesh or [])
    category_filter = set(args.category or [])

    for category in cfg.get("categories", []):
        category_name = str(category["name"])
        if category_filter and category_name not in category_filter:
            continue
        mesh_ids = list_mesh_ids(category, explicit=args.mesh)
        if mesh_filter:
            mesh_ids = [mesh_id for mesh_id in mesh_ids if mesh_id in mesh_filter]
        for mesh_id in mesh_ids:
            for stage in stages:
                if args.limit and attempted >= args.limit:
                    break
                attempted += 1
                records.append(
                    evaluate_parity_record(
                        cfg,
                        category,
                        mesh_id,
                        stage=stage,
                        backend=args.backend,
                        max_boxes=args.max_boxes,
                        max_tets=args.max_tets,
                        chamfer_points=args.chamfer_points,
                        tolerance=args.tolerance,
                    )
                )
            if args.limit and attempted >= args.limit:
                break
        if args.limit and attempted >= args.limit:
            break

    payload: dict[str, Any] = {
        "config": args.config,
        "stages": stages,
        "backend_requested": args.backend,
        "rust_available": smart_rust.using_rust(),
        "rust_backend_path": smart_rust.backend_path(),
        "tolerance": args.tolerance,
        "max_boxes": args.max_boxes,
        "max_tets": args.max_tets,
        "chamfer_points": args.chamfer_points,
        "elapsed_sec": time.time() - started,
        "summary": summarize(records, args.tolerance),
        "records": [asdict(record) for record in records],
    }

    output = Path(args.output) if args.output else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["output_path"] = str(output)

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if output is not None:
        print(f"wrote {output}")
    return 0 if payload["summary"]["failed"] == 0 and payload["summary"]["failed_parity"] == 0 else 1


def evaluate_parity_record(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    stage: str,
    backend: str,
    max_boxes: int,
    max_tets: int,
    chamfer_points: int,
    tolerance: float,
) -> ParityRecord:
    started = time.time()
    category_name = str(category["name"])
    source_path = normalized_mesh_path(cfg, category, mesh_id)
    if not source_path.exists():
        source_path = Path(category["mesh_root"]) / mesh_id / "model.obj"
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
    tetra_dir = mesh_tetra_dir(cfg, category, mesh_id)
    tetra_path = tetra_dir / "tetra.msh"
    surface_path = tetra_dir / "tetra.msh__sf.obj"
    bbox_path = bbox_dir_for_render(cfg, category, mesh_id, stage)

    record = ParityRecord(
        category=category_name,
        mesh_id=mesh_id,
        stage=stage,
        status="failed",
        elapsed_sec=0.0,
        mesh_path=str(source_path),
        surface_path=str(surface_path),
        tetra_path=str(tetra_path),
        bbox_path=str(bbox_path) if bbox_path is not None else None,
    )

    try:
        if bbox_path is None or not bbox_path.exists():
            record.status = "missing_output"
            record.error = f"No bbox result found for stage={stage} mesh={mesh_id}"
            return record
        if not source_path.exists():
            record.status = "missing_input"
            record.error = f"Missing source mesh OBJ: {source_path}"
            return record
        if not surface_path.exists():
            record.status = "missing_input"
            record.error = f"Missing tetra surface OBJ: {surface_path}"
            return record
        if not tetra_path.exists():
            record.status = "missing_input"
            record.error = f"Missing tetra mesh: {tetra_path}"
            return record

        source_mesh = _load_mesh(source_path)
        surface_mesh = _load_mesh(surface_path)
        bbox_meshes = _load_bbox_meshes(bbox_path)
        if not bbox_meshes:
            raise ValueError(f"No bbox OBJ files found in {bbox_path}")
        diagnostics = _bbox_volume_diagnostics(bbox_meshes)
        manifold_metrics = EvaluationMetrics(source_mesh, surface_mesh, bbox_meshes).compute(
            chamfer_points=chamfer_points
        )

        use_rust = backend == "rust" or (backend == "auto" and smart_rust.using_rust())
        if use_rust:
            clipping_metrics = rust_tet_clipping_metrics(
                tetra_path=tetra_path,
                surface_mesh=surface_mesh,
                bbox_meshes=bbox_meshes,
                max_boxes=max_boxes,
                max_tets=max_tets,
            )
        else:
            clipping_metrics = tet_clipping_metrics(
                tetra_path=tetra_path,
                surface_mesh=surface_mesh,
                bbox_meshes=bbox_meshes,
                max_boxes=max_boxes,
                max_tets=max_tets,
            )

        num_tets = int(clipping_metrics.pop("_num_tets"))
        num_tets_used = int(clipping_metrics.pop("_num_tets_used"))
        diffs = {
            key: abs(float(manifold_metrics[key]) - float(clipping_metrics[key]))
            for key in METRIC_KEYS
        }
        max_abs_diff = max(diffs.values(), default=0.0)

        record.status = "success"
        record.num_boxes = len(bbox_meshes)
        record.num_tets = num_tets
        record.num_tets_used = num_tets_used
        record.clipping_backend = "rust" if use_rust else "python"
        record.max_abs_diff = max_abs_diff
        record.passed = max_abs_diff <= tolerance
        record.manifold = {key: float(manifold_metrics[key]) for key in METRIC_KEYS}
        record.tet_clipping = {key: float(clipping_metrics[key]) for key in METRIC_KEYS}
        record.abs_diff = diffs
        record.max_bbox_mesh_hull_abs_diff = max(
            [row["abs_diff"] for row in diagnostics], default=0.0
        )
    except Exception as exc:
        record.error = str(exc)
    finally:
        record.elapsed_sec = time.time() - started
    return record


def summarize(records: list[ParityRecord], tolerance: float) -> dict[str, Any]:
    successes = [record for record in records if record.status == "success"]
    missing = [record for record in records if record.status.startswith("missing_")]
    failed = [
        record
        for record in records
        if record.status != "success" and not record.status.startswith("missing_")
    ]
    failed_parity = [
        record for record in successes if record.passed is False or (record.max_abs_diff or 0.0) > tolerance
    ]
    summary: dict[str, Any] = {
        "total": len(records),
        "success": len(successes),
        "failed": len(failed),
        "missing": len(missing),
        "passed_parity": len(successes) - len(failed_parity),
        "failed_parity": len(failed_parity),
        "tolerance": tolerance,
        "max_abs_diff": 0.0,
        "avg_elapsed_sec": None,
    }
    if successes:
        summary["max_abs_diff"] = max(float(record.max_abs_diff or 0.0) for record in successes)
        summary["avg_elapsed_sec"] = sum(record.elapsed_sec for record in successes) / len(successes)
        for key in METRIC_KEYS:
            values = [
                float(record.abs_diff[key])
                for record in successes
                if record.abs_diff is not None and key in record.abs_diff
            ]
            summary[f"max_abs_diff_{key}"] = max(values) if values else 0.0
            summary[f"avg_abs_diff_{key}"] = sum(values) / len(values) if values else 0.0
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
