#!/usr/bin/env python3
"""Run post-MCTS local refinement and keep only non-worse outputs.

This is the quality-safe version of the hybrid MCTS + local search idea. It
evaluates an input bbox stage, runs `smart local_refine`, evaluates the refined
output, and copies the selected result into an explicit guarded stage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from smart.pipeline.config import load_config, workspace_path  # noqa: E402
from smart.pipeline.manifest import ManifestWriter, StageRecord  # noqa: E402
from smart.pipeline.stages import bbox_dir_for_render, list_mesh_ids  # noqa: E402
from smart.quality import GuardedSelection, select_quality_guarded_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expanded_200.yaml")
    parser.add_argument("--category", help="Single category to run")
    parser.add_argument("--categories", default="", help="Comma-separated category filter for batch runs")
    parser.add_argument("--mesh", action="append", help="Mesh id; repeat for multiple meshes. Requires --category.")
    parser.add_argument("--mesh-limit", type=int, default=1, help="Used when --mesh is omitted")
    parser.add_argument("--per-category-limit", type=int, default=None, help="Alias for --mesh-limit in multi-category runs")
    parser.add_argument("--input-stage", default="mcts_guarded", help="Stage to refine and guard against")
    parser.add_argument("--stage", default="local_refine_guarded", help="Guarded output stage name")
    parser.add_argument("--max-step", type=int, default=100)
    parser.add_argument("--action-unit", type=float, default=0.005)
    parser.add_argument("--reward-backend", default="manifold_stateful")
    parser.add_argument("--backend", default="rust_stateful")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--covered-tolerance",
        type=float,
        default=0.0,
        help="Extra tolerance for Avg_Covered drops when guarding local refinement.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("improved", "not_worse"),
        default="improved",
        help="Select local_refine only when it improves quality, or also when it is merely not worse.",
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output", default="runs/bench_exact/quality_guarded_local_refine.json")
    parser.add_argument("--only-existing-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-local-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse the latest local_refine output instead of rerunning the local_refine stage.",
    )
    parser.add_argument("--print-full-report", action="store_true", help="Print the full JSON report instead of a compact summary")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    category_meshes = _select_category_meshes(cfg, args)
    run_tag = args.run_tag or time.strftime("local_refine_guard_%Y%m%d_%H%M%S")
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = ManifestWriter(workspace_path(cfg, "manifests"))
    report: dict[str, Any] = {
        "config": args.config,
        "input_stage": args.input_stage,
        "stage": args.stage,
        "max_step": args.max_step,
        "action_unit": args.action_unit,
        "metric_tolerance": args.metric_tolerance,
        "covered_tolerance": args.covered_tolerance,
        "selection_mode": args.selection_mode,
        "reward_backend": args.reward_backend,
        "backend": args.backend,
        "reuse_local_refine": args.reuse_local_refine,
        "run_tag": run_tag,
        "categories": sorted(category_meshes),
        "meshes": category_meshes,
        "records": {},
    }

    for category_name, mesh_ids in category_meshes.items():
        for index, mesh_id in enumerate(mesh_ids, start=1):
            mesh_tag = f"{run_tag}_{category_name}_{index:03d}"
            baseline = _run_eval(args, category=category_name, mesh_id=mesh_id, stage=args.input_stage, exp_tag=f"{mesh_tag}_input")
            if args.reuse_local_refine:
                refined = {
                    "command": None,
                    "elapsed_sec": 0.0,
                    "returncode": 0,
                    "reused": True,
                }
            else:
                refined = _run_local_refine(args, category=category_name, mesh_id=mesh_id, exp_tag=mesh_tag)
            if refined.get("returncode") == 0:
                refined.update(_run_eval(args, category=category_name, mesh_id=mesh_id, stage="local_refine", exp_tag=f"{mesh_tag}_local_refine"))
            runs = {"input": baseline, "local_refine": refined}
            selection = _select_local_refine(args, runs)
            selected_run = runs.get(selection.selected_label, {})
            selected_bbox = _selected_bbox_path(selected_run)
            guarded_output = None
            error = None
            status = "success"
            started = time.time()
            if selected_bbox is None:
                status = "failed"
                error = "selected run has no bbox_path"
            else:
                guarded_output = _guarded_bbox_output(cfg, args.stage, category_name, mesh_id, mesh_tag)
                try:
                    _copy_bbox_dir(Path(selected_bbox), guarded_output, force=args.force)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            record = StageRecord.now(
                stage=args.stage,
                category=category_name,
                mesh_id=mesh_id,
                status=status,
                started_at=started,
                output_path=guarded_output,
                metadata={
                    "selection": selection.to_dict(),
                    "selected_source_bbox": selected_bbox,
                    "input_summary": baseline.get("summary"),
                    "local_refine_summary": refined.get("summary"),
                    "local_refine_elapsed_sec": refined.get("elapsed_sec"),
                    "local_refine_reused": bool(refined.get("reused")),
                },
                error=error,
            )
            writer.append(record)
            report["records"][f"{category_name}/{mesh_id}"] = {
                "category": category_name,
                "mesh_id": mesh_id,
                "runs": runs,
                "selection": selection.to_dict(),
                "guarded_record": asdict(record),
            }
            report["aggregate"] = _aggregate_records(report["records"])
            _write_json(output, report)

    report["aggregate"] = _aggregate_records(report["records"])
    _write_json(output, report)
    printed = report if args.print_full_report else _compact_report(report, output)
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0 if all(item["guarded_record"]["status"] == "success" for item in report["records"].values()) else 1


def _run_local_refine(args: argparse.Namespace, *, category: str, mesh_id: str, exp_tag: str) -> dict[str, Any]:
    command = [
        args.python,
        "-m",
        "smart",
        "--config",
        args.config,
        "--set",
        f"local_refine.input_stage={args.input_stage}",
        "--set",
        f"local_refine.max_step={args.max_step}",
        "--set",
        f"local_refine.action_unit={args.action_unit}",
        "--set",
        f"local_refine.reward_backend={args.reward_backend}",
        "--set",
        f"local_refine.backend={args.backend}",
        "--set",
        f"local_refine.timeout_sec={args.timeout}",
        "--set",
        f"local_refine.exp_tag={exp_tag}",
        "local_refine",
        "--category",
        category,
        "--mesh",
        mesh_id,
        "--force",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=max(int(args.timeout) + 90, 90),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "elapsed_sec": time.perf_counter() - started,
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "timeout": True,
        }
    return {
        "command": command,
        "elapsed_sec": time.perf_counter() - started,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _metric_tolerances(args: argparse.Namespace) -> dict[str, float]:
    if float(args.covered_tolerance) <= 0.0:
        return {}
    return {"Avg_Covered": float(args.covered_tolerance)}


def _run_eval(args: argparse.Namespace, *, category: str, mesh_id: str, stage: str, exp_tag: str) -> dict[str, Any]:
    output = _repo_path(args.output)
    eval_path = output.with_name(f"{output.stem}_{exp_tag}_eval.json")
    command = [
        args.python,
        "-m",
        "smart",
        "--config",
        args.config,
        "evaluate",
        "--stage",
        stage,
        "--category",
        category,
        "--mesh",
        mesh_id,
        "--chamfer-points",
        str(args.chamfer_points),
        "--output",
        str(eval_path),
        "--json",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=max(int(args.eval_timeout), 10),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 0,
            "evaluation_command": command,
            "evaluation_path": str(eval_path),
            "evaluation_time_sec": time.perf_counter() - started,
            "evaluation_returncode": 124,
            "evaluation_stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "evaluation_stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "evaluation_timeout": True,
        }
    result: dict[str, Any] = {
        "returncode": 0,
        "evaluation_command": command,
        "evaluation_path": str(eval_path),
        "evaluation_time_sec": time.perf_counter() - started,
        "evaluation_returncode": completed.returncode,
        "evaluation_stdout_tail": completed.stdout[-2000:],
        "evaluation_stderr_tail": completed.stderr[-2000:],
    }
    if completed.returncode == 0:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        result["summary"] = payload["summary"]
        result["records"] = payload["records"]
    return result


def _select_category_meshes(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, list[str]]:
    if args.mesh and not args.category:
        raise SystemExit("--mesh requires --category")
    allowed = {item.strip() for item in str(args.categories or "").split(",") if item.strip()}
    if args.category:
        allowed.add(args.category)
    limit = args.per_category_limit if args.per_category_limit is not None else args.mesh_limit
    out: dict[str, list[str]] = {}
    for category in cfg.get("categories", []):
        name = str(category["name"])
        if allowed and name not in allowed:
            continue
        if args.mesh:
            meshes = list(args.mesh)
        else:
            meshes = []
            for mesh_id in list_mesh_ids(category):
                if args.only_existing_input and bbox_dir_for_render(cfg, category, mesh_id, args.input_stage) is None:
                    continue
                meshes.append(mesh_id)
                if limit > 0 and len(meshes) >= limit:
                    break
        out[name] = meshes
    if not out:
        raise SystemExit("No categories selected")
    return out


def _selected_bbox_path(run: dict[str, Any]) -> str | None:
    records = run.get("records") or []
    if not records:
        return None
    value = records[0].get("bbox_path")
    return str(value) if value else None


def _guarded_bbox_output(
    cfg: dict[str, Any],
    stage: str,
    category: str,
    mesh_id: str,
    run_tag: str,
) -> Path:
    return workspace_path(cfg, stage, category, run_tag, "result", "updated0", mesh_id, "bboxs_steps0")


def _copy_bbox_dir(source: Path, destination: Path, *, force: bool) -> None:
    if destination.exists():
        if not force:
            return
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _repo_path(path: str | Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else REPO_ROOT / out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compact_report(report: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "output": str(output),
        "stage": report.get("stage"),
        "input_stage": report.get("input_stage"),
        "reuse_local_refine": report.get("reuse_local_refine"),
        "selection_mode": report.get("selection_mode"),
        "run_tag": report.get("run_tag"),
        "aggregate": report.get("aggregate", {}),
    }


def _select_local_refine(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> GuardedSelection:
    selection = select_quality_guarded_run(
        runs,
        baseline_label="input",
        candidate_labels=["local_refine"],
        tolerance=args.metric_tolerance,
        metric_tolerances=_metric_tolerances(args),
    )
    if args.selection_mode == "not_worse":
        return selection

    comparison = selection.comparisons.get("local_refine", {})
    if comparison.get("improved"):
        return GuardedSelection(
            selected_label="local_refine",
            baseline_label=selection.baseline_label,
            eligible_labels=selection.eligible_labels,
            rejected_labels=selection.rejected_labels,
            reason="candidate_quality_improved",
            comparisons=selection.comparisons,
        )

    rejected = {key: list(value) for key, value in selection.rejected_labels.items()}
    if "local_refine" not in rejected:
        rejected["local_refine"] = ["not_improved"]
    return GuardedSelection(
        selected_label="input",
        baseline_label=selection.baseline_label,
        eligible_labels=selection.eligible_labels,
        rejected_labels=rejected,
        reason="baseline_selected",
        comparisons=selection.comparisons,
    )


def _aggregate_records(records: dict[str, Any]) -> dict[str, Any]:
    total = len(records)
    status_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    local_not_worse = 0
    local_improved = 0
    local_worse = 0
    local_times: list[float] = []
    local_deltas: list[dict[str, float]] = []
    selected_local_deltas: list[dict[str, float]] = []
    categories: dict[str, dict[str, int]] = {}
    for record in records.values():
        guarded = record.get("guarded_record", {})
        status = str(guarded.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        category = str(record.get("category", "unknown"))
        cat = categories.setdefault(category, {"total": 0, "success": 0, "local_selected": 0, "input_selected": 0})
        cat["total"] += 1
        if status == "success":
            cat["success"] += 1
        selection = record.get("selection", {})
        selected = str(selection.get("selected_label", "unknown"))
        selected_counts[selected] = selected_counts.get(selected, 0) + 1
        if selected == "local_refine":
            cat["local_selected"] += 1
        if selected == "input":
            cat["input_selected"] += 1
        reason = str(selection.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        comparison = selection.get("comparisons", {}).get("local_refine")
        if isinstance(comparison, dict):
            deltas = comparison.get("deltas")
            if isinstance(deltas, dict):
                numeric_deltas = {key: float(value) for key, value in deltas.items()}
                local_deltas.append(numeric_deltas)
                if selected == "local_refine":
                    selected_local_deltas.append(numeric_deltas)
            if comparison.get("not_worse"):
                local_not_worse += 1
            if comparison.get("improved"):
                local_improved += 1
            if not comparison.get("not_worse"):
                local_worse += 1
        local_time = float(record.get("runs", {}).get("local_refine", {}).get("elapsed_sec", 0.0) or 0.0)
        if local_time > 0.0:
            local_times.append(local_time)
    return {
        "total": total,
        "status_counts": status_counts,
        "selected_counts": selected_counts,
        "reason_counts": reason_counts,
        "local_not_worse": local_not_worse,
        "local_improved": local_improved,
        "local_worse": local_worse,
        "mean_local_refine_elapsed_sec": sum(local_times) / len(local_times) if local_times else None,
        "mean_local_deltas": _mean_deltas(local_deltas),
        "mean_selected_local_deltas": _mean_deltas(selected_local_deltas),
        "categories": categories,
    }


def _mean_deltas(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
    }


if __name__ == "__main__":
    raise SystemExit(main())
