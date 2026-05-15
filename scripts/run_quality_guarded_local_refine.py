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
from smart.local_refine_gate import load_local_refine_gate, score_local_refine_gate  # noqa: E402
from smart.quality import GuardedSelection, quality_gain_score, select_quality_guarded_run  # noqa: E402


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
    parser.add_argument(
        "--from-input-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select meshes from the input stage manifest instead of the configured mesh roots.",
    )
    parser.add_argument("--max-step", type=int, default=100)
    parser.add_argument("--action-unit", type=float, default=0.005)
    parser.add_argument("--reward-backend", default="manifold_stateful")
    parser.add_argument("--backend", default="rust_stateful")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--quality-weights",
        default="",
        help="Optional comma-separated metric weights for final-return trace labels, e.g. Avg_BVS=1,Avg_vIoU=1",
    )
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
    parser.add_argument("--gate-path", default="", help="Optional local-refine gate JSON; skips local refine below threshold")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--final-return-trace-output",
        default="",
        help="Optional JSONL output. Local-refine action traces are relabeled with final exact quality gain.",
    )
    parser.add_argument(
        "--trace-actions-dir",
        default="",
        help="Directory for temporary local_refine action traces when --final-return-trace-output is set.",
    )
    parser.add_argument("--print-full-report", action="store_true", help="Print the full JSON report instead of a compact summary")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    category_meshes = _select_category_meshes(cfg, args)
    quality_weights = _parse_quality_weights(args.quality_weights)
    gate_payload = load_local_refine_gate(_repo_path(args.gate_path)) if args.gate_path else None
    run_tag = args.run_tag or time.strftime("local_refine_guard_%Y%m%d_%H%M%S")
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_trace_output = _repo_path(args.final_return_trace_output) if args.final_return_trace_output else None
    if final_trace_output is not None:
        final_trace_output.parent.mkdir(parents=True, exist_ok=True)
        final_trace_output.write_text("", encoding="utf-8")
    trace_actions_dir = _trace_actions_dir(args, output, final_trace_output)
    if trace_actions_dir is not None:
        trace_actions_dir.mkdir(parents=True, exist_ok=True)
    writer = ManifestWriter(workspace_path(cfg, "manifests"))
    report: dict[str, Any] = {
        "config": args.config,
        "input_stage": args.input_stage,
        "stage": args.stage,
        "from_input_manifest": args.from_input_manifest,
        "max_step": args.max_step,
        "action_unit": args.action_unit,
        "metric_tolerance": args.metric_tolerance,
        "covered_tolerance": args.covered_tolerance,
        "quality_weights": quality_weights,
        "selection_mode": args.selection_mode,
        "reward_backend": args.reward_backend,
        "backend": args.backend,
        "reuse_local_refine": args.reuse_local_refine,
        "gate_path": args.gate_path,
        "gate_threshold": args.gate_threshold,
        "final_return_trace_output": str(final_trace_output) if final_trace_output is not None else "",
        "run_tag": run_tag,
        "categories": sorted(category_meshes),
        "meshes": category_meshes,
        "records": {},
    }

    for category_name, mesh_ids in category_meshes.items():
        for index, mesh_id in enumerate(mesh_ids, start=1):
            mesh_tag = f"{run_tag}_{category_name}_{index:03d}"
            baseline = _run_eval(args, category=category_name, mesh_id=mesh_id, stage=args.input_stage, exp_tag=f"{mesh_tag}_input")
            gate_decision = _gate_decision(gate_payload, args, category_name, mesh_id, baseline)
            if gate_decision.get("skip_local_refine"):
                refined = {
                    "command": None,
                    "elapsed_sec": 0.0,
                    "returncode": 0,
                    "skipped_by_gate": True,
                    "gate": gate_decision,
                }
            elif args.reuse_local_refine:
                refined = {
                    "command": None,
                    "elapsed_sec": 0.0,
                    "returncode": 0,
                    "reused": True,
                    "gate": gate_decision,
                }
            else:
                trace_path = _local_refine_trace_path(trace_actions_dir, mesh_tag)
                refined = _run_local_refine(args, category=category_name, mesh_id=mesh_id, exp_tag=mesh_tag, trace_path=trace_path)
                refined["gate"] = gate_decision
            if refined.get("returncode") == 0 and not refined.get("skipped_by_gate"):
                refined.update(_run_eval(args, category=category_name, mesh_id=mesh_id, stage="local_refine", exp_tag=f"{mesh_tag}_local_refine"))
            runs = {"input": baseline, "local_refine": refined}
            selection = _select_local_refine(args, runs)
            final_trace_rows = _append_final_return_trace_rows(
                final_trace_output,
                category=category_name,
                mesh_id=mesh_id,
                runs=runs,
                selection=selection.to_dict(),
                quality_weights=quality_weights,
            )
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
                    "local_refine_skipped_by_gate": bool(refined.get("skipped_by_gate")),
                    "gate": gate_decision,
                    "final_return_trace_rows": final_trace_rows,
                },
                error=error,
            )
            writer.append(record)
            report["records"][f"{category_name}/{mesh_id}"] = {
                "category": category_name,
                "mesh_id": mesh_id,
                "runs": runs,
                "selection": selection.to_dict(),
                "final_return_trace_rows": final_trace_rows,
                "guarded_record": asdict(record),
            }
            report["aggregate"] = _aggregate_records(report["records"])
            _write_json(output, report)

    report["aggregate"] = _aggregate_records(report["records"])
    _write_json(output, report)
    printed = report if args.print_full_report else _compact_report(report, output)
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0 if all(item["guarded_record"]["status"] == "success" for item in report["records"].values()) else 1


def _run_local_refine(
    args: argparse.Namespace,
    *,
    category: str,
    mesh_id: str,
    exp_tag: str,
    trace_path: Path | None = None,
) -> dict[str, Any]:
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
    ]
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--set", f"local_refine.trace_actions_path={trace_path}"])
    command.extend(
        [
            "local_refine",
            "--category",
            category,
            "--mesh",
            mesh_id,
            "--force",
        ]
    )
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
            "trace_actions_path": str(trace_path) if trace_path is not None else "",
        }
    return {
        "command": command,
        "elapsed_sec": time.perf_counter() - started,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "trace_actions_path": str(trace_path) if trace_path is not None else "",
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
    if args.from_input_manifest:
        out = _select_meshes_from_manifest(cfg, args.input_stage, allowed, args)
        if not out:
            raise SystemExit(f"No meshes found in manifest for stage={args.input_stage}")
        return out
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


def _select_meshes_from_manifest(
    cfg: dict[str, Any],
    stage: str,
    allowed: set[str],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    limit = args.per_category_limit if args.per_category_limit is not None else args.mesh_limit
    manifest = workspace_path(cfg, "manifests", f"{stage}.jsonl")
    if not manifest.exists():
        return {}
    configured_categories = {str(category["name"]) for category in cfg.get("categories", [])}
    selected: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    with manifest.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            category = str(record.get("category") or "")
            mesh_id = str(record.get("mesh_id") or "")
            if not category or not mesh_id:
                continue
            if allowed and category not in allowed:
                continue
            if category not in configured_categories:
                continue
            if record.get("status") != "success" or not record.get("output_path"):
                continue
            key = (category, mesh_id)
            if key in seen:
                continue
            seen.add(key)
            meshes = selected.setdefault(category, [])
            if limit <= 0 or len(meshes) < limit:
                meshes.append(mesh_id)
    return selected


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


def _trace_actions_dir(args: argparse.Namespace, output: Path, final_trace_output: Path | None) -> Path | None:
    if args.trace_actions_dir:
        return _repo_path(args.trace_actions_dir)
    if final_trace_output is None:
        return None
    return output.with_name(f"{output.stem}_local_refine_traces")


def _local_refine_trace_path(trace_actions_dir: Path | None, exp_tag: str) -> Path | None:
    if trace_actions_dir is None:
        return None
    return trace_actions_dir / f"{_safe_tag(exp_tag)}.jsonl"


def _append_final_return_trace_rows(
    output: Path | None,
    *,
    category: str,
    mesh_id: str,
    runs: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    quality_weights: dict[str, float],
) -> int:
    if output is None:
        return 0
    run = runs.get("local_refine", {})
    trace_path = str(run.get("trace_actions_path", "") or "")
    if not trace_path:
        return 0
    path = Path(trace_path)
    if not path.exists():
        return 0
    comparison = selection.get("comparisons", {}).get("local_refine") or {}
    final_not_worse = bool(comparison.get("not_worse", False))
    final_improved = bool(comparison.get("improved", False))
    final_deltas = dict(comparison.get("deltas", {}))
    final_worse_metrics = list(comparison.get("worse_metrics", []))
    final_improved_metrics = list(comparison.get("improved_metrics", []))
    quality_score = _final_return_quality_score(
        comparison,
        weights=quality_weights,
        not_worse=final_not_worse,
        worse_metrics=final_worse_metrics,
    )
    selected_label = str(selection.get("selected_label", ""))
    rows_written = 0
    with output.open("a", encoding="utf-8") as out_file:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            action_reward = float(row.get("reward", 0.0) or 0.0)
            row.update(
                {
                    "schema_version": 4,
                    "record_type": "local_refine_final_return",
                    "original_record_type": str(row.get("record_type", "")),
                    "category": str(row.get("category", category) or category),
                    "mesh": str(row.get("mesh", mesh_id) or mesh_id),
                    "run_label": "local_refine",
                    "selected_run": bool(selected_label == "local_refine"),
                    "selection_label": selected_label,
                    "selection_reason": str(selection.get("reason", "")),
                    "action_reward": action_reward,
                    "reward": float(quality_score),
                    "final_quality_score": float(quality_score),
                    "final_not_worse": final_not_worse,
                    "final_improved": final_improved,
                    "final_deltas": final_deltas,
                    "final_worse_metrics": final_worse_metrics,
                    "final_improved_metrics": final_improved_metrics,
                }
            )
            out_file.write(json.dumps(row, sort_keys=True) + "\n")
            rows_written += 1
    return rows_written


def _final_return_quality_score(
    comparison: dict[str, Any],
    *,
    weights: dict[str, float],
    not_worse: bool,
    worse_metrics: list[str],
) -> float:
    score = quality_gain_score(comparison, weights=weights)
    if not_worse:
        return float(score)
    deltas = comparison.get("deltas", {})
    violation = 0.0
    for metric in worse_metrics:
        violation += abs(float(deltas.get(metric, 0.0))) * float(weights.get(metric, 1.0))
    return -max(abs(float(score)), violation, 1.0e-9)


def _parse_quality_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid --quality-weights item: {item!r}")
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value)
    return weights


def _safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compact_report(report: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "output": str(output),
        "stage": report.get("stage"),
        "input_stage": report.get("input_stage"),
        "from_input_manifest": report.get("from_input_manifest"),
        "reuse_local_refine": report.get("reuse_local_refine"),
        "gate_path": report.get("gate_path"),
        "gate_threshold": report.get("gate_threshold"),
        "selection_mode": report.get("selection_mode"),
        "final_return_trace_output": report.get("final_return_trace_output"),
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


def _gate_decision(
    gate_payload: dict[str, Any] | None,
    args: argparse.Namespace,
    category: str,
    mesh_id: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if gate_payload is None:
        return {"enabled": False, "skip_local_refine": False}
    summary = baseline.get("summary")
    if not isinstance(summary, dict):
        return {
            "enabled": True,
            "skip_local_refine": False,
            "reason": "missing_input_summary",
            "threshold": float(args.gate_threshold),
        }
    row = {
        "category": category,
        "mesh_id": mesh_id,
        "input_Avg_num_box": summary.get("Avg_num_box", 0.0),
        "input_Avg_BVS": summary.get("Avg_BVS", 0.0),
        "input_Avg_MOV": summary.get("Avg_MOV", 0.0),
        "input_Avg_TOV": summary.get("Avg_TOV", 0.0),
        "input_Avg_Covered": summary.get("Avg_Covered", 0.0),
        "input_Avg_vIoU": summary.get("Avg_vIoU", 0.0),
        "input_Avg_cub_CD": summary.get("Avg_cub_CD", 0.0),
    }
    probability = score_local_refine_gate(gate_payload, row)
    threshold = float(args.gate_threshold)
    skip = probability < threshold
    return {
        "enabled": True,
        "skip_local_refine": skip,
        "probability": probability,
        "threshold": threshold,
        "reason": "below_threshold" if skip else "above_threshold",
    }


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
    gate_scored = 0
    gate_skipped = 0
    gate_probabilities: list[float] = []
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
        gate = record.get("runs", {}).get("local_refine", {}).get("gate")
        if isinstance(gate, dict) and gate.get("enabled"):
            gate_scored += 1
            if gate.get("skip_local_refine"):
                gate_skipped += 1
            if gate.get("probability") is not None:
                gate_probabilities.append(float(gate["probability"]))
    return {
        "total": total,
        "status_counts": status_counts,
        "selected_counts": selected_counts,
        "reason_counts": reason_counts,
        "local_not_worse": local_not_worse,
        "local_improved": local_improved,
        "local_worse": local_worse,
        "mean_local_refine_elapsed_sec": sum(local_times) / len(local_times) if local_times else None,
        "gate_scored": gate_scored,
        "gate_skipped": gate_skipped,
        "mean_gate_probability": sum(gate_probabilities) / len(gate_probabilities) if gate_probabilities else None,
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
