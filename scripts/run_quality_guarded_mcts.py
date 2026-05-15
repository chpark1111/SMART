#!/usr/bin/env python3
"""Run baseline and prior-guided MCTS, then keep the non-worse result.

This is a research runner for learned action priors. It does not change the
paper reproduction defaults: baseline and prior trajectories are both evaluated
with SMART metrics, and the selected bbox output is copied to an explicit
guarded stage only when it is no worse than the baseline.
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
from smart.pipeline.stages import latest_bbox_dir, list_mesh_ids, stage_root  # noqa: E402
from smart.quality import quality_gain_score, select_quality_guarded_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl_search_experimental.yaml")
    parser.add_argument("--category", help="Single category to run")
    parser.add_argument("--categories", default="", help="Comma-separated category filter for batch runs")
    parser.add_argument("--mesh", action="append", help="Mesh id; repeat for multiple meshes. Requires --category.")
    parser.add_argument("--mesh-limit", type=int, default=1, help="Used when --mesh is omitted")
    parser.add_argument("--per-category-limit", type=int, default=None, help="Alias for --mesh-limit in multi-category runs")
    parser.add_argument("--prior-path", required=True)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument(
        "--prior-weights",
        default="",
        help="Comma-separated learned-prior weights. When set, all weights are run as separate guarded candidates.",
    )
    parser.add_argument(
        "--adaptive-prior-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With multiple --prior-weights, run candidates sequentially and stop "
            "after the first guarded quality improvement. This saves runtime but "
            "can miss a later weight with a larger improvement."
        ),
    )
    parser.add_argument(
        "--adaptive-stop-mode",
        choices=("improved", "not_worse"),
        default="improved",
        help=(
            "Adaptive prior-weight stop rule. 'improved' stops only after exact "
            "metrics improve over baseline. 'not_worse' also stops after a "
            "faster non-worse candidate, trading possible later improvements for speed."
        ),
    )
    parser.add_argument("--mcts-iter", type=int, default=20)
    parser.add_argument("--max-step", type=int, default=20)
    parser.add_argument("--mcts-timeout", type=int, default=300)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--reward-backend", default="manifold_stateful")
    parser.add_argument("--mcts-backend", default="auto")
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--selection-objective",
        choices=("legacy", "quality_score"),
        default="legacy",
        help=(
            "legacy may keep faster identical candidates; quality_score selects "
            "only non-worse candidates with positive scalar SMART metric gain."
        ),
    )
    parser.add_argument(
        "--quality-weights",
        default="",
        help="Optional comma-separated metric weights, e.g. Avg_BVS=1,Avg_vIoU=2",
    )
    parser.add_argument("--stage", default="mcts_guarded")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output", default="runs/bench_exact/quality_guarded_mcts.json")
    parser.add_argument(
        "--only-existing-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only run meshes that already have refine outputs.",
    )
    parser.add_argument("--print-full-report", action="store_true", help="Print the full JSON report instead of a compact summary")
    parser.add_argument("--force", action="store_true", help="Overwrite existing guarded output dirs")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    category_meshes = _select_category_meshes(cfg, args)
    prior_weights = _parse_prior_weights(args)
    quality_weights = _parse_quality_weights(args.quality_weights)
    run_tag = args.run_tag or time.strftime("quality_guard_%Y%m%d_%H%M%S")
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": args.config,
        "categories": sorted(category_meshes),
        "meshes": category_meshes,
        "prior_path": args.prior_path,
        "prior_weight": args.prior_weight,
        "prior_weights": prior_weights,
        "adaptive_prior_weights": args.adaptive_prior_weights,
        "adaptive_stop_mode": args.adaptive_stop_mode,
        "mcts_iter": args.mcts_iter,
        "max_step": args.max_step,
        "reward_backend": args.reward_backend,
        "mcts_backend": args.mcts_backend,
        "stage": args.stage,
        "selection_objective": args.selection_objective,
        "quality_weights": quality_weights,
        "run_tag": run_tag,
        "records": {},
    }
    writer = ManifestWriter(workspace_path(cfg, "manifests"))

    for category_name, mesh_ids in category_meshes.items():
        for index, mesh_id in enumerate(mesh_ids, start=1):
            mesh_tag = f"{run_tag}_{category_name}_{index:03d}"
            baseline = _run_pair_member(args, category=category_name, mesh_id=mesh_id, exp_tag=f"{mesh_tag}_baseline", prior=False)
            runs = {"baseline": baseline}
            candidate_labels: list[str] = []
            skipped_candidate_labels: list[str] = []
            adaptive_stop_reason = None
            for weight in prior_weights:
                label = "prior" if len(prior_weights) == 1 else f"prior_{_weight_slug(weight)}"
                candidate_labels.append(label)
                runs[label] = _run_pair_member(
                    args,
                    category=category_name,
                    mesh_id=mesh_id,
                    exp_tag=f"{mesh_tag}_{label}",
                    prior=True,
                    prior_weight=weight,
                )
                if args.adaptive_prior_weights and len(prior_weights) > 1:
                    partial_selection = select_quality_guarded_run(
                        runs,
                        baseline_label="baseline",
                        candidate_labels=candidate_labels,
                        tolerance=args.metric_tolerance,
                        selection_objective=args.selection_objective,
                        quality_score_weights=quality_weights,
                    )
                    adaptive_stop_reason = _adaptive_stop_reason(
                        partial_selection,
                        mode=args.adaptive_stop_mode,
                    )
                    if adaptive_stop_reason:
                        break
            if len(candidate_labels) < len(prior_weights):
                skipped_candidate_labels = [
                    f"prior_{_weight_slug(weight)}"
                    for weight in prior_weights[len(candidate_labels):]
                ]
            selection = select_quality_guarded_run(
                runs,
                baseline_label="baseline",
                candidate_labels=candidate_labels,
                tolerance=args.metric_tolerance,
                selection_objective=args.selection_objective,
                quality_score_weights=quality_weights,
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
                    "baseline_summary": baseline.get("summary"),
                    "candidate_summaries": {
                        label: runs.get(label, {}).get("summary")
                        for label in candidate_labels
                    },
                    "baseline_elapsed_sec": baseline.get("elapsed_sec"),
                    "candidate_elapsed_sec": {
                        label: runs.get(label, {}).get("elapsed_sec")
                        for label in candidate_labels
                    },
                    "candidate_quality_score": {
                        label: quality_gain_score(selection.comparisons.get(label), weights=quality_weights)
                        for label in candidate_labels
                    },
                    "skipped_candidate_labels": skipped_candidate_labels,
                    "adaptive_stop_reason": adaptive_stop_reason,
                },
                error=error,
            )
            writer.append(record)
            report["records"][f"{category_name}/{mesh_id}"] = {
                "category": category_name,
                "mesh_id": mesh_id,
                "runs": runs,
                "candidate_labels": candidate_labels,
                "skipped_candidate_labels": skipped_candidate_labels,
                "adaptive_stop_reason": adaptive_stop_reason,
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


def _run_pair_member(
    args: argparse.Namespace,
    *,
    category: str,
    mesh_id: str,
    exp_tag: str,
    prior: bool,
    prior_weight: float | None = None,
) -> dict[str, Any]:
    run = _run_mcts(args, category=category, mesh_id=mesh_id, exp_tag=exp_tag, prior=prior, prior_weight=prior_weight)
    if run["returncode"] == 0:
        run.update(_run_eval(args, category=category, mesh_id=mesh_id, exp_tag=exp_tag))
    return run


def _run_mcts(
    args: argparse.Namespace,
    *,
    category: str,
    mesh_id: str,
    exp_tag: str,
    prior: bool,
    prior_weight: float | None = None,
) -> dict[str, Any]:
    weight = float(args.prior_weight if prior_weight is None else prior_weight)
    command = [
        args.python,
        "-m",
        "smart",
        "--config",
        args.config,
        "--set",
        f"mcts.mcts_iter={args.mcts_iter}",
        "--set",
        f"mcts.max_step={args.max_step}",
        "--set",
        f"mcts.reward_backend={args.reward_backend}",
        "--set",
        f"mcts.backend={args.mcts_backend}",
        "--set",
        f"mcts.timeout_sec={args.mcts_timeout}",
        "--set",
        f"mcts.exp_tag={exp_tag}",
        "--set",
        f"mcts.action_prior_weight={weight if prior else 0.0}",
    ]
    if prior:
        command.extend(
            [
                "--set",
                f"mcts.action_prior_path={args.prior_path}",
                "--set",
                "mcts.allow_search_order_changes=true",
            ]
        )
    command.extend(["mcts", "--category", category, "--mesh", mesh_id, "--force"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=max(int(args.mcts_timeout) + 90, 90),
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


def _run_eval(args: argparse.Namespace, *, category: str, mesh_id: str, exp_tag: str) -> dict[str, Any]:
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
        "mcts",
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
            "evaluation_command": command,
            "evaluation_path": str(eval_path),
            "evaluation_time_sec": time.perf_counter() - started,
            "evaluation_returncode": 124,
            "evaluation_stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "evaluation_stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "evaluation_timeout": True,
        }
    result: dict[str, Any] = {
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


def _find_category(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for category in cfg.get("categories", []):
        if category.get("name") == name:
            return category
    raise SystemExit(f"Unknown category in config: {name}")


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
            meshes = _candidate_meshes(cfg, category, limit, args.only_existing_refine)
        out[name] = meshes
    if not out:
        raise SystemExit("No categories selected")
    return out


def _candidate_meshes(
    cfg: dict[str, Any],
    category: dict[str, Any],
    limit: int,
    only_existing_refine: bool,
) -> list[str]:
    selected: list[str] = []
    refine_root = stage_root(cfg, "refine", category)
    for mesh_id in list_mesh_ids(category):
        if only_existing_refine and latest_bbox_dir(refine_root, mesh_id) is None:
            continue
        selected.append(mesh_id)
        if limit > 0 and len(selected) >= limit:
            break
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


def _parse_prior_weights(args: argparse.Namespace) -> list[float]:
    if str(args.prior_weights or "").strip():
        weights = [
            float(item.strip())
            for item in str(args.prior_weights).split(",")
            if item.strip()
        ]
    else:
        weights = [float(args.prior_weight)]
    if not weights:
        raise SystemExit("No prior weights configured")
    return weights


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


def _weight_slug(weight: float) -> str:
    text = ("%g" % float(weight)).replace("-", "m").replace(".", "p")
    return f"w{text}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compact_report(report: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "output": str(output),
        "stage": report.get("stage"),
        "run_tag": report.get("run_tag"),
        "prior_path": report.get("prior_path"),
        "prior_weights": report.get("prior_weights"),
        "adaptive_prior_weights": report.get("adaptive_prior_weights"),
        "adaptive_stop_mode": report.get("adaptive_stop_mode"),
        "selection_objective": report.get("selection_objective"),
        "aggregate": report.get("aggregate", {}),
    }


def _aggregate_records(records: dict[str, Any]) -> dict[str, Any]:
    total = len(records)
    status_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    rejected_candidate_labels = 0
    meshes_with_candidate_rejection = 0
    meshes_all_candidates_worse = 0
    prior_not_worse = 0
    prior_improved = 0
    speedups: list[float] = []
    skipped_candidate_labels = 0
    candidate_runs_executed = 0
    candidate_runs_failed = 0
    max_candidate_labels = 0
    adaptive_stops: dict[str, int] = {}
    categories: dict[str, dict[str, int]] = {}
    for record in records.values():
        guarded = record.get("guarded_record", {})
        status = str(guarded.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        category = str(record.get("category", "unknown"))
        cat = categories.setdefault(category, {"total": 0, "success": 0, "prior_selected": 0, "baseline_selected": 0})
        cat["total"] += 1
        if status == "success":
            cat["success"] += 1
        selection = record.get("selection", {})
        selected = str(selection.get("selected_label", "unknown"))
        selected_counts[selected] = selected_counts.get(selected, 0) + 1
        if selected != "baseline" and selected != "unknown":
            cat["prior_selected"] += 1
        if selected == "baseline":
            cat["baseline_selected"] += 1
        reason = str(selection.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        candidate_comparisons = [
            comparison
            for label, comparison in selection.get("comparisons", {}).items()
            if str(label) != "baseline"
        ]
        rejected_count = sum(
            1
            for label in selection.get("rejected_labels", {})
            if str(label) != "baseline"
        )
        rejected_candidate_labels += rejected_count
        if rejected_count:
            meshes_with_candidate_rejection += 1
        if candidate_comparisons and any(comparison.get("not_worse") for comparison in candidate_comparisons):
            prior_not_worse += 1
        if candidate_comparisons and any(comparison.get("improved") for comparison in candidate_comparisons):
            prior_improved += 1
        if candidate_comparisons and not any(comparison.get("not_worse") for comparison in candidate_comparisons):
            meshes_all_candidates_worse += 1
        runs = record.get("runs", {})
        baseline_time = float(runs.get("baseline", {}).get("elapsed_sec", 0.0) or 0.0)
        candidate_times = [
            float(run.get("elapsed_sec", 0.0) or 0.0)
            for label, run in runs.items()
            if str(label) != "baseline" and isinstance(run, dict)
        ]
        candidate_runs = [
            run
            for label, run in runs.items()
            if str(label) != "baseline" and isinstance(run, dict)
        ]
        candidate_runs_executed += len(candidate_runs)
        candidate_runs_failed += sum(1 for run in candidate_runs if run.get("returncode") not in (None, 0))
        candidate_times = [value for value in candidate_times if value > 0.0]
        if baseline_time > 0.0 and candidate_times:
            speedups.append(baseline_time / min(candidate_times))
        labels = record.get("candidate_labels") or []
        skipped = record.get("skipped_candidate_labels") or []
        if isinstance(labels, list) and isinstance(skipped, list):
            max_candidate_labels = max(max_candidate_labels, len(labels) + len(skipped))
        skipped_candidate_labels += len(skipped) if isinstance(skipped, list) else 0
        stop_reason = str(record.get("adaptive_stop_reason") or "")
        if stop_reason:
            adaptive_stops[stop_reason] = adaptive_stops.get(stop_reason, 0) + 1
    possible_candidate_runs = total * max_candidate_labels if max_candidate_labels else candidate_runs_executed
    possible_total_mcts_runs = total + possible_candidate_runs
    executed_total_mcts_runs = total + candidate_runs_executed
    return {
        "total": total,
        "status_counts": status_counts,
        "selected_counts": selected_counts,
        "reason_counts": reason_counts,
        "prior_rejected_by_quality": rejected_candidate_labels,
        "candidate_rejections_by_quality": rejected_candidate_labels,
        "meshes_with_candidate_rejection": meshes_with_candidate_rejection,
        "prior_not_worse": prior_not_worse,
        "prior_improved": prior_improved,
        "prior_worse": meshes_all_candidates_worse,
        "meshes_all_candidates_worse": meshes_all_candidates_worse,
        "mean_prior_speedup": sum(speedups) / len(speedups) if speedups else None,
        "skipped_candidate_labels": skipped_candidate_labels,
        "candidate_runs_executed": candidate_runs_executed,
        "candidate_runs_failed": candidate_runs_failed,
        "possible_candidate_runs": possible_candidate_runs,
        "candidate_runs_skipped": max(possible_candidate_runs - candidate_runs_executed, 0),
        "candidate_run_reduction": (
            max(possible_candidate_runs - candidate_runs_executed, 0) / possible_candidate_runs
            if possible_candidate_runs
            else None
        ),
        "executed_total_mcts_runs": executed_total_mcts_runs,
        "possible_total_mcts_runs": possible_total_mcts_runs,
        "total_mcts_run_reduction": (
            max(possible_total_mcts_runs - executed_total_mcts_runs, 0) / possible_total_mcts_runs
            if possible_total_mcts_runs
            else None
        ),
        "adaptive_stops": adaptive_stops,
        "categories": categories,
    }


def _adaptive_stop_reason(selection: Any, *, mode: str = "improved") -> str | None:
    """Stop after a prior candidate proves a guarded quality improvement.

    The default keeps adaptive mode conservative: exact SMART metrics must
    already prefer a learned-search output over baseline before we skip later
    weights. The optional not-worse mode still requires a non-worse exact
    candidate, but it can miss a later weight with larger quality gains.
    """

    reason = getattr(selection, "reason", "")
    if reason in {"candidate_quality_improved", "candidate_quality_score_improved"}:
        return reason
    if mode == "not_worse" and reason == "candidate_not_worse_faster":
        return reason
    return None


if __name__ == "__main__":
    raise SystemExit(main())
