#!/usr/bin/env python3
"""Benchmark a fixed SMART MCTS action prior by category.

This script does not train a prior. It evaluates a packaged or generated prior
against the exact SMART reward path by running baseline MCTS and prior-guided
MCTS on the same meshes, then comparing paper metrics category by category.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from smart.pipeline.config import load_config  # noqa: E402
from smart.pipeline.stages import latest_bbox_dir, list_mesh_ids, stage_root  # noqa: E402


METRIC_KEYS = (
    "Avg_num_box",
    "Avg_BVS",
    "Avg_MOV",
    "Avg_TOV",
    "Avg_Covered",
    "Avg_vIoU",
    "Avg_cub_CD",
)
LOWER_IS_BETTER = ("Avg_BVS", "Avg_MOV", "Avg_TOV", "Avg_cub_CD")
HIGHER_IS_BETTER = ("Avg_Covered", "Avg_vIoU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expanded_full.yaml")
    parser.add_argument("--categories", default="", help="Comma-separated category filter")
    parser.add_argument("--per-category-limit", type=int, default=20)
    parser.add_argument("--prior-path", required=True)
    parser.add_argument("--weights", default="0,0.1")
    parser.add_argument("--mcts-iter", type=int, default=20)
    parser.add_argument("--max-step", type=int, default=20)
    parser.add_argument("--mcts-timeout", type=int, default=300)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--reward-backend", default="manifold_stateful")
    parser.add_argument("--mcts-backend", default="auto")
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    parser.add_argument("--output", default="runs/bench_exact/fixed_prior_by_category.json")
    parser.add_argument("--exp-prefix", default="")
    parser.add_argument(
        "--only-existing-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only benchmark meshes that already have refine outputs.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs from --output if present")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort on the first failed run instead of recording it and continuing.",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    weights = [float(item) for item in args.weights.split(",") if item.strip()]
    if 0.0 not in weights:
        weights.insert(0, 0.0)

    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    allowed = {item.strip() for item in args.categories.split(",") if item.strip()}
    run_tag = args.exp_prefix or time.strftime("fixed_prior_%Y%m%d_%H%M%S")
    if args.resume and output.exists():
        results = json.loads(output.read_text(encoding="utf-8"))
        results.update(
            {
                "config": args.config,
                "prior_path": args.prior_path,
                "weights": weights,
                "mcts_iter": args.mcts_iter,
                "max_step": args.max_step,
                "mcts_timeout": args.mcts_timeout,
                "eval_timeout": args.eval_timeout,
                "reward_backend": args.reward_backend,
                "mcts_backend": args.mcts_backend,
                "run_tag": run_tag,
            }
        )
        results.setdefault("categories", {})
    else:
        results = {
            "config": args.config,
            "prior_path": args.prior_path,
            "weights": weights,
            "mcts_iter": args.mcts_iter,
            "max_step": args.max_step,
            "mcts_timeout": args.mcts_timeout,
            "eval_timeout": args.eval_timeout,
            "reward_backend": args.reward_backend,
            "mcts_backend": args.mcts_backend,
            "run_tag": run_tag,
            "categories": {},
        }
    _write_json(output, results)

    for category in cfg.get("categories", []):
        name = str(category["name"])
        if allowed and name not in allowed:
            continue
        meshes = _select_meshes(cfg, category, args.per_category_limit, args.only_existing_refine)
        category_record = results["categories"].get(name)
        if not isinstance(category_record, dict):
            category_record = {
                "configured_meshes": len(list_mesh_ids(category)),
                "selected_meshes": meshes,
                "per_mesh": {},
            }
        category_record["configured_meshes"] = len(list_mesh_ids(category))
        category_record["selected_meshes"] = meshes
        category_record.setdefault("per_mesh", {})
        results["categories"][name] = category_record
        _write_json(output, results)

        for mesh_idx, mesh_id in enumerate(meshes, start=1):
            mesh_record = category_record["per_mesh"].get(mesh_id)
            if not isinstance(mesh_record, dict):
                mesh_record = {"runs": {}}
            category_record["per_mesh"][mesh_id] = mesh_record
            for weight in weights:
                label = _weight_label(weight)
                exp_tag = f"{run_tag}_{name}_{mesh_idx:03d}_{label}"
                existing_run = mesh_record.get("runs", {}).get(label)
                if (
                    args.resume
                    and isinstance(existing_run, dict)
                    and existing_run.get("returncode") == 0
                    and existing_run.get("evaluation_returncode") == 0
                    and "summary" in existing_run
                ):
                    mesh_record["metric_diffs_vs_w0"] = _metric_diffs(mesh_record["runs"])
                    mesh_record["speedups_vs_w0"] = _speedups(mesh_record["runs"])
                    mesh_record["quality_vs_w0"] = _quality_vs_w0(
                        mesh_record["runs"], args.metric_tolerance
                    )
                    continue
                run_record = _run_mcts(args, category=name, mesh_id=mesh_id, weight=weight, exp_tag=exp_tag)
                eval_record = (
                    _run_eval(args, category=name, mesh_id=mesh_id, output=output, exp_tag=exp_tag)
                    if run_record["returncode"] == 0
                    else {"evaluation_returncode": None}
                )
                run_record.update(eval_record)
                mesh_record["runs"][label] = run_record
                mesh_record["metric_diffs_vs_w0"] = _metric_diffs(mesh_record["runs"])
                mesh_record["speedups_vs_w0"] = _speedups(mesh_record["runs"])
                mesh_record["quality_vs_w0"] = _quality_vs_w0(
                    mesh_record["runs"], args.metric_tolerance
                )
                category_record["aggregate"] = _aggregate_category(category_record["per_mesh"], args.metric_tolerance)
                results["aggregate"] = _aggregate_all(results["categories"], args.metric_tolerance)
                _write_json(output, results)
                if args.stop_on_failure and (
                    run_record["returncode"] != 0 or eval_record["evaluation_returncode"] != 0
                ):
                    print(json.dumps(results, indent=2, sort_keys=True))
                    return int(run_record["returncode"] or eval_record["evaluation_returncode"])

    results["aggregate"] = _aggregate_all(results["categories"], args.metric_tolerance)
    _write_json(output, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _select_meshes(
    cfg: dict[str, Any],
    category: dict[str, Any],
    limit: int,
    only_existing_refine: bool,
) -> list[str]:
    meshes: list[str] = []
    refine_root = stage_root(cfg, "refine", category)
    for mesh_id in list_mesh_ids(category):
        if only_existing_refine and latest_bbox_dir(refine_root, mesh_id) is None:
            continue
        meshes.append(mesh_id)
        if limit > 0 and len(meshes) >= limit:
            break
    return meshes


def _run_mcts(
    args: argparse.Namespace,
    *,
    category: str,
    mesh_id: str,
    weight: float,
    exp_tag: str,
) -> dict[str, Any]:
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
        f"mcts.action_prior_weight={weight}",
    ]
    if weight != 0.0:
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


def _run_eval(
    args: argparse.Namespace,
    *,
    category: str,
    mesh_id: str,
    output: Path,
    exp_tag: str,
) -> dict[str, Any]:
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


def _weight_label(weight: float) -> str:
    return "w%s" % str(weight).replace("-", "m").replace(".", "p")


def _metric_diffs(runs: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline = runs.get("w0p0") or runs.get("w0")
    if not baseline or "summary" not in baseline:
        return {}
    out: dict[str, dict[str, float]] = {}
    baseline_summary = baseline["summary"]
    for label, run in runs.items():
        summary = run.get("summary")
        if not summary:
            continue
        out[label] = {
            key: abs(float(summary[key]) - float(baseline_summary[key]))
            for key in METRIC_KEYS
        }
    return out


def _quality_vs_w0(runs: dict[str, Any], tolerance: float) -> dict[str, Any]:
    baseline = runs.get("w0p0") or runs.get("w0")
    if not baseline or "summary" not in baseline:
        return {}
    baseline_summary = baseline["summary"]
    out: dict[str, Any] = {}
    for label, run in runs.items():
        summary = run.get("summary")
        if not summary:
            continue
        deltas: dict[str, float] = {}
        worse: list[str] = []
        improved: list[str] = []
        for key in LOWER_IS_BETTER:
            delta = float(summary[key]) - float(baseline_summary[key])
            deltas[key] = delta
            if delta > tolerance:
                worse.append(key)
            elif delta < -tolerance:
                improved.append(key)
        for key in HIGHER_IS_BETTER:
            delta = float(summary[key]) - float(baseline_summary[key])
            deltas[key] = delta
            if delta < -tolerance:
                worse.append(key)
            elif delta > tolerance:
                improved.append(key)
        out[label] = {
            "deltas": deltas,
            "improved_metrics": improved,
            "worse_metrics": worse,
            "not_worse": not worse,
            "improved": bool(improved) and not worse,
            "worse": bool(worse),
        }
    return out


def _speedups(runs: dict[str, Any]) -> dict[str, float]:
    baseline = runs.get("w0p0") or runs.get("w0")
    if not baseline or baseline.get("returncode") != 0 or baseline.get("evaluation_returncode") != 0:
        return {}
    base_time = float(baseline.get("elapsed_sec", 0.0) or 0.0)
    if base_time <= 0.0:
        return {}
    return {
        label: base_time / float(run.get("elapsed_sec", 1.0))
        for label, run in runs.items()
        if run.get("returncode") == 0
        and run.get("evaluation_returncode") == 0
        if float(run.get("elapsed_sec", 0.0) or 0.0) > 0.0
    }


def _aggregate_category(per_mesh: dict[str, Any], tolerance: float) -> dict[str, Any]:
    return _aggregate_mesh_records(per_mesh.values(), tolerance)


def _aggregate_all(categories: dict[str, Any], tolerance: float) -> dict[str, Any]:
    out = {}
    for category, record in categories.items():
        out[category] = _aggregate_category(record.get("per_mesh", {}), tolerance)
    all_meshes = []
    for record in categories.values():
        all_meshes.extend(record.get("per_mesh", {}).values())
    out["all"] = _aggregate_mesh_records(all_meshes, tolerance)
    return out


def _aggregate_mesh_records(mesh_records: Any, tolerance: float) -> dict[str, Any]:
    records = list(mesh_records)
    speedups: dict[str, list[float]] = {}
    metric_diffs: dict[str, dict[str, list[float]]] = {}
    identical: dict[str, int] = {}
    totals: dict[str, int] = {}
    quality_totals: dict[str, int] = {}
    quality_not_worse: dict[str, int] = {}
    quality_improved: dict[str, int] = {}
    quality_worse: dict[str, int] = {}
    quality_worse_metrics: dict[str, dict[str, int]] = {}
    quality_improved_metrics: dict[str, dict[str, int]] = {}
    for mesh_record in records:
        for label, speedup in mesh_record.get("speedups_vs_w0", {}).items():
            speedups.setdefault(label, []).append(float(speedup))
        for label, diffs in mesh_record.get("metric_diffs_vs_w0", {}).items():
            totals[label] = totals.get(label, 0) + 1
            if all(float(value) <= tolerance for value in diffs.values()):
                identical[label] = identical.get(label, 0) + 1
            for key, value in diffs.items():
                metric_diffs.setdefault(label, {}).setdefault(key, []).append(float(value))
        for label, quality in mesh_record.get("quality_vs_w0", {}).items():
            quality_totals[label] = quality_totals.get(label, 0) + 1
            if quality.get("not_worse"):
                quality_not_worse[label] = quality_not_worse.get(label, 0) + 1
            if quality.get("improved"):
                quality_improved[label] = quality_improved.get(label, 0) + 1
            if quality.get("worse"):
                quality_worse[label] = quality_worse.get(label, 0) + 1
            for key in quality.get("worse_metrics", []):
                quality_worse_metrics.setdefault(label, {}).setdefault(key, 0)
                quality_worse_metrics[label][key] += 1
            for key in quality.get("improved_metrics", []):
                quality_improved_metrics.setdefault(label, {}).setdefault(key, 0)
                quality_improved_metrics[label][key] += 1
    return {
        "count": len(records),
        "mean_speedup": {
            label: sum(values) / len(values)
            for label, values in speedups.items()
            if values
        },
        "quality_totals": quality_totals,
        "quality_not_worse": quality_not_worse,
        "quality_improved": quality_improved,
        "quality_worse": quality_worse,
        "quality_worse_metrics": quality_worse_metrics,
        "quality_improved_metrics": quality_improved_metrics,
        "metric_identical": identical,
        "metric_totals": totals,
        "mean_metric_abs_diff": {
            label: {
                key: sum(values) / len(values)
                for key, values in per_metric.items()
                if values
            }
            for label, per_metric in metric_diffs.items()
        },
        "max_metric_abs_diff": {
            label: {
                key: max(values)
                for key, values in per_metric.items()
                if values
            }
            for label, per_metric in metric_diffs.items()
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
