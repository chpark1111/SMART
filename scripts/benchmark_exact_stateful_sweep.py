from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from smart.pipeline.config import load_config


METRIC_KEYS = (
    "Avg_num_box",
    "Avg_BVS",
    "Avg_MOV",
    "Avg_TOV",
    "Avg_Covered",
    "Avg_vIoU",
    "Avg_cub_CD",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact legacy Manifold and exact stateful Manifold backends "
            "across configured SMART meshes."
        )
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument("--stage", choices=["refine", "mcts"], default="mcts")
    parser.add_argument("--categories", default="", help="Comma-separated category filter")
    parser.add_argument("--mesh", action="append", help="Limit to specific mesh id; repeatable")
    parser.add_argument("--target-limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--refine-max-step", type=int, default=50)
    parser.add_argument("--mcts-iter", type=int, default=20)
    parser.add_argument("--mcts-max-step", type=int, default=20)
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    parser.add_argument("--output", default="runs/bench_exact/exact_stateful_sweep.json")
    parser.add_argument(
        "--trace-actions",
        action="store_true",
        help="Write action traces for each backend and summarize first divergence vs manifold.",
    )
    parser.add_argument(
        "--trace-dir",
        default="runs/bench_exact/traces",
        help="Directory used with --trace-actions.",
    )
    parser.add_argument(
        "--include-bridge",
        action="store_true",
        help="Also benchmark reward_backend=manifold_bridge as an exact bridge baseline.",
    )
    parser.add_argument(
        "--stateful-control-backend",
        choices=["auto", "rust_stateful"],
        default="auto",
        help=(
            "Control backend for manifold_stateful. Default auto preserves the "
            "legacy MCTS tree; rust_stateful is a search-order-changing experiment."
        ),
    )
    parser.add_argument(
        "--stateful-union-cache",
        action="store_true",
        help=(
            "Enable stateful union_except_i cache. Off by default because the "
            "cache changes boolean grouping and can move near-tie actions."
        ),
    )
    parser.add_argument(
        "--include-properties-volume",
        action="store_true",
        help=(
            "Also benchmark reward_backend=manifold_stateful with "
            "manifold_volume_method=properties. This is an opt-in research speed "
            "probe; the default backend still uses GetMesh signed volume."
        ),
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    cfg = load_config(args.config)
    targets = _targets(cfg, args.categories, args.mesh, args.target_limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backends = [
        {"name": "manifold", "reward_backend": "manifold", "control_backend": "auto"},
        {
            "name": "manifold_stateful",
            "reward_backend": "manifold_stateful",
            "control_backend": args.stateful_control_backend,
            "volume_method": "mesh",
        },
    ]
    if args.include_properties_volume:
        backends.append(
            {
                "name": "manifold_stateful_properties",
                "reward_backend": "manifold_stateful",
                "control_backend": args.stateful_control_backend,
                "volume_method": "properties",
            }
        )
    if args.include_bridge:
        backends.insert(
            1,
            {
                "name": "manifold_bridge",
                "reward_backend": "manifold_bridge",
                "control_backend": "auto",
                "volume_method": "mesh",
            },
        )

    results: dict[str, Any] = {
        "config": args.config,
        "stage": args.stage,
        "repeat": args.repeat,
        "mcts_iter": args.mcts_iter,
        "mcts_max_step": args.mcts_max_step,
        "refine_max_step": args.refine_max_step,
        "chamfer_points": args.chamfer_points,
        "metric_tolerance": args.metric_tolerance,
        "stateful_control_backend": args.stateful_control_backend,
        "stateful_union_cache": args.stateful_union_cache,
        "include_properties_volume": args.include_properties_volume,
        "targets": {},
    }

    for target in targets:
        key = _target_key(target)
        target_record: dict[str, Any] = {
            "category": target["category"],
            "mesh": target["mesh"],
            "runs": {},
        }
        results["targets"][key] = target_record
        _write_json(output_path, results)

        for backend in backends:
            run_record = _run_stage(args, target, backend)
            target_record["runs"][backend["name"]] = run_record
            _write_json(output_path, results)
            if run_record["returncode"] != 0 or not run_record["pipeline_success"]:
                print(json.dumps(results, indent=2, sort_keys=True))
                return int(run_record["returncode"] or 1)
            eval_record = _run_eval(args, target, backend["name"], output_path)
            run_record.update(eval_record)
            target_record["runs"][backend["name"]] = run_record
            target_record["speedup_vs_manifold"] = _speedups(target_record["runs"])
            target_record["metric_diffs_vs_manifold"] = _metric_diffs(target_record["runs"])
            if args.trace_actions:
                target_record["trace_diffs_vs_manifold"] = _trace_diffs(target_record["runs"])
            _write_json(output_path, results)
            if run_record["returncode"] != 0 or eval_record["evaluation_returncode"] != 0:
                print(json.dumps(results, indent=2, sort_keys=True))
                return int(run_record["returncode"] or eval_record["evaluation_returncode"])

    results["aggregate"] = _aggregate(results["targets"], args.metric_tolerance)
    _write_json(output_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _targets(
    cfg: dict[str, Any],
    categories: str,
    meshes: list[str] | None,
    limit: int,
) -> list[dict[str, str]]:
    allowed_categories = {item.strip() for item in categories.split(",") if item.strip()}
    allowed_meshes = set(meshes or [])
    out: list[dict[str, str]] = []
    for category in cfg.get("categories", []):
        category_name = str(category["name"])
        if allowed_categories and category_name not in allowed_categories:
            continue
        for mesh in category.get("meshes", []):
            mesh_id = str(mesh)
            if allowed_meshes and mesh_id not in allowed_meshes:
                continue
            out.append({"category": category_name, "mesh": mesh_id})
            if limit > 0 and len(out) >= limit:
                return out
    return out


def _run_stage(
    args: argparse.Namespace,
    target: dict[str, str],
    backend: dict[str, str],
) -> dict[str, Any]:
    tag = _exp_tag(args, target, backend["name"])
    command = [sys.executable, "-m", "smart", "--config", args.config]
    if args.stage == "refine":
        command.extend(
            [
                "--set",
                f"refine.max_step={args.refine_max_step}",
                "--set",
                f"refine.reward_backend={backend['reward_backend']}",
                "--set",
                f"refine.manifold_volume_method={backend.get('volume_method', 'mesh')}",
                "--set",
                f"refine.backend={backend['control_backend']}",
                "--set",
                f"refine.exp_tag={tag}",
            ]
        )
        if backend["reward_backend"] == "manifold_stateful":
            command.extend(
                ["--set", f"refine.stateful_union_cache={str(args.stateful_union_cache).lower()}"]
            )
    else:
        command.extend(
            [
                "--set",
                f"mcts.mcts_iter={args.mcts_iter}",
                "--set",
                f"mcts.max_step={args.mcts_max_step}",
                "--set",
                f"mcts.reward_backend={backend['reward_backend']}",
                "--set",
                f"mcts.manifold_volume_method={backend.get('volume_method', 'mesh')}",
                "--set",
                f"mcts.backend={backend['control_backend']}",
                "--set",
                f"mcts.exp_tag={tag}",
            ]
        )
        if backend["reward_backend"] == "manifold_stateful":
            command.extend(
                ["--set", f"mcts.stateful_union_cache={str(args.stateful_union_cache).lower()}"]
            )
        if backend["control_backend"] in {"rust", "rust_stateful"}:
            command.extend(["--set", "mcts.allow_search_order_changes=true"])
    trace_path = None
    if args.trace_actions:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = (
            trace_dir
            / f"{args.stage}_{target['category']}_{target['mesh'][:10]}_{backend['name']}.jsonl"
        ).resolve()
        command.extend(["--set", f"{args.stage}.trace_actions_path={trace_path}"])
    command.extend([args.stage, "--category", target["category"], "--mesh", target["mesh"], "--force"])

    elapsed_runs = []
    completed = None
    for _ in range(args.repeat):
        if trace_path is not None and trace_path.exists():
            trace_path.unlink()
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        elapsed_runs.append(time.perf_counter() - started)
        if completed.returncode != 0:
            break
    assert completed is not None
    result: dict[str, Any] = {
        "backend": backend,
        "command": command,
        "elapsed_sec": sum(elapsed_runs) / len(elapsed_runs),
        "elapsed_runs_sec": elapsed_runs,
        "returncode": completed.returncode,
        "pipeline_success": _pipeline_success(completed.stdout, args.stage),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }
    if trace_path is not None:
        result["trace_actions_path"] = str(trace_path)
        result["trace_actions_count"] = _trace_count(trace_path)
    return result


def _pipeline_success(stdout: str, stage: str) -> bool:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return False
    stage_summary = payload.get(stage)
    if not isinstance(stage_summary, dict):
        return False
    return int(stage_summary.get("success", 0)) > 0 and int(stage_summary.get("failed", 0)) == 0


def _run_eval(
    args: argparse.Namespace,
    target: dict[str, str],
    backend_name: str,
    output_path: Path,
) -> dict[str, Any]:
    key = _target_key(target)
    eval_path = output_path.with_name(f"{output_path.stem}_{args.stage}_{key}_{backend_name}_eval.json")
    command = [
        sys.executable,
        "-m",
        "smart",
        "--config",
        args.config,
        "evaluate",
        "--stage",
        args.stage,
        "--category",
        target["category"],
        "--mesh",
        target["mesh"],
        "--chamfer-points",
        str(args.chamfer_points),
        "--output",
        str(eval_path),
        "--json",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result: dict[str, Any] = {
        "evaluation_command": command,
        "evaluation_path": str(eval_path),
        "evaluation_time_sec": time.perf_counter() - started,
        "evaluation_returncode": completed.returncode,
        "evaluation_stdout_tail": completed.stdout[-1000:],
        "evaluation_stderr_tail": completed.stderr[-1000:],
    }
    if completed.returncode == 0 and eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        result["summary"] = payload["summary"]
        records = payload.get("records") or []
        if records:
            result["bbox_path"] = records[0].get("bbox_path")
            rust_stats = _rust_stats_path(records[0].get("bbox_path"))
            if rust_stats is not None:
                result["rust_stats_path"] = str(rust_stats)
                result["rust_stats"] = json.loads(rust_stats.read_text(encoding="utf-8"))
    return result


def _rust_stats_path(bbox_path: str | None) -> Path | None:
    if not bbox_path:
        return None
    path = Path(bbox_path)
    if len(path.parents) < 4:
        return None
    stats = path.parents[3] / "rust_stats.json"
    return stats if stats.exists() else None


def _speedups(runs: dict[str, Any]) -> dict[str, float]:
    baseline = runs.get("manifold")
    if baseline is None:
        return {}
    baseline_time = float(baseline["elapsed_sec"])
    return {
        name: baseline_time / float(record["elapsed_sec"])
        for name, record in runs.items()
        if float(record["elapsed_sec"]) > 0
    }


def _metric_diffs(runs: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline = runs.get("manifold", {}).get("summary")
    if baseline is None:
        return {}
    out: dict[str, dict[str, float]] = {}
    for name, record in runs.items():
        summary = record.get("summary")
        if summary is None:
            continue
        out[name] = {}
        for key in METRIC_KEYS:
            left = baseline.get(key)
            right = summary.get(key)
            out[name][key] = 0.0 if left is None and right is None else abs(float(left) - float(right))
    return out


def _trace_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def _trace_diffs(runs: dict[str, Any]) -> dict[str, Any]:
    baseline_path = runs.get("manifold", {}).get("trace_actions_path")
    if not baseline_path:
        return {}
    baseline = _load_trace(Path(baseline_path))
    out: dict[str, Any] = {}
    for name, record in runs.items():
        trace_path = record.get("trace_actions_path")
        if not trace_path:
            continue
        candidate = _load_trace(Path(trace_path))
        out[name] = _compare_traces(baseline, candidate)
    return out


def _load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _compare_traces(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    limit = min(len(baseline), len(candidate))
    max_reward_diff = 0.0
    for idx in range(limit):
        left = baseline[idx]
        right = candidate[idx]
        max_reward_diff = max(
            max_reward_diff,
            abs(float(left.get("reward", 0.0)) - float(right.get("reward", 0.0))),
        )
        if int(left.get("action", -1)) != int(right.get("action", -1)):
            return {
                "first_divergence_index": idx,
                "baseline_len": len(baseline),
                "candidate_len": len(candidate),
                "max_reward_abs_diff_before_divergence": max_reward_diff,
                "baseline_action": int(left.get("action", -1)),
                "candidate_action": int(right.get("action", -1)),
                "baseline_record": left,
                "candidate_record": right,
            }
    return {
        "first_divergence_index": None if len(baseline) == len(candidate) else limit,
        "baseline_len": len(baseline),
        "candidate_len": len(candidate),
        "max_reward_abs_diff_before_divergence": max_reward_diff,
    }


def _aggregate(targets: dict[str, Any], tolerance: float) -> dict[str, Any]:
    speedups: dict[str, list[float]] = {}
    max_diffs: dict[str, dict[str, float]] = {}
    parity_failures: dict[str, list[str]] = {}
    for target_key, target in targets.items():
        for backend, value in target.get("speedup_vs_manifold", {}).items():
            speedups.setdefault(backend, []).append(float(value))
        for backend, diffs in target.get("metric_diffs_vs_manifold", {}).items():
            backend_max = max_diffs.setdefault(backend, {key: 0.0 for key in METRIC_KEYS})
            failed = False
            for key, value in diffs.items():
                diff = float(value)
                backend_max[key] = max(backend_max[key], diff)
                if diff > tolerance:
                    failed = True
            if failed:
                parity_failures.setdefault(backend, []).append(target_key)
    return {
        "mean_speedup_vs_manifold": {
            backend: sum(values) / len(values) for backend, values in speedups.items() if values
        },
        "max_metric_diffs_vs_manifold": max_diffs,
        "parity_failures": parity_failures,
        "target_count": len(targets),
    }


def _exp_tag(args: argparse.Namespace, target: dict[str, str], backend_name: str) -> str:
    return "%s_%s_%s_%s" % (
        args.stage,
        target["category"],
        target["mesh"][:10],
        backend_name,
    )


def _target_key(target: dict[str, str]) -> str:
    return f"{target['category']}__{target['mesh']}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
