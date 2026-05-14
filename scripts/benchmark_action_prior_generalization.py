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
            "Trace SMART MCTS on several meshes, build leave-one-out action priors, "
            "and evaluate whether the prior generalizes without replacing exact reward."
        )
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument("--categories", default="", help="Comma-separated category filter")
    parser.add_argument("--target-limit", type=int, default=0, help="Limit target count for smoke runs")
    parser.add_argument("--mcts-iter", type=int, default=20)
    parser.add_argument("--max-step", type=int, default=20)
    parser.add_argument("--reward-backend", default="manifold_stateful")
    parser.add_argument("--mcts-backend", default="rust_stateful")
    parser.add_argument("--weights", default="0,0.5,1.0")
    parser.add_argument("--prior-model", choices=["counts", "linear", "mlp"], default="counts")
    parser.add_argument("--linear-epochs", type=int, default=100, help="Deprecated alias for --prior-epochs")
    parser.add_argument("--prior-epochs", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="PyTorch device for --prior-model mlp")
    parser.add_argument("--transposition-table", action="store_true")
    parser.add_argument("--transposition-table-size", type=int, default=8192)
    parser.add_argument(
        "--include-action-logits",
        action="store_true",
        help=(
            "Also include per-action logits. This is only portable when the train "
            "and target action layouts are comparable, so leave it off for category-general tests."
        ),
    )
    parser.add_argument("--output", default="runs/bench_exact/action_prior_generalization.json")
    parser.add_argument(
        "--global-prior-output",
        default="",
        help=(
            "Optional path to write one prior trained from all generated traces after "
            "leave-one-out validation completes."
        ),
    )
    parser.add_argument("--force-traces", action="store_true")
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-9,
        help="Tolerance used to report metric-identical weights in the aggregate summary",
    )
    args = parser.parse_args()
    if args.prior_epochs <= 0:
        args.prior_epochs = args.linear_epochs

    cfg = load_config(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = output_path.with_suffix("")
    trace_dir.mkdir(parents=True, exist_ok=True)

    targets = _targets(cfg, args.categories, args.target_limit)
    weights = [float(item) for item in args.weights.split(",") if item.strip()]
    results: dict[str, Any] = {
        "config": args.config,
        "mcts_iter": args.mcts_iter,
        "max_step": args.max_step,
        "reward_backend": args.reward_backend,
        "mcts_backend": args.mcts_backend,
        "weights": weights,
        "prior_model": args.prior_model,
        "transposition_table": args.transposition_table,
        "transposition_table_size": args.transposition_table_size,
        "include_action_logits": args.include_action_logits,
        "targets": {},
    }

    trace_paths: dict[str, Path] = {}
    for target in targets:
        key = _target_key(target)
        trace_path = trace_dir / f"{key}.trace.jsonl"
        trace_paths[key] = trace_path
        if args.force_traces and trace_path.exists():
            trace_path.unlink()
        if not trace_path.exists():
            record = _run_mcts(
                args,
                target,
                weight=0.0,
                prior_path=None,
                trace_path=trace_path,
                exp_tag=f"loo_trace_{key}",
            )
            results["targets"].setdefault(key, {})["trace_run"] = record
            _write_json(output_path, results)
            if record["returncode"] != 0:
                print(json.dumps(results, indent=2, sort_keys=True))
                return int(record["returncode"])

    for target in targets:
        key = _target_key(target)
        train_traces = [path for other, path in trace_paths.items() if other != key]
        target_record: dict[str, Any] = results["targets"].setdefault(key, {})
        target_record.update({"category": target["category"], "mesh": target["mesh"]})
        if not train_traces:
            target_record["error"] = "need at least two targets for leave-one-out prior"
            _write_json(output_path, results)
            continue

        prior_path = trace_dir / f"{key}.loo_prior.json"
        target_record["prior_build"] = _build_prior(
            args,
            train_traces,
            prior_path,
            include_action_logits=args.include_action_logits,
        )
        _write_json(output_path, results)
        if target_record["prior_build"]["returncode"] != 0:
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(target_record["prior_build"]["returncode"])

        target_record["runs"] = {}
        for weight in weights:
            tag = "loo_%s_w%s" % (key, str(weight).replace(".", "p").replace("-", "m"))
            run_record = _run_mcts(
                args,
                target,
                weight=weight,
                prior_path=prior_path if weight != 0.0 else None,
                trace_path=None,
                exp_tag=tag,
            )
            eval_record = _run_eval(args, target, output_path, tag)
            run_record.update(eval_record)
            target_record["runs"][str(weight)] = run_record
            target_record["metric_diffs_vs_weight0"] = _metric_diffs(target_record["runs"])
            target_record["speedup_vs_weight0"] = _speedups(target_record["runs"])
            _write_json(output_path, results)
            if run_record["returncode"] != 0 or eval_record["evaluation_returncode"] != 0:
                print(json.dumps(results, indent=2, sort_keys=True))
                return int(run_record["returncode"] or eval_record["evaluation_returncode"])

    results["aggregate"] = _aggregate(results["targets"], float(args.metric_tolerance))
    if args.global_prior_output:
        global_prior_path = Path(args.global_prior_output)
        results["global_prior"] = _build_prior(
            args,
            list(trace_paths.values()),
            global_prior_path,
            include_action_logits=args.include_action_logits,
        )
        if results["global_prior"]["returncode"] != 0:
            _write_json(output_path, results)
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(results["global_prior"]["returncode"])
    _write_json(output_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _targets(cfg: dict[str, Any], categories: str, limit: int) -> list[dict[str, str]]:
    allowed = {item.strip() for item in categories.split(",") if item.strip()}
    out = []
    for category in cfg.get("categories", []):
        name = str(category["name"])
        if allowed and name not in allowed:
            continue
        for mesh in category.get("meshes", []):
            out.append({"category": name, "mesh": str(mesh)})
            if limit > 0 and len(out) >= limit:
                return out
    return out


def _run_mcts(
    args: argparse.Namespace,
    target: dict[str, str],
    *,
    weight: float,
    prior_path: Path | None,
    trace_path: Path | None,
    exp_tag: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
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
        f"mcts.exp_tag={exp_tag}",
        "--set",
        f"mcts.action_prior_weight={weight}",
    ]
    if prior_path is not None:
        command.extend(["--set", f"mcts.action_prior_path={prior_path}"])
    if trace_path is not None:
        command.extend(["--set", f"mcts.trace_actions_path={trace_path}"])
    if args.mcts_backend in {"rust", "rust_stateful"} or weight != 0.0:
        command.extend(["--set", "mcts.allow_search_order_changes=true"])
    if args.transposition_table:
        command.extend(
            [
                "--set",
                "mcts.transposition_table=true",
                "--set",
                f"mcts.transposition_table_size={args.transposition_table_size}",
            ]
        )
    command.extend(["mcts", "--category", target["category"], "--mesh", target["mesh"], "--force"])
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "elapsed_sec": time.perf_counter() - started,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def _run_eval(
    args: argparse.Namespace,
    target: dict[str, str],
    output_path: Path,
    tag: str,
) -> dict[str, Any]:
    eval_path = output_path.with_name(f"{output_path.stem}_{tag}_eval.json")
    command = [
        sys.executable,
        "-m",
        "smart",
        "--config",
        args.config,
        "evaluate",
        "--stage",
        "mcts",
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
    if completed.returncode == 0:
        result["summary"] = json.loads(eval_path.read_text(encoding="utf-8"))["summary"]
    return result


def _metric_diffs(records: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline = records.get("0.0") or records.get("0")
    if not baseline or "summary" not in baseline:
        return {}
    baseline_summary = baseline["summary"]
    out = {}
    for weight, record in records.items():
        summary = record.get("summary")
        if summary is None:
            continue
        out[weight] = {
            key: abs(float(baseline_summary[key]) - float(summary[key]))
            for key in METRIC_KEYS
        }
    return out


def _speedups(records: dict[str, Any]) -> dict[str, float]:
    baseline = records.get("0.0") or records.get("0")
    if not baseline:
        return {}
    baseline_time = float(baseline["elapsed_sec"])
    return {
        weight: baseline_time / float(record["elapsed_sec"])
        for weight, record in records.items()
        if float(record.get("elapsed_sec", 0.0)) > 0.0
    }


def _aggregate(targets: dict[str, Any], metric_tolerance: float) -> dict[str, Any]:
    speedups: dict[str, list[float]] = {}
    max_metric_diffs: dict[str, dict[str, float]] = {}
    for target in targets.values():
        for weight, value in target.get("speedup_vs_weight0", {}).items():
            speedups.setdefault(weight, []).append(float(value))
        for weight, diffs in target.get("metric_diffs_vs_weight0", {}).items():
            current = max_metric_diffs.setdefault(weight, {key: 0.0 for key in METRIC_KEYS})
            for key, value in diffs.items():
                current[key] = max(current[key], float(value))
    metric_identical_weights = []
    for weight, diffs in sorted(max_metric_diffs.items()):
        if all(abs(value) <= metric_tolerance for value in diffs.values()):
            metric_identical_weights.append(weight)

    mean_speedups = {
        weight: sum(values) / len(values)
        for weight, values in sorted(speedups.items())
        if values
    }
    recommended_weight = None
    recommended_speedup = 0.0
    for weight in metric_identical_weights:
        speedup = float(mean_speedups.get(weight, 0.0))
        if recommended_weight is None or speedup > recommended_speedup:
            recommended_weight = weight
            recommended_speedup = speedup

    return {
        "mean_speedup_vs_weight0": mean_speedups,
        "max_metric_diffs_vs_weight0": max_metric_diffs,
        "metric_tolerance": metric_tolerance,
        "metric_identical_weights": metric_identical_weights,
        "recommended_weight": recommended_weight,
        "recommended_mean_speedup": recommended_speedup,
    }


def _target_key(target: dict[str, str]) -> str:
    return f"{target['category']}_{target['mesh']}"


def _build_prior(
    args: argparse.Namespace,
    traces: list[Path],
    output: Path,
    *,
    include_action_logits: bool,
) -> dict[str, Any]:
    if args.prior_model in {"linear", "mlp"}:
        command = [
            sys.executable,
            "scripts/train_action_prior_from_traces.py",
            *[str(path) for path in traces],
            "--output",
            str(output),
            "--model-type",
            args.prior_model,
            "--epochs",
            str(args.prior_epochs),
        ]
        if args.prior_model == "mlp":
            command.extend(["--hidden-size", str(args.hidden_size), "--device", str(args.device)])
    else:
        command = [
            sys.executable,
            "scripts/build_action_prior_from_traces.py",
            *[str(path) for path in traces],
            "--output",
            str(output),
            "--min-reward",
            "0.0",
        ]
    if include_action_logits and args.prior_model == "counts":
        command.append("--include-action-logits")
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "elapsed_sec": time.perf_counter() - started,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
        "train_traces": [str(path) for path in traces],
        "prior_path": str(output),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
