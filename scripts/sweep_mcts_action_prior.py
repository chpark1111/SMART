from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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
        description="Run an opt-in exact-reward MCTS action-prior sweep."
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument("--category", default="airplane")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--mcts-iter", type=int, default=20)
    parser.add_argument("--max-step", type=int, default=20)
    parser.add_argument("--reward-backend", default="manifold_stateful")
    parser.add_argument("--mcts-backend", default="rust_stateful")
    parser.add_argument("--weights", default="0,0.05,0.1,0.2")
    parser.add_argument("--prior-path", default="")
    parser.add_argument("--trace-path", default="")
    parser.add_argument("--make-prior", action="store_true")
    parser.add_argument(
        "--include-action-logits",
        action="store_true",
        help=(
            "When building a prior, also include per-action logits. This is useful "
            "for same-mesh/search-layout experiments and remains opt-in because it "
            "can change MCTS search order."
        ),
    )
    parser.add_argument("--transposition-table", action="store_true")
    parser.add_argument("--transposition-table-size", type=int, default=8192)
    parser.add_argument(
        "--stateful-unscored-apply",
        action="store_true",
        help=(
            "Forward mcts.stateful_unscored_apply=true. This is an experimental "
            "exact path and should be compared against weight 0 baselines before use."
        ),
    )
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--output", default="runs/bench_exact/mcts_action_prior_sweep.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights = [float(item) for item in args.weights.split(",") if item.strip()]

    trace_path = Path(args.trace_path) if args.trace_path else output_path.with_suffix(".trace.jsonl")
    prior_path = Path(args.prior_path) if args.prior_path else output_path.with_suffix(".prior.json")

    results: dict[str, Any] = {
        "config": args.config,
        "category": args.category,
        "mesh": args.mesh,
        "mcts_iter": args.mcts_iter,
        "max_step": args.max_step,
        "reward_backend": args.reward_backend,
        "mcts_backend": args.mcts_backend,
        "trace_path": str(trace_path),
        "prior_path": str(prior_path),
        "include_action_logits": args.include_action_logits,
        "transposition_table": args.transposition_table,
        "transposition_table_size": args.transposition_table_size,
        "stateful_unscored_apply": args.stateful_unscored_apply,
        "weights": {},
    }

    needs_prior = any(weight != 0.0 for weight in weights)
    if needs_prior and (args.make_prior or not prior_path.exists()):
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_result = _run_mcts(
            args,
            weight=0.0,
            prior_path=None,
            trace_path=trace_path,
            exp_tag="trace",
        )
        results["trace_run"] = trace_result
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        if trace_result["returncode"] != 0:
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(trace_result["returncode"])

        build_command = [
            sys.executable,
            "scripts/build_action_prior_from_traces.py",
            str(trace_path),
            "--output",
            str(prior_path),
            "--min-reward",
            "0.0",
        ]
        if args.include_action_logits:
            build_command.append("--include-action-logits")
        build_started = time.perf_counter()
        completed = subprocess.run(build_command, text=True, capture_output=True, check=False)
        results["prior_build"] = {
            "command": build_command,
            "elapsed_sec": time.perf_counter() - build_started,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        if completed.returncode != 0:
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(completed.returncode)

    for weight in weights:
        tag = "prior_w%s" % str(weight).replace(".", "p").replace("-", "m")
        stage_result = _run_mcts(
            args,
            weight=weight,
            prior_path=prior_path if weight != 0.0 else None,
            trace_path=None,
            exp_tag=tag,
        )
        eval_result = _run_eval(args, output_path, tag)
        stage_result.update(eval_result)
        results["weights"][str(weight)] = stage_result
        results["metric_diffs_vs_weight0"] = _metric_diffs(results["weights"])
        results["speedup_vs_weight0"] = _speedups(results["weights"])
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        if stage_result["returncode"] != 0 or eval_result["evaluation_returncode"] != 0:
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(stage_result["returncode"] or eval_result["evaluation_returncode"])

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _run_mcts(
    args: argparse.Namespace,
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
    if args.transposition_table:
        command.extend(
            [
                "--set",
                "mcts.transposition_table=true",
                "--set",
                "mcts.allow_search_order_changes=true",
                "--set",
                f"mcts.transposition_table_size={args.transposition_table_size}",
            ]
        )
    if args.stateful_unscored_apply:
        command.extend(["--set", "mcts.stateful_unscored_apply=true"])
    command.extend(["mcts", "--category", args.category, "--mesh", args.mesh, "--force"])
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "elapsed_sec": time.perf_counter() - started,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def _run_eval(args: argparse.Namespace, output_path: Path, tag: str) -> dict[str, Any]:
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
        args.category,
        "--mesh",
        args.mesh,
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
        if float(record["elapsed_sec"]) > 0
    }


if __name__ == "__main__":
    raise SystemExit(main())
