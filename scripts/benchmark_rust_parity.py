from __future__ import annotations

import argparse
import json
import os
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
        description="Compare SMART Python fallback kernels against Rust kernels."
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument("--category", default="table")
    parser.add_argument("--mesh", default="1692563658149377630047043c6a0c50")
    parser.add_argument(
        "--stage",
        action="append",
        choices=["merge", "refine", "mcts"],
        help="Stage to run; repeat for multiple stages. Defaults to merge/refine/mcts.",
    )
    parser.add_argument("--eval-stage", default="mcts", choices=["merge", "refine", "mcts"])
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only time stages; useful for merge-only runs that produce txt segments rather than bbox OBJ outputs.",
    )
    parser.add_argument("--refine-max-step", type=int, default=50)
    parser.add_argument("--mcts-iter", type=int, default=30)
    parser.add_argument("--mcts-max-step", type=int, default=5)
    parser.add_argument("--chamfer-points", type=int, default=512)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each stage this many times and report the mean elapsed time.",
    )
    parser.add_argument("--output", default="runs/bench_exact/rust_parity_benchmark.json")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    stages = args.stage or ["merge", "refine", "mcts"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "config": args.config,
        "category": args.category,
        "mesh": args.mesh,
        "stages": stages,
        "variants": {},
    }

    for variant, disable_rust in (("python_fallback", True), ("rust_enabled", False)):
        env = os.environ.copy()
        if disable_rust:
            env["SMART_DISABLE_RUST"] = "1"
        else:
            env.pop("SMART_DISABLE_RUST", None)

        variant_result: dict[str, Any] = {"stage_times": {}}
        for stage in stages:
            command = _stage_command(args, stage)
            elapsed_runs = []
            completed = None
            for _ in range(args.repeat):
                started = time.perf_counter()
                completed = subprocess.run(command, env=env, text=True, check=False)
                elapsed_runs.append(time.perf_counter() - started)
                if completed.returncode != 0:
                    break
            assert completed is not None
            variant_result["stage_times"][stage] = {
                "elapsed_sec": sum(elapsed_runs) / len(elapsed_runs),
                "elapsed_runs_sec": elapsed_runs,
                "returncode": completed.returncode,
                "command": command,
            }
            if completed.returncode != 0:
                variant_result["error"] = f"{stage} failed rc={completed.returncode}"
                results["variants"][variant] = variant_result
                output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
                return completed.returncode

        if args.skip_eval:
            results["variants"][variant] = variant_result
            continue

        eval_path = output_path.with_name(f"{output_path.stem}_{variant}_eval.json")
        eval_command = [
            sys.executable,
            "-m",
            "smart",
            "--config",
            args.config,
            "evaluate",
            "--stage",
            args.eval_stage,
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
        completed = subprocess.run(eval_command, env=env, text=True, check=False)
        variant_result["evaluation_time_sec"] = time.perf_counter() - started
        variant_result["evaluation_returncode"] = completed.returncode
        variant_result["evaluation_path"] = str(eval_path)
        if completed.returncode != 0:
            variant_result["error"] = f"evaluation failed rc={completed.returncode}"
            results["variants"][variant] = variant_result
            output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
            return completed.returncode

        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        variant_result["summary"] = evaluation["summary"]
        results["variants"][variant] = variant_result

    if not args.skip_eval:
        results["metric_diffs"] = _metric_diffs(
            results["variants"]["python_fallback"]["summary"],
            results["variants"]["rust_enabled"]["summary"],
        )
    results["speedup"] = _speedups(
        results["variants"]["python_fallback"]["stage_times"],
        results["variants"]["rust_enabled"]["stage_times"],
    )
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _stage_command(args: argparse.Namespace, stage: str) -> list[str]:
    command = [sys.executable, "-m", "smart", "--config", args.config]
    if stage == "refine":
        command.extend(["--set", f"refine.max_step={args.refine_max_step}"])
    if stage == "mcts":
        command.extend(
            [
                "--set",
                f"mcts.mcts_iter={args.mcts_iter}",
                "--set",
                f"mcts.max_step={args.mcts_max_step}",
            ]
        )
    command.extend(
        [
            stage,
            "--category",
            args.category,
            "--mesh",
            args.mesh,
            "--force",
        ]
    )
    return command


def _metric_diffs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    diffs = {}
    for key in METRIC_KEYS:
        diffs[key] = abs(float(left[key]) - float(right[key]))
    return diffs


def _speedups(left_times: dict[str, Any], right_times: dict[str, Any]) -> dict[str, float]:
    speedups = {}
    for stage, left in left_times.items():
        right = right_times.get(stage)
        if not right:
            continue
        rust_time = float(right["elapsed_sec"])
        speedups[stage] = float(left["elapsed_sec"]) / rust_time if rust_time > 0 else 0.0
    return speedups


if __name__ == "__main__":
    raise SystemExit(main())
