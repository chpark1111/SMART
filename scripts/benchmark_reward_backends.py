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
        description="Compare exact refine/MCTS reward backends on one SMART mesh."
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument("--category", default="table")
    parser.add_argument("--mesh", default="1692563658149377630047043c6a0c50")
    parser.add_argument("--stage", choices=["refine", "mcts"], default="refine")
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Reward backend to test. Repeat to compare more than two. Defaults to manifold, manifold_bridge, and manifold_stateful.",
    )
    parser.add_argument("--refine-max-step", type=int, default=10)
    parser.add_argument(
        "--refine-backend",
        default=None,
        help="Optional refine control backend override, e.g. python, rust, rust_stateful.",
    )
    parser.add_argument("--mcts-iter", type=int, default=10)
    parser.add_argument("--mcts-max-step", type=int, default=5)
    parser.add_argument(
        "--mcts-backend",
        default=None,
        help="Optional MCTS control backend override, e.g. python, rust, rust_stateful.",
    )
    parser.add_argument(
        "--candidate-backend",
        default="exact",
        choices=["exact", "bitset_topk"],
        help="Candidate helper to use while benchmarking reward backends.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", default="runs/bench_exact/reward_backend_benchmark.json")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    backends = args.backend or ["manifold", "manifold_bridge", "manifold_stateful"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "config": args.config,
        "category": args.category,
        "mesh": args.mesh,
        "stage": args.stage,
        "backends": {},
    }

    for backend in backends:
        stage_result = _run_stage(args, backend)
        results["backends"][backend] = stage_result
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        if stage_result["returncode"] != 0:
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(stage_result["returncode"])

        eval_result = _run_eval(args, backend, output_path)
        stage_result.update(eval_result)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        if eval_result["evaluation_returncode"] != 0:
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(eval_result["evaluation_returncode"])

    baseline = backends[0]
    results["speedup_vs_baseline"] = _speedups(results["backends"], baseline)
    results["metric_diffs_vs_baseline"] = _metric_diffs(results["backends"], baseline)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _run_stage(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "smart",
        "--config",
        args.config,
    ]
    if args.stage == "refine":
        command.extend(
            [
                "--set",
                f"refine.max_step={args.refine_max_step}",
                "--set",
                f"refine.reward_backend={backend}",
                "--set",
                f"refine.candidate_backend={args.candidate_backend}",
                "--set",
                f"refine.candidate_top_k={args.candidate_top_k}",
            ]
        )
        if args.refine_backend:
            command.extend(["--set", f"refine.backend={args.refine_backend}"])
    else:
        command.extend(
            [
                "--set",
                f"mcts.mcts_iter={args.mcts_iter}",
                "--set",
                f"mcts.max_step={args.mcts_max_step}",
                "--set",
                f"mcts.reward_backend={backend}",
                "--set",
                f"mcts.candidate_backend={args.candidate_backend}",
                "--set",
                f"mcts.candidate_top_k={args.candidate_top_k}",
            ]
        )
        if args.mcts_backend:
            command.extend(["--set", f"mcts.backend={args.mcts_backend}"])
    command.extend(
        [
            args.stage,
            "--category",
            args.category,
            "--mesh",
            args.mesh,
            "--force",
        ]
    )
    elapsed_runs = []
    completed = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        elapsed_runs.append(time.perf_counter() - started)
        if completed.returncode != 0:
            break
    assert completed is not None
    return {
        "command": command,
        "elapsed_sec": sum(elapsed_runs) / len(elapsed_runs),
        "elapsed_runs_sec": elapsed_runs,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def _run_eval(args: argparse.Namespace, backend: str, output_path: Path) -> dict[str, Any]:
    eval_path = output_path.with_name(
        f"{output_path.stem}_{args.stage}_{backend}_eval.json"
    )
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


def _speedups(results: dict[str, Any], baseline: str) -> dict[str, float]:
    baseline_time = float(results[baseline]["elapsed_sec"])
    return {
        backend: baseline_time / float(record["elapsed_sec"])
        for backend, record in results.items()
        if float(record["elapsed_sec"]) > 0
    }


def _metric_diffs(results: dict[str, Any], baseline: str) -> dict[str, dict[str, float]]:
    baseline_summary = results[baseline].get("summary")
    if baseline_summary is None:
        return {}
    out: dict[str, dict[str, float]] = {}
    for backend, record in results.items():
        summary = record.get("summary")
        if summary is None:
            continue
        out[backend] = {
            key: abs(float(baseline_summary[key]) - float(summary[key]))
            for key in METRIC_KEYS
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
