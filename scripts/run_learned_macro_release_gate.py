#!/usr/bin/env python3
"""Run the learned macro-skill release gate end to end.

This supervisor is intentionally conservative:

1. wait for an already running collection process, if supplied;
2. run a fresh matched benchmark at the 150-case gate;
3. continue collection to the 500+ gate;
4. run the fresh matched benchmark/readiness/tests again.

The script does not make learned controllers default.  It only gathers the
evidence needed to decide whether that is safe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "experiments/macro_search/runs/parameterized_skills_4k/dataset_500_20260603"
RUN_CONFIG = DATASET / "learned_macro_500_20260603.yaml"
RUN_LOG = DATASET / "chunk_run_log.jsonl"
RUN_ROOT = REPO_ROOT / "runs/learned_macro_500_20260603"
REPORT_ROOT = REPO_ROOT / "experiments/macro_search/runs/parameterized_skills_4k"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_counts() -> dict[str, Any]:
    counts: dict[str, int] = {}
    failures: dict[str, int] = {}
    timeouts = 0
    if RUN_LOG.exists():
        for line in RUN_LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dry_run"):
                continue
            category = str(row.get("category", "unknown"))
            mesh_count = len(row.get("meshes") or [])
            if row.get("returncode") == 0:
                counts[category] = counts.get(category, 0) + mesh_count
            elif row.get("returncode") is not None:
                failures[category] = failures.get(category, 0) + mesh_count
                timeouts += int(bool(row.get("timed_out")))
    return {
        "success_counts": dict(sorted(counts.items())),
        "success_total": sum(counts.values()),
        "failures": dict(sorted(failures.items())),
        "timeouts": timeouts,
    }


def _print_counts(label: str) -> None:
    print(json.dumps({"label": label, **_read_counts()}, sort_keys=True), flush=True)


def _run(cmd: list[str], *, check: bool = True) -> int:
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def _wait_for_pid(pid: int, *, label: str, poll_sec: float) -> None:
    if pid <= 0:
        return
    print(f"waiting for {label} pid={pid}", flush=True)
    while _pid_alive(pid):
        _print_counts(f"{label}_progress")
        time.sleep(poll_sec)
    print(f"{label} pid={pid} exited", flush=True)
    _print_counts(f"{label}_done")


def _collect(target_per_category: int, timeout_sec: int) -> None:
    _run(
        [
            sys.executable,
            "scripts/run_macro_replay_balanced_chunks.py",
            "--balanced-round-robin",
            "--target-per-category",
            str(target_per_category),
            "--chunk-size",
            "1",
            "--chunk-timeout-sec",
            str(timeout_sec),
        ]
    )
    _print_counts(f"collect_{target_per_category}_done")


def _matched_benchmark(require_min_cases: int, tag: str, jobs: int) -> Path:
    out_dir = REPORT_ROOT / f"fresh_matched_learned_macro_500_{tag}_{_timestamp()}"
    _run(
        [
            sys.executable,
            "experiments/macro_search/run_fresh_matched_macro_benchmark.py",
            "--run-root",
            str(RUN_ROOT.relative_to(REPO_ROOT)),
            "--tetra-root",
            str((RUN_ROOT / "tetra").relative_to(REPO_ROOT)),
            "--bbox-root",
            str(RUN_ROOT.relative_to(REPO_ROOT)),
            "--out-dir",
            str(out_dir.relative_to(REPO_ROOT)),
            "--require-min-cases",
            str(require_min_cases),
            "--max-cases",
            "0",
            "--jobs",
            str(jobs),
            "--progress-every",
            "10",
        ]
    )
    return out_dir


def _verify() -> None:
    _run([sys.executable, "-m", "smart", "learned-release-readiness", "--json"])
    _run([sys.executable, "-m", "pytest", "-q", "tests"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-runner-pid", type=int, default=0)
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--chunk-timeout-sec", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-150-benchmark", action="store_true")
    parser.add_argument("--skip-500-benchmark", action="store_true")
    args = parser.parse_args()

    print(
        json.dumps(
            {
                "status": "started",
                "dataset": str(DATASET.relative_to(REPO_ROOT)),
                "config": str(RUN_CONFIG.relative_to(REPO_ROOT)),
                "current_runner_pid": args.current_runner_pid,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _print_counts("initial")

    _wait_for_pid(args.current_runner_pid, label="gate150_runner", poll_sec=args.poll_sec)

    if not args.skip_150_benchmark:
        _matched_benchmark(150, "seed150_3cat", args.jobs)
        _verify()

    _collect(167, args.chunk_timeout_sec)

    if not args.skip_500_benchmark:
        _matched_benchmark(500, "seed500_3cat", args.jobs)
        _verify()

    _print_counts("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
