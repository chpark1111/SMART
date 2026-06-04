#!/usr/bin/env python3
"""Wait for extra learned-macro artifacts and run the combined release gate.

The first 500-candidate dataset can contain fewer than 500 runnable prepared
states after tetra/preseg timeouts.  This supervisor combines the original
artifact root with an extra artifact root, waits until enough runnable states
exist, then runs the matched benchmark plus the release-readiness checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MACRO_DIR = REPO_ROOT / "experiments/macro_search"
if str(MACRO_DIR) not in sys.path:
    sys.path.insert(0, str(MACRO_DIR))

from replay_parameterized_skills import scan_available_cases  # noqa: E402


DEFAULT_REPORT_ROOT = REPO_ROOT / "experiments/macro_search/runs/parameterized_skills_4k"


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


def _read_chunk_counts(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    timeouts = 0
    rows = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            if row.get("dry_run"):
                continue
            category = str(row.get("category", "unknown"))
            mesh_count = len(row.get("meshes") or [])
            if row.get("returncode") == 0:
                counts[category] += mesh_count
            elif row.get("returncode") is not None:
                failures[category] += mesh_count
                timeouts += int(bool(row.get("timed_out")))
    return {
        "rows": rows,
        "success_counts": dict(sorted(counts.items())),
        "success_total": sum(counts.values()),
        "failures": dict(sorted(failures.items())),
        "timeouts": timeouts,
    }


def _scan_count(tetra_roots: list[Path], bbox_roots: list[Path], *, bbox_source: str = "embedded") -> int:
    if not tetra_roots:
        return 0
    scanned = scan_available_cases(tetra_roots, bbox_roots=bbox_roots, bbox_source=bbox_source)
    return len(scanned)


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-run-root",
        type=Path,
        default=Path("runs/learned_macro_500_20260603"),
    )
    parser.add_argument(
        "--extra-run-root",
        type=Path,
        action="append",
        default=[Path("runs/learned_macro_extra_180_20260604")],
    )
    parser.add_argument(
        "--extra-chunk-log",
        type=Path,
        default=Path(
            "experiments/macro_search/runs/parameterized_skills_4k/"
            "dataset_extra_180_20260604/chunk_run_log.jsonl"
        ),
    )
    parser.add_argument("--collector-pid", type=int, default=0)
    parser.add_argument("--require-min-cases", type=int, default=500)
    parser.add_argument(
        "--min-extra-success",
        type=int,
        default=25,
        help="Require this many completed extra chunks before running the gate.",
    )
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--max-wait-sec", type=float, default=0.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_REPORT_ROOT / "combined_500_gate_state.json")
    parser.add_argument(
        "--bbox-source",
        choices=["embedded", "refine", "mcts", "any"],
        default="embedded",
        help="which bbox artifact stage to use for the combined release gate",
    )
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    base = args.base_run_root
    extras = list(args.extra_run_root or [])
    tetra_roots = [base / "tetra", *[root / "tetra" for root in extras]]
    bbox_roots = [base, *extras]
    start = time.perf_counter()

    while True:
        scanned = _scan_count(tetra_roots, bbox_roots, bbox_source=args.bbox_source)
        counts = _read_chunk_counts(args.extra_chunk_log)
        extra_success = int(counts["success_total"])
        ready = scanned >= args.require_min_cases and extra_success >= args.min_extra_success
        payload = {
            "status": "waiting" if not ready else "ready",
            "scanned_cases": scanned,
            "required_cases": args.require_min_cases,
            "extra_success": extra_success,
            "min_extra_success": args.min_extra_success,
            "collector_pid": args.collector_pid,
            "collector_alive": _pid_alive(args.collector_pid) if args.collector_pid else None,
            "tetra_roots": [str(path) for path in tetra_roots],
            "bbox_roots": [str(path) for path in bbox_roots],
            "bbox_source": args.bbox_source,
            "extra_chunk_counts": counts,
            "elapsed_sec": time.perf_counter() - start,
            "updated_at": _timestamp(),
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        _write_state(args.state_path, payload)

        if ready:
            break
        if args.collector_pid and not _pid_alive(args.collector_pid):
            payload["status"] = "collector_exited_before_gate"
            _write_state(args.state_path, payload)
            raise SystemExit(
                f"collector pid {args.collector_pid} exited before {args.require_min_cases} runnable cases"
            )
        if args.max_wait_sec > 0 and time.perf_counter() - start > args.max_wait_sec:
            payload["status"] = "timeout_before_gate"
            _write_state(args.state_path, payload)
            raise SystemExit(f"timed out before {args.require_min_cases} runnable cases")
        time.sleep(args.poll_sec)

    out_dir = DEFAULT_REPORT_ROOT / f"fresh_matched_learned_macro_combined_500_{_timestamp()}"
    cmd = [
        sys.executable,
        "experiments/macro_search/run_fresh_matched_macro_benchmark.py",
        "--run-root",
        str(base),
        "--tetra-root",
        str(base / "tetra"),
        "--bbox-root",
        str(base),
        "--out-dir",
        str(out_dir.relative_to(REPO_ROOT)),
        "--require-min-cases",
        str(args.require_min_cases),
        "--max-cases",
        "0",
        "--jobs",
        str(args.jobs),
        "--progress-every",
        "10",
        "--bbox-source",
        args.bbox_source,
    ]
    for root in extras:
        cmd.extend(["--extra-tetra-root", str(root / "tetra")])
        cmd.extend(["--bbox-root", str(root)])
    _run(cmd)

    if not args.skip_readiness:
        _run([sys.executable, "-m", "smart", "learned-release-readiness", "--json"])
    if not args.skip_pytest:
        _run([sys.executable, "-m", "pytest", "-q", "tests"])

    final_payload = {
        "status": "complete",
        "report_dir": str(out_dir),
        "report": str(out_dir / "fresh_matched_report.json"),
        "scanned_cases": _scan_count(tetra_roots, bbox_roots, bbox_source=args.bbox_source),
        "bbox_source": args.bbox_source,
        "completed_at": _timestamp(),
    }
    print(json.dumps(final_payload, sort_keys=True), flush=True)
    _write_state(args.state_path, final_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
