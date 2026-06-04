#!/usr/bin/env python3
"""Run learned macro replay expansion in resumable, category-balanced chunks.

This is the supported release-readiness runner for the opt-in learned
macro-skill path.  It executes a SMART dataset config by category/mesh chunks,
records every chunk in JSONL, skips already handled chunks on resume, and can
interleave categories so validation does not overfit one category before the
others have coverage.

The older research runner under ``experiments/macro_search`` may have extra
temporary knobs, but this script is tracked and intended for reproducible
release-gate evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "experiments/macro_search/runs/parameterized_skills_4k/dataset_500_20260603"
DEFAULT_CONFIG = DEFAULT_DATASET / "learned_macro_500_20260603.yaml"
DEFAULT_LOG = DEFAULT_DATASET / "chunk_run_log.jsonl"


def _load_config(path: Path) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from smart.pipeline.config import load_config

    return load_config(path)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("--chunk-size must be positive")
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def _log_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _successful_chunk_keys(path: Path) -> set[tuple[str, tuple[str, ...]]]:
    out: set[tuple[str, tuple[str, ...]]] = set()
    for row in _log_rows(path):
        returncode = row.get("returncode")
        if returncode is None or int(returncode) != 0 or bool(row.get("dry_run")):
            continue
        category = str(row.get("category", ""))
        meshes = tuple(str(item) for item in row.get("meshes", []) if str(item))
        if category and meshes:
            out.add((category, meshes))
    return out


def _failed_chunk_keys(path: Path) -> set[tuple[str, tuple[str, ...]]]:
    out: set[tuple[str, tuple[str, ...]]] = set()
    for row in _log_rows(path):
        returncode = row.get("returncode")
        if returncode is None or int(returncode) == 0 or bool(row.get("dry_run")):
            continue
        category = str(row.get("category", ""))
        meshes = tuple(str(item) for item in row.get("meshes", []) if str(item))
        if category and meshes:
            out.add((category, meshes))
    return out


def _successful_mesh_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in _log_rows(path):
        returncode = row.get("returncode")
        if returncode is None or int(returncode) != 0 or bool(row.get("dry_run")):
            continue
        category = str(row.get("category", ""))
        meshes = [str(item) for item in row.get("meshes", []) if str(item)]
        if category:
            counts[category] += len(meshes)
    return counts


def _chunk_plan(
    categories: list[tuple[str, list[str]]],
    *,
    chunk_size: int,
    balanced_round_robin: bool,
) -> list[tuple[int, str, int, list[str]]]:
    per_category: list[list[tuple[int, str, int, list[str]]]] = []
    global_index = 0
    for category, meshes in categories:
        items: list[tuple[int, str, int, list[str]]] = []
        for local_chunk_index, mesh_chunk in enumerate(_chunks(meshes, chunk_size)):
            items.append((global_index, category, local_chunk_index, mesh_chunk))
            global_index += 1
        per_category.append(items)

    if not balanced_round_robin:
        return [item for items in per_category for item in items]

    out: list[tuple[int, str, int, list[str]]] = []
    max_len = max((len(items) for items in per_category), default=0)
    for idx in range(max_len):
        for items in per_category:
            if idx < len(items):
                out.append(items[idx])
    return out


def _run_command(cmd: list[str], *, cwd: Path, dry_run: bool, timeout_sec: float = 0.0) -> tuple[int, bool]:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0, False
    if timeout_sec <= 0:
        return subprocess.run(cmd, cwd=cwd).returncode, False
    proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
    try:
        return proc.wait(timeout=timeout_sec), False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        return 124, True


def _categories_from_config(config_path: Path, selected: set[str]) -> list[tuple[str, list[str]]]:
    cfg = _load_config(config_path)
    categories: list[tuple[str, list[str]]] = []
    for category in cfg.get("categories", []):
        name = str(category.get("name", ""))
        if selected and name not in selected:
            continue
        meshes = [str(item) for item in category.get("meshes", [])]
        if meshes:
            categories.append((name, meshes))
    return categories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--category", action="append", help="Limit to one category; repeatable")
    parser.add_argument("--start-chunk", type=int, default=0, help="Skip chunks before this scheduled index")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Run at most this many scheduled chunks")
    parser.add_argument(
        "--balanced-round-robin",
        action="store_true",
        help="Interleave categories instead of exhausting one category first.",
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=0,
        help="Stop scheduling a category once this many successful meshes are logged.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build-tools-first", action="store_true")
    parser.add_argument(
        "--skip-successful",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip chunks already recorded with returncode 0 in the log.",
    )
    parser.add_argument(
        "--skip-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip chunks already recorded with nonzero returncode in the log. Use --no-skip-failed to retry failures.",
    )
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--chunk-timeout-sec",
        type=float,
        default=0.0,
        help="Optional wall-clock timeout per chunk. 0 disables timeout.",
    )
    args = parser.parse_args()

    selected_categories = set(args.category or [])
    categories = _categories_from_config(args.config, selected_categories)
    if not categories:
        raise SystemExit("no categories/meshes found in config")

    successful_chunks = _successful_chunk_keys(args.log) if args.skip_successful else set()
    failed_chunks = _failed_chunk_keys(args.log) if args.skip_failed else set()
    successful_counts = _successful_mesh_counts(args.log) if args.skip_successful else Counter()

    if args.build_tools_first:
        rc, _ = _run_command(
            [sys.executable, "-m", "smart", "--config", str(args.config), "build-tools"],
            cwd=REPO_ROOT,
            dry_run=args.dry_run,
            timeout_sec=args.chunk_timeout_sec,
        )
        if rc != 0:
            return rc

    executed = 0
    failures = 0
    plan = _chunk_plan(
        categories,
        chunk_size=args.chunk_size,
        balanced_round_robin=args.balanced_round_robin,
    )
    for schedule_index, (global_chunk_index, category, local_chunk_index, mesh_chunk) in enumerate(plan):
        if schedule_index < args.start_chunk:
            continue
        if args.target_per_category and successful_counts[category] >= args.target_per_category:
            continue
        chunk_key = (category, tuple(mesh_chunk))
        if chunk_key in successful_chunks:
            continue
        if chunk_key in failed_chunks:
            continue
        if args.limit_chunks and executed >= args.limit_chunks:
            break

        cmd = [
            sys.executable,
            "-m",
            "smart",
            "--config",
            str(args.config),
            "run",
            "--category",
            category,
        ]
        if args.force:
            cmd.append("--force")
        for mesh_id in mesh_chunk:
            cmd.extend(["--mesh", mesh_id])

        start = time.perf_counter()
        rc, timed_out = _run_command(
            cmd,
            cwd=REPO_ROOT,
            dry_run=args.dry_run,
            timeout_sec=args.chunk_timeout_sec,
        )
        elapsed = time.perf_counter() - start
        record = {
            "global_chunk_index": global_chunk_index,
            "schedule_index": schedule_index,
            "category": category,
            "local_chunk_index": local_chunk_index,
            "mesh_count": len(mesh_chunk),
            "meshes": mesh_chunk,
            "returncode": rc,
            "timed_out": bool(timed_out),
            "timeout_sec": float(args.chunk_timeout_sec),
            "elapsed_sec": elapsed,
            "config": str(args.config),
            "dry_run": bool(args.dry_run),
        }
        _append_jsonl(args.log, record)
        executed += 1

        if rc == 0 and not args.dry_run:
            successful_counts[category] += len(mesh_chunk)
        if rc != 0:
            failures += 1
            if not args.continue_on_error:
                print(json.dumps({"executed_chunks": executed, "failures": failures}, sort_keys=True))
                return rc

    print(
        json.dumps(
            {
                "executed_chunks": executed,
                "failures": failures,
                "log": str(args.log),
                "dry_run": bool(args.dry_run),
                "successful_counts": dict(sorted(successful_counts.items())),
            },
            sort_keys=True,
        )
    )
    return 1 if failures and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
