#!/usr/bin/env python3
"""Collect exact-scored MCTS candidate traces for RL policy training."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expanded_full.yaml")
    parser.add_argument("--categories", default="", help="Comma-separated category filter")
    parser.add_argument("--per-category-limit", type=int, default=3)
    parser.add_argument("--trace-root", default="runs/bench_exact/candidate_traces")
    parser.add_argument("--tag", default="", help="Trace filename tag. Default uses current timestamp")
    parser.add_argument("--mcts-iter", type=int, default=20)
    parser.add_argument("--max-step", type=int, default=20)
    parser.add_argument("--candidate-top-k", type=int, default=4)
    parser.add_argument("--reward-backend", default="manifold_stateful")
    parser.add_argument("--mcts-backend", default="auto")
    parser.add_argument("--mcts-timeout", type=int, default=300)
    parser.add_argument("--only-existing-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="runs/bench_exact/candidate_trace_collection.json")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    trace_root = _repo_path(args.trace_root)
    trace_root.mkdir(parents=True, exist_ok=True)
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    allowed = {item.strip() for item in str(args.categories or "").split(",") if item.strip()}

    report: dict[str, Any] = {
        "config": args.config,
        "trace_root": str(trace_root),
        "tag": tag,
        "mcts_iter": args.mcts_iter,
        "max_step": args.max_step,
        "candidate_top_k": args.candidate_top_k,
        "reward_backend": args.reward_backend,
        "mcts_backend": args.mcts_backend,
        "categories": {},
    }
    _write_json(output, report)

    for category in cfg.get("categories", []):
        name = str(category["name"])
        if allowed and name not in allowed:
            continue
        meshes = _select_meshes(cfg, category, args.per_category_limit, args.only_existing_refine)
        trace_path = trace_root / f"{tag}_{name}_candidates.jsonl"
        if args.force and trace_path.exists() and not args.dry_run:
            trace_path.unlink()
        category_record = {
            "selected_meshes": meshes,
            "trace_path": str(trace_path),
            "commands": [],
        }
        report["categories"][name] = category_record
        _write_json(output, report)
        for mesh_idx, mesh_id in enumerate(meshes, start=1):
            command = _command(args, category=name, mesh_id=mesh_id, trace_path=trace_path, mesh_idx=mesh_idx, tag=tag)
            record: dict[str, Any] = {
                "mesh_id": mesh_id,
                "command": command,
            }
            if args.dry_run:
                record["returncode"] = None
            else:
                started = time.perf_counter()
                try:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        text=True,
                        capture_output=True,
                        timeout=max(int(args.mcts_timeout) + 90, 90),
                    )
                    record.update(
                        {
                            "returncode": completed.returncode,
                            "elapsed_sec": time.perf_counter() - started,
                            "stdout_tail": completed.stdout[-2000:],
                            "stderr_tail": completed.stderr[-2000:],
                            "trace_lines": _line_count(trace_path),
                        }
                    )
                except subprocess.TimeoutExpired as exc:
                    record.update(
                        {
                            "returncode": 124,
                            "elapsed_sec": time.perf_counter() - started,
                            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                            "trace_lines": _line_count(trace_path),
                            "timeout": True,
                        }
                    )
            category_record["commands"].append(record)
            category_record["trace_lines"] = _line_count(trace_path)
            _write_json(output, report)
            if record.get("returncode") not in (0, None):
                break

    report["aggregate"] = _aggregate(report["categories"])
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(
        command.get("returncode") in (0, None)
        for category in report["categories"].values()
        for command in category.get("commands", [])
    ) else 1


def _select_meshes(
    cfg: dict[str, Any],
    category: dict[str, Any],
    limit: int,
    only_existing_refine: bool,
) -> list[str]:
    refine_root = stage_root(cfg, "refine", category)
    meshes: list[str] = []
    for mesh_id in list_mesh_ids(category):
        if only_existing_refine and latest_bbox_dir(refine_root, mesh_id) is None:
            continue
        meshes.append(mesh_id)
        if limit > 0 and len(meshes) >= limit:
            break
    return meshes


def _command(args: argparse.Namespace, *, category: str, mesh_id: str, trace_path: Path, mesh_idx: int, tag: str) -> list[str]:
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
        f"mcts.exp_tag={tag}_candidate_trace_{category}_{mesh_idx:03d}",
        "--set",
        f"mcts.candidate_trace_path={trace_path}",
        "--set",
        f"mcts.candidate_trace_top_k={args.candidate_top_k}",
        "mcts",
        "--category",
        category,
        "--mesh",
        mesh_id,
    ]
    if args.force:
        command.append("--force")
    return command


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file)


def _aggregate(categories: dict[str, Any]) -> dict[str, Any]:
    total_meshes = 0
    total_success = 0
    total_lines = 0
    total_seconds = 0.0
    for category in categories.values():
        total_lines += int(category.get("trace_lines", 0) or 0)
        for command in category.get("commands", []):
            total_meshes += 1
            if command.get("returncode") == 0:
                total_success += 1
            total_seconds += float(command.get("elapsed_sec", 0.0) or 0.0)
    return {
        "total_meshes": total_meshes,
        "success": total_success,
        "failed": total_meshes - total_success,
        "trace_lines": total_lines,
        "elapsed_sec": total_seconds,
    }


def _repo_path(path: str | Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else REPO_ROOT / out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
