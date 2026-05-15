#!/usr/bin/env python3
"""Collect SMART MCTS action traces from configured meshes.

The script is intentionally conservative: it reuses the normal `smart run`
pipeline, records all stage manifests, and only selects meshes that do not have
an MCTS result yet. Mesh2Tet failures stay recorded by the pipeline and the next
batch can continue from the remaining candidates.
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

from smart.pipeline.config import load_config, workspace_path  # noqa: E402
from smart.pipeline.stages import latest_bbox_dir, list_mesh_ids, stage_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SMART batches and append category-general MCTS trace data."
    )
    parser.add_argument("--config", default="configs/expanded_200.yaml")
    parser.add_argument("--categories", default="", help="Comma-separated category names. Default: all config categories")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--target-success-per-category", type=int, default=0)
    parser.add_argument("--max-batches-per-category", type=int, default=1)
    parser.add_argument("--trace-root", default="runs/bench_exact/trace_collection")
    parser.add_argument("--tag", default="", help="Trace filename tag. Default uses current timestamp")
    parser.add_argument("--mcts-iter", type=int, default=20)
    parser.add_argument("--mcts-max-step", type=int, default=20)
    parser.add_argument("--refine-max-step", type=int, default=200)
    parser.add_argument("--reward-backend", default="manifold_stateful")
    parser.add_argument("--mcts-backend", default="")
    parser.add_argument("--tetra-manifold-timeout", type=int, default=180)
    parser.add_argument("--tetra-ftetwild-timeout", type=int, default=300)
    parser.add_argument("--refine-timeout", type=int, default=300)
    parser.add_argument("--mcts-timeout", type=int, default=300)
    parser.add_argument(
        "--ignore-existing-traces",
        action="store_true",
        help="Do not skip meshes that already appear in JSONL files under --trace-root.",
    )
    parser.add_argument("--force", action="store_true", help="Pass --force to smart run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    trace_root = Path(args.trace_root)
    if not trace_root.is_absolute():
        trace_root = REPO_ROOT / trace_root
    trace_root.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")

    allowed = {item.strip() for item in args.categories.split(",") if item.strip()}
    summary: dict[str, Any] = {
        "config": args.config,
        "trace_root": str(trace_root),
        "tag": tag,
        "categories": {},
        "commands": [],
    }
    for category in cfg.get("categories", []):
        name = str(category["name"])
        if allowed and name not in allowed:
            continue
        category_summary = _collect_category(args, cfg, category, trace_root, tag)
        summary["categories"][name] = category_summary
        summary["commands"].extend(category_summary["commands"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    failures = [
        command for command in summary["commands"]
        if command.get("returncode") not in (0, None)
    ]
    return 1 if failures else 0


def _collect_category(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    category: dict[str, Any],
    trace_root: Path,
    tag: str,
) -> dict[str, Any]:
    name = str(category["name"])
    all_meshes = list_mesh_ids(category)
    candidates = _unprocessed_meshes(cfg, category)
    already_done = len(all_meshes) - len(candidates)
    traced_meshes = set() if args.ignore_existing_traces else _traced_meshes(trace_root, name)
    if traced_meshes:
        candidates = [mesh_id for mesh_id in candidates if mesh_id not in traced_meshes]
    target = int(args.target_success_per_category or 0)
    if target > 0:
        remaining_to_target = max(target - already_done, 0)
        max_meshes = min(len(candidates), remaining_to_target)
    else:
        max_meshes = len(candidates)
    max_batches = max(int(args.max_batches_per_category), 0)
    batches: list[list[str]] = []
    cursor = 0
    while cursor < max_meshes and len(batches) < max_batches:
        batch = candidates[cursor: cursor + max(int(args.batch_size), 1)]
        if not batch:
            break
        batches.append(batch)
        cursor += len(batch)

    commands = []
    for batch_idx, batch in enumerate(batches, start=1):
        trace_path = trace_root / f"{tag}_{name}_batch{batch_idx:03d}.jsonl"
        command = _smart_command(args, name, batch, trace_path)
        command_record: dict[str, Any] = {
            "category": name,
            "batch": batch_idx,
            "meshes": batch,
            "trace_path": str(trace_path),
            "command": command,
        }
        if args.dry_run:
            command_record["returncode"] = None
        else:
            started = time.time()
            result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
            command_record.update(
                {
                    "returncode": result.returncode,
                    "seconds": time.time() - started,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                    "trace_lines": _line_count(trace_path),
                }
            )
            if result.returncode != 0:
                commands.append(command_record)
                break
        commands.append(command_record)

    return {
        "configured_meshes": len(all_meshes),
        "already_done_or_skipped_by_mcts": already_done,
        "already_traced": len(traced_meshes.intersection(all_meshes)),
        "candidates_before": len(candidates),
        "planned_batches": len(batches),
        "commands": commands,
    }


def _unprocessed_meshes(cfg: dict[str, Any], category: dict[str, Any]) -> list[str]:
    out = []
    root = stage_root(cfg, "mcts", category)
    known_tetra_failures = _latest_stage_statuses(cfg, "tetra", str(category["name"]))
    known_refine_failures = _latest_stage_statuses(cfg, "refine", str(category["name"]))
    known_mcts_failures = _latest_stage_statuses(cfg, "mcts", str(category["name"]))
    for mesh_id in list_mesh_ids(category):
        if known_tetra_failures.get(mesh_id) == "failed":
            continue
        if known_refine_failures.get(mesh_id) == "failed":
            continue
        if known_mcts_failures.get(mesh_id) == "failed":
            continue
        if latest_bbox_dir(root, mesh_id) is None:
            out.append(mesh_id)
    return out


def _latest_stage_statuses(cfg: dict[str, Any], stage: str, category_name: str) -> dict[str, str]:
    path = workspace_path(cfg, "manifests", f"{stage}.jsonl")
    latest: dict[str, tuple[float, str]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("category", "")) != category_name:
                continue
            mesh_id = str(record.get("mesh_id", ""))
            if not mesh_id:
                continue
            finished_at = float(record.get("finished_at", 0.0) or 0.0)
            previous = latest.get(mesh_id)
            if previous is None or finished_at >= previous[0]:
                latest[mesh_id] = (finished_at, str(record.get("status", "")))
    return {mesh_id: status for mesh_id, (_finished, status) in latest.items()}


def _traced_meshes(trace_root: Path, category_name: str) -> set[str]:
    meshes: set[str] = set()
    for trace_path in sorted(trace_root.glob("*.jsonl")):
        with trace_path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(record.get("category", "")) != category_name:
                    continue
                mesh_id = str(record.get("mesh", ""))
                if mesh_id:
                    meshes.add(mesh_id)
    return meshes


def _smart_command(
    args: argparse.Namespace,
    category: str,
    meshes: list[str],
    trace_path: Path,
) -> list[str]:
    command = [
        args.python,
        "-m",
        "smart",
        "--config",
        args.config,
        "--set",
        f"tetra.manifold_timeout_sec={args.tetra_manifold_timeout}",
        "--set",
        f"tetra.ftetwild_timeout_sec={args.tetra_ftetwild_timeout}",
        "--set",
        f"refine.reward_backend={args.reward_backend}",
        "--set",
        f"refine.max_step={args.refine_max_step}",
        "--set",
        f"refine.timeout_sec={args.refine_timeout}",
        "--set",
        "refine.summary_metrics=false",
        "--set",
        f"mcts.reward_backend={args.reward_backend}",
        "--set",
        f"mcts.mcts_iter={args.mcts_iter}",
        "--set",
        f"mcts.max_step={args.mcts_max_step}",
        "--set",
        f"mcts.timeout_sec={args.mcts_timeout}",
        "--set",
        "mcts.summary_metrics=false",
        "--set",
        f"mcts.trace_actions_path={trace_path}",
    ]
    if args.mcts_backend:
        command.extend(["--set", f"mcts.backend={args.mcts_backend}"])
    command.extend(["run", "--category", category])
    if args.force:
        command.append("--force")
    for mesh in meshes:
        command.extend(["--mesh", mesh])
    return command


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file)


if __name__ == "__main__":
    raise SystemExit(main())
