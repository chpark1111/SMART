#!/usr/bin/env python3
"""Summarize SMART pipeline failures from manifest JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        default="runs/expanded_full/manifests",
        help="Directory containing stage JSONL manifests.",
    )
    parser.add_argument("--top-slow", type=int, default=5)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def classify(record: dict[str, Any]) -> str:
    error = str(record.get("error") or "").lower()
    metadata = record.get("metadata") or {}
    if record.get("status") != "failed":
        return str(record.get("status", "unknown"))
    if "sigsegv" in error or "segmentation" in error or "rc=-11" in error:
        return "crash_sigsegv"
    if "sigterm" in error or "rc=-15" in error:
        return "killed_sigterm"
    if "sigkill" in error or "rc=-9" in error:
        return "killed_sigkill"
    if "timed out" in error or "timeout" in error or "rc=124" in error:
        return "timeout"
    if "missing" in error:
        return "missing_input"
    if "tetra element count below minimum" in error:
        return "validation_too_few_tets"
    if "surface face count below minimum" in error:
        return "validation_surface_too_small"
    if "surface is not watertight" in error:
        return "validation_non_watertight_surface"
    if "multiple connected components" in error:
        return "validation_multi_component_surface"
    if "validation failed" in error:
        return "validation"
    attempts = metadata.get("attempts") if isinstance(metadata, dict) else None
    if attempts:
        joined = " ".join(str(attempt.get("failure", "")).lower() for attempt in attempts)
        if "sigsegv" in joined:
            return "crash_sigsegv"
        if "timed out" in joined:
            return "timeout"
        if "tetra element count below minimum" in joined:
            return "validation_too_few_tets"
        if "surface is not watertight" in joined:
            return "validation_non_watertight_surface"
        if "multiple connected components" in joined:
            return "validation_multi_component_surface"
    return "failed_other"


def load_records(manifest_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(manifest_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record.setdefault("stage", path.stem)
                records.append(record)
    return records


def main() -> int:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    records = load_records(manifest_dir)
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    failed: list[dict[str, Any]] = []
    for record in records:
        label = classify(record)
        by_stage[str(record.get("stage", "unknown"))][label] += 1
        if record.get("status") == "failed":
            failed.append({**record, "failure_class": label})

    slow_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        try:
            elapsed = float(record["finished_at"]) - float(record["started_at"])
        except Exception:
            continue
        slow_by_stage[str(record.get("stage", "unknown"))].append(
            {
                "category": record.get("category"),
                "mesh_id": record.get("mesh_id"),
                "status": record.get("status"),
                "elapsed_sec": elapsed,
                "error": record.get("error"),
            }
        )

    report = {
        "manifest_dir": str(manifest_dir),
        "num_records": len(records),
        "status_by_stage": {stage: dict(counter) for stage, counter in sorted(by_stage.items())},
        "failed": failed,
        "slowest": {
            stage: sorted(items, key=lambda item: item["elapsed_sec"], reverse=True)[: args.top_slow]
            for stage, items in sorted(slow_by_stage.items())
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
