#!/usr/bin/env python3
"""Export a local-refine guard report as a small gating dataset.

The target is whether local_refine produced an exact SMART-metric improvement
under the guard. This is intentionally separate from the paper pipeline: it is
for training or analyzing a policy that decides when post-MCTS local search is
worth running.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("Avg_num_box", "Avg_BVS", "Avg_MOV", "Avg_TOV", "Avg_Covered", "Avg_vIoU", "Avg_cub_CD")
DELTA_METRICS = ("Avg_BVS", "Avg_MOV", "Avg_TOV", "Avg_Covered", "Avg_vIoU", "Avg_cub_CD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="JSON report from scripts/run_quality_guarded_local_refine.py")
    parser.add_argument("--output", required=True, help="Path to write CSV or JSONL")
    parser.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = [_row_from_record(key, record) for key, record in sorted(report.get("records", {}).items())]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "jsonl":
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    else:
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "output": str(output),
        "format": args.format,
        "rows": len(rows),
        "positive": sum(1 for row in rows if int(row["label_improved"]) == 1),
        "selected_local_refine": sum(1 for row in rows if int(row["selected_local_refine"]) == 1),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _row_from_record(key: str, record: dict[str, Any]) -> dict[str, Any]:
    runs = record.get("runs", {})
    baseline = runs.get("input", {}).get("summary") or {}
    local = runs.get("local_refine", {}).get("summary") or {}
    selection = record.get("selection", {})
    comparison = selection.get("comparisons", {}).get("local_refine", {})
    deltas = comparison.get("deltas") or {}
    selected = selection.get("selected_label") == "local_refine"
    improved = bool(comparison.get("improved"))
    not_worse = bool(comparison.get("not_worse"))
    row: dict[str, Any] = {
        "key": key,
        "category": record.get("category"),
        "mesh_id": record.get("mesh_id"),
        "label_improved": int(improved),
        "label_not_worse": int(not_worse),
        "selected_local_refine": int(selected),
        "selection_reason": selection.get("reason"),
        "local_refine_elapsed_sec": _float_or_empty(runs.get("local_refine", {}).get("elapsed_sec")),
    }
    for metric in METRICS:
        row[f"input_{metric}"] = _float_or_empty(baseline.get(metric))
        row[f"local_{metric}"] = _float_or_empty(local.get(metric))
    for metric in DELTA_METRICS:
        row[f"delta_{metric}"] = _float_or_empty(deltas.get(metric))
    return row


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _float_or_empty(value: Any) -> float | str:
    if value is None:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
