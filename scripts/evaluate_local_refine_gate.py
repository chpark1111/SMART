#!/usr/bin/env python3
"""Evaluate local-refine gate thresholds on an exported gate dataset.

This is a fast offline check: it uses the already-computed input/local metrics
from scripts/export_local_refine_gate_dataset.py and predicts which cases would
run local_refine under each threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from smart.local_refine_gate import load_local_refine_gate, score_local_refine_gate  # noqa: E402


METRICS = ("Avg_BVS", "Avg_MOV", "Avg_TOV", "Avg_Covered", "Avg_vIoU", "Avg_cub_CD")
LOWER_IS_BETTER = ("Avg_BVS", "Avg_MOV", "Avg_TOV", "Avg_cub_CD")
HIGHER_IS_BETTER = ("Avg_Covered", "Avg_vIoU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV exported by export_local_refine_gate_dataset.py")
    parser.add_argument("--gate-path", required=True, help="local_refine_gate JSON")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--output", default="", help="Optional JSON report output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_local_refine_gate(args.gate_path)
    rows = _load_rows(args.dataset)
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    probabilities = [score_local_refine_gate(payload, row) for row in rows]
    report = {
        "dataset": args.dataset,
        "gate_path": args.gate_path,
        "rows": len(rows),
        "positive": sum(_label(row) for row in rows),
        "full_local_refine_time_sec": _sum_time(rows),
        "input_mean": _mean_selected(rows, selected_local=set()),
        "full_guarded_mean": _mean_selected(rows, selected_local={idx for idx, row in enumerate(rows) if _label(row)}),
        "thresholds": [
            _threshold_report(rows, probabilities, threshold)
            for threshold in thresholds
        ],
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_compact(report), indent=2, sort_keys=True))
    return 0


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _threshold_report(rows: list[dict[str, Any]], probabilities: list[float], threshold: float) -> dict[str, Any]:
    run_local = {idx for idx, probability in enumerate(probabilities) if probability >= threshold}
    true_positive = {idx for idx, row in enumerate(rows) if _label(row)}
    selected_local = run_local & true_positive
    run_time = sum(_time(rows[idx]) for idx in run_local)
    full_time = _sum_time(rows)
    full_mean = _mean_selected(rows, selected_local=true_positive)
    selected_mean = _mean_selected(rows, selected_local=selected_local)
    return {
        "threshold": threshold,
        "run_local_refine": len(run_local),
        "skip_local_refine": len(rows) - len(run_local),
        "true_positive": len(selected_local),
        "false_positive": len(run_local - true_positive),
        "false_negative": len(true_positive - run_local),
        "precision": len(selected_local) / len(run_local) if run_local else 0.0,
        "recall": len(selected_local) / len(true_positive) if true_positive else 0.0,
        "local_refine_time_sec": run_time,
        "local_refine_time_saved_pct": 100.0 * (1.0 - run_time / full_time) if full_time else 0.0,
        "selected_mean": selected_mean,
        "delta_vs_full_guarded": _delta(selected_mean, full_mean),
        "quality_same_as_full_guarded": _quality_same(selected_mean, full_mean),
    }


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "rows": report["rows"],
        "positive": report["positive"],
        "full_local_refine_time_sec": round(report["full_local_refine_time_sec"], 3),
        "thresholds": [],
    }
    for row in report["thresholds"]:
        compact["thresholds"].append(
            {
                "threshold": row["threshold"],
                "run": row["run_local_refine"],
                "skip": row["skip_local_refine"],
                "tp": row["true_positive"],
                "fp": row["false_positive"],
                "fn": row["false_negative"],
                "saved_pct": round(row["local_refine_time_saved_pct"], 1),
                "same_as_full": row["quality_same_as_full_guarded"],
                "BVS_delta": round(row["delta_vs_full_guarded"]["Avg_BVS"], 6),
                "vIoU_delta": round(row["delta_vs_full_guarded"]["Avg_vIoU"], 6),
            }
        )
    return compact


def _mean_selected(rows: list[dict[str, Any]], *, selected_local: set[int]) -> dict[str, float]:
    return {
        metric: sum(_metric(row, metric, local=idx in selected_local) for idx, row in enumerate(rows)) / len(rows)
        for metric in METRICS
    }


def _delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {metric: candidate[metric] - baseline[metric] for metric in METRICS}


def _quality_same(candidate: dict[str, float], baseline: dict[str, float], tolerance: float = 1.0e-12) -> bool:
    for metric in LOWER_IS_BETTER:
        if candidate[metric] > baseline[metric] + tolerance:
            return False
    for metric in HIGHER_IS_BETTER:
        if candidate[metric] < baseline[metric] - tolerance:
            return False
    return True


def _metric(row: dict[str, Any], metric: str, *, local: bool) -> float:
    return float(row[("local_" if local else "input_") + metric])


def _label(row: dict[str, Any]) -> int:
    return int(float(row.get("label_improved", 0) or 0))


def _time(row: dict[str, Any]) -> float:
    return float(row.get("local_refine_elapsed_sec", 0.0) or 0.0)


def _sum_time(rows: list[dict[str, Any]]) -> float:
    return sum(_time(row) for row in rows)


if __name__ == "__main__":
    raise SystemExit(main())
