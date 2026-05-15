from __future__ import annotations

from dataclasses import dataclass, asdict
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
LOWER_IS_BETTER = ("Avg_BVS", "Avg_MOV", "Avg_TOV", "Avg_cub_CD")
HIGHER_IS_BETTER = ("Avg_Covered", "Avg_vIoU")


@dataclass(frozen=True)
class QualityComparison:
    deltas: dict[str, float]
    improved_metrics: list[str]
    worse_metrics: list[str]
    not_worse: bool
    improved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedSelection:
    selected_label: str
    baseline_label: str
    eligible_labels: list[str]
    rejected_labels: dict[str, list[str]]
    reason: str
    comparisons: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_quality(
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    *,
    tolerance: float = 1e-9,
    metric_tolerances: dict[str, float] | None = None,
) -> QualityComparison:
    metric_tolerances = metric_tolerances or {}
    deltas: dict[str, float] = {}
    worse: list[str] = []
    improved: list[str] = []
    for key in LOWER_IS_BETTER:
        metric_tolerance = float(metric_tolerances.get(key, tolerance))
        delta = _metric(candidate_summary, key) - _metric(baseline_summary, key)
        deltas[key] = delta
        if delta > metric_tolerance:
            worse.append(key)
        elif delta < -metric_tolerance:
            improved.append(key)
    for key in HIGHER_IS_BETTER:
        metric_tolerance = float(metric_tolerances.get(key, tolerance))
        delta = _metric(candidate_summary, key) - _metric(baseline_summary, key)
        deltas[key] = delta
        if delta < -metric_tolerance:
            worse.append(key)
        elif delta > metric_tolerance:
            improved.append(key)
    return QualityComparison(
        deltas=deltas,
        improved_metrics=improved,
        worse_metrics=worse,
        not_worse=not worse,
        improved=bool(improved) and not worse,
    )


def select_quality_guarded_run(
    runs: dict[str, dict[str, Any]],
    *,
    baseline_label: str = "baseline",
    candidate_labels: list[str] | None = None,
    tolerance: float = 1e-9,
    metric_tolerances: dict[str, float] | None = None,
) -> GuardedSelection:
    baseline = runs.get(baseline_label)
    if not _successful_run(baseline):
        fallback = _first_successful_label(runs, candidate_labels)
        if fallback is None:
            return GuardedSelection(
                selected_label=baseline_label,
                baseline_label=baseline_label,
                eligible_labels=[],
                rejected_labels={},
                reason="no_successful_runs",
                comparisons={},
            )
        return GuardedSelection(
            selected_label=fallback,
            baseline_label=baseline_label,
            eligible_labels=[fallback],
            rejected_labels={},
            reason="baseline_failed",
            comparisons={},
        )

    labels = candidate_labels or [label for label in runs if label != baseline_label]
    baseline_summary = baseline["summary"]
    comparisons: dict[str, QualityComparison] = {}
    eligible = [baseline_label]
    rejected: dict[str, list[str]] = {}
    for label in labels:
        run = runs.get(label)
        if not _successful_run(run):
            rejected[label] = ["run_failed"]
            continue
        comparison = compare_quality(
            run["summary"],
            baseline_summary,
            tolerance=tolerance,
            metric_tolerances=metric_tolerances,
        )
        comparisons[label] = comparison
        if comparison.not_worse:
            eligible.append(label)
        else:
            rejected[label] = list(comparison.worse_metrics)

    selected = max(
        eligible,
        key=lambda label: (
            _improved_count(comparisons.get(label)),
            _stage_speed(runs.get(label)),
        ),
    )
    if selected == baseline_label:
        reason = "baseline_selected"
    elif _improved_count(comparisons.get(selected)) > 0:
        reason = "candidate_quality_improved"
    else:
        reason = "candidate_not_worse_faster"
    return GuardedSelection(
        selected_label=selected,
        baseline_label=baseline_label,
        eligible_labels=eligible,
        rejected_labels=rejected,
        reason=reason,
        comparisons={label: comparison.to_dict() for label, comparison in comparisons.items()},
    )


def _metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if value is None:
        raise ValueError(f"Missing metric {key}")
    return float(value)


def _successful_run(run: dict[str, Any] | None) -> bool:
    if not isinstance(run, dict):
        return False
    if run.get("returncode") not in (None, 0):
        return False
    if run.get("evaluation_returncode") not in (None, 0):
        return False
    return isinstance(run.get("summary"), dict)


def _first_successful_label(runs: dict[str, dict[str, Any]], labels: list[str] | None) -> str | None:
    for label in labels or runs.keys():
        if _successful_run(runs.get(label)):
            return label
    return None


def _improved_count(comparison: QualityComparison | None) -> int:
    if comparison is None:
        return 0
    return len(comparison.improved_metrics) if comparison.not_worse else -1


def _stage_speed(run: dict[str, Any] | None) -> float:
    if not isinstance(run, dict):
        return 0.0
    elapsed = float(run.get("elapsed_sec", 0.0) or 0.0)
    return 1.0 / elapsed if elapsed > 0.0 else 0.0
