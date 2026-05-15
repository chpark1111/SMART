from __future__ import annotations

import smart
from scripts.run_quality_guarded_mcts import (
    _adaptive_stop_reason,
    _aggregate_records,
    _final_return_quality_score,
)
from smart.quality import compare_quality, quality_gain_score, select_quality_guarded_run


def _summary(**overrides):
    data = {
        "Avg_num_box": 4.0,
        "Avg_BVS": 2.0,
        "Avg_MOV": 1.0,
        "Avg_TOV": 0.8,
        "Avg_Covered": 0.99,
        "Avg_vIoU": 0.5,
        "Avg_cub_CD": 0.0,
    }
    data.update(overrides)
    return data


def _run(summary, elapsed=1.0):
    return {
        "returncode": 0,
        "evaluation_returncode": 0,
        "elapsed_sec": elapsed,
        "summary": summary,
    }


def test_compare_quality_detects_not_worse_improvement() -> None:
    comparison = compare_quality(
        _summary(Avg_BVS=1.9, Avg_Covered=0.995),
        _summary(),
    )

    assert comparison.not_worse is True
    assert comparison.improved is True
    assert comparison.worse_metrics == []
    assert set(comparison.improved_metrics) == {"Avg_BVS", "Avg_Covered"}
    assert smart.compare_quality is compare_quality


def test_compare_quality_supports_metric_specific_tolerance() -> None:
    strict = compare_quality(
        _summary(Avg_BVS=1.9, Avg_Covered=0.9895),
        _summary(Avg_Covered=0.99),
    )
    tolerant = compare_quality(
        _summary(Avg_BVS=1.9, Avg_Covered=0.9895),
        _summary(Avg_Covered=0.99),
        metric_tolerances={"Avg_Covered": 0.001},
    )

    assert strict.not_worse is False
    assert strict.worse_metrics == ["Avg_Covered"]
    assert tolerant.not_worse is True
    assert tolerant.improved is True
    assert "Avg_BVS" in tolerant.improved_metrics


def test_quality_guard_rejects_worse_prior() -> None:
    selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=2.0),
            "prior": _run(_summary(Avg_Covered=0.98), elapsed=1.0),
        },
        candidate_labels=["prior"],
    )

    assert selection.selected_label == "baseline"
    assert selection.reason == "baseline_selected"
    assert selection.rejected_labels == {"prior": ["Avg_Covered"]}
    assert smart.select_quality_guarded_run is select_quality_guarded_run


def test_quality_guard_accepts_identical_faster_prior() -> None:
    selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=2.0),
            "prior": _run(_summary(), elapsed=1.0),
        },
        candidate_labels=["prior"],
    )

    assert selection.selected_label == "prior"
    assert selection.reason == "candidate_not_worse_faster"
    assert selection.eligible_labels == ["baseline", "prior"]


def test_quality_guard_prefers_quality_improvement() -> None:
    selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=1.0),
            "prior": _run(_summary(Avg_TOV=0.7), elapsed=2.0),
        },
        candidate_labels=["prior"],
    )

    assert selection.selected_label == "prior"
    assert selection.reason == "candidate_quality_improved"


def test_quality_guard_handles_multiple_prior_candidates() -> None:
    selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=1.0),
            "prior_w0p05": _run(_summary(Avg_Covered=0.98), elapsed=0.5),
            "prior_w0p1": _run(_summary(), elapsed=0.6),
            "prior_w0p2": _run(_summary(Avg_BVS=1.8), elapsed=2.0),
        },
        candidate_labels=["prior_w0p05", "prior_w0p1", "prior_w0p2"],
    )

    assert selection.selected_label == "prior_w0p2"
    assert selection.reason == "candidate_quality_improved"
    assert selection.rejected_labels == {"prior_w0p05": ["Avg_Covered"]}
    assert selection.eligible_labels == ["baseline", "prior_w0p1", "prior_w0p2"]


def test_quality_score_objective_ignores_faster_identical_prior() -> None:
    selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=2.0),
            "prior": _run(_summary(), elapsed=1.0),
        },
        candidate_labels=["prior"],
        selection_objective="quality_score",
    )

    assert selection.selected_label == "baseline"
    assert selection.reason == "baseline_selected"


def test_quality_score_objective_prefers_larger_metric_gain() -> None:
    selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=1.0),
            "small_gain": _run(_summary(Avg_TOV=0.75), elapsed=0.5),
            "large_gain": _run(_summary(Avg_BVS=1.8), elapsed=2.0),
        },
        candidate_labels=["small_gain", "large_gain"],
        selection_objective="quality_score",
    )

    assert selection.selected_label == "large_gain"
    assert selection.reason == "candidate_quality_score_improved"
    assert quality_gain_score(selection.comparisons["large_gain"]) > quality_gain_score(selection.comparisons["small_gain"])
    assert smart.quality_gain_score is quality_gain_score


def test_final_return_score_penalizes_guard_failure() -> None:
    comparison = compare_quality(
        _summary(Avg_BVS=2.1, Avg_MOV=0.7),
        _summary(),
    ).to_dict()

    assert quality_gain_score(comparison) > 0
    assert comparison["not_worse"] is False
    assert _final_return_quality_score(
        comparison,
        weights={},
        not_worse=False,
        worse_metrics=comparison["worse_metrics"],
    ) < 0


def test_adaptive_guard_stops_only_after_quality_improvement() -> None:
    identical_selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=2.0),
            "prior_w0p05": _run(_summary(), elapsed=1.0),
        },
        candidate_labels=["prior_w0p05"],
    )
    improved_selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=2.0),
            "prior_w0p05": _run(_summary(Avg_BVS=1.8), elapsed=3.0),
        },
        candidate_labels=["prior_w0p05"],
    )

    assert identical_selection.reason == "candidate_not_worse_faster"
    assert _adaptive_stop_reason(identical_selection) is None
    assert _adaptive_stop_reason(identical_selection, mode="not_worse") == "candidate_not_worse_faster"
    assert improved_selection.reason == "candidate_quality_improved"
    assert _adaptive_stop_reason(improved_selection) == "candidate_quality_improved"

    quality_score_selection = select_quality_guarded_run(
        {
            "baseline": _run(_summary(), elapsed=2.0),
            "prior_w0p05": _run(_summary(Avg_BVS=1.8), elapsed=3.0),
        },
        candidate_labels=["prior_w0p05"],
        selection_objective="quality_score",
    )
    assert quality_score_selection.reason == "candidate_quality_score_improved"
    assert _adaptive_stop_reason(quality_score_selection) == "candidate_quality_score_improved"


def test_guard_aggregate_counts_skipped_candidate_runs() -> None:
    records = {
        "airplane/a": {
            "category": "airplane",
            "runs": {
                "baseline": _run(_summary(), elapsed=2.0),
                "prior_w0p2": _run(_summary(), elapsed=1.0),
            },
            "candidate_labels": ["prior_w0p2"],
            "skipped_candidate_labels": ["prior_w0p1", "prior_w0p05"],
            "adaptive_stop_reason": "candidate_not_worse_faster",
            "selection": select_quality_guarded_run(
                {
                    "baseline": _run(_summary(), elapsed=2.0),
                    "prior_w0p2": _run(_summary(), elapsed=1.0),
                },
                candidate_labels=["prior_w0p2"],
            ).to_dict(),
            "guarded_record": {"status": "success"},
        },
        "chair/b": {
            "category": "chair",
            "runs": {
                "baseline": _run(_summary(), elapsed=2.0),
                "prior_w0p2": _run(_summary(Avg_Covered=0.98), elapsed=1.0),
                "prior_w0p1": _run(_summary(), elapsed=1.5),
                "prior_w0p05": _run(_summary(), elapsed=1.6),
            },
            "candidate_labels": ["prior_w0p2", "prior_w0p1", "prior_w0p05"],
            "skipped_candidate_labels": [],
            "adaptive_stop_reason": None,
            "selection": select_quality_guarded_run(
                {
                    "baseline": _run(_summary(), elapsed=2.0),
                    "prior_w0p2": _run(_summary(Avg_Covered=0.98), elapsed=1.0),
                    "prior_w0p1": _run(_summary(), elapsed=1.5),
                    "prior_w0p05": _run(_summary(), elapsed=1.6),
                },
                candidate_labels=["prior_w0p2", "prior_w0p1", "prior_w0p05"],
            ).to_dict(),
            "guarded_record": {"status": "success"},
        },
    }

    aggregate = _aggregate_records(records)

    assert aggregate["possible_candidate_runs"] == 6
    assert aggregate["candidate_runs_executed"] == 4
    assert aggregate["candidate_runs_skipped"] == 2
    assert aggregate["executed_total_mcts_runs"] == 6
    assert aggregate["possible_total_mcts_runs"] == 8
    assert aggregate["skipped_candidate_labels"] == 2
