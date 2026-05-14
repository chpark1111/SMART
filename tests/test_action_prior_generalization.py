from __future__ import annotations

from scripts.benchmark_action_prior_generalization import _aggregate


def test_action_prior_aggregate_recommends_fastest_metric_identical_weight() -> None:
    targets = {
        "a": {
            "speedup_vs_weight0": {"0.0": 1.0, "0.1": 1.2, "0.2": 1.4},
            "metric_diffs_vs_weight0": {
                "0.0": _diffs(),
                "0.1": _diffs(),
                "0.2": _diffs(Avg_vIoU=0.01),
            },
        },
        "b": {
            "speedup_vs_weight0": {"0.0": 1.0, "0.1": 1.1, "0.2": 1.3},
            "metric_diffs_vs_weight0": {
                "0.0": _diffs(),
                "0.1": _diffs(),
                "0.2": _diffs(),
            },
        },
    }

    aggregate = _aggregate(targets, metric_tolerance=1e-9)

    assert aggregate["metric_identical_weights"] == ["0.0", "0.1"]
    assert aggregate["recommended_weight"] == "0.1"
    assert abs(aggregate["recommended_mean_speedup"] - 1.15) < 1e-12
    assert aggregate["max_metric_diffs_vs_weight0"]["0.2"]["Avg_vIoU"] == 0.01


def _diffs(**overrides: float) -> dict[str, float]:
    keys = (
        "Avg_num_box",
        "Avg_BVS",
        "Avg_MOV",
        "Avg_TOV",
        "Avg_Covered",
        "Avg_vIoU",
        "Avg_cub_CD",
    )
    out = {key: 0.0 for key in keys}
    out.update(overrides)
    return out
