from __future__ import annotations

import csv

import pytest

from smart.local_refine_gate import (
    GATE_INPUT_METRICS,
    load_gate_dataset,
    load_local_refine_gate,
    score_local_refine_gate,
    train_local_refine_gate,
)


def test_load_gate_dataset_uses_only_input_features(tmp_path) -> None:
    dataset_path = tmp_path / "gate.csv"
    _write_gate_rows(dataset_path)

    dataset = load_gate_dataset(dataset_path)

    assert len(dataset.rows) == 4
    assert dataset.categories == ["airplane", "chair"]
    assert "local_Avg_BVS" not in dataset.feature_names
    assert "delta_Avg_BVS" not in dataset.feature_names
    assert dataset.feature_names[-len(GATE_INPUT_METRICS) :] == [f"input_{metric}" for metric in GATE_INPUT_METRICS]
    assert dataset.labels == [1, 0, 1, 0]


def test_train_local_refine_gate_with_torch(tmp_path) -> None:
    pytest.importorskip("torch")
    dataset_path = tmp_path / "gate.csv"
    output_path = tmp_path / "gate_model.json"
    _write_gate_rows(dataset_path)

    payload = train_local_refine_gate(
        dataset_path,
        output=output_path,
        hidden_size=4,
        epochs=5,
        device="cpu",
        leave_one_out=True,
    )
    loaded = load_local_refine_gate(output_path)

    assert payload["policy_type"] == "local_refine_gate"
    assert loaded["metadata"]["trainer_backend"] == "torch"
    assert loaded["metadata"]["rows"] == 4
    assert loaded["metadata"]["leave_one_out"]["n"] == 4
    probability = score_local_refine_gate(
        loaded,
        {
            "category": "airplane",
            "input_Avg_num_box": "10",
            "input_Avg_BVS": "1.0",
            "input_Avg_MOV": "1.0",
            "input_Avg_TOV": "0.7",
            "input_Avg_Covered": "0.999",
            "input_Avg_vIoU": "0.7",
            "input_Avg_cub_CD": "0.0",
        },
    )
    assert 0.0 <= probability <= 1.0


def _write_gate_rows(path) -> None:
    fieldnames = [
        "key",
        "category",
        "mesh_id",
        "label_improved",
        "label_not_worse",
        "selected_local_refine",
        "local_refine_elapsed_sec",
    ]
    fieldnames.extend(f"input_{metric}" for metric in GATE_INPUT_METRICS)
    fieldnames.extend(f"local_{metric}" for metric in GATE_INPUT_METRICS)
    fieldnames.extend(f"delta_{metric}" for metric in GATE_INPUT_METRICS if metric != "Avg_num_box")
    rows = [
        _row("airplane", "a1", 1, 1.0, 1.0, 0.7, 0.999, 0.70),
        _row("airplane", "a2", 0, 2.0, 1.7, 0.9, 0.998, 0.52),
        _row("chair", "c1", 1, 1.1, 0.9, 0.6, 0.999, 0.68),
        _row("chair", "c2", 0, 2.2, 1.9, 1.0, 0.998, 0.50),
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(category: str, mesh: str, label: int, bvs: float, mov: float, tov: float, covered: float, viou: float):
    row = {
        "key": f"{category}/{mesh}",
        "category": category,
        "mesh_id": mesh,
        "label_improved": label,
        "label_not_worse": label,
        "selected_local_refine": label,
        "local_refine_elapsed_sec": 1.0,
        "input_Avg_num_box": 10.0,
        "input_Avg_BVS": bvs,
        "input_Avg_MOV": mov,
        "input_Avg_TOV": tov,
        "input_Avg_Covered": covered,
        "input_Avg_vIoU": viou,
        "input_Avg_cub_CD": 0.0,
    }
    for metric in GATE_INPUT_METRICS:
        row[f"local_{metric}"] = row[f"input_{metric}"]
    for metric in GATE_INPUT_METRICS:
        if metric != "Avg_num_box":
            row[f"delta_{metric}"] = 0.0
    return row
