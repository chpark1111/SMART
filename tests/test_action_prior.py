from __future__ import annotations

import json

import pytest

import smart
from smart.action_prior import (
    build_action_prior_from_traces,
    build_linear_action_prior_from_traces,
    build_mlp_action_prior_from_traces,
    load_action_prior,
)
from smart.cli import main


def test_build_action_prior_from_trace_records(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps({"coord_idx": 0, "scale_idx": 0, "action": 0, "reward": 0.5}),
                json.dumps({"coord_idx": 3, "scale_idx": 1, "action": 7, "reward": 1.0}),
                json.dumps({"coord_idx": 6, "scale_idx": 0, "action": 12, "reward": -1.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prior.json"

    prior = build_action_prior_from_traces(
        [trace],
        output=output,
        include_action_logits=True,
    )

    assert output.exists()
    assert prior["schema_version"] == 2
    assert prior["policy_type"] == "coord_scale_count_prior"
    assert prior["num_action_scale"] == 2
    assert prior["metadata"]["records_seen"] == 3
    assert prior["metadata"]["records_used"] == 2
    assert "action_logits" in prior
    assert set(prior["coord_scale_logits"]) == {
        "0:0",
        "0:1",
        "1:0",
        "1:1",
        "2:0",
        "2:1",
        "3:0",
        "3:1",
        "4:0",
        "4:1",
        "5:0",
        "5:1",
        "6:0",
    }
    assert smart.build_action_prior_from_traces is build_action_prior_from_traces
    assert smart.build_linear_action_prior_from_traces is build_linear_action_prior_from_traces
    assert smart.build_mlp_action_prior_from_traces is build_mlp_action_prior_from_traces
    assert smart.load_action_prior is load_action_prior


def test_build_action_prior_infers_wider_scale_space(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "category": "airplane",
                        "mesh": "mesh-a",
                        "coord_idx": 2,
                        "scale_idx": 3,
                        "num_action_scale": 4,
                        "action": 11,
                        "reward": 0.5,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prior = build_action_prior_from_traces([trace], output=tmp_path / "prior.json")

    assert prior["num_action_scale"] == 4
    assert "5:3" in prior["coord_scale_logits"]
    assert prior["metadata"]["categories"] == ["airplane"]
    assert prior["metadata"]["num_meshes"] == 1


def test_linear_action_prior_trains_and_scores_context(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    rows = [
        {
            "category": "airplane",
            "mesh": "a",
            "coord_idx": 3,
            "scale_idx": 1,
            "num_action_scale": 2,
            "action": 7,
            "reward": 1.0,
            "bvs": 2.0,
            "step": 2,
            "max_step": 20,
            "num_bbox": 8,
            "action_unit": 0.02,
        },
        {
            "category": "chair",
            "mesh": "b",
            "coord_idx": 0,
            "scale_idx": 0,
            "num_action_scale": 2,
            "action": 0,
            "reward": 1.0,
            "bvs": 1.1,
            "step": 1,
            "max_step": 20,
            "num_bbox": 6,
            "action_unit": 0.02,
        },
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "linear_prior.json"

    payload = build_linear_action_prior_from_traces(
        [trace],
        output=output,
        epochs=5,
        learning_rate=0.01,
    )
    prior = load_action_prior(output)
    logits = prior.action_logits_for(
        [0, 7, 12],
        num_action_scale=2,
        context={
            "category": "airplane",
            "bvs": 2.0,
            "step": 2,
            "max_step": 20,
            "num_bbox": 8,
            "action_unit": 0.02,
        },
    )

    assert payload["policy_type"] == "coord_scale_linear_prior"
    assert payload["metadata"]["categories"] == ["airplane", "chair"]
    assert len(logits) == 3
    assert all(isinstance(value, float) for value in logits)


def test_mlp_action_prior_trains_with_torch_and_scores_context(tmp_path) -> None:
    pytest.importorskip("torch")
    trace = tmp_path / "trace.jsonl"
    rows = [
        {
            "category": "airplane",
            "mesh": "a",
            "coord_idx": 3,
            "scale_idx": 1,
            "num_action_scale": 2,
            "action": 7,
            "reward": 1.0,
            "bvs": 2.0,
            "step": 2,
            "max_step": 20,
            "num_bbox": 8,
            "action_unit": 0.02,
        },
        {
            "category": "chair",
            "mesh": "b",
            "coord_idx": 0,
            "scale_idx": 0,
            "num_action_scale": 2,
            "action": 0,
            "reward": 1.0,
            "bvs": 1.1,
            "step": 1,
            "max_step": 20,
            "num_bbox": 6,
            "action_unit": 0.02,
        },
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "mlp_prior.json"

    payload = build_mlp_action_prior_from_traces(
        [trace],
        output=output,
        epochs=2,
        learning_rate=0.01,
        hidden_size=4,
        device="cpu",
    )
    prior = load_action_prior(output)
    logits = prior.action_logits_for(
        [0, 7, 12],
        num_action_scale=2,
        context={
            "category": "airplane",
            "bvs": 2.0,
            "step": 2,
            "max_step": 20,
            "num_bbox": 8,
            "action_unit": 0.02,
        },
    )

    assert payload["policy_type"] == "coord_scale_mlp_prior"
    assert payload["metadata"]["trainer_backend"] == "torch"
    assert payload["metadata"]["device"] == "cpu"
    assert len(logits) == 3
    assert all(isinstance(value, float) for value in logits)


def test_cli_build_prior(tmp_path, capsys) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"coord_idx": 0, "scale_idx": 0, "action": 0, "reward": 1.0}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prior.json"

    rc = main(["--config", "configs/smoke_5.yaml", "build-prior", str(trace), "--output", str(output)])

    assert rc == 0
    assert output.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["records_seen"] == 1
    assert printed["records_used"] == 1


def test_cli_build_linear_prior(tmp_path, capsys) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps({"category": "airplane", "coord_idx": 0, "scale_idx": 0, "reward": 1.0}),
                json.dumps({"category": "airplane", "coord_idx": 3, "scale_idx": 1, "reward": 0.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "linear_prior.json"

    rc = main(
        [
            "--config",
            "configs/smoke_5.yaml",
            "build-prior",
            str(trace),
            "--output",
            str(output),
            "--model-type",
            "linear",
            "--epochs",
            "3",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert payload["policy_type"] == "coord_scale_linear_prior"
    assert printed["model_type"] == "linear"


def test_cli_build_mlp_prior(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps({"category": "airplane", "coord_idx": 0, "scale_idx": 0, "reward": 1.0}),
                json.dumps({"category": "airplane", "coord_idx": 3, "scale_idx": 1, "reward": 0.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "mlp_prior.json"

    rc = main(
        [
            "--config",
            "configs/smoke_5.yaml",
            "build-prior",
            str(trace),
            "--output",
            str(output),
            "--model-type",
            "mlp",
            "--epochs",
            "2",
            "--hidden-size",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert payload["policy_type"] == "coord_scale_mlp_prior"
    assert printed["model_type"] == "mlp"
    assert printed["trainer_backend"] == "torch"
