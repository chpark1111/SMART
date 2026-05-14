from __future__ import annotations

import json

import smart
from smart.action_prior import build_action_prior_from_traces
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
