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
