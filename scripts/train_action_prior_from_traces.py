from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart.action_prior import build_action_prior_from_traces, build_linear_action_prior_from_traces


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the first opt-in SMART action prior from accepted action traces. "
            "This trainer currently emits the count-based category-general baseline; "
            "exact SMART reward remains the evaluator at inference time."
        )
    )
    parser.add_argument("traces", nargs="+", help="JSONL trace files from --trace_actions_path")
    parser.add_argument("--output", required=True, help="output prior JSON path")
    parser.add_argument(
        "--model-type",
        choices=["counts", "linear"],
        default="counts",
        help="Policy family. linear is a lightweight state-aware coord/scale scorer.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing weight for unseen coord/scale actions",
    )
    parser.add_argument(
        "--reward-weighted",
        action="store_true",
        help="weight counts by positive reward instead of plain accepted-action counts",
    )
    parser.add_argument(
        "--num-action-scale",
        type=int,
        default=0,
        help="Override coord/scale key count. Default infers from trace schema and scale_idx values.",
    )
    parser.add_argument(
        "--include-action-logits",
        action="store_true",
        help="Also include per-action logits for same-layout experiments.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="linear policy training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="linear policy learning rate")
    parser.add_argument("--l2", type=float, default=1e-4, help="linear policy L2 regularization")
    args = parser.parse_args()

    if args.model_type == "linear":
        payload = build_linear_action_prior_from_traces(
            args.traces,
            output=args.output,
            min_reward=0.0,
            smoothing=args.alpha,
            reward_power=1.0 if args.reward_weighted else 0.0,
            num_action_scale=args.num_action_scale or None,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    else:
        payload = build_action_prior_from_traces(
            args.traces,
            output=args.output,
            min_reward=0.0,
            smoothing=args.alpha,
            reward_power=1.0 if args.reward_weighted else 0.0,
            include_action_logits=args.include_action_logits,
            num_action_scale=args.num_action_scale or None,
        )
    payload["metadata"]["trainer"] = "scripts/train_action_prior_from_traces.py"
    payload["metadata"]["model_type"] = args.model_type
    payload["metadata"]["reward_weighted"] = bool(args.reward_weighted)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
