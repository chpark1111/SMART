from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart.action_prior import build_action_prior_from_traces


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an opt-in MCTS action-prior JSON from SMART trace_actions_path JSONL. "
            "The prior only changes action ordering when action_prior_weight > 0."
        )
    )
    parser.add_argument("traces", nargs="+", help="Trace JSONL file(s) from SMART refine/MCTS")
    parser.add_argument("--output", required=True, help="Path to write action-prior JSON")
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Only positive/improving actions at or above this reward contribute",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Additive count smoothing before logit conversion",
    )
    parser.add_argument(
        "--reward-power",
        type=float,
        default=1.0,
        help="Exponent applied to positive rewards before accumulation",
    )
    parser.add_argument(
        "--include-action-logits",
        action="store_true",
        help="Also write per-action logits. Useful for same-mesh/search-layout experiments; less category-general.",
    )
    parser.add_argument(
        "--num-action-scale",
        type=int,
        default=0,
        help="Override coord/scale key count. Default infers from trace schema and scale_idx values.",
    )
    args = parser.parse_args()

    output = build_action_prior_from_traces(
        args.traces,
        output=args.output,
        min_reward=args.min_reward,
        smoothing=args.smoothing,
        reward_power=args.reward_power,
        include_action_logits=args.include_action_logits,
        num_action_scale=args.num_action_scale or None,
    )
    print(json.dumps(output["metadata"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
