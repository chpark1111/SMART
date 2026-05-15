from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart.action_prior import (
    build_action_prior_from_traces,
    build_linear_action_prior_from_traces,
    build_mlp_action_prior_from_traces,
    build_policy_gradient_action_prior_from_traces,
    build_policy_value_action_prior_from_traces,
)


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
        choices=["counts", "linear", "mlp", "rl-mlp", "pg-agent", "policy-value"],
        default="counts",
        help="Policy family. learned models only guide action ordering.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing weight for unseen coord/scale actions",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Only trace actions with reward at or above this value are used.",
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
    parser.add_argument("--epochs", type=int, default=200, help="learned policy training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="learned policy learning rate")
    parser.add_argument("--l2", type=float, default=1e-4, help="learned policy L2 regularization")
    parser.add_argument("--hidden-size", type=int, default=16, help="hidden units for --model-type mlp")
    parser.add_argument("--device", default="auto", help="PyTorch device for --model-type mlp: auto, mps, cuda, or cpu")
    parser.add_argument(
        "--advantage-baseline",
        choices=["category", "mesh", "global", "none"],
        default="category",
        help="Reward baseline for --model-type rl-mlp",
    )
    parser.add_argument("--advantage-clip", type=float, default=5.0, help="normalized advantage clip for --model-type rl-mlp")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy bonus for --model-type rl-mlp")
    parser.add_argument("--max-logit-abs", type=float, default=8.0, help="calibrate RL prior logits to this max absolute value")
    parser.add_argument("--value-epochs", type=int, default=0, help="policy-value value-head epochs; default reuses --epochs")
    parser.add_argument(
        "--value-learning-rate",
        type=float,
        default=0.0,
        help="policy-value value-head learning rate; default reuses --learning-rate",
    )
    parser.add_argument("--value-clip", type=float, default=5.0, help="policy-value normalized action-value target clip")
    parser.add_argument(
        "--policy-base-prior",
        default="",
        help="For --model-type policy-value, reuse this action policy and train only the value head.",
    )
    parser.add_argument("--accepted-weight", type=float, default=1.0, help="pg-agent loss weight for accepted SMART trace records")
    parser.add_argument("--candidate-weight", type=float, default=1.0, help="pg-agent loss weight for mcts_candidate records")
    parser.add_argument(
        "--selected-candidate-weight",
        type=float,
        default=1.0,
        help="extra pg-agent multiplier for selected mcts_candidate rows",
    )
    parser.add_argument(
        "--category-balance",
        action="store_true",
        help="rebalance pg-agent examples so airplane/chair/table contribute similar total loss",
    )
    args = parser.parse_args()

    if args.model_type == "linear":
        payload = build_linear_action_prior_from_traces(
            args.traces,
            output=args.output,
            min_reward=args.min_reward,
            smoothing=args.alpha,
            reward_power=1.0 if args.reward_weighted else 0.0,
            num_action_scale=args.num_action_scale or None,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    elif args.model_type == "mlp":
        payload = build_mlp_action_prior_from_traces(
            args.traces,
            output=args.output,
            min_reward=args.min_reward,
            smoothing=args.alpha,
            reward_power=1.0 if args.reward_weighted else 0.0,
            num_action_scale=args.num_action_scale or None,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            hidden_size=args.hidden_size,
            device=args.device,
        )
    elif args.model_type == "rl-mlp":
        from smart.action_prior import build_rl_mlp_action_prior_from_traces

        payload = build_rl_mlp_action_prior_from_traces(
            args.traces,
            output=args.output,
            min_reward=args.min_reward,
            smoothing=args.alpha,
            num_action_scale=args.num_action_scale or None,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            hidden_size=args.hidden_size,
            device=args.device,
            advantage_baseline=args.advantage_baseline,
            advantage_clip=args.advantage_clip,
            entropy_coef=args.entropy_coef,
            max_logit_abs=args.max_logit_abs,
        )
    elif args.model_type == "pg-agent":
        payload = build_policy_gradient_action_prior_from_traces(
            args.traces,
            output=args.output,
            min_reward=args.min_reward,
            smoothing=args.alpha,
            num_action_scale=args.num_action_scale or None,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            hidden_size=args.hidden_size,
            device=args.device,
            advantage_baseline=args.advantage_baseline,
            advantage_clip=args.advantage_clip,
            entropy_coef=args.entropy_coef,
            max_logit_abs=args.max_logit_abs,
            accepted_weight=args.accepted_weight,
            candidate_weight=args.candidate_weight,
            selected_candidate_weight=args.selected_candidate_weight,
            category_balance=args.category_balance,
        )
    elif args.model_type == "policy-value":
        payload = build_policy_value_action_prior_from_traces(
            args.traces,
            output=args.output,
            policy_base_prior=args.policy_base_prior or None,
            min_reward=args.min_reward,
            smoothing=args.alpha,
            num_action_scale=args.num_action_scale or None,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            hidden_size=args.hidden_size,
            device=args.device,
            advantage_baseline=args.advantage_baseline,
            advantage_clip=args.advantage_clip,
            entropy_coef=args.entropy_coef,
            max_logit_abs=args.max_logit_abs,
            accepted_weight=args.accepted_weight,
            candidate_weight=args.candidate_weight,
            selected_candidate_weight=args.selected_candidate_weight,
            category_balance=args.category_balance,
            value_epochs=args.value_epochs or None,
            value_learning_rate=args.value_learning_rate or None,
            value_clip=args.value_clip,
        )
    else:
        payload = build_action_prior_from_traces(
            args.traces,
            output=args.output,
            min_reward=args.min_reward,
            smoothing=args.alpha,
            reward_power=1.0 if args.reward_weighted else 0.0,
            include_action_logits=args.include_action_logits,
            num_action_scale=args.num_action_scale or None,
        )
    payload["metadata"]["trainer"] = "scripts/train_action_prior_from_traces.py"
    payload["metadata"].setdefault("model_type", args.model_type)
    payload["metadata"]["reward_weighted"] = bool(args.reward_weighted)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
