#!/usr/bin/env python3
"""Train a PyTorch gate that predicts useful post-MCTS local refinement.

Input is the CSV/JSONL exported by scripts/export_local_refine_gate_dataset.py.
The model only sees pre-local-refine fields, so it can be used before spending
time on the optional local search stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smart.local_refine_gate import train_local_refine_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV/JSONL gate dataset from export_local_refine_gate_dataset.py")
    parser.add_argument("--output", required=True, help="Path to write the gate model JSON")
    parser.add_argument(
        "--target",
        default="label_improved",
        choices=("label_improved", "label_not_worse", "selected_local_refine"),
        help="Supervised target column",
    )
    parser.add_argument("--hidden-size", type=int, default=8, help="Hidden layer size; 0 uses logistic regression")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, mps, cuda, or cpu")
    parser.add_argument(
        "--no-leave-one-out",
        action="store_true",
        help="Skip leave-one-out validation and only train the final model",
    )
    parser.add_argument(
        "--positive-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="Class weighting for BCE loss",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = train_local_refine_gate(
        args.dataset,
        output=args.output,
        target=args.target,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        leave_one_out=not args.no_leave_one_out,
        positive_weight=args.positive_weight,
    )
    metadata = dict(payload["metadata"])
    metadata["output"] = str(Path(args.output))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
