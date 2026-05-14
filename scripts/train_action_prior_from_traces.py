from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an opt-in SMART MCTS action prior from accepted action traces. "
            "The output guides action sampling only; exact reward stays unchanged."
        )
    )
    parser.add_argument("traces", nargs="+", help="JSONL trace files from --trace_actions_path")
    parser.add_argument("--output", required=True, help="output prior JSON path")
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
    args = parser.parse_args()

    counts: dict[str, float] = defaultdict(float)
    examples = 0
    meshes = set()
    sources: dict[str, int] = defaultdict(int)

    for trace in args.traces:
        path = Path(trace)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = _coord_scale_key(row)
                reward = float(row.get("reward", 0.0))
                weight = max(reward, 0.0) if args.reward_weighted else 1.0
                if weight == 0.0:
                    continue
                counts[key] += weight
                examples += 1
                if row.get("mesh"):
                    meshes.add(str(row["mesh"]))
                if row.get("source"):
                    sources[str(row["source"])] += 1

    keys = ["%d:%d" % (coord_idx, scale_idx) for coord_idx in range(6) for scale_idx in range(2)]
    keys.append("6:0")
    for key in counts:
        if key not in keys:
            keys.append(key)

    alpha = max(float(args.alpha), 0.0)
    total = sum(counts.values()) + alpha * len(keys)
    default_prob = alpha / total if total > 0.0 else 1.0 / max(len(keys), 1)
    default_logit = math.log(max(default_prob, 1e-300))
    logits = {}
    for key in keys:
        prob = (counts.get(key, 0.0) + alpha) / total if total > 0.0 else default_prob
        logits[key] = math.log(max(prob, 1e-300))

    payload: dict[str, Any] = {
        "type": "coord_scale_action_prior",
        "coord_scale_logits": logits,
        "default_logit": default_logit,
        "num_examples": examples,
        "num_meshes": len(meshes),
        "sources": dict(sorted(sources.items())),
        "reward_weighted": bool(args.reward_weighted),
        "alpha": alpha,
        "note": (
            "Use with --action_prior_path and --action_prior_weight. This changes "
            "MCTS action sampling order only and does not replace exact SMART reward."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _coord_scale_key(row: dict[str, Any]) -> str:
    coord_idx = int(row.get("coord_idx", 6))
    scale_idx = int(row.get("scale_idx", 0))
    if coord_idx >= 6:
        return "6:0"
    return "%d:%d" % (coord_idx, scale_idx)


if __name__ == "__main__":
    raise SystemExit(main())
