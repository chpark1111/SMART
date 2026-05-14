from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def build_action_prior_from_traces(
    traces: Iterable[str | Path],
    *,
    output: str | Path,
    min_reward: float = 0.0,
    smoothing: float = 1.0,
    reward_power: float = 1.0,
    include_action_logits: bool = False,
) -> dict[str, Any]:
    """Build an opt-in MCTS action prior without changing SMART's exact reward."""

    counts: dict[str, float] = defaultdict(float)
    action_counts: dict[str, float] = defaultdict(float)
    total = 0.0
    action_total = 0.0
    kept = 0
    seen = 0
    trace_paths = [Path(path) for path in traces]

    for trace_path in trace_paths:
        with trace_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                seen += 1
                record = json.loads(line)
                reward = float(record.get("reward", 0.0))
                if reward < min_reward:
                    continue
                coord_idx = int(record.get("coord_idx", 6))
                scale_idx = int(record.get("scale_idx", 0))
                key = f"{coord_idx}:{scale_idx if coord_idx != 6 else 0}"
                weight = max(reward, 0.0) ** reward_power
                if weight == 0.0:
                    weight = 1.0
                counts[key] += weight
                action_counts[str(int(record.get("action", 0)))] += weight
                total += weight
                action_total += weight
                kept += 1

    all_keys = [f"{coord}:{scale}" for coord in range(6) for scale in range(2)] + ["6:0"]
    denom = total + smoothing * len(all_keys)
    if denom <= 0.0:
        raise ValueError("action-prior denominator must be positive")
    priors = {}
    for key in all_keys:
        prob = (counts.get(key, 0.0) + smoothing) / denom
        priors[key] = math.log(prob)

    payload: dict[str, Any] = {
        "coord_scale_logits": priors,
        "default_logit": math.log(smoothing / denom),
        "metadata": {
            "source": "smart.action_prior.build_action_prior_from_traces",
            "trace_files": [str(path) for path in trace_paths],
            "records_seen": seen,
            "records_used": kept,
            "min_reward": min_reward,
            "smoothing": smoothing,
            "reward_power": reward_power,
        },
    }
    if include_action_logits and action_counts:
        action_denom = action_total + smoothing * len(action_counts)
        payload["action_logits"] = {
            action: math.log((count + smoothing) / action_denom)
            for action, count in sorted(action_counts.items(), key=lambda item: int(item[0]))
        }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
