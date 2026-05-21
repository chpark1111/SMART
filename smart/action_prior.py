from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
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
    num_action_scale: int | None = None,
) -> dict[str, Any]:
    """Build an opt-in MCTS action prior without changing SMART's exact reward."""

    counts: dict[str, float] = defaultdict(float)
    action_counts: dict[str, float] = defaultdict(float)
    categories: set[str] = set()
    meshes: set[str] = set()
    reward_backends: set[str] = set()
    volume_methods: set[str] = set()
    max_scale_idx = -1
    max_trace_num_action_scale = 0
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
                if record.get("category"):
                    categories.add(str(record["category"]))
                if record.get("mesh"):
                    meshes.add(str(record["mesh"]))
                if record.get("reward_backend"):
                    reward_backends.add(str(record["reward_backend"]))
                if record.get("manifold_volume_method"):
                    volume_methods.add(str(record["manifold_volume_method"]))
                reward = float(record.get("reward", 0.0))
                if reward < min_reward:
                    continue
                coord_idx = int(record.get("coord_idx", 6))
                scale_idx = int(record.get("scale_idx", 0))
                if coord_idx != 6:
                    max_scale_idx = max(max_scale_idx, scale_idx)
                max_trace_num_action_scale = max(
                    max_trace_num_action_scale,
                    int(record.get("num_action_scale", 0) or 0),
                )
                key = f"{coord_idx}:{scale_idx if coord_idx != 6 else 0}"
                weight = max(reward, 0.0) ** reward_power
                if weight == 0.0:
                    weight = 1.0
                counts[key] += weight
                action_counts[str(int(record.get("action", 0)))] += weight
                total += weight
                action_total += weight
                kept += 1

    inferred_num_action_scale = max(max_scale_idx + 1, max_trace_num_action_scale, 2)
    if num_action_scale is None:
        num_action_scale = inferred_num_action_scale
    num_action_scale = max(int(num_action_scale), inferred_num_action_scale, 1)
    all_keys = coord_scale_keys(num_action_scale)
    denom = total + smoothing * len(all_keys)
    if denom <= 0.0:
        raise ValueError("action-prior denominator must be positive")
    priors = {}
    for key in all_keys:
        prob = (counts.get(key, 0.0) + smoothing) / denom
        priors[key] = math.log(prob)

    payload: dict[str, Any] = {
        "schema_version": 2,
        "policy_type": "coord_scale_count_prior",
        "coord_scale_logits": priors,
        "default_logit": math.log(smoothing / denom),
        "num_action_scale": num_action_scale,
        "metadata": {
            "source": "smart.action_prior.build_action_prior_from_traces",
            "model_type": "counts",
            "trace_files": [str(path) for path in trace_paths],
            "records_seen": seen,
            "records_used": kept,
            "categories": sorted(categories),
            "num_meshes": len(meshes),
            "meshes": sorted(meshes),
            "reward_backends": sorted(reward_backends),
            "manifold_volume_methods": sorted(volume_methods),
            "min_reward": min_reward,
            "smoothing": smoothing,
            "reward_power": reward_power,
            "inferred_num_action_scale": inferred_num_action_scale,
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


def build_linear_action_prior_from_traces(
    traces: Iterable[str | Path],
    *,
    output: str | Path,
    min_reward: float = 0.0,
    smoothing: float = 1.0,
    reward_power: float = 0.0,
    num_action_scale: int | None = None,
    epochs: int = 200,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
) -> dict[str, Any]:
    """Train a small category-general linear policy over coord/scale actions.

    This is intentionally lightweight: it learns action-ordering logits only.
    SMART's exact reward and final evaluation remain outside the model.
    """

    records: list[dict[str, Any]] = []
    categories: set[str] = set()
    meshes: set[str] = set()
    max_scale_idx = -1
    max_trace_num_action_scale = 0
    trace_paths = [Path(path) for path in traces]
    seen = 0

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
                if coord_idx != 6:
                    max_scale_idx = max(max_scale_idx, scale_idx)
                max_trace_num_action_scale = max(
                    max_trace_num_action_scale,
                    int(record.get("num_action_scale", 0) or 0),
                )
                if record.get("category"):
                    categories.add(str(record["category"]))
                if record.get("mesh"):
                    meshes.add(str(record["mesh"]))
                records.append(record)

    inferred_num_action_scale = max(max_scale_idx + 1, max_trace_num_action_scale, 2)
    if num_action_scale is None:
        num_action_scale = inferred_num_action_scale
    num_action_scale = max(int(num_action_scale), inferred_num_action_scale, 1)
    classes = coord_scale_keys(num_action_scale)
    class_to_idx = {key: idx for idx, key in enumerate(classes)}
    category_list = sorted(categories)
    feature_names = linear_feature_names(category_list)

    if not records:
        raise ValueError("no trace records passed the prior training filter")

    weights = [[0.0 for _ in classes] for _ in feature_names]
    bias = [math.log(1.0 / len(classes)) for _ in classes]
    sample_weights = []
    examples = []
    for record in records:
        key = _record_coord_scale_key(record)
        if key not in class_to_idx:
            continue
        reward = max(float(record.get("reward", 0.0)), 0.0)
        sample_weight = reward**reward_power if reward_power != 0.0 else 1.0
        if sample_weight == 0.0:
            sample_weight = 1.0
        examples.append((linear_features(record, category_list), class_to_idx[key]))
        sample_weights.append(sample_weight)

    if not examples:
        raise ValueError("no trainable coord/scale examples found")

    for _ in range(max(int(epochs), 0)):
        grad_w = [[0.0 for _ in classes] for _ in feature_names]
        grad_b = [0.0 for _ in classes]
        total_weight = sum(sample_weights) + 1e-12
        for (features, label), sample_weight in zip(examples, sample_weights):
            logits = _linear_logits(features, weights, bias)
            probs = _softmax(logits)
            for class_idx in range(len(classes)):
                delta = (probs[class_idx] - (1.0 if class_idx == label else 0.0)) * sample_weight
                grad_b[class_idx] += delta
                for feat_idx, value in enumerate(features):
                    grad_w[feat_idx][class_idx] += delta * value
        for feat_idx in range(len(feature_names)):
            for class_idx in range(len(classes)):
                grad = grad_w[feat_idx][class_idx] / total_weight + l2 * weights[feat_idx][class_idx]
                weights[feat_idx][class_idx] -= learning_rate * grad
        for class_idx in range(len(classes)):
            bias[class_idx] -= learning_rate * grad_b[class_idx] / total_weight

    # Blend with smoothed counts for sane behavior on tiny trace sets.
    count_payload = build_action_prior_from_traces(
        trace_paths,
        output=Path(output).with_suffix(".counts.tmp.json"),
        min_reward=min_reward,
        smoothing=smoothing,
        reward_power=reward_power,
        include_action_logits=False,
        num_action_scale=num_action_scale,
    )
    try:
        Path(output).with_suffix(".counts.tmp.json").unlink()
    except OSError:
        pass

    payload: dict[str, Any] = {
        "schema_version": 2,
        "policy_type": "coord_scale_linear_prior",
        "classes": classes,
        "num_action_scale": num_action_scale,
        "feature_names": feature_names,
        "categories": category_list,
        "weights": weights,
        "bias": bias,
        "fallback_coord_scale_logits": count_payload["coord_scale_logits"],
        "default_logit": count_payload["default_logit"],
        "metadata": {
            "source": "smart.action_prior.build_linear_action_prior_from_traces",
            "model_type": "linear",
            "trace_files": [str(path) for path in trace_paths],
            "records_seen": seen,
            "records_used": len(examples),
            "categories": category_list,
            "num_meshes": len(meshes),
            "meshes": sorted(meshes),
            "min_reward": min_reward,
            "smoothing": smoothing,
            "reward_power": reward_power,
            "epochs": int(epochs),
            "learning_rate": learning_rate,
            "l2": l2,
        },
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_mlp_action_prior_from_traces(
    traces: Iterable[str | Path],
    *,
    output: str | Path,
    min_reward: float = 0.0,
    smoothing: float = 1.0,
    reward_power: float = 0.0,
    num_action_scale: int | None = None,
    epochs: int = 200,
    learning_rate: float = 0.03,
    l2: float = 1e-4,
    hidden_size: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    """Train a small PyTorch MLP policy over coord/scale actions.

    The model is intentionally modest and JSON-serializable. Training uses
    PyTorch, with `device="auto"` preferring Apple Silicon MPS when available.
    Inference from the saved JSON remains lightweight inside SMART's MCTS path.
    """

    records: list[dict[str, Any]] = []
    categories: set[str] = set()
    meshes: set[str] = set()
    max_scale_idx = -1
    max_trace_num_action_scale = 0
    trace_paths = [Path(path) for path in traces]
    seen = 0

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
                if coord_idx != 6:
                    max_scale_idx = max(max_scale_idx, scale_idx)
                max_trace_num_action_scale = max(
                    max_trace_num_action_scale,
                    int(record.get("num_action_scale", 0) or 0),
                )
                if record.get("category"):
                    categories.add(str(record["category"]))
                if record.get("mesh"):
                    meshes.add(str(record["mesh"]))
                records.append(record)

    inferred_num_action_scale = max(max_scale_idx + 1, max_trace_num_action_scale, 2)
    if num_action_scale is None:
        num_action_scale = inferred_num_action_scale
    num_action_scale = max(int(num_action_scale), inferred_num_action_scale, 1)
    classes = coord_scale_keys(num_action_scale)
    class_to_idx = {key: idx for idx, key in enumerate(classes)}
    category_list = sorted(categories)
    feature_names = linear_feature_names(category_list)

    if not records:
        raise ValueError("no trace records passed the prior training filter")

    examples = []
    sample_weights = []
    for record in records:
        key = _record_coord_scale_key(record)
        if key not in class_to_idx:
            continue
        reward = max(float(record.get("reward", 0.0)), 0.0)
        sample_weight = reward**reward_power if reward_power != 0.0 else 1.0
        if sample_weight == 0.0:
            sample_weight = 1.0
        examples.append((linear_features(record, category_list), class_to_idx[key]))
        sample_weights.append(sample_weight)

    if not examples:
        raise ValueError("no trainable coord/scale examples found")

    hidden_size = max(int(hidden_size), 1)
    input_dim = len(feature_names)
    output_dim = len(classes)
    init_input_weights = [
        [0.01 * math.sin((feat_idx + 1) * (hidden_idx + 1)) for hidden_idx in range(hidden_size)]
        for feat_idx in range(input_dim)
    ]
    init_hidden_bias = [0.0 for _ in range(hidden_size)]
    init_output_weights = [
        [0.01 * math.cos((hidden_idx + 1) * (class_idx + 1)) for class_idx in range(output_dim)]
        for hidden_idx in range(hidden_size)
    ]
    init_output_bias = [math.log(1.0 / output_dim) for _ in range(output_dim)]

    torch = _import_torch()
    torch_device = _select_torch_device(torch, device)
    x = torch.tensor([features for features, _ in examples], dtype=torch.float32, device=torch_device)
    y = torch.tensor([label for _, label in examples], dtype=torch.long, device=torch_device)
    sw = torch.tensor(sample_weights, dtype=torch.float32, device=torch_device)
    input_w = torch.tensor(init_input_weights, dtype=torch.float32, device=torch_device, requires_grad=True)
    hidden_b = torch.tensor(init_hidden_bias, dtype=torch.float32, device=torch_device, requires_grad=True)
    output_w = torch.tensor(init_output_weights, dtype=torch.float32, device=torch_device, requires_grad=True)
    output_b = torch.tensor(init_output_bias, dtype=torch.float32, device=torch_device, requires_grad=True)
    optimizer = torch.optim.Adam([input_w, hidden_b, output_w, output_b], lr=float(learning_rate))
    total_weight = sw.sum().clamp_min(1e-12)
    for _ in range(max(int(epochs), 0)):
        hidden = torch.tanh(x.matmul(input_w) + hidden_b)
        logits = hidden.matmul(output_w) + output_b
        losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        penalty = (
            input_w.square().mean()
            + output_w.square().mean()
        )
        loss = (losses * sw).sum() / total_weight + float(l2) * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    input_weights = input_w.detach().cpu().tolist()
    hidden_bias = hidden_b.detach().cpu().tolist()
    output_weights = output_w.detach().cpu().tolist()
    output_bias = output_b.detach().cpu().tolist()

    count_payload = build_action_prior_from_traces(
        trace_paths,
        output=Path(output).with_suffix(".counts.tmp.json"),
        min_reward=min_reward,
        smoothing=smoothing,
        reward_power=reward_power,
        include_action_logits=False,
        num_action_scale=num_action_scale,
    )
    try:
        Path(output).with_suffix(".counts.tmp.json").unlink()
    except OSError:
        pass

    payload: dict[str, Any] = {
        "schema_version": 2,
        "policy_type": "coord_scale_mlp_prior",
        "classes": classes,
        "num_action_scale": num_action_scale,
        "feature_names": feature_names,
        "categories": category_list,
        "hidden_size": hidden_size,
        "activation": "tanh",
        "input_weights": input_weights,
        "hidden_bias": hidden_bias,
        "output_weights": output_weights,
        "output_bias": output_bias,
        "fallback_coord_scale_logits": count_payload["coord_scale_logits"],
        "default_logit": count_payload["default_logit"],
        "metadata": {
            "source": "smart.action_prior.build_mlp_action_prior_from_traces",
            "model_type": "mlp",
            "trace_files": [str(path) for path in trace_paths],
            "records_seen": seen,
            "records_used": len(examples),
            "categories": category_list,
            "num_meshes": len(meshes),
            "meshes": sorted(meshes),
            "min_reward": min_reward,
            "smoothing": smoothing,
            "reward_power": reward_power,
            "epochs": int(epochs),
            "learning_rate": learning_rate,
            "l2": l2,
            "hidden_size": hidden_size,
            "trainer_backend": "torch",
            "torch_version": str(torch.__version__),
            "device": str(torch_device),
        },
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_rl_mlp_action_prior_from_traces(
    traces: Iterable[str | Path],
    *,
    output: str | Path,
    min_reward: float = -1.0e18,
    smoothing: float = 1.0,
    num_action_scale: int | None = None,
    epochs: int = 300,
    learning_rate: float = 0.01,
    l2: float = 1e-4,
    hidden_size: int = 32,
    device: str = "auto",
    advantage_baseline: str = "category",
    advantage_clip: float = 5.0,
    entropy_coef: float = 0.01,
    max_logit_abs: float = 8.0,
) -> dict[str, Any]:
    """Train an offline policy-gradient action prior from exact SMART rewards.

    This is intentionally an opt-in research model. It does not replace SMART's
    exact geometric reward; it only exports logits used to order MCTS actions.
    """

    records: list[dict[str, Any]] = []
    categories: set[str] = set()
    meshes: set[str] = set()
    max_scale_idx = -1
    max_trace_num_action_scale = 0
    trace_paths = [Path(path) for path in traces]
    seen = 0

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
                if coord_idx != 6:
                    max_scale_idx = max(max_scale_idx, scale_idx)
                max_trace_num_action_scale = max(
                    max_trace_num_action_scale,
                    int(record.get("num_action_scale", 0) or 0),
                )
                if record.get("category"):
                    categories.add(str(record["category"]))
                if record.get("mesh"):
                    meshes.add(str(record["mesh"]))
                records.append(record)

    inferred_num_action_scale = max(max_scale_idx + 1, max_trace_num_action_scale, 2)
    if num_action_scale is None:
        num_action_scale = inferred_num_action_scale
    num_action_scale = max(int(num_action_scale), inferred_num_action_scale, 1)
    classes = coord_scale_keys(num_action_scale)
    class_to_idx = {key: idx for idx, key in enumerate(classes)}
    category_list = sorted(categories)
    feature_names = linear_feature_names(category_list)

    if not records:
        raise ValueError("no trace records passed the RL prior training filter")

    baseline_map: dict[str, float] = {}
    rewards_by_key: dict[str, list[float]] = defaultdict(list)
    global_rewards: list[float] = []
    for record in records:
        reward = float(record.get("reward", 0.0))
        global_rewards.append(reward)
        if advantage_baseline == "mesh":
            key = str(record.get("mesh", ""))
        elif advantage_baseline == "none":
            key = "__zero__"
        elif advantage_baseline == "global":
            key = "__global__"
        else:
            key = str(record.get("category", ""))
        rewards_by_key[key].append(reward)
    global_baseline = sum(global_rewards) / max(len(global_rewards), 1)
    for key, values in rewards_by_key.items():
        baseline_map[key] = 0.0 if advantage_baseline == "none" else sum(values) / len(values)

    examples = []
    raw_advantages = []
    for record in records:
        action_key = _record_coord_scale_key(record)
        if action_key not in class_to_idx:
            continue
        if advantage_baseline == "mesh":
            baseline_key = str(record.get("mesh", ""))
        elif advantage_baseline == "none":
            baseline_key = "__zero__"
        elif advantage_baseline == "global":
            baseline_key = "__global__"
        else:
            baseline_key = str(record.get("category", ""))
        reward = float(record.get("reward", 0.0))
        advantage = reward - baseline_map.get(baseline_key, global_baseline)
        examples.append((linear_features(record, category_list), class_to_idx[action_key]))
        raw_advantages.append(advantage)

    if not examples:
        raise ValueError("no trainable coord/scale examples found")

    mean_advantage = sum(raw_advantages) / len(raw_advantages)
    variance = sum((value - mean_advantage) ** 2 for value in raw_advantages) / max(len(raw_advantages), 1)
    std_advantage = max(math.sqrt(variance), 1e-12)
    clipped_advantages = [
        max(-float(advantage_clip), min(float(advantage_clip), (value - mean_advantage) / std_advantage))
        for value in raw_advantages
    ]

    hidden_size = max(int(hidden_size), 1)
    input_dim = len(feature_names)
    output_dim = len(classes)
    init_input_weights = [
        [0.01 * math.sin((feat_idx + 1) * (hidden_idx + 1)) for hidden_idx in range(hidden_size)]
        for feat_idx in range(input_dim)
    ]
    init_hidden_bias = [0.0 for _ in range(hidden_size)]
    init_output_weights = [
        [0.01 * math.cos((hidden_idx + 1) * (class_idx + 1)) for class_idx in range(output_dim)]
        for hidden_idx in range(hidden_size)
    ]
    init_output_bias = [math.log(1.0 / output_dim) for _ in range(output_dim)]

    torch = _import_torch()
    torch_device = _select_torch_device(torch, device)
    x = torch.tensor([features for features, _ in examples], dtype=torch.float32, device=torch_device)
    y = torch.tensor([label for _, label in examples], dtype=torch.long, device=torch_device)
    advantages = torch.tensor(clipped_advantages, dtype=torch.float32, device=torch_device)
    input_w = torch.tensor(init_input_weights, dtype=torch.float32, device=torch_device, requires_grad=True)
    hidden_b = torch.tensor(init_hidden_bias, dtype=torch.float32, device=torch_device, requires_grad=True)
    output_w = torch.tensor(init_output_weights, dtype=torch.float32, device=torch_device, requires_grad=True)
    output_b = torch.tensor(init_output_bias, dtype=torch.float32, device=torch_device, requires_grad=True)
    optimizer = torch.optim.Adam([input_w, hidden_b, output_w, output_b], lr=float(learning_rate))
    for _ in range(max(int(epochs), 0)):
        hidden = torch.tanh(x.matmul(input_w) + hidden_b)
        logits = hidden.matmul(output_w) + output_b
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        selected_log_probs = log_probs.gather(1, y.view(-1, 1)).squeeze(1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        penalty = input_w.square().mean() + output_w.square().mean()
        loss = -(advantages * selected_log_probs).mean() - float(entropy_coef) * entropy + float(l2) * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logit_scale = 1.0
    with torch.no_grad():
        hidden = torch.tanh(x.matmul(input_w) + hidden_b)
        logits = hidden.matmul(output_w) + output_b
        max_abs_logit = float(logits.abs().max().detach().cpu().item()) if logits.numel() else 0.0
        if max_logit_abs > 0.0 and max_abs_logit > max_logit_abs:
            logit_scale = float(max_logit_abs) / max_abs_logit
            output_w.mul_(logit_scale)
            output_b.mul_(logit_scale)

    input_weights = input_w.detach().cpu().tolist()
    hidden_bias = hidden_b.detach().cpu().tolist()
    output_weights = output_w.detach().cpu().tolist()
    output_bias = output_b.detach().cpu().tolist()

    fallback_min_reward = max(float(min_reward), 0.0)
    count_payload = build_action_prior_from_traces(
        trace_paths,
        output=Path(output).with_suffix(".counts.tmp.json"),
        min_reward=fallback_min_reward,
        smoothing=smoothing,
        reward_power=0.0,
        include_action_logits=False,
        num_action_scale=num_action_scale,
    )
    try:
        Path(output).with_suffix(".counts.tmp.json").unlink()
    except OSError:
        pass

    payload: dict[str, Any] = {
        "schema_version": 2,
        "policy_type": "coord_scale_mlp_prior",
        "classes": classes,
        "num_action_scale": num_action_scale,
        "feature_names": feature_names,
        "categories": category_list,
        "hidden_size": hidden_size,
        "activation": "tanh",
        "input_weights": input_weights,
        "hidden_bias": hidden_bias,
        "output_weights": output_weights,
        "output_bias": output_bias,
        "fallback_coord_scale_logits": count_payload["coord_scale_logits"],
        "default_logit": count_payload["default_logit"],
        "metadata": {
            "source": "smart.action_prior.build_rl_mlp_action_prior_from_traces",
            "model_type": "offline_rl_mlp",
            "trace_files": [str(path) for path in trace_paths],
            "records_seen": seen,
            "records_used": len(examples),
            "categories": category_list,
            "num_meshes": len(meshes),
            "meshes": sorted(meshes),
            "min_reward": min_reward,
            "fallback_min_reward": fallback_min_reward,
            "smoothing": smoothing,
            "epochs": int(epochs),
            "learning_rate": learning_rate,
            "l2": l2,
            "hidden_size": hidden_size,
            "trainer_backend": "torch",
            "torch_version": str(torch.__version__),
            "device": str(torch_device),
            "advantage_baseline": advantage_baseline,
            "advantage_clip": advantage_clip,
            "entropy_coef": entropy_coef,
            "max_logit_abs": max_logit_abs,
            "logit_scale": logit_scale,
            "pre_scale_max_abs_logit": max_abs_logit,
            "reward_mean": global_baseline,
            "advantage_mean": mean_advantage,
            "advantage_std": std_advantage,
        },
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_policy_gradient_action_prior_from_traces(
    traces: Iterable[str | Path],
    *,
    output: str | Path,
    min_reward: float = -1.0e18,
    smoothing: float = 1.0,
    num_action_scale: int | None = None,
    epochs: int = 120,
    learning_rate: float = 0.005,
    l2: float = 1e-4,
    hidden_size: int = 48,
    device: str = "auto",
    advantage_baseline: str = "category",
    advantage_clip: float = 5.0,
    entropy_coef: float = 0.005,
    max_logit_abs: float = 6.0,
    accepted_weight: float = 1.0,
    candidate_weight: float = 1.0,
    selected_candidate_weight: float = 1.0,
    category_balance: bool = False,
) -> dict[str, Any]:
    """Train an action-level offline policy-gradient prior.

    Unlike the coord/scale prior, this model scores concrete SMART action ids.
    It sees the bbox index and local action layout, so it can guide MCTS toward
    different boxes without replacing the exact geometric reward.
    """

    records: list[dict[str, Any]] = []
    categories: set[str] = set()
    meshes: set[str] = set()
    max_scale_idx = -1
    max_trace_num_action_scale = 0
    trace_paths = [Path(path) for path in traces]
    seen = 0

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
                if coord_idx != 6:
                    max_scale_idx = max(max_scale_idx, scale_idx)
                max_trace_num_action_scale = max(
                    max_trace_num_action_scale,
                    int(record.get("num_action_scale", 0) or 0),
                )
                if record.get("category"):
                    categories.add(str(record["category"]))
                if record.get("mesh"):
                    meshes.add(str(record["mesh"]))
                records.append(record)

    inferred_num_action_scale = max(max_scale_idx + 1, max_trace_num_action_scale, 2)
    if num_action_scale is None:
        num_action_scale = inferred_num_action_scale
    num_action_scale = max(int(num_action_scale), inferred_num_action_scale, 1)
    category_list = sorted(categories)
    feature_names = action_feature_names(category_list, num_action_scale)

    if not records:
        raise ValueError("no trace records passed the policy-gradient prior training filter")

    rewards_by_key: dict[str, list[float]] = defaultdict(list)
    global_rewards: list[float] = []
    for record in records:
        reward = float(record.get("reward", 0.0))
        global_rewards.append(reward)
        if advantage_baseline == "mesh":
            key = str(record.get("mesh", ""))
        elif advantage_baseline == "none":
            key = "__zero__"
        elif advantage_baseline == "global":
            key = "__global__"
        else:
            key = str(record.get("category", ""))
        rewards_by_key[key].append(reward)
    global_baseline = sum(global_rewards) / max(len(global_rewards), 1)
    baseline_map = {
        key: (0.0 if advantage_baseline == "none" else sum(values) / len(values))
        for key, values in rewards_by_key.items()
    }
    candidate_rewards_by_group: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if not _is_action_candidate_record(record):
            continue
        candidate_rewards_by_group[_candidate_group_key(record)].append(float(record.get("reward", 0.0)))
    candidate_baseline_by_group = {
        key: sum(values) / len(values)
        for key, values in candidate_rewards_by_group.items()
        if len(values) > 1
    }

    flat_features: list[list[float]] = []
    segment_starts: list[int] = []
    segment_lengths: list[int] = []
    selected_offsets: list[int] = []
    raw_advantages: list[float] = []
    example_weights: list[float] = []
    example_categories: list[str] = []
    used_records = 0
    candidate_records_used = 0

    for record in records:
        trace_num_action_scale = max(
            int(record.get("num_action_scale", 0) or num_action_scale),
            1,
        )
        per_bbox = 6 * trace_num_action_scale + 1
        selected_action = int(record.get("action", -1))
        if selected_action < 0:
            continue
        selected_bbox = selected_action // per_bbox
        num_bbox = max(int(record.get("num_bbox", 0) or 0), selected_bbox + 1)
        num_actions = num_bbox * per_bbox
        if selected_action >= num_actions:
            continue

        start = len(flat_features)
        for action in range(num_actions):
            flat_features.append(
                action_features(
                    record,
                    action,
                    action_num_action_scale=trace_num_action_scale,
                    model_num_action_scale=num_action_scale,
                    categories=category_list,
                )
            )
        segment_starts.append(start)
        segment_lengths.append(num_actions)
        selected_offsets.append(selected_action)

        if advantage_baseline == "mesh":
            baseline_key = str(record.get("mesh", ""))
        elif advantage_baseline == "none":
            baseline_key = "__zero__"
        elif advantage_baseline == "global":
            baseline_key = "__global__"
        else:
            baseline_key = str(record.get("category", ""))
        reward = float(record.get("reward", 0.0))
        if _is_action_candidate_record(record):
            group_key = _candidate_group_key(record)
            raw_advantages.append(reward - candidate_baseline_by_group.get(group_key, global_baseline))
            weight = float(candidate_weight)
            if bool(record.get("selected", False)):
                weight *= float(selected_candidate_weight)
            candidate_records_used += 1
        else:
            raw_advantages.append(reward - baseline_map.get(baseline_key, global_baseline))
            weight = float(accepted_weight)
        example_weights.append(max(weight, 0.0))
        example_categories.append(str(record.get("category", "")))
        used_records += 1

    if not flat_features or not used_records:
        raise ValueError("no trainable action-level policy-gradient examples found")

    mean_advantage = sum(raw_advantages) / len(raw_advantages)
    variance = sum((value - mean_advantage) ** 2 for value in raw_advantages) / max(len(raw_advantages), 1)
    std_advantage = max(math.sqrt(variance), 1e-12)
    clipped_advantages = [
        max(-float(advantage_clip), min(float(advantage_clip), (value - mean_advantage) / std_advantage))
        for value in raw_advantages
    ]
    if category_balance and example_weights:
        category_counts = Counter(example_categories)
        category_count = max(len(category_counts), 1)
        total_examples = max(len(example_weights), 1)
        example_weights = [
            weight * total_examples / max(category_count * category_counts.get(category, 1), 1)
            for weight, category in zip(example_weights, example_categories)
        ]
    mean_example_weight = sum(example_weights) / max(len(example_weights), 1)
    if mean_example_weight > 0.0:
        example_weights = [weight / mean_example_weight for weight in example_weights]

    hidden_size = max(int(hidden_size), 1)
    input_dim = len(feature_names)
    init_input_weights = [
        [0.01 * math.sin((feat_idx + 1) * (hidden_idx + 1)) for hidden_idx in range(hidden_size)]
        for feat_idx in range(input_dim)
    ]
    init_hidden_bias = [0.0 for _ in range(hidden_size)]
    init_output_weights = [0.01 * math.cos(hidden_idx + 1) for hidden_idx in range(hidden_size)]
    init_output_bias = 0.0

    torch = _import_torch()
    torch_device = _select_torch_device(torch, device)
    x = torch.tensor(flat_features, dtype=torch.float32, device=torch_device)
    starts = [int(value) for value in segment_starts]
    lengths = [int(value) for value in segment_lengths]
    selected = [int(value) for value in selected_offsets]
    max_segment_length = max(lengths)
    row_ids: list[int] = []
    col_ids: list[int] = []
    for row_idx, length in enumerate(lengths):
        row_ids.extend([row_idx] * length)
        col_ids.extend(range(length))
    row_index = torch.tensor(row_ids, dtype=torch.long, device=torch_device)
    col_index = torch.tensor(col_ids, dtype=torch.long, device=torch_device)
    selected_index = torch.tensor(selected, dtype=torch.long, device=torch_device).view(-1, 1)
    advantages = torch.tensor(clipped_advantages, dtype=torch.float32, device=torch_device)
    weights = torch.tensor(example_weights, dtype=torch.float32, device=torch_device)
    weight_sum = torch.clamp(weights.sum(), min=1.0e-12)
    input_w = torch.tensor(init_input_weights, dtype=torch.float32, device=torch_device, requires_grad=True)
    hidden_b = torch.tensor(init_hidden_bias, dtype=torch.float32, device=torch_device, requires_grad=True)
    output_w = torch.tensor(init_output_weights, dtype=torch.float32, device=torch_device, requires_grad=True)
    output_b = torch.tensor(init_output_bias, dtype=torch.float32, device=torch_device, requires_grad=True)
    optimizer = torch.optim.Adam([input_w, hidden_b, output_w, output_b], lr=float(learning_rate))

    for _ in range(max(int(epochs), 0)):
        hidden = torch.tanh(x.matmul(input_w) + hidden_b)
        logits = hidden.matmul(output_w) + output_b
        padded_logits = torch.full(
            (len(lengths), max_segment_length),
            -torch.inf,
            dtype=logits.dtype,
            device=torch_device,
        )
        padded_logits[row_index, col_index] = logits
        log_probs = torch.nn.functional.log_softmax(padded_logits, dim=1)
        selected_log_probs = log_probs.gather(1, selected_index).squeeze(1)
        policy_term = -((weights * advantages * selected_log_probs).sum() / weight_sum)
        probs = log_probs.exp()
        finite_log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
        entropy_per_example = -(probs * finite_log_probs).sum(dim=1)
        entropy = (weights * entropy_per_example).sum() / weight_sum
        penalty = input_w.square().mean() + output_w.square().mean()
        loss = policy_term - float(entropy_coef) * entropy + float(l2) * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logit_scale = 1.0
    with torch.no_grad():
        hidden = torch.tanh(x.matmul(input_w) + hidden_b)
        logits = hidden.matmul(output_w) + output_b
        max_abs_logit = float(logits.abs().max().detach().cpu().item()) if logits.numel() else 0.0
        if max_logit_abs > 0.0 and max_abs_logit > max_logit_abs:
            logit_scale = float(max_logit_abs) / max_abs_logit
            output_w.mul_(logit_scale)
            output_b.mul_(logit_scale)

    fallback_min_reward = max(float(min_reward), 0.0)
    count_payload = build_action_prior_from_traces(
        trace_paths,
        output=Path(output).with_suffix(".counts.tmp.json"),
        min_reward=fallback_min_reward,
        smoothing=smoothing,
        reward_power=0.0,
        include_action_logits=False,
        num_action_scale=num_action_scale,
    )
    try:
        Path(output).with_suffix(".counts.tmp.json").unlink()
    except OSError:
        pass

    payload: dict[str, Any] = {
        "schema_version": 2,
        "policy_type": "action_mlp_prior",
        "num_action_scale": num_action_scale,
        "feature_names": feature_names,
        "categories": category_list,
        "hidden_size": hidden_size,
        "activation": "tanh",
        "action_input_weights": input_w.detach().cpu().tolist(),
        "action_hidden_bias": hidden_b.detach().cpu().tolist(),
        "action_output_weights": output_w.detach().cpu().tolist(),
        "action_output_bias": float(output_b.detach().cpu().item()),
        "fallback_coord_scale_logits": count_payload["coord_scale_logits"],
        "default_logit": count_payload["default_logit"],
        "metadata": {
            "source": "smart.action_prior.build_policy_gradient_action_prior_from_traces",
            "model_type": "policy_gradient_agent",
            "trace_files": [str(path) for path in trace_paths],
            "records_seen": seen,
            "records_used": used_records,
            "candidate_records_used": candidate_records_used,
            "candidate_actions_seen": len(flat_features),
            "categories": category_list,
            "num_meshes": len(meshes),
            "meshes": sorted(meshes),
            "min_reward": min_reward,
            "fallback_min_reward": fallback_min_reward,
            "smoothing": smoothing,
            "epochs": int(epochs),
            "learning_rate": learning_rate,
            "l2": l2,
            "hidden_size": hidden_size,
            "trainer_backend": "torch",
            "torch_version": str(torch.__version__),
            "device": str(torch_device),
            "advantage_baseline": advantage_baseline,
            "advantage_clip": advantage_clip,
            "entropy_coef": entropy_coef,
            "max_logit_abs": max_logit_abs,
            "accepted_weight": accepted_weight,
            "candidate_weight": candidate_weight,
            "selected_candidate_weight": selected_candidate_weight,
            "category_balance": bool(category_balance),
            "mean_example_weight": mean_example_weight,
            "logit_scale": logit_scale,
            "pre_scale_max_abs_logit": max_abs_logit,
            "reward_mean": global_baseline,
            "advantage_mean": mean_advantage,
            "advantage_std": std_advantage,
        },
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_policy_value_action_prior_from_traces(
    traces: Iterable[str | Path],
    *,
    output: str | Path,
    policy_base_prior: str | Path | None = None,
    min_reward: float = -1.0e18,
    smoothing: float = 1.0,
    num_action_scale: int | None = None,
    epochs: int = 120,
    learning_rate: float = 0.005,
    l2: float = 1e-4,
    hidden_size: int = 48,
    device: str = "auto",
    advantage_baseline: str = "category",
    advantage_clip: float = 5.0,
    entropy_coef: float = 0.005,
    max_logit_abs: float = 6.0,
    accepted_weight: float = 1.0,
    candidate_weight: float = 1.0,
    selected_candidate_weight: float = 1.0,
    category_balance: bool = False,
    value_epochs: int | None = None,
    value_learning_rate: float | None = None,
    value_clip: float = 5.0,
    value_positive_weight: float = 1.0,
    value_negative_weight: float = 1.0,
    value_zero_weight: float = 1.0,
    value_auto_balance: bool = False,
    value_group_balance: bool = False,
    value_pairwise_weight: float = 0.0,
    value_pairwise_margin: float = 0.1,
    value_pairwise_max_pairs: int = 200000,
    value_validation_mesh_fraction: float = 0.0,
) -> dict[str, Any]:
    """Train an action-level policy prior plus an action-value head.

    The policy head guides action ordering; the value head predicts normalized
    exact-reward advantage for a concrete action. Both are opt-in search biases:
    SMART's exact geometric reward still decides accepted rollouts and final
    evaluation.
    """

    trace_paths = [Path(path) for path in traces]
    base_policy_metadata: dict[str, Any] = {}
    if policy_base_prior:
        with Path(policy_base_prior).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not payload.get("action_input_weights") or not payload.get("action_output_weights"):
            raise ValueError("policy_base_prior must contain an action-level MLP policy")
        payload.setdefault("metadata", {})
        base_policy_metadata = dict(payload.get("metadata", {}))
        if num_action_scale is not None:
            payload["num_action_scale"] = max(int(payload.get("num_action_scale", 0) or 0), int(num_action_scale))
    else:
        policy_tmp = Path(output).with_suffix(".policy.tmp.json")
        payload = build_policy_gradient_action_prior_from_traces(
            trace_paths,
            output=policy_tmp,
            min_reward=min_reward,
            smoothing=smoothing,
            num_action_scale=num_action_scale,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            hidden_size=hidden_size,
            device=device,
            advantage_baseline=advantage_baseline,
            advantage_clip=advantage_clip,
            entropy_coef=entropy_coef,
            max_logit_abs=max_logit_abs,
            accepted_weight=accepted_weight,
            candidate_weight=candidate_weight,
            selected_candidate_weight=selected_candidate_weight,
            category_balance=category_balance,
        )
        try:
            policy_tmp.unlink()
        except OSError:
            pass

    examples = _action_value_examples(
        trace_paths,
        min_reward=min_reward,
        num_action_scale=int(payload["num_action_scale"]),
        categories=[str(item) for item in payload.get("categories", [])],
        advantage_baseline=advantage_baseline,
        advantage_clip=value_clip,
        accepted_weight=accepted_weight,
        candidate_weight=candidate_weight,
        selected_candidate_weight=selected_candidate_weight,
        category_balance=category_balance,
        value_positive_weight=value_positive_weight,
        value_negative_weight=value_negative_weight,
        value_zero_weight=value_zero_weight,
        value_auto_balance=value_auto_balance,
        value_group_balance=value_group_balance,
    )
    if not examples["features"]:
        raise ValueError("no trainable action-value examples found")

    expected_feature_dim = len(payload.get("action_input_weights", []))
    if expected_feature_dim > 0:
        examples["features"] = _fixed_feature_width(
            examples["features"],
            expected_feature_dim,
        )

    train_indices, validation_indices = _value_train_validation_indices(
        examples.get("mesh_ids", []),
        validation_fraction=float(value_validation_mesh_fraction),
    )

    torch = _import_torch()
    torch_device = _select_torch_device(torch, device)
    all_features = torch.tensor(examples["features"], dtype=torch.float32, device=torch_device)
    all_targets = torch.tensor(examples["targets"], dtype=torch.float32, device=torch_device)
    all_weights = torch.tensor(examples["weights"], dtype=torch.float32, device=torch_device)
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long, device=torch_device)
    features = all_features.index_select(0, train_index_tensor)
    targets = all_targets.index_select(0, train_index_tensor)
    weights = all_weights.index_select(0, train_index_tensor)
    weight_sum = torch.clamp(weights.sum(), min=1.0e-12)
    input_w = torch.tensor(payload["action_input_weights"], dtype=torch.float32, device=torch_device)
    hidden_b = torch.tensor(payload["action_hidden_bias"], dtype=torch.float32, device=torch_device)
    hidden_size = max(int(payload.get("hidden_size", hidden_size) or hidden_size), 1)
    value_w = torch.zeros(hidden_size, dtype=torch.float32, device=torch_device, requires_grad=True)
    value_b = torch.tensor(0.0, dtype=torch.float32, device=torch_device, requires_grad=True)
    pairwise_pairs = _value_pairwise_pairs(
        examples.get("targets", []),
        examples.get("group_keys", []),
        train_indices,
        max_pairs=int(value_pairwise_max_pairs),
    )
    if pairwise_pairs:
        pair_high = torch.tensor([item[0] for item in pairwise_pairs], dtype=torch.long, device=torch_device)
        pair_low = torch.tensor([item[1] for item in pairwise_pairs], dtype=torch.long, device=torch_device)
        pair_weights = torch.tensor([item[2] for item in pairwise_pairs], dtype=torch.float32, device=torch_device)
        pair_weight_sum = torch.clamp(pair_weights.sum(), min=1.0e-12)
    else:
        pair_high = pair_low = pair_weights = pair_weight_sum = None
    optimizer = torch.optim.Adam(
        [value_w, value_b],
        lr=float(value_learning_rate if value_learning_rate is not None else learning_rate),
    )
    for _ in range(max(int(value_epochs if value_epochs is not None else epochs), 0)):
        hidden = torch.tanh(features.matmul(input_w) + hidden_b)
        values = hidden.matmul(value_w) + value_b
        loss = ((values - targets).square() * weights).sum() / weight_sum + float(l2) * value_w.square().mean()
        if (
            float(value_pairwise_weight) > 0.0
            and pair_high is not None
            and pair_low is not None
            and pair_weights is not None
            and pair_weight_sum is not None
        ):
            pair_margin = torch.relu(float(value_pairwise_margin) - (values.index_select(0, pair_high) - values.index_select(0, pair_low)))
            ranking_loss = (pair_margin * pair_weights).sum() / pair_weight_sum
            loss = loss + float(value_pairwise_weight) * ranking_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        all_hidden = torch.tanh(all_features.matmul(input_w) + hidden_b)
        all_values = all_hidden.matmul(value_w) + value_b
        train_metrics = _value_prediction_metrics(
            all_values,
            all_targets,
            all_weights,
            train_indices,
            torch=torch,
        )
        validation_metrics = _value_prediction_metrics(
            all_values,
            all_targets,
            all_weights,
            validation_indices,
            torch=torch,
        )

    payload["policy_type"] = "action_policy_value_prior"
    payload["action_value_output_weights"] = value_w.detach().cpu().tolist()
    payload["action_value_output_bias"] = float(value_b.detach().cpu().item())
    metadata_update: dict[str, Any] = {
        "source": "smart.action_prior.build_policy_value_action_prior_from_traces",
        "model_type": "policy_value_agent",
        "trace_files": [str(path) for path in trace_paths],
        "value_trace_files": [str(path) for path in trace_paths],
        "policy_base_prior": str(policy_base_prior or ""),
        "records_seen": int(examples["records_used"]),
        "records_used": int(examples["records_used"]),
        "candidate_records_used": int(examples["candidate_records_used"]),
        "value_records_used": int(examples["records_used"]),
        "value_candidate_records_used": int(examples["candidate_records_used"]),
        "value_categories": list(examples.get("categories_used", [])),
        "value_epochs": int(value_epochs if value_epochs is not None else epochs),
        "value_learning_rate": float(value_learning_rate if value_learning_rate is not None else learning_rate),
        "value_clip": float(value_clip),
        "value_advantage_baseline": str(advantage_baseline),
        "value_positive_weight": float(value_positive_weight),
        "value_negative_weight": float(value_negative_weight),
        "value_zero_weight": float(value_zero_weight),
        "value_auto_balance": bool(value_auto_balance),
        "value_group_balance": bool(value_group_balance),
        "value_class_multipliers": dict(examples.get("class_multipliers", {})),
        "value_group_balance_groups": int(examples.get("group_balance_groups", 0)),
        "value_pairwise_weight": float(value_pairwise_weight),
        "value_pairwise_margin": float(value_pairwise_margin),
        "value_pairwise_max_pairs": int(value_pairwise_max_pairs),
        "value_pairwise_pairs": int(len(pairwise_pairs)),
        "value_validation_mesh_fraction": float(value_validation_mesh_fraction),
        "value_train_examples": int(len(train_indices)),
        "value_validation_examples": int(len(validation_indices)),
        "value_train_metrics": train_metrics,
        "value_validation_metrics": validation_metrics,
        "value_target_mean": float(examples["target_mean"]),
        "value_target_std": float(examples["target_std"]),
        "value_positive_examples": int(examples["positive_examples"]),
        "value_negative_examples": int(examples["negative_examples"]),
        "value_zero_examples": int(examples["zero_examples"]),
    }
    if base_policy_metadata:
        metadata_update["policy_base_model_type"] = str(base_policy_metadata.get("model_type", ""))
        metadata_update["policy_base_trace_files"] = list(base_policy_metadata.get("trace_files", []))
    payload["metadata"].update(metadata_update)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def coord_scale_keys(num_action_scale: int) -> list[str]:
    """Return SMART coord/scale prior keys for the legacy action order."""

    if num_action_scale < 1:
        raise ValueError("num_action_scale must be positive")
    return [f"{coord}:{scale}" for coord in range(6) for scale in range(num_action_scale)] + ["6:0"]


def load_action_prior(path: str | Path, *, inference_device: str = "json") -> "LoadedActionPrior":
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return LoadedActionPrior(json.load(file), inference_device=inference_device, base_dir=path.parent)


class LoadedActionPrior:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        inference_device: str = "json",
        base_dir: str | Path | None = None,
    ) -> None:
        self.payload = payload
        self.policy_type = str(payload.get("policy_type") or payload.get("type") or "coord_scale_count_prior")
        self.base_dir = Path(base_dir) if base_dir is not None else Path(".")
        self.default_logit = float(payload.get("default_logit", 0.0))
        self.category_priors: dict[str, LoadedActionPrior] = {}
        self.fallback_prior: LoadedActionPrior | None = None
        if self.policy_type == "category_dispatch_prior":
            self._init_category_dispatch(payload, inference_device=inference_device)
        self.coord_scale_logits = {
            str(key): float(value)
            for key, value in payload.get("coord_scale_logits", payload.get("priors", {})).items()
        }
        self.action_logits = {
            int(key): float(value)
            for key, value in payload.get("action_logits", {}).items()
        }
        self.classes = [str(item) for item in payload.get("classes", [])]
        self.categories = [str(item) for item in payload.get("categories", [])]
        self.feature_names = [str(item) for item in payload.get("feature_names", [])]
        self.weights = [
            [float(value) for value in row]
            for row in payload.get("weights", [])
        ]
        self.bias = [float(value) for value in payload.get("bias", [])]
        self.fallback_coord_scale_logits = {
            str(key): float(value)
            for key, value in payload.get("fallback_coord_scale_logits", {}).items()
        }
        self.hidden_size = int(payload.get("hidden_size", 0) or 0)
        self.input_weights = [
            [float(value) for value in row]
            for row in payload.get("input_weights", [])
        ]
        self.hidden_bias = [float(value) for value in payload.get("hidden_bias", [])]
        self.output_weights = [
            [float(value) for value in row]
            for row in payload.get("output_weights", [])
        ]
        self.output_bias = [float(value) for value in payload.get("output_bias", [])]
        self.action_input_weights = [
            [float(value) for value in row]
            for row in payload.get("action_input_weights", [])
        ]
        self.action_hidden_bias = [float(value) for value in payload.get("action_hidden_bias", [])]
        self.action_output_weights = [float(value) for value in payload.get("action_output_weights", [])]
        self.action_output_bias = float(payload.get("action_output_bias", 0.0) or 0.0)
        self.action_value_output_weights = [
            float(value) for value in payload.get("action_value_output_weights", [])
        ]
        self.action_value_output_bias = float(payload.get("action_value_output_bias", 0.0) or 0.0)
        self.model_num_action_scale = int(payload.get("num_action_scale", 0) or 0)
        self.inference_device = str(inference_device or "json")
        self._torch = None
        self._torch_device = None
        self._torch_action_input_weights = None
        self._torch_action_hidden_bias = None
        self._torch_action_output_weights = None
        self._torch_action_output_bias = None
        self._torch_action_value_output_weights = None
        self._torch_action_value_output_bias = None
        self._cpp_action_policy = None
        if self.inference_device.lower() not in {"", "json", "python", "cpp", "native", "c++"}:
            self._init_torch_inference(self.inference_device)

    def _init_category_dispatch(self, payload: dict[str, Any], *, inference_device: str) -> None:
        prior_paths = payload.get("category_prior_paths", {})
        if not isinstance(prior_paths, dict):
            raise ValueError("category_dispatch_prior requires category_prior_paths")
        for category, prior_path in prior_paths.items():
            path = self._resolve_child_prior_path(str(prior_path))
            with path.open("r", encoding="utf-8") as file:
                self.category_priors[str(category).lower()] = LoadedActionPrior(
                    json.load(file),
                    inference_device=inference_device,
                    base_dir=path.parent,
                )
        fallback_path = str(payload.get("fallback_prior_path", "") or "")
        if fallback_path:
            path = self._resolve_child_prior_path(fallback_path)
            with path.open("r", encoding="utf-8") as file:
                self.fallback_prior = LoadedActionPrior(
                    json.load(file),
                    inference_device=inference_device,
                    base_dir=path.parent,
                )

    def _resolve_child_prior_path(self, prior_path: str) -> Path:
        path = Path(prior_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def _dispatch_prior(self, context: dict[str, Any]) -> "LoadedActionPrior | None":
        category = str(context.get("category", "") or "").lower()
        if category and category in self.category_priors:
            return self.category_priors[category]
        return self.fallback_prior

    def action_logits_for(
        self,
        actions: Iterable[int],
        *,
        num_action_scale: int,
        context: dict[str, Any] | None = None,
    ) -> list[float]:
        actions = [int(action) for action in actions]
        if self.policy_type == "category_dispatch_prior":
            prior = self._dispatch_prior(context or {})
            if prior is None:
                return [self.default_logit for _ in actions]
            return prior.action_logits_for(
                actions,
                num_action_scale=int(num_action_scale),
                context=context or {},
            )
        if self.policy_type in {"action_mlp_prior", "action_policy_value_prior"}:
            cpp_values = self._action_mlp_logits_values_for_cpp(
                actions,
                num_action_scale=int(num_action_scale),
                context=context or {},
            )
            if cpp_values is not None:
                return cpp_values[0]
            torch_logits = self._action_mlp_logits_for_torch(
                actions,
                num_action_scale=int(num_action_scale),
                context=context or {},
            )
            if torch_logits is not None:
                return torch_logits
            return self._action_mlp_logits_for(
                actions,
                num_action_scale=int(num_action_scale),
                context=context or {},
            )
        class_logits = self._class_logits(context or {})
        out = []
        per_bbox = 6 * int(num_action_scale) + 1
        for action in actions:
            if action in self.action_logits:
                out.append(self.action_logits[action])
                continue
            local = action % per_bbox
            if local == per_bbox - 1:
                key = "6:0"
            else:
                key = "%d:%d" % (local // int(num_action_scale), local % int(num_action_scale))
            out.append(float(class_logits.get(key, self.default_logit)))
        return out

    def action_values_for(
        self,
        actions: Iterable[int],
        *,
        num_action_scale: int,
        context: dict[str, Any] | None = None,
    ) -> list[float]:
        actions = [int(action) for action in actions]
        if self.policy_type == "category_dispatch_prior":
            prior = self._dispatch_prior(context or {})
            if prior is None:
                return [0.0 for _ in actions]
            return prior.action_values_for(
                actions,
                num_action_scale=int(num_action_scale),
                context=context or {},
            )
        if not self.action_value_output_weights:
            return [0.0 for _ in actions]
        cpp_values = self._action_mlp_logits_values_for_cpp(
            actions,
            num_action_scale=int(num_action_scale),
            context=context or {},
        )
        if cpp_values is not None:
            return cpp_values[1]
        torch_values = self._action_values_for_torch(
            actions,
            num_action_scale=int(num_action_scale),
            context=context or {},
        )
        if torch_values is not None:
            return torch_values
        model_num_action_scale = max(self.model_num_action_scale, int(num_action_scale), 1)
        out = []
        for action in actions:
            hidden = self._action_mlp_hidden_for(
                int(action),
                num_action_scale=int(num_action_scale),
                model_num_action_scale=model_num_action_scale,
                context=context or {},
            )
            value = self.action_value_output_bias
            for hidden_idx, hidden_value in enumerate(hidden):
                if hidden_idx >= len(self.action_value_output_weights):
                    break
                value += hidden_value * self.action_value_output_weights[hidden_idx]
            out.append(float(value))
        return out

    def action_logits_values_for(
        self,
        actions: Iterable[int],
        *,
        num_action_scale: int,
        context: dict[str, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        actions = [int(action) for action in actions]
        context = context or {}
        if self.policy_type == "category_dispatch_prior":
            prior = self._dispatch_prior(context)
            if prior is None:
                return [self.default_logit for _ in actions], [0.0 for _ in actions]
            return prior.action_logits_values_for(
                actions,
                num_action_scale=int(num_action_scale),
                context=context,
            )
        if self.policy_type in {"action_mlp_prior", "action_policy_value_prior"}:
            cpp_values = self._action_mlp_logits_values_for_cpp(
                actions,
                num_action_scale=int(num_action_scale),
                context=context,
            )
            if cpp_values is not None:
                logits, values = cpp_values
                if not self.action_value_output_weights:
                    values = [0.0 for _ in actions]
                return logits, values
        return (
            self.action_logits_for(
                actions,
                num_action_scale=int(num_action_scale),
                context=context,
            ),
            self.action_values_for(
                actions,
                num_action_scale=int(num_action_scale),
                context=context,
            ),
        )

    def _class_logits(self, context: dict[str, Any]) -> dict[str, float]:
        if self.policy_type == "coord_scale_mlp_prior":
            return self._mlp_class_logits(context)
        if self.policy_type != "coord_scale_linear_prior":
            return self.coord_scale_logits
        if not self.classes or not self.weights or not self.bias:
            return self.fallback_coord_scale_logits or self.coord_scale_logits
        features = linear_features(context, self.categories)
        logits = _linear_logits(features, self.weights, self.bias)
        out = {
            key: float(logits[idx])
            for idx, key in enumerate(self.classes)
            if idx < len(logits)
        }
        fallback = self.fallback_coord_scale_logits or self.coord_scale_logits
        for key, value in fallback.items():
            out.setdefault(key, value)
        return out

    def _mlp_class_logits(self, context: dict[str, Any]) -> dict[str, float]:
        if (
            not self.classes
            or not self.input_weights
            or not self.hidden_bias
            or not self.output_weights
            or not self.output_bias
        ):
            return self.fallback_coord_scale_logits or self.coord_scale_logits
        features = linear_features(context, self.categories)
        _, hidden = _mlp_hidden(features, self.input_weights, self.hidden_bias)
        logits = _mlp_logits_from_hidden(hidden, self.output_weights, self.output_bias)
        out = {
            key: float(logits[idx])
            for idx, key in enumerate(self.classes)
            if idx < len(logits)
        }
        fallback = self.fallback_coord_scale_logits or self.coord_scale_logits
        for key, value in fallback.items():
            out.setdefault(key, value)
        return out

    def _action_mlp_logits_for(
        self,
        actions: list[int],
        *,
        num_action_scale: int,
        context: dict[str, Any],
    ) -> list[float]:
        if (
            not self.action_input_weights
            or not self.action_hidden_bias
            or not self.action_output_weights
        ):
            class_logits = self._class_logits(context)
            per_bbox = 6 * int(num_action_scale) + 1
            out = []
            for action in actions:
                local = int(action) % per_bbox
                if local == per_bbox - 1:
                    key = "6:0"
                else:
                    key = "%d:%d" % (local // int(num_action_scale), local % int(num_action_scale))
                out.append(float(class_logits.get(key, self.default_logit)))
            return out

        model_num_action_scale = max(self.model_num_action_scale, int(num_action_scale), 1)
        out = []
        for action in actions:
            hidden = self._action_mlp_hidden_for(
                int(action),
                num_action_scale=int(num_action_scale),
                model_num_action_scale=model_num_action_scale,
                context=context,
            )
            logit = self.action_output_bias
            for hidden_idx, value in enumerate(hidden):
                if hidden_idx >= len(self.action_output_weights):
                    break
                logit += value * self.action_output_weights[hidden_idx]
            out.append(float(logit))
        return out

    def _action_mlp_hidden_for(
        self,
        action: int,
        *,
        num_action_scale: int,
        model_num_action_scale: int,
        context: dict[str, Any],
    ) -> list[float]:
        features = action_features(
            context,
            int(action),
            action_num_action_scale=int(num_action_scale),
            model_num_action_scale=int(model_num_action_scale),
            categories=self.categories,
        )
        _, hidden = _mlp_hidden(features, self.action_input_weights, self.action_hidden_bias)
        return hidden

    def _action_mlp_logits_values_for_cpp(
        self,
        actions: list[int],
        *,
        num_action_scale: int,
        context: dict[str, Any],
    ) -> tuple[list[float], list[float]] | None:
        if self.inference_device.lower() not in {"", "json", "cpp", "native", "c++"}:
            return None
        if (
            not self.action_input_weights
            or not self.action_hidden_bias
            or not self.action_output_weights
        ):
            return None
        try:
            from . import cpp as smart_cpp
        except Exception:
            return None
        if not smart_cpp.using_cpp():
            return None
        try:
            if self._cpp_action_policy is None:
                policy_cls = getattr(smart_cpp, "ActionMlpPolicy", None)
                if policy_cls is None:
                    return smart_cpp.native_action_mlp_logits_values(
                        actions,
                        action_num_action_scale=int(num_action_scale),
                        model_num_action_scale=max(self.model_num_action_scale, int(num_action_scale), 1),
                        context=context,
                        categories=self.categories,
                        action_input_weights=self.action_input_weights,
                        action_hidden_bias=self.action_hidden_bias,
                        action_output_weights=self.action_output_weights,
                        action_output_bias=self.action_output_bias,
                        action_value_output_weights=self.action_value_output_weights,
                        action_value_output_bias=self.action_value_output_bias,
                    )
                self._cpp_action_policy = policy_cls(
                    self.categories,
                    self.action_input_weights,
                    self.action_hidden_bias,
                    self.action_output_weights,
                    self.action_output_bias,
                    self.action_value_output_weights,
                    self.action_value_output_bias,
                )
            logits, values = self._cpp_action_policy.logits_values(
                actions,
                int(num_action_scale),
                max(self.model_num_action_scale, int(num_action_scale), 1),
                context,
            )
            return [float(value) for value in logits], [float(value) for value in values]
        except Exception:
            return None

    def _init_torch_inference(self, device: str) -> None:
        if self.policy_type not in {"action_mlp_prior", "action_policy_value_prior"}:
            return
        if not self.action_input_weights or not self.action_hidden_bias or not self.action_output_weights:
            return
        torch = _import_torch()
        torch_device = _select_torch_device(torch, device)
        self._torch = torch
        self._torch_device = torch_device
        self._torch_action_input_weights = torch.tensor(
            self.action_input_weights,
            dtype=torch.float32,
            device=torch_device,
        )
        self._torch_action_hidden_bias = torch.tensor(
            self.action_hidden_bias,
            dtype=torch.float32,
            device=torch_device,
        )
        self._torch_action_output_weights = torch.tensor(
            self.action_output_weights,
            dtype=torch.float32,
            device=torch_device,
        )
        self._torch_action_output_bias = torch.tensor(
            self.action_output_bias,
            dtype=torch.float32,
            device=torch_device,
        )
        if self.action_value_output_weights:
            self._torch_action_value_output_weights = torch.tensor(
                self.action_value_output_weights,
                dtype=torch.float32,
                device=torch_device,
            )
            self._torch_action_value_output_bias = torch.tensor(
                self.action_value_output_bias,
                dtype=torch.float32,
                device=torch_device,
            )

    def _batched_action_features(
        self,
        actions: list[int],
        *,
        num_action_scale: int,
        context: dict[str, Any],
    ) -> list[list[float]]:
        model_num_action_scale = max(self.model_num_action_scale, int(num_action_scale), 1)
        expected_features = len(self.action_input_weights)
        features = [
            action_features(
                context,
                int(action),
                action_num_action_scale=int(num_action_scale),
                model_num_action_scale=model_num_action_scale,
                categories=self.categories,
            )
            for action in actions
        ]
        if expected_features <= 0:
            return features
        fixed = []
        for row in features:
            if len(row) < expected_features:
                fixed.append(row + [0.0] * (expected_features - len(row)))
            else:
                fixed.append(row[:expected_features])
        return fixed

    def _torch_hidden_for_actions(
        self,
        actions: list[int],
        *,
        num_action_scale: int,
        context: dict[str, Any],
    ):
        if (
            self._torch is None
            or self._torch_action_input_weights is None
            or self._torch_action_hidden_bias is None
        ):
            return None
        if not actions:
            return None
        features = self._batched_action_features(
            actions,
            num_action_scale=int(num_action_scale),
            context=context,
        )
        if not features:
            return None
        with self._torch.no_grad():
            x = self._torch.tensor(features, dtype=self._torch.float32, device=self._torch_device)
            return self._torch.tanh(
                x.matmul(self._torch_action_input_weights) + self._torch_action_hidden_bias
            )

    def _action_mlp_logits_for_torch(
        self,
        actions: list[int],
        *,
        num_action_scale: int,
        context: dict[str, Any],
    ) -> list[float] | None:
        if self._torch_action_output_weights is None or self._torch_action_output_bias is None:
            return None
        hidden = self._torch_hidden_for_actions(
            actions,
            num_action_scale=int(num_action_scale),
            context=context,
        )
        if hidden is None:
            return None
        with self._torch.no_grad():
            logits = hidden.matmul(self._torch_action_output_weights) + self._torch_action_output_bias
            return [float(value) for value in logits.detach().cpu().tolist()]

    def _action_values_for_torch(
        self,
        actions: list[int],
        *,
        num_action_scale: int,
        context: dict[str, Any],
    ) -> list[float] | None:
        if self._torch_action_value_output_weights is None or self._torch_action_value_output_bias is None:
            return None
        hidden = self._torch_hidden_for_actions(
            actions,
            num_action_scale=int(num_action_scale),
            context=context,
        )
        if hidden is None:
            return None
        with self._torch.no_grad():
            values = hidden.matmul(self._torch_action_value_output_weights) + self._torch_action_value_output_bias
            return [float(value) for value in values.detach().cpu().tolist()]


def linear_feature_names(categories: list[str]) -> list[str]:
    return [
        "bias_context",
        "bvs",
        "bvs_minus_one",
        "abs_bvs_minus_one",
        "step_fraction",
        "action_unit",
        "num_bbox_scaled",
        "cover_penalty_scaled",
        "pen_rate",
        "mcts_not_updated_fraction",
        "mcts_best_reward",
        "mcts_escape_active",
    ] + [f"category={category}" for category in categories]


def linear_features(record: dict[str, Any], categories: list[str]) -> list[float]:
    bvs = float(record.get("bvs", 1.0) or 1.0)
    step = float(record.get("step", 0.0) or 0.0)
    max_step = max(float(record.get("max_step", 0.0) or 0.0), 1.0)
    step_fraction = step / max_step if "max_step" in record else step / 150.0
    num_bbox = float(record.get("num_bbox", 0.0) or 0.0)
    action_unit = float(record.get("action_unit", 0.0) or 0.0)
    cover_penalty = float(record.get("cover_penalty", 100.0) or 100.0)
    pen_rate = float(record.get("pen_rate", 1.0) or 1.0)
    mcts_not_updated = float(
        record.get("mcts_not_updated", record.get("not_updated", 0.0)) or 0.0
    )
    mcts_best_reward = float(
        record.get("mcts_best_reward", record.get("best_reward", 0.0)) or 0.0
    )
    mcts_escape_active = 1.0 if bool(
        record.get("mcts_escape_active", record.get("escape_active", False))
    ) else 0.0
    category = str(record.get("category", ""))
    features = [
        1.0,
        bvs,
        bvs - 1.0,
        abs(bvs - 1.0),
        step_fraction,
        action_unit,
        num_bbox / 32.0,
        cover_penalty / 100.0,
        pen_rate,
        mcts_not_updated / max(max_step, 1.0),
        mcts_best_reward,
        mcts_escape_active,
    ]
    features.extend(1.0 if category == item else 0.0 for item in categories)
    return features


def action_feature_names(categories: list[str], num_action_scale: int) -> list[str]:
    names = list(linear_feature_names(categories))
    names.extend(
        [
            "bbox_idx_norm",
            "bbox_centered",
            "bbox_reverse_norm",
            "local_action_norm",
            "is_axis_action",
            "is_recenter_action",
            "is_lower_coord",
            "is_upper_coord",
            "signed_scale_norm",
            "abs_scale_norm",
            "scale_idx_norm",
            "is_min_bound",
            "is_max_bound",
            "bbox_valid_feature",
            "bbox_dim_x",
            "bbox_dim_y",
            "bbox_dim_z",
            "bbox_volume_ratio",
            "bbox_log_volume_ratio",
            "bbox_center_x",
            "bbox_center_y",
            "bbox_center_z",
            "bbox_extent_ratio",
            "action_delta",
            "action_delta_abs",
            "action_shrinks_box",
            "action_expands_box",
            "action_new_volume_ratio",
            "action_volume_delta_ratio",
            "action_invalid_after",
        ]
    )
    names.extend([f"axis={axis}" for axis in ("x", "y", "z")])
    names.extend([f"coord={coord_idx}" for coord_idx in range(7)])
    names.extend([f"scale_idx={scale_idx}" for scale_idx in range(max(int(num_action_scale), 1))])
    return names


def action_features(
    record: dict[str, Any],
    action: int,
    *,
    action_num_action_scale: int,
    model_num_action_scale: int,
    categories: list[str],
) -> list[float]:
    action_num_action_scale = max(int(action_num_action_scale), 1)
    model_num_action_scale = max(int(model_num_action_scale), 1)
    per_bbox = 6 * action_num_action_scale + 1
    action = int(action)
    bbox_idx = max(action // per_bbox, 0)
    local = action % per_bbox
    if local == per_bbox - 1:
        coord_idx = 6
        scale_idx = 0
    else:
        coord_idx = local // action_num_action_scale
        scale_idx = local % action_num_action_scale
    num_bbox = max(int(record.get("num_bbox", 0) or 0), bbox_idx + 1, 1)
    signed_scale = _action_scale_value(scale_idx, action_num_action_scale) if coord_idx < 6 else 0.0
    max_scale = max(abs(_action_scale_value(0, action_num_action_scale)), 1.0)
    if action_num_action_scale > 1:
        scale_idx_norm = scale_idx / float(action_num_action_scale - 1)
    else:
        scale_idx_norm = 0.0
    features = list(linear_features(record, categories))
    features.extend(
        [
            bbox_idx / float(num_bbox),
            (bbox_idx - (num_bbox - 1) * 0.5) / max(float(num_bbox), 1.0),
            (num_bbox - 1 - bbox_idx) / float(num_bbox),
            local / float(max(per_bbox - 1, 1)),
            1.0 if coord_idx < 6 else 0.0,
            1.0 if coord_idx >= 6 else 0.0,
            1.0 if coord_idx < 6 and coord_idx % 2 == 0 else 0.0,
            1.0 if coord_idx < 6 and coord_idx % 2 == 1 else 0.0,
            signed_scale / max_scale,
            abs(signed_scale) / max_scale,
            scale_idx_norm,
        ]
    )
    features.extend(
        _bbox_action_geometry_features(
            record,
            bbox_idx=bbox_idx,
            coord_idx=coord_idx,
            signed_delta=signed_scale * float(record.get("action_unit", 0.0) or 0.0),
        )
    )
    axis_idx = coord_idx // 2 if coord_idx < 6 else -1
    features.extend(1.0 if axis_idx == idx else 0.0 for idx in range(3))
    features.extend(1.0 if coord_idx == idx else 0.0 for idx in range(7))
    features.extend(1.0 if scale_idx == idx and coord_idx < 6 else 0.0 for idx in range(model_num_action_scale))
    return features


def _bbox_action_geometry_features(
    record: dict[str, Any],
    *,
    bbox_idx: int,
    coord_idx: int,
    signed_delta: float,
) -> list[float]:
    bounds = _record_bbox_bounds(record, bbox_idx)
    volume_sum = max(float(record.get("volume_sum", 0.0) or 0.0), 1.0e-12)
    if bounds is None:
        dims = [
            float(record.get("bbox_dim_x", 0.0) or 0.0),
            float(record.get("bbox_dim_y", 0.0) or 0.0),
            float(record.get("bbox_dim_z", 0.0) or 0.0),
        ]
        center = [
            float(record.get("bbox_center_x", 0.0) or 0.0),
            float(record.get("bbox_center_y", 0.0) or 0.0),
            float(record.get("bbox_center_z", 0.0) or 0.0),
        ]
        bbox_volume = float(record.get("bbox_volume", 0.0) or 0.0)
        valid = 1.0 if bbox_volume > 0.0 and all(dim > 0.0 for dim in dims) else 0.0
        new_volume = float(record.get("action_new_bbox_volume", bbox_volume) or 0.0)
        invalid_after = 1.0 if bool(record.get("action_invalid_after", False)) else 0.0
    else:
        dims = [
            max(0.0, bounds[3] - bounds[0]),
            max(0.0, bounds[4] - bounds[1]),
            max(0.0, bounds[5] - bounds[2]),
        ]
        center = [
            0.5 * (bounds[0] + bounds[3]),
            0.5 * (bounds[1] + bounds[4]),
            0.5 * (bounds[2] + bounds[5]),
        ]
        bbox_volume = dims[0] * dims[1] * dims[2]
        valid = 1.0 if bbox_volume > 0.0 else 0.0
        candidate = list(bounds)
        if 0 <= coord_idx < 6:
            candidate[coord_idx] += float(signed_delta)
        new_dims = [
            candidate[3] - candidate[0],
            candidate[4] - candidate[1],
            candidate[5] - candidate[2],
        ]
        invalid_after = 1.0 if any(dim <= 0.0 for dim in new_dims) else 0.0
        new_volume = 0.0 if invalid_after else float(new_dims[0] * new_dims[1] * new_dims[2])
    min_dim = min(dims) if dims else 0.0
    max_dim = max(dims) if dims else 0.0
    extent_ratio = min_dim / max(max_dim, 1.0e-12)
    bbox_volume_ratio = bbox_volume / volume_sum
    new_volume_ratio = new_volume / volume_sum
    action_delta = float(signed_delta) if 0 <= coord_idx < 6 else 0.0
    shrinks = 0.0
    expands = 0.0
    if 0 <= coord_idx < 3:
        shrinks = 1.0 if action_delta > 0.0 else 0.0
        expands = 1.0 if action_delta < 0.0 else 0.0
    elif 3 <= coord_idx < 6:
        shrinks = 1.0 if action_delta < 0.0 else 0.0
        expands = 1.0 if action_delta > 0.0 else 0.0
    return [
        1.0 if 0 <= coord_idx < 3 else 0.0,
        1.0 if 3 <= coord_idx < 6 else 0.0,
        valid,
        float(dims[0] if len(dims) > 0 else 0.0),
        float(dims[1] if len(dims) > 1 else 0.0),
        float(dims[2] if len(dims) > 2 else 0.0),
        float(bbox_volume_ratio),
        float(math.log1p(max(bbox_volume_ratio, 0.0))),
        float(center[0] if len(center) > 0 else 0.0),
        float(center[1] if len(center) > 1 else 0.0),
        float(center[2] if len(center) > 2 else 0.0),
        float(extent_ratio),
        float(action_delta),
        float(abs(action_delta)),
        shrinks,
        expands,
        float(new_volume_ratio),
        float(new_volume_ratio - bbox_volume_ratio),
        invalid_after,
    ]


def _record_bbox_bounds(record: dict[str, Any], bbox_idx: int) -> list[float] | None:
    if all(key in record for key in ("bbox_min_x", "bbox_min_y", "bbox_min_z", "bbox_max_x", "bbox_max_y", "bbox_max_z")):
        return [
            float(record.get("bbox_min_x", 0.0) or 0.0),
            float(record.get("bbox_min_y", 0.0) or 0.0),
            float(record.get("bbox_min_z", 0.0) or 0.0),
            float(record.get("bbox_max_x", 0.0) or 0.0),
            float(record.get("bbox_max_y", 0.0) or 0.0),
            float(record.get("bbox_max_z", 0.0) or 0.0),
        ]
    bounds = record.get("bbox_bounds")
    if isinstance(bounds, list) and 0 <= int(bbox_idx) < len(bounds):
        row = bounds[int(bbox_idx)]
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            return [float(value) for value in row[:6]]
    return None


def _action_scale_value(scale_idx: int, num_action_scale: int) -> float:
    num_action_scale = max(int(num_action_scale), 1)
    half = max(num_action_scale // 2, 1)
    if scale_idx < half:
        return float(-(2 ** (half - 1 - scale_idx)))
    return float(2 ** (scale_idx - half))


def _record_coord_scale_key(record: dict[str, Any]) -> str:
    coord_idx = int(record.get("coord_idx", 6))
    scale_idx = int(record.get("scale_idx", 0))
    if coord_idx >= 6:
        return "6:0"
    return f"{coord_idx}:{scale_idx}"


def _candidate_group_key(record: dict[str, Any]) -> str:
    record_type = str(record.get("record_type", ""))
    original_record_type = str(record.get("original_record_type", ""))
    if str(record.get("record_type", "")) == "local_refine_branch_final_return":
        branch_group = str(record.get("branch_group", ""))
        if branch_group:
            return branch_group
        return "|".join(
            [
                str(record.get("category", "")),
                str(record.get("mesh", "")),
                "local_refine_branch",
                str(record.get("source", "")),
            ]
        )
    if record_type == "local_refine_candidate" or (
        record_type == "local_refine_final_return"
        and original_record_type == "local_refine_candidate"
    ):
        return "|".join(
            [
                str(record.get("category", "")),
                str(record.get("mesh", "")),
                "local_refine",
                str(record.get("step", "")),
                str(record.get("source", "")),
            ]
        )
    return "|".join(
        [
            str(record.get("category", "")),
            str(record.get("mesh", "")),
            str(record.get("mcts_iter", "")),
            str(record.get("rollout_step", "")),
            str(record.get("node_id", "")),
        ]
    )


def _is_action_candidate_record(record: dict[str, Any]) -> bool:
    record_type = str(record.get("record_type", ""))
    original_record_type = str(record.get("original_record_type", ""))
    if record_type in {
        "mcts_candidate",
        "local_refine_candidate",
        "local_refine_branch_final_return",
    }:
        return True
    return (
        record_type in {"mcts_final_return", "local_refine_final_return"}
        and original_record_type in {"mcts_candidate", "local_refine_candidate"}
    )


def _action_value_examples(
    trace_paths: list[Path],
    *,
    min_reward: float,
    num_action_scale: int,
    categories: list[str],
    advantage_baseline: str,
    advantage_clip: float,
    accepted_weight: float,
    candidate_weight: float,
    selected_candidate_weight: float,
    category_balance: bool,
    value_positive_weight: float = 1.0,
    value_negative_weight: float = 1.0,
    value_zero_weight: float = 1.0,
    value_auto_balance: bool = False,
    value_group_balance: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        with trace_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                reward = float(record.get("reward", 0.0))
                if reward < min_reward:
                    continue
                records.append(record)

    rewards_by_key: dict[str, list[float]] = defaultdict(list)
    global_rewards: list[float] = []
    for record in records:
        reward = float(record.get("reward", 0.0))
        global_rewards.append(reward)
        if advantage_baseline == "mesh":
            key = str(record.get("mesh", ""))
        elif advantage_baseline == "none":
            key = "__zero__"
        elif advantage_baseline == "global":
            key = "__global__"
        else:
            key = str(record.get("category", ""))
        rewards_by_key[key].append(reward)
    global_baseline = sum(global_rewards) / max(len(global_rewards), 1)
    baseline_map = {
        key: (0.0 if advantage_baseline == "none" else sum(values) / len(values))
        for key, values in rewards_by_key.items()
    }

    candidate_rewards_by_group: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if _is_action_candidate_record(record):
            candidate_rewards_by_group[_candidate_group_key(record)].append(float(record.get("reward", 0.0)))
    candidate_baseline_by_group = {
        key: sum(values) / len(values)
        for key, values in candidate_rewards_by_group.items()
        if len(values) > 1
    }

    features: list[list[float]] = []
    raw_targets: list[float] = []
    weights: list[float] = []
    target_signs: list[str] = []
    example_categories: list[str] = []
    mesh_ids: list[str] = []
    group_keys: list[str] = []
    candidate_records_used = 0
    positive_examples = 0
    negative_examples = 0
    zero_examples = 0
    for record in records:
        action = int(record.get("action", -1))
        if action < 0:
            continue
        action_num_action_scale = max(int(record.get("num_action_scale", 0) or num_action_scale), 1)
        per_bbox = 6 * action_num_action_scale + 1
        selected_bbox = action // per_bbox
        num_bbox = max(int(record.get("num_bbox", 0) or 0), selected_bbox + 1)
        if action >= num_bbox * per_bbox:
            continue
        if advantage_baseline == "mesh":
            baseline_key = str(record.get("mesh", ""))
        elif advantage_baseline == "none":
            baseline_key = "__zero__"
        elif advantage_baseline == "global":
            baseline_key = "__global__"
        else:
            baseline_key = str(record.get("category", ""))
        reward = float(record.get("reward", 0.0))
        if _is_action_candidate_record(record):
            target = reward - candidate_baseline_by_group.get(_candidate_group_key(record), global_baseline)
            weight = float(candidate_weight)
            if bool(record.get("selected", False)):
                weight *= float(selected_candidate_weight)
            candidate_records_used += 1
        else:
            target = reward - baseline_map.get(baseline_key, global_baseline)
            weight = float(accepted_weight)
        if target > 1.0e-12:
            weight *= float(value_positive_weight)
            positive_examples += 1
            target_sign = "positive"
        elif target < -1.0e-12:
            weight *= float(value_negative_weight)
            negative_examples += 1
            target_sign = "negative"
        else:
            weight *= float(value_zero_weight)
            zero_examples += 1
            target_sign = "zero"
        feature_record = dict(record)
        feature_record["num_bbox"] = num_bbox
        features.append(
            action_features(
                feature_record,
                action,
                action_num_action_scale=action_num_action_scale,
                model_num_action_scale=num_action_scale,
                categories=categories,
            )
        )
        raw_targets.append(target)
        weights.append(max(weight, 0.0))
        target_signs.append(target_sign)
        example_categories.append(str(record.get("category", "")))
        mesh_ids.append(str(record.get("mesh", "")))
        group_keys.append(_value_pairwise_group_key(record))

    if not raw_targets:
        return {
            "features": [],
            "targets": [],
            "weights": [],
            "records_used": 0,
            "candidate_records_used": 0,
            "positive_examples": 0,
            "negative_examples": 0,
            "zero_examples": 0,
            "target_mean": 0.0,
            "target_std": 1.0,
            "mesh_ids": [],
            "group_keys": [],
            "categories_used": [],
            "class_multipliers": {"positive": 1.0, "negative": 1.0, "zero": 1.0},
            "group_balance_groups": 0,
        }

    target_mean = sum(raw_targets) / len(raw_targets)
    variance = sum((value - target_mean) ** 2 for value in raw_targets) / max(len(raw_targets), 1)
    target_std = max(math.sqrt(variance), 1.0e-12)
    clipped_targets = [
        max(-float(advantage_clip), min(float(advantage_clip), (value - target_mean) / target_std))
        for value in raw_targets
    ]
    class_counts = {
        "positive": positive_examples,
        "negative": negative_examples,
        "zero": zero_examples,
    }
    class_multipliers = {"positive": 1.0, "negative": 1.0, "zero": 1.0}
    if value_auto_balance and weights:
        active_counts = {name: count for name, count in class_counts.items() if count > 0}
        active_total = sum(active_counts.values())
        active_class_count = max(len(active_counts), 1)
        for name, count in active_counts.items():
            class_multipliers[name] = active_total / max(active_class_count * count, 1)
        weights = [
            weight * class_multipliers.get(sign, 1.0)
            for weight, sign in zip(weights, target_signs)
        ]
    group_balance_groups = 0
    if value_group_balance and weights:
        group_weight_sums: dict[str, float] = defaultdict(float)
        for group_key, weight in zip(group_keys, weights):
            group_weight_sums[str(group_key)] += float(weight)
        nonzero_groups = {
            group_key: total
            for group_key, total in group_weight_sums.items()
            if total > 0.0
        }
        group_balance_groups = len(nonzero_groups)
        if group_balance_groups:
            target_group_weight = sum(nonzero_groups.values()) / group_balance_groups
            weights = [
                (
                    weight * (target_group_weight / nonzero_groups[str(group_key)])
                    if nonzero_groups.get(str(group_key), 0.0) > 0.0
                    else weight
                )
                for weight, group_key in zip(weights, group_keys)
            ]
    if category_balance and weights:
        category_counts = Counter(example_categories)
        category_count = max(len(category_counts), 1)
        total_examples = max(len(weights), 1)
        weights = [
            weight * total_examples / max(category_count * category_counts.get(category, 1), 1)
            for weight, category in zip(weights, example_categories)
        ]
    mean_weight = sum(weights) / max(len(weights), 1)
    if mean_weight > 0.0:
        weights = [weight / mean_weight for weight in weights]
    return {
        "features": features,
        "targets": clipped_targets,
        "weights": weights,
        "records_used": len(features),
        "candidate_records_used": candidate_records_used,
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "zero_examples": zero_examples,
        "target_mean": target_mean,
        "target_std": target_std,
        "mesh_ids": mesh_ids,
        "group_keys": group_keys,
        "categories_used": sorted({category for category in example_categories if category}),
        "class_multipliers": class_multipliers,
        "group_balance_groups": group_balance_groups,
    }


def _value_train_validation_indices(
    mesh_ids: list[str],
    *,
    validation_fraction: float,
) -> tuple[list[int], list[int]]:
    total = len(mesh_ids)
    if total == 0:
        return [], []
    meshes = sorted({mesh for mesh in mesh_ids if mesh})
    if validation_fraction <= 0.0 or len(meshes) < 2:
        return list(range(total)), []
    validation_count = int(round(len(meshes) * min(max(float(validation_fraction), 0.0), 0.9)))
    validation_count = min(max(validation_count, 1), len(meshes) - 1)
    validation_meshes = set(meshes[-validation_count:])
    train = [idx for idx, mesh in enumerate(mesh_ids) if mesh not in validation_meshes]
    validation = [idx for idx, mesh in enumerate(mesh_ids) if mesh in validation_meshes]
    if not train or not validation:
        return list(range(total)), []
    return train, validation


def _value_pairwise_group_key(record: dict[str, Any]) -> str:
    record_type = str(record.get("record_type", ""))
    original_record_type = str(record.get("original_record_type", ""))
    if record_type == "local_refine_branch_final_return":
        branch_group = str(record.get("branch_group", ""))
        if branch_group:
            return branch_group
        return "|".join(
            [
                str(record.get("category", "")),
                str(record.get("mesh", "")),
                "local_refine_branch",
                str(record.get("source", "")),
            ]
        )
    if record_type == "local_refine_candidate" or (
        record_type == "local_refine_final_return"
        and original_record_type == "local_refine_candidate"
    ):
        return "|".join(
            [
                str(record.get("category", "")),
                str(record.get("mesh", "")),
                "local_refine",
                str(record.get("step", "")),
                str(record.get("source", "")),
            ]
        )
    if record_type == "mcts_candidate" or (
        record_type == "mcts_final_return"
        and original_record_type == "mcts_candidate"
    ):
        return "|".join(
            [
                str(record.get("category", "")),
                str(record.get("mesh", "")),
                str(record.get("mcts_iter", "")),
                str(record.get("rollout_step", "")),
                str(record.get("node_id", "")),
            ]
        )
    return "|".join(
        [
            str(record.get("category", "")),
            str(record.get("mesh", "")),
        ]
    )


def _value_pairwise_pairs(
    targets: list[float],
    group_keys: list[str],
    indices: list[int],
    *,
    max_pairs: int,
) -> list[tuple[int, int, float]]:
    if max_pairs <= 0 or not indices:
        return []
    old_to_local = {int(old_idx): local_idx for local_idx, old_idx in enumerate(indices)}
    by_group: dict[str, list[int]] = defaultdict(list)
    for old_idx in indices:
        if old_idx >= len(group_keys) or old_idx >= len(targets):
            continue
        by_group[str(group_keys[old_idx])].append(old_idx)
    pairs: list[tuple[int, int, float]] = []
    for _, group_indices in sorted(by_group.items()):
        positive = [
            idx
            for idx in group_indices
            if float(targets[idx]) > 1.0e-12
        ]
        non_positive = [
            idx
            for idx in group_indices
            if float(targets[idx]) <= 1.0e-12
        ]
        if not positive or not non_positive:
            continue
        positive.sort(key=lambda idx: float(targets[idx]), reverse=True)
        non_positive.sort(key=lambda idx: float(targets[idx]))
        for high in positive:
            high_target = float(targets[high])
            for low in non_positive:
                diff = high_target - float(targets[low])
                if diff <= 1.0e-12:
                    continue
                pairs.append((old_to_local[high], old_to_local[low], diff))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _value_prediction_metrics(
    values: Any,
    targets: Any,
    weights: Any,
    indices: list[int],
    *,
    torch: Any,
) -> dict[str, float | int | None]:
    if not indices:
        return {
            "examples": 0,
            "weighted_mse": None,
            "weighted_mae": None,
            "sign_accuracy": None,
        }
    index_tensor = torch.tensor(indices, dtype=torch.long, device=values.device)
    selected_values = values.index_select(0, index_tensor)
    selected_targets = targets.index_select(0, index_tensor)
    selected_weights = weights.index_select(0, index_tensor)
    weight_sum = torch.clamp(selected_weights.sum(), min=1.0e-12)
    errors = selected_values - selected_targets
    weighted_mse = (errors.square() * selected_weights).sum() / weight_sum
    weighted_mae = (errors.abs() * selected_weights).sum() / weight_sum
    non_zero = selected_targets.abs() > 1.0e-12
    if bool(non_zero.any().detach().cpu().item()):
        sign_accuracy = (
            (torch.sign(selected_values[non_zero]) == torch.sign(selected_targets[non_zero]))
            .to(dtype=torch.float32)
            .mean()
        )
        sign_accuracy_value: float | None = float(sign_accuracy.detach().cpu().item())
    else:
        sign_accuracy_value = None
    return {
        "examples": int(len(indices)),
        "weighted_mse": float(weighted_mse.detach().cpu().item()),
        "weighted_mae": float(weighted_mae.detach().cpu().item()),
        "sign_accuracy": sign_accuracy_value,
    }


def _fixed_feature_width(features: list[list[float]], expected_dim: int) -> list[list[float]]:
    expected_dim = max(int(expected_dim), 0)
    if expected_dim == 0:
        return features
    out = []
    for row in features:
        if len(row) < expected_dim:
            out.append(list(row) + [0.0] * (expected_dim - len(row)))
        else:
            out.append(list(row[:expected_dim]))
    return out


def _linear_logits(features: list[float], weights: list[list[float]], bias: list[float]) -> list[float]:
    logits = list(bias)
    for feat_idx, value in enumerate(features):
        if feat_idx >= len(weights):
            break
        row = weights[feat_idx]
        for class_idx in range(min(len(logits), len(row))):
            logits[class_idx] += value * row[class_idx]
    return logits


def _mlp_hidden(
    features: list[float],
    input_weights: list[list[float]],
    hidden_bias: list[float],
) -> tuple[list[float], list[float]]:
    hidden_pre = list(hidden_bias)
    for feat_idx, value in enumerate(features):
        if feat_idx >= len(input_weights):
            break
        row = input_weights[feat_idx]
        for hidden_idx in range(min(len(hidden_pre), len(row))):
            hidden_pre[hidden_idx] += value * row[hidden_idx]
    hidden = [math.tanh(value) for value in hidden_pre]
    return hidden_pre, hidden


def _mlp_logits_from_hidden(
    hidden: list[float],
    output_weights: list[list[float]],
    output_bias: list[float],
) -> list[float]:
    logits = list(output_bias)
    for hidden_idx, value in enumerate(hidden):
        if hidden_idx >= len(output_weights):
            break
        row = output_weights[hidden_idx]
        for class_idx in range(min(len(logits), len(row))):
            logits[class_idx] += value * row[class_idx]
    return logits


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for --model-type mlp. Install SMART with the "
            "pipeline extra or install torch, then retry."
        ) from exc
    return torch


def _select_torch_device(torch: Any, device: str):
    requested = str(device or "auto").lower()
    if requested == "auto":
        if _torch_device_works(torch, "mps"):
            requested = "mps"
        elif _torch_device_works(torch, "cuda"):
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "mps":
        if not _torch_device_works(torch, "mps"):
            raise RuntimeError("PyTorch MPS device was requested but is not available")
    elif requested == "cuda":
        if not _torch_device_works(torch, "cuda"):
            raise RuntimeError("PyTorch CUDA device was requested but is not available")
    elif requested != "cpu":
        raise ValueError(f"unsupported torch device: {device!r}")
    return torch.device(requested)


def _torch_device_works(torch: Any, device: str) -> bool:
    try:
        if device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_built()):
                return False
        elif device == "cuda":
            if not torch.cuda.is_available():
                return False
        else:
            return device == "cpu"
        tensor = torch.tensor([0.0], device=device)
        _ = tensor + 1.0
        return True
    except Exception:
        return False


def _softmax(logits: list[float]) -> list[float]:
    offset = max(logits)
    exps = [math.exp(value - offset) for value in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exps]
