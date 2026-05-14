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


def coord_scale_keys(num_action_scale: int) -> list[str]:
    """Return SMART coord/scale prior keys for the legacy action order."""

    if num_action_scale < 1:
        raise ValueError("num_action_scale must be positive")
    return [f"{coord}:{scale}" for coord in range(6) for scale in range(num_action_scale)] + ["6:0"]


def load_action_prior(path: str | Path) -> "LoadedActionPrior":
    with Path(path).open("r", encoding="utf-8") as file:
        return LoadedActionPrior(json.load(file))


class LoadedActionPrior:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.policy_type = str(payload.get("policy_type") or payload.get("type") or "coord_scale_count_prior")
        self.default_logit = float(payload.get("default_logit", 0.0))
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

    def action_logits_for(
        self,
        actions: Iterable[int],
        *,
        num_action_scale: int,
        context: dict[str, Any] | None = None,
    ) -> list[float]:
        class_logits = self._class_logits(context or {})
        out = []
        per_bbox = 6 * int(num_action_scale) + 1
        for action in actions:
            action = int(action)
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
    ]
    features.extend(1.0 if category == item else 0.0 for item in categories)
    return features


def _record_coord_scale_key(record: dict[str, Any]) -> str:
    coord_idx = int(record.get("coord_idx", 6))
    scale_idx = int(record.get("scale_idx", 0))
    if coord_idx >= 6:
        return "6:0"
    return f"{coord_idx}:{scale_idx}"


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
