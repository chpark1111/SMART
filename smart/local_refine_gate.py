from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GATE_INPUT_METRICS = (
    "Avg_num_box",
    "Avg_BVS",
    "Avg_MOV",
    "Avg_TOV",
    "Avg_Covered",
    "Avg_vIoU",
    "Avg_cub_CD",
)

GATE_DERIVED_FEATURES = (
    "input_uncovered",
    "input_log_num_box",
    "input_bvs_per_box",
    "input_mov_per_box",
    "input_tov_per_box",
    "input_mov_to_tov",
    "input_bvs_to_mov",
    "input_tov_to_mov",
    "input_viou_times_covered",
    "input_tightness_pressure",
)

GATE_RICH_FEATURES = (
    "input_low_viou",
    "input_quality_pressure",
    "input_quality_pressure_per_box",
    "input_box_density_pressure",
    "input_coverage_tightness_gap",
    "input_log_bvs",
    "input_log_mov",
    "input_log_tov",
    "input_log_quality_pressure",
    "input_bvs_x_uncovered",
    "input_mov_x_uncovered",
    "input_tov_x_uncovered",
    "input_low_viou_x_uncovered",
    "input_low_viou_x_tov",
    "input_bvs_x_num_box",
    "input_mov_x_num_box",
    "input_tov_x_num_box",
)

GATE_CATEGORY_INTERACTION_FEATURES = (
    "input_tightness_pressure",
    "input_quality_pressure",
    "input_uncovered",
    "input_low_viou",
    "input_log_num_box",
)

GATE_FEATURE_SETS = ("basic", "expanded", "rich")


@dataclass(frozen=True)
class GateDataset:
    rows: list[dict[str, Any]]
    categories: list[str]
    stages: list[str]
    feature_names: list[str]
    features: list[list[float]]
    labels: list[int]


def load_gate_dataset(path: str | Path, *, target: str = "label_improved", feature_set: str = "expanded") -> GateDataset:
    """Load rows exported by experiments/scripts/export_local_refine_gate_dataset.py."""

    rows = _load_rows(path)
    if not rows:
        raise ValueError(f"empty gate dataset: {path}")
    feature_set = _validate_feature_set(feature_set)
    categories = sorted({str(row.get("category") or "unknown") for row in rows})
    stages = sorted({str(row.get("input_stage") or "unknown") for row in rows})
    feature_names = gate_feature_names(categories, stages=stages, feature_set=feature_set)
    features = [gate_features(row, categories, feature_names=feature_names) for row in rows]
    labels = [_int_field(row, target) for row in rows]
    return GateDataset(rows=rows, categories=categories, stages=stages, feature_names=feature_names, features=features, labels=labels)


def train_local_refine_gate(
    dataset_path: str | Path,
    *,
    output: str | Path,
    target: str = "label_improved",
    hidden_size: int = 8,
    epochs: int = 200,
    learning_rate: float = 0.03,
    weight_decay: float = 1.0e-3,
    seed: int = 7,
    device: str = "auto",
    leave_one_out: bool = True,
    positive_weight: str = "balanced",
    feature_set: str = "expanded",
) -> dict[str, Any]:
    """Train a small PyTorch gate for deciding whether local_refine is worth running.

    The model sees only pre-local-refine information: category and input-stage
    SMART metrics. It never sees local_refine metrics or deltas during
    inference, so it can be used before paying the local search cost.
    """

    dataset = load_gate_dataset(dataset_path, target=target, feature_set=feature_set)
    torch = _import_torch()
    torch_device = _select_torch_device(torch, device)
    cv_predictions: list[float] | None = None
    cv_summary: dict[str, Any] | None = None
    if leave_one_out and len(dataset.labels) >= 3 and len(set(dataset.labels)) > 1:
        cv_predictions = _leave_one_out_predictions(
            torch,
            torch_device,
            dataset,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            positive_weight=positive_weight,
        )
        cv_summary = _prediction_summary(dataset.labels, cv_predictions)

    mean, std = _fit_scaler(dataset.features)
    final_model = _fit_torch_model(
        torch,
        torch_device,
        dataset.features,
        dataset.labels,
        mean,
        std,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        positive_weight=positive_weight,
    )
    train_predictions = _predict_torch_model(torch, torch_device, final_model, dataset.features, mean, std)
    train_summary = _prediction_summary(dataset.labels, train_predictions)
    payload = _model_payload(
        dataset,
        mean,
        std,
        final_model,
        target=target,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=str(torch_device),
        torch_version=str(torch.__version__),
        positive_weight=positive_weight,
        feature_set=feature_set,
        train_summary=train_summary,
        cv_summary=cv_summary,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_local_refine_gate(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("policy_type") != "local_refine_gate":
        raise ValueError(f"not a local_refine_gate model: {path}")
    return payload


def score_local_refine_gate(payload: dict[str, Any], row: dict[str, Any]) -> float:
    categories = list(payload["categories"])
    feature_names = payload.get("feature_names") or gate_feature_names(
        categories,
        feature_set=str(payload.get("feature_set") or payload.get("metadata", {}).get("feature_set") or "basic"),
    )
    features = gate_features(row, categories, feature_names=feature_names)
    mean = [float(value) for value in payload["feature_mean"]]
    std = [float(value) for value in payload["feature_std"]]
    x = [((value - mean[idx]) / std[idx]) if std[idx] else 0.0 for idx, value in enumerate(features)]
    weights = payload["weights"]
    if payload.get("hidden_size", 0) > 0:
        hidden = []
        for col in range(int(payload["hidden_size"])):
            value = float(weights["hidden_bias"][col])
            for row_idx, feature in enumerate(x):
                value += feature * float(weights["input"][row_idx][col])
            hidden.append(math.tanh(value))
        logit = float(weights["output_bias"])
        for value, weight in zip(hidden, weights["output"]):
            logit += value * float(weight)
    else:
        logit = float(weights["bias"])
        for value, weight in zip(x, weights["linear"]):
            logit += value * float(weight)
    return 1.0 / (1.0 + math.exp(-max(min(logit, 60.0), -60.0)))


def gate_feature_names(
    categories: Iterable[str],
    *,
    stages: Iterable[str] | None = None,
    feature_set: str = "expanded",
) -> list[str]:
    feature_set = _validate_feature_set(feature_set)
    names = [f"category={category}" for category in categories]
    names.extend(f"input_stage={stage}" for stage in stages or [])
    names.extend(f"input_{metric}" for metric in GATE_INPUT_METRICS)
    if feature_set in {"expanded", "rich"}:
        names.extend(GATE_DERIVED_FEATURES)
    if feature_set == "rich":
        names.extend(GATE_RICH_FEATURES)
        for category in categories:
            names.extend(
                f"category_interaction={category}:{feature}" for feature in GATE_CATEGORY_INTERACTION_FEATURES
            )
    return names


def gate_features(
    row: dict[str, Any],
    categories: Iterable[str],
    *,
    feature_names: Iterable[str] | None = None,
) -> list[float]:
    categories = list(categories)
    names = list(feature_names or gate_feature_names(categories, feature_set="expanded"))
    category = str(row.get("category") or "unknown")
    return [_feature_value(row, name, category=category, categories=categories) for name in names]


def _feature_value(row: dict[str, Any], name: str, *, category: str, categories: list[str]) -> float:
    if name.startswith("category="):
        return 1.0 if category == name.split("=", 1)[1] else 0.0
    if name.startswith("input_stage="):
        return 1.0 if str(row.get("input_stage") or "unknown") == name.split("=", 1)[1] else 0.0
    if name.startswith("category_interaction="):
        spec = name.split("=", 1)[1]
        expected_category, feature_name = spec.split(":", 1)
        if category != expected_category:
            return 0.0
        return _feature_value(row, feature_name, category=category, categories=categories)
    if name.startswith("input_Avg_"):
        return _float_field(row, name)
    metrics = {metric: _float_field(row, f"input_{metric}") for metric in GATE_INPUT_METRICS}
    num_box = max(metrics["Avg_num_box"], 1.0)
    bvs = metrics["Avg_BVS"]
    mov = metrics["Avg_MOV"]
    tov = metrics["Avg_TOV"]
    covered = metrics["Avg_Covered"]
    viou = metrics["Avg_vIoU"]
    cub_cd = metrics["Avg_cub_CD"]
    eps = 1.0e-9
    low_viou = max(0.0, 1.0 - viou)
    uncovered = max(0.0, 1.0 - covered)
    quality_pressure = max(0.0, bvs) + max(0.0, mov) + max(0.0, tov) + max(0.0, cub_cd) + low_viou
    if name == "input_uncovered":
        return uncovered
    if name == "input_log_num_box":
        return math.log1p(max(metrics["Avg_num_box"], 0.0))
    if name == "input_bvs_per_box":
        return bvs / num_box
    if name == "input_mov_per_box":
        return mov / num_box
    if name == "input_tov_per_box":
        return tov / num_box
    if name == "input_mov_to_tov":
        return mov / max(abs(tov), eps)
    if name == "input_bvs_to_mov":
        return bvs / max(abs(mov), eps)
    if name == "input_tov_to_mov":
        return tov / max(abs(mov), eps)
    if name == "input_viou_times_covered":
        return viou * covered
    if name == "input_tightness_pressure":
        return bvs + mov + tov - viou
    if name == "input_low_viou":
        return low_viou
    if name == "input_quality_pressure":
        return quality_pressure
    if name == "input_quality_pressure_per_box":
        return quality_pressure / num_box
    if name == "input_box_density_pressure":
        return math.log1p(max(metrics["Avg_num_box"], 0.0)) * quality_pressure
    if name == "input_coverage_tightness_gap":
        return uncovered + quality_pressure
    if name == "input_log_bvs":
        return math.log1p(max(bvs, 0.0))
    if name == "input_log_mov":
        return math.log1p(max(mov, 0.0))
    if name == "input_log_tov":
        return math.log1p(max(tov, 0.0))
    if name == "input_log_quality_pressure":
        return math.log1p(quality_pressure)
    if name == "input_bvs_x_uncovered":
        return bvs * uncovered
    if name == "input_mov_x_uncovered":
        return mov * uncovered
    if name == "input_tov_x_uncovered":
        return tov * uncovered
    if name == "input_low_viou_x_uncovered":
        return low_viou * uncovered
    if name == "input_low_viou_x_tov":
        return low_viou * tov
    if name == "input_bvs_x_num_box":
        return bvs * num_box
    if name == "input_mov_x_num_box":
        return mov * num_box
    if name == "input_tov_x_num_box":
        return tov * num_box
    return _float_field(row, name)


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _validate_feature_set(feature_set: str) -> str:
    feature_set = str(feature_set or "expanded")
    if feature_set not in GATE_FEATURE_SETS:
        raise ValueError(f"unsupported gate feature_set {feature_set!r}; expected one of {GATE_FEATURE_SETS}")
    return feature_set


def _fit_scaler(features: list[list[float]]) -> tuple[list[float], list[float]]:
    if not features:
        raise ValueError("cannot fit scaler on empty feature matrix")
    cols = len(features[0])
    mean = []
    std = []
    for col in range(cols):
        values = [row[col] for row in features]
        avg = sum(values) / len(values)
        var = sum((value - avg) ** 2 for value in values) / max(len(values) - 1, 1)
        scale = math.sqrt(var)
        if scale < 1.0e-12:
            scale = 1.0
        mean.append(avg)
        std.append(scale)
    return mean, std


def _fit_torch_model(
    torch: Any,
    device: Any,
    features: list[list[float]],
    labels: list[int],
    mean: list[float],
    std: list[float],
    *,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    positive_weight: str,
) -> dict[str, Any]:
    if len(features) != len(labels):
        raise ValueError("feature and label counts differ")
    torch.manual_seed(int(seed))
    x = _torch_tensor(torch, device, _normalize_features(features, mean, std))
    y = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)
    pos_weight_tensor = _pos_weight_tensor(torch, device, labels, positive_weight)
    input_dim = len(features[0])
    hidden_size = max(int(hidden_size), 0)
    if hidden_size > 0:
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
        ).to(device)
    else:
        model = torch.nn.Linear(input_dim, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    for _ in range(max(int(epochs), 0)):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
    return {"model": model, "hidden_size": hidden_size}


def _predict_torch_model(
    torch: Any,
    device: Any,
    model_bundle: dict[str, Any],
    features: list[list[float]],
    mean: list[float],
    std: list[float],
) -> list[float]:
    model = model_bundle["model"]
    x = _torch_tensor(torch, device, _normalize_features(features, mean, std))
    with torch.no_grad():
        logits = model(x).detach().cpu().view(-1).tolist()
    return [1.0 / (1.0 + math.exp(-max(min(float(logit), 60.0), -60.0))) for logit in logits]


def _leave_one_out_predictions(
    torch: Any,
    device: Any,
    dataset: GateDataset,
    *,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    positive_weight: str,
) -> list[float]:
    predictions = []
    for held_out in range(len(dataset.labels)):
        train_features = [row for idx, row in enumerate(dataset.features) if idx != held_out]
        train_labels = [label for idx, label in enumerate(dataset.labels) if idx != held_out]
        mean, std = _fit_scaler(train_features)
        model = _fit_torch_model(
            torch,
            device,
            train_features,
            train_labels,
            mean,
            std,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + held_out,
            positive_weight=positive_weight,
        )
        pred = _predict_torch_model(torch, device, model, [dataset.features[held_out]], mean, std)[0]
        predictions.append(pred)
    return predictions


def _prediction_summary(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    if len(labels) != len(probabilities):
        raise ValueError("label and probability counts differ")
    threshold_summary = _metrics_at_threshold(labels, probabilities, 0.5)
    best_f1 = threshold_summary
    for threshold in [idx / 100.0 for idx in range(1, 100)]:
        metrics = _metrics_at_threshold(labels, probabilities, threshold)
        if (metrics["f1"], metrics["accuracy"]) > (best_f1["f1"], best_f1["accuracy"]):
            best_f1 = metrics
    positive = sum(labels)
    majority_accuracy = max(positive, len(labels) - positive) / len(labels) if labels else 0.0
    return {
        "n": len(labels),
        "positive": positive,
        "positive_rate": positive / len(labels) if labels else 0.0,
        "majority_accuracy": majority_accuracy,
        "roc_auc": _roc_auc(labels, probabilities),
        "threshold_0_5": threshold_summary,
        "best_f1_threshold": best_f1,
    }


def _metrics_at_threshold(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, Any]:
    preds = [1 if prob >= threshold else 0 for prob in probabilities]
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _roc_auc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = [prob for label, prob in zip(labels, probabilities) if label == 1]
    negatives = [prob for label, prob in zip(labels, probabilities) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def _model_payload(
    dataset: GateDataset,
    mean: list[float],
    std: list[float],
    model_bundle: dict[str, Any],
    *,
    target: str,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    torch_version: str,
    positive_weight: str,
    feature_set: str,
    train_summary: dict[str, Any],
    cv_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    model = model_bundle["model"]
    hidden_size = int(model_bundle["hidden_size"])
    weights: dict[str, Any]
    if hidden_size > 0:
        first = model[0]
        second = model[2]
        weights = {
            "input": _transpose(first.weight.detach().cpu().tolist()),
            "hidden_bias": first.bias.detach().cpu().tolist(),
            "output": second.weight.detach().cpu().view(-1).tolist(),
            "output_bias": float(second.bias.detach().cpu().view(-1).tolist()[0]),
        }
    else:
        weights = {
            "linear": model.weight.detach().cpu().view(-1).tolist(),
            "bias": float(model.bias.detach().cpu().view(-1).tolist()[0]),
        }
    return {
        "schema_version": 2,
        "policy_type": "local_refine_gate",
        "feature_set": feature_set,
        "target": target,
        "categories": dataset.categories,
        "stages": dataset.stages,
        "feature_names": dataset.feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "hidden_size": hidden_size,
        "activation": "tanh" if hidden_size > 0 else "linear",
        "weights": weights,
        "metadata": {
            "source": "smart.local_refine_gate.train_local_refine_gate",
            "model_type": "pytorch_mlp_gate" if hidden_size > 0 else "pytorch_logistic_gate",
            "trainer_backend": "torch",
            "torch_version": torch_version,
            "device": device,
            "rows": len(dataset.rows),
            "positive": sum(dataset.labels),
            "epochs": int(epochs),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": int(seed),
            "positive_weight": positive_weight,
            "feature_set": feature_set,
            "train": train_summary,
            "leave_one_out": cv_summary,
        },
    }


def _normalize_features(features: list[list[float]], mean: list[float], std: list[float]) -> list[list[float]]:
    return [[(value - mean[idx]) / std[idx] for idx, value in enumerate(row)] for row in features]


def _torch_tensor(torch: Any, device: Any, values: list[list[float]]) -> Any:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _pos_weight_tensor(torch: Any, device: Any, labels: list[int], mode: str) -> Any:
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError("positive_weight must be 'balanced' or 'none'")
    positive = sum(labels)
    negative = len(labels) - positive
    if positive <= 0 or negative <= 0:
        return None
    return torch.tensor([negative / positive], dtype=torch.float32, device=device)


def _transpose(values: list[list[float]]) -> list[list[float]]:
    if not values:
        return []
    return [list(col) for col in zip(*values)]


def _float_field(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def _int_field(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if value in (None, ""):
        return 0
    return int(float(value))


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for local-refine gate training. Install with "
            'python -m pip install -e ".[pipeline]" or install torch.'
        ) from exc
    return torch


def _select_torch_device(torch: Any, device: str) -> Any:
    requested = str(device).lower()
    if requested == "auto":
        if _torch_device_works(torch, "mps"):
            requested = "mps"
        elif _torch_device_works(torch, "cuda"):
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "mps" and not _torch_device_works(torch, "mps"):
        raise RuntimeError("requested PyTorch MPS device is not available")
    if requested == "cuda" and not _torch_device_works(torch, "cuda"):
        raise RuntimeError("requested PyTorch CUDA device is not available")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported PyTorch device: {device!r}")
    return torch.device(requested)


def _torch_device_works(torch: Any, device: str) -> bool:
    try:
        if device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_built()):
                return False
            if hasattr(torch.backends.mps, "is_available") and not torch.backends.mps.is_available():
                return False
        if device == "cuda" and not torch.cuda.is_available():
            return False
        tensor = torch.tensor([0.0], device=device)
        return float(tensor.cpu()[0]) == 0.0
    except Exception:
        return False
