from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .local_refine_gate import (
    _fit_scaler,
    _fit_torch_model,
    _import_torch,
    _leave_one_out_predictions,
    _predict_torch_model,
    _prediction_summary,
    _select_torch_device,
    _transpose,
)


PRUNING_GATE_METRICS = (
    "Avg_num_box",
    "Avg_BVS",
    "Avg_MOV",
    "Avg_TOV",
    "Avg_Covered",
    "Avg_vIoU",
    "Avg_cub_CD",
)

PRUNING_GATE_DERIVED_FEATURES = (
    "baseline_uncovered",
    "baseline_log_num_box",
    "baseline_bvs_per_box",
    "baseline_mov_per_box",
    "baseline_tov_per_box",
    "baseline_mov_to_tov",
    "baseline_bvs_to_mov",
    "baseline_tov_to_mov",
    "baseline_viou_times_covered",
    "baseline_tightness_pressure",
    "candidate_log_fallback_disabled",
    "candidate_log_proxy_exact",
    "candidate_log_fallback_exact",
    "candidate_prune_to_proxy",
    "candidate_prune_to_baseline_time",
    "candidate_cache_hit_rate",
    "candidate_speedup_log",
    "bbox_geom_available",
    "bbox_geom_num_files",
    "bbox_geom_volume_sum",
    "bbox_geom_volume_mean",
    "bbox_geom_volume_std",
    "bbox_geom_volume_min",
    "bbox_geom_volume_max",
    "bbox_geom_extent_mean_x",
    "bbox_geom_extent_mean_y",
    "bbox_geom_extent_mean_z",
    "bbox_geom_extent_std_x",
    "bbox_geom_extent_std_y",
    "bbox_geom_extent_std_z",
    "bbox_geom_extent_min",
    "bbox_geom_extent_max",
    "bbox_geom_aspect_mean",
    "bbox_geom_aspect_max",
    "bbox_geom_center_std_x",
    "bbox_geom_center_std_y",
    "bbox_geom_center_std_z",
    "bbox_geom_total_aabb_volume",
    "bbox_geom_volume_fill_ratio",
    "bbox_geom_pairwise_aabb_overlap",
    "bbox_geom_pairwise_aabb_overlap_ratio",
)

PRUNING_GATE_FEATURE_SETS = ("basic", "expanded")


@dataclass(frozen=True)
class PruningGateDataset:
    rows: list[dict[str, Any]]
    categories: list[str]
    profiles: list[str]
    feature_names: list[str]
    features: list[list[float]]
    labels: list[int]


def load_pruning_gate_dataset(
    path: str | Path,
    *,
    target: str = "label_safe_pruned",
    only_pruned: bool = True,
    feature_set: str = "expanded",
) -> PruningGateDataset:
    rows = [_with_derived_labels(row) for row in _load_rows(path)]
    if only_pruned:
        rows = [row for row in rows if _int_field(row, "label_pruned") == 1]
    if not rows:
        raise ValueError(f"empty pruning gate dataset after filtering: {path}")
    feature_set = _validate_feature_set(feature_set)
    categories = sorted({str(row.get("category") or "unknown") for row in rows})
    profiles = sorted({str(row.get("profile") or "unknown") for row in rows})
    feature_names = pruning_gate_feature_names(categories, profiles, feature_set=feature_set)
    features = [pruning_gate_features(row, categories, profiles, feature_names=feature_names) for row in rows]
    labels = [_int_field(row, target) for row in rows]
    return PruningGateDataset(
        rows=rows,
        categories=categories,
        profiles=profiles,
        feature_names=feature_names,
        features=features,
        labels=labels,
    )


def train_pruning_gate(
    dataset_path: str | Path,
    *,
    output: str | Path,
    target: str = "label_safe_pruned",
    only_pruned: bool = True,
    hidden_size: int = 8,
    epochs: int = 200,
    learning_rate: float = 0.03,
    weight_decay: float = 1.0e-3,
    seed: int = 7,
    device: str = "auto",
    leave_one_out: bool = True,
    group_by: str = "",
    positive_weight: str = "balanced",
    feature_set: str = "expanded",
) -> dict[str, Any]:
    dataset = load_pruning_gate_dataset(
        dataset_path,
        target=target,
        only_pruned=only_pruned,
        feature_set=feature_set,
    )
    torch = _import_torch()
    torch_device = _select_torch_device(torch, device)
    cv_summary = None
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
    group_summary = None
    if group_by:
        group_predictions = _group_out_predictions(
            torch,
            torch_device,
            dataset,
            group_by=group_by,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            positive_weight=positive_weight,
        )
        group_summary = _prediction_summary(dataset.labels, group_predictions)

    mean, std = _fit_scaler(dataset.features)
    model_bundle = _fit_torch_model(
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
    train_predictions = _predict_torch_model(
        torch,
        torch_device,
        model_bundle,
        dataset.features,
        mean,
        std,
    )
    payload = _model_payload(
        dataset,
        mean,
        std,
        model_bundle,
        target=target,
        only_pruned=only_pruned,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=str(torch_device),
        torch_version=str(torch.__version__),
        positive_weight=positive_weight,
        feature_set=feature_set,
        train_summary=_prediction_summary(dataset.labels, train_predictions),
        cv_summary=cv_summary,
        group_by=group_by,
        group_summary=group_summary,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_pruning_gate(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("policy_type") != "candidate_pruning_gate":
        raise ValueError(f"not a candidate_pruning_gate model: {path}")
    return payload


def score_pruning_gate(payload: dict[str, Any], row: dict[str, Any]) -> float:
    categories = list(payload["categories"])
    profiles = list(payload["profiles"])
    feature_names = payload.get("feature_names") or pruning_gate_feature_names(
        categories,
        profiles,
        feature_set=str(payload.get("feature_set") or payload.get("metadata", {}).get("feature_set") or "basic"),
    )
    features = pruning_gate_features(row, categories, profiles, feature_names=feature_names)
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


def pruning_gate_feature_names(
    categories: Iterable[str],
    profiles: Iterable[str],
    *,
    feature_set: str = "expanded",
) -> list[str]:
    feature_set = _validate_feature_set(feature_set)
    names = [f"category={category}" for category in categories]
    names.extend(f"profile={profile}" for profile in profiles)
    names.extend(f"baseline_{metric}" for metric in PRUNING_GATE_METRICS)
    if feature_set == "expanded":
        names.extend(PRUNING_GATE_DERIVED_FEATURES)
    return names


def pruning_gate_features(
    row: dict[str, Any],
    categories: Iterable[str],
    profiles: Iterable[str],
    *,
    feature_names: Iterable[str] | None = None,
) -> list[float]:
    categories = list(categories)
    profiles = list(profiles)
    names = list(feature_names or pruning_gate_feature_names(categories, profiles))
    category = str(row.get("category") or "unknown")
    profile = str(row.get("profile") or "unknown")
    return [
        _feature_value(row, name, category=category, profile=profile)
        for name in names
    ]


def _feature_value(row: dict[str, Any], name: str, *, category: str, profile: str) -> float:
    if name.startswith("category="):
        return 1.0 if category == name.split("=", 1)[1] else 0.0
    if name.startswith("profile="):
        return 1.0 if profile == name.split("=", 1)[1] else 0.0
    if name.startswith("baseline_Avg_"):
        return _float_field(row, name)
    if name == "candidate_log_fallback_disabled":
        return math.log1p(max(_float_field(row, "candidate_fallback_disabled"), 0.0))
    if name == "candidate_log_proxy_exact":
        return math.log1p(max(_float_field(row, "candidate_proxy_exact"), 0.0))
    if name == "candidate_log_fallback_exact":
        return math.log1p(max(_float_field(row, "candidate_fallback_exact"), 0.0))
    if name == "candidate_prune_to_proxy":
        return _float_field(row, "candidate_fallback_disabled") / max(_float_field(row, "candidate_proxy_exact"), 1.0)
    if name == "candidate_prune_to_baseline_time":
        return _float_field(row, "candidate_fallback_disabled") / max(_float_field(row, "baseline_elapsed_sec"), 1.0e-9)
    if name == "candidate_cache_hit_rate":
        hits = _float_field(row, "candidate_reward_cache_hits")
        misses = _float_field(row, "candidate_reward_cache_misses")
        return hits / max(hits + misses, 1.0)
    if name == "candidate_speedup_log":
        return math.log(max(_float_field(row, "speedup_vs_baseline"), 1.0e-9))
    metrics = {metric: _float_field(row, f"baseline_{metric}") for metric in PRUNING_GATE_METRICS}
    num_box = max(metrics["Avg_num_box"], 1.0)
    bvs = metrics["Avg_BVS"]
    mov = metrics["Avg_MOV"]
    tov = metrics["Avg_TOV"]
    covered = metrics["Avg_Covered"]
    viou = metrics["Avg_vIoU"]
    eps = 1.0e-9
    if name == "baseline_uncovered":
        return max(0.0, 1.0 - covered)
    if name == "baseline_log_num_box":
        return math.log1p(max(metrics["Avg_num_box"], 0.0))
    if name == "baseline_bvs_per_box":
        return bvs / num_box
    if name == "baseline_mov_per_box":
        return mov / num_box
    if name == "baseline_tov_per_box":
        return tov / num_box
    if name == "baseline_mov_to_tov":
        return mov / max(abs(tov), eps)
    if name == "baseline_bvs_to_mov":
        return bvs / max(abs(mov), eps)
    if name == "baseline_tov_to_mov":
        return tov / max(abs(mov), eps)
    if name == "baseline_viou_times_covered":
        return viou * covered
    if name == "baseline_tightness_pressure":
        return bvs + mov + tov - viou
    return _float_field(row, name)


def _model_payload(
    dataset: PruningGateDataset,
    mean: list[float],
    std: list[float],
    model_bundle: dict[str, Any],
    *,
    target: str,
    only_pruned: bool,
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
    group_by: str,
    group_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    model = model_bundle["model"]
    hidden_size = int(model_bundle["hidden_size"])
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
        "schema_version": 1,
        "policy_type": "candidate_pruning_gate",
        "feature_set": feature_set,
        "target": target,
        "only_pruned": bool(only_pruned),
        "categories": dataset.categories,
        "profiles": dataset.profiles,
        "feature_names": dataset.feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "hidden_size": hidden_size,
        "activation": "tanh" if hidden_size > 0 else "linear",
        "weights": weights,
        "metadata": {
            "source": "smart.candidate_pruning_gate.train_pruning_gate",
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
            "group_out": {
                "group_by": group_by,
                "summary": group_summary,
            } if group_summary is not None else None,
        },
    }


def _group_out_predictions(
    torch: Any,
    device: Any,
    dataset: PruningGateDataset,
    *,
    group_by: str,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    positive_weight: str,
) -> list[float]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(dataset.rows):
        groups.setdefault(_group_key(row, group_by), []).append(idx)
    if len(groups) < 2:
        raise ValueError(f"group_by={group_by!r} produced fewer than 2 groups")
    predictions = [0.0 for _ in dataset.labels]
    for group_idx, indices in enumerate(groups.values()):
        held_out = set(indices)
        train_features = [row for idx, row in enumerate(dataset.features) if idx not in held_out]
        train_labels = [label for idx, label in enumerate(dataset.labels) if idx not in held_out]
        if not train_labels:
            probability = sum(dataset.labels) / len(dataset.labels)
            for idx in indices:
                predictions[idx] = probability
            continue
        if len(set(train_labels)) <= 1:
            probability = sum(train_labels) / len(train_labels)
            for idx in indices:
                predictions[idx] = probability
            continue
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
            seed=seed + group_idx,
            positive_weight=positive_weight,
        )
        held_out_features = [dataset.features[idx] for idx in indices]
        held_out_predictions = _predict_torch_model(
            torch,
            device,
            model,
            held_out_features,
            mean,
            std,
        )
        for idx, probability in zip(indices, held_out_predictions):
            predictions[idx] = probability
    return predictions


def _group_key(row: dict[str, Any], group_by: str) -> str:
    if group_by == "category_profile":
        return f"{row.get('category', '')}::{row.get('profile', '')}"
    if group_by == "category_mesh":
        return f"{row.get('category', '')}::{row.get('mesh_id', '')}"
    return str(row.get(group_by, ""))


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


def _with_derived_labels(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    pruned = _numeric(row.get("candidate_fallback_disabled")) > 0.0
    not_worse = _int_field(row, "label_not_worse") == 1
    if "label_pruned" not in row:
        row["label_pruned"] = int(pruned)
    if "label_safe_pruned" not in row:
        row["label_safe_pruned"] = int(pruned and not_worse)
    if "label_pruning_regression" not in row:
        row["label_pruning_regression"] = int(pruned and not not_worse)
    if "label_metric_identical" not in row:
        deltas = [_float_field(row, f"delta_{metric}") for metric in PRUNING_GATE_METRICS]
        row["label_metric_identical"] = int(all(value == 0.0 for value in deltas))
    return row


def _validate_feature_set(feature_set: str) -> str:
    feature_set = str(feature_set or "expanded")
    if feature_set not in PRUNING_GATE_FEATURE_SETS:
        raise ValueError(
            f"unsupported pruning gate feature_set {feature_set!r}; "
            f"expected one of {PRUNING_GATE_FEATURE_SETS}"
        )
    return feature_set


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


def _numeric(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
