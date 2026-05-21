from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .pipeline.config import REPO_ROOT, workspace_path
from .pipeline.stages import bbox_dir_for_render, list_mesh_ids, normalized_mesh_path


@dataclass
class BoxProposalRecord:
    category: str
    mesh_id: str
    mesh_path: str
    bbox_dir: str
    boxes: list[list[float]]


def export_box_proposal_dataset(
    cfg: dict[str, Any],
    *,
    output: str | Path,
    stage: str = "mcts",
    category_name: str | None = None,
    meshes: list[str] | None = None,
    from_manifest: bool = False,
    max_boxes: int = 32,
    label_format: str = "corners",
) -> dict[str, Any]:
    """Export pseudo-label rows for learned bbox proposal training.

    The labels are generated from existing SMART bbox OBJ outputs. They are
    deliberately plain JSONL so future models can reuse the same dataset.
    """

    records: list[BoxProposalRecord] = []
    manifest_meshes = _manifest_meshes(cfg, stage) if from_manifest else None
    allowed_meshes = set(meshes or [])
    for category in cfg.get("categories", []):
        name = str(category.get("name", ""))
        if category_name and name != category_name:
            continue
        if manifest_meshes is not None:
            mesh_ids = manifest_meshes.get(name, [])
        else:
            mesh_ids = list_mesh_ids(category, explicit=meshes)
        for mesh_id in mesh_ids:
            if allowed_meshes and mesh_id not in allowed_meshes:
                continue
            mesh_path = normalized_mesh_path(cfg, category, mesh_id)
            if not mesh_path.exists():
                mesh_path = Path(category["mesh_root"]) / mesh_id / "model.obj"
                if not mesh_path.is_absolute():
                    mesh_path = REPO_ROOT / mesh_path
            bbox_dir = bbox_dir_for_render(cfg, category, mesh_id, stage)
            if bbox_dir is None or not bbox_dir.exists() or not mesh_path.exists():
                continue
            boxes = (
                load_box_corners_from_dir(bbox_dir)
                if str(label_format) == "corners"
                else load_box_basis_from_dir(bbox_dir)
                if str(label_format) == "basis"
                else load_box_params_from_dir(bbox_dir)
            )
            if not boxes:
                continue
            boxes = sorted(boxes, key=_box_volume, reverse=True)[: max(int(max_boxes), 1)]
            records.append(
                BoxProposalRecord(
                    category=name,
                    mesh_id=str(mesh_id),
                    mesh_path=str(mesh_path),
                    bbox_dir=str(bbox_dir),
                    boxes=boxes,
                )
            )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.__dict__, sort_keys=True) + "\n")
    categories = sorted({record.category for record in records})
    return {
        "output": str(out),
        "stage": stage,
        "records": len(records),
        "categories": categories,
        "max_boxes": int(max_boxes),
        "label_format": str(label_format),
    }


def train_box_proposal_model(
    dataset_path: str | Path,
    *,
    output: str | Path,
    num_points: int = 1024,
    max_boxes: int = 16,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1.0e-3,
    hidden_size: int = 128,
    device: str = "auto",
    seed: int = 0,
    loss_mode: str = "matched",
    architecture: str = "query_pointnet",
    validation_fraction: float = 0.0,
    structured_basis: bool = True,
    coverage_loss_weight: float = 0.0,
    coverage_temperature: float = 0.05,
    compactness_loss_weight: float = 0.0,
) -> dict[str, Any]:
    torch = _import_torch()
    torch_device = _select_torch_device(torch, device)
    records = _load_dataset_records(dataset_path)
    if not records:
        raise ValueError(f"empty box proposal dataset: {dataset_path}")
    random.Random(seed).shuffle(records)
    val_count = 0
    if float(validation_fraction) > 0.0 and len(records) > 1:
        val_count = min(max(int(round(len(records) * float(validation_fraction))), 1), len(records) - 1)
    val_records = records[:val_count]
    train_records = records[val_count:] if val_records else records
    categories = sorted({str(record["category"]) for record in records})
    category_to_idx = {name: idx for idx, name in enumerate(categories)}
    box_dim = _infer_box_dim(records)
    network_box_dim = _network_box_dim(box_dim, structured_basis)
    model = _PointNetBoxProposal(
        max_boxes=max_boxes,
        box_dim=box_dim,
        network_box_dim=network_box_dim,
        num_categories=max(len(categories), 1),
        hidden_size=hidden_size,
        architecture=architecture,
        structured_basis=structured_basis,
        torch=torch,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None

    for _ in range(max(int(epochs), 0)):
        random.Random(seed + len(losses)).shuffle(train_records)
        epoch_loss = 0.0
        batches = 0
        for batch in _batched(train_records, max(int(batch_size), 1)):
            points, category_ids, targets, mask = _proposal_batch(
                batch,
                categories=category_to_idx,
                num_points=num_points,
                max_boxes=max_boxes,
                box_dim=box_dim,
                rng=rng,
                torch=torch,
                device=torch_device,
            )
            pred_boxes, pred_logits = model(points, category_ids)
            box_loss, objectness_loss = _proposal_losses(
                pred_boxes,
                pred_logits,
                targets,
                mask,
                points=points,
                box_dim=box_dim,
                loss_mode=loss_mode,
                coverage_loss_weight=coverage_loss_weight,
                coverage_temperature=coverage_temperature,
                compactness_loss_weight=compactness_loss_weight,
                torch=torch,
            )
            loss = box_loss + 0.2 * objectness_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            batches += 1
        losses.append(epoch_loss / max(batches, 1))
        if val_records:
            val_loss = _evaluate_proposal_loss(
                model,
                val_records,
                categories=category_to_idx,
                num_points=num_points,
                max_boxes=max_boxes,
                box_dim=box_dim,
                batch_size=batch_size,
                loss_mode=loss_mode,
                coverage_loss_weight=coverage_loss_weight,
                coverage_temperature=coverage_temperature,
                compactness_loss_weight=compactness_loss_weight,
                rng=rng,
                torch=torch,
                device=torch_device,
            )
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "schema_version": 1,
        "model_type": _box_model_type_for_architecture(architecture),
        "architecture": str(architecture),
        "structured_basis": bool(structured_basis),
        "max_boxes": int(max_boxes),
        "box_dim": int(box_dim),
        "network_box_dim": int(network_box_dim),
        "label_format": _label_format_for_box_dim(box_dim),
        "num_points": int(num_points),
        "hidden_size": int(hidden_size),
        "categories": categories,
        "model_state": model.state_dict(),
        "metadata": {
            "dataset": str(dataset_path),
            "records": len(records),
            "train_records": len(train_records),
            "validation_records": len(val_records),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "device": str(torch_device),
            "loss_mode": str(loss_mode),
            "architecture": str(architecture),
            "structured_basis": bool(structured_basis),
            "coverage_loss_weight": float(coverage_loss_weight),
            "coverage_temperature": float(coverage_temperature),
            "compactness_loss_weight": float(compactness_loss_weight),
            "losses": losses,
            "validation_losses": val_losses,
            "best_validation_loss": best_val_loss if val_records else None,
        },
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    return {k: v for k, v in checkpoint["metadata"].items() if k not in {"losses", "validation_losses"}} | {
        "output": str(out),
        "final_loss": losses[-1] if losses else None,
        "best_validation_loss": best_val_loss if val_records else None,
        "categories": categories,
        "max_boxes": int(max_boxes),
        "box_dim": int(box_dim),
        "loss_mode": str(loss_mode),
        "architecture": str(architecture),
        "structured_basis": bool(structured_basis),
        "coverage_loss_weight": float(coverage_loss_weight),
        "coverage_temperature": float(coverage_temperature),
        "compactness_loss_weight": float(compactness_loss_weight),
    }


def predict_box_proposals(
    model_path: str | Path,
    mesh_path: str | Path,
    *,
    output_dir: str | Path,
    category: str = "",
    mesh_id: str | None = None,
    num_points: int | None = None,
    score_threshold: float = 0.5,
    max_boxes: int | None = None,
    min_boxes: int = 0,
    nms_iou_threshold: float = 1.0,
    coverage_calibration_target: float = 0.0,
    coverage_calibration_max_scale: float = 2.0,
    legacy_layout: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    torch = _import_torch()
    checkpoint = torch.load(model_path, map_location="cpu")
    categories = [str(item) for item in checkpoint.get("categories", [])]
    category_to_idx = {name: idx for idx, name in enumerate(categories)}
    ckpt_max_boxes = int(checkpoint.get("max_boxes", 16))
    box_dim = int(checkpoint.get("box_dim", 6))
    network_box_dim = int(checkpoint.get("network_box_dim", box_dim))
    use_max_boxes = min(int(max_boxes or ckpt_max_boxes), ckpt_max_boxes)
    architecture = _architecture_from_checkpoint(checkpoint)
    structured_basis = bool(checkpoint.get("structured_basis", False))
    model = _PointNetBoxProposal(
        max_boxes=ckpt_max_boxes,
        box_dim=box_dim,
        network_box_dim=network_box_dim,
        num_categories=max(len(categories), 1),
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        architecture=architecture,
        structured_basis=structured_basis,
        torch=torch,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sampled_points = sample_mesh_points(mesh_path, int(num_points or checkpoint.get("num_points", 1024)), seed=seed)
    point_tensor = torch.tensor(sampled_points, dtype=torch.float32).unsqueeze(0)
    category_id = torch.tensor([category_to_idx.get(str(category), 0)], dtype=torch.long)
    with torch.no_grad():
        pred_boxes, logits = model(point_tensor, category_id)
        if box_dim == 6:
            centers = pred_boxes[:, :, :3]
            sizes = torch.exp(pred_boxes[:, :, 3:6]).clamp(min=1.0e-5)
        scores = torch.sigmoid(logits)
    all_boxes: list[tuple[float, list[float]]] = []
    for idx in range(use_max_boxes):
        score = float(scores[0, idx].cpu().item())
        if box_dim == 6:
            center = centers[0, idx].cpu().numpy().astype(float)
            size = sizes[0, idx].cpu().numpy().astype(float)
            box = _sanitize_axis_box_np(
                np.asarray([center[0], center[1], center[2], size[0], size[1], size[2]], dtype=np.float64)
            ).tolist()
        elif box_dim == 12:
            box = _sanitize_basis_np(pred_boxes[0, idx].cpu().numpy().reshape(-1).astype(np.float64)).tolist()
        else:
            box = [float(value) for value in pred_boxes[0, idx].cpu().numpy().reshape(-1).tolist()]
        all_boxes.append((score, box))
    all_boxes.sort(key=lambda item: item[0], reverse=True)
    boxes = [item for item in all_boxes if item[0] >= float(score_threshold)]
    if float(nms_iou_threshold) < 1.0:
        boxes = _proposal_nms(boxes, iou_threshold=float(nms_iou_threshold))
    required = min(max(int(min_boxes), 0), use_max_boxes)
    if len(boxes) < required:
        selected_ids = {id(item) for item in boxes}
        for item in all_boxes:
            if id(item) in selected_ids:
                continue
            if float(nms_iou_threshold) < 1.0 and any(
                _proposal_iou(item[1], kept[1]) > float(nms_iou_threshold) for kept in boxes
            ):
                continue
            boxes.append(item)
            if len(boxes) >= required:
                break
    boxes = boxes[:use_max_boxes]
    if not boxes and use_max_boxes > 0:
        idx = int(torch.argmax(scores[0, :use_max_boxes]).cpu().item())
        if box_dim == 6:
            center = centers[0, idx].cpu().numpy().astype(float)
            size = sizes[0, idx].cpu().numpy().astype(float)
            box = _sanitize_axis_box_np(
                np.asarray([center[0], center[1], center[2], size[0], size[1], size[2]], dtype=np.float64)
            ).tolist()
        elif box_dim == 12:
            box = _sanitize_basis_np(pred_boxes[0, idx].cpu().numpy().reshape(-1).astype(np.float64)).tolist()
        else:
            box = [float(value) for value in pred_boxes[0, idx].cpu().numpy().reshape(-1).tolist()]
        boxes.append(
            (
                float(scores[0, idx].cpu().item()),
                box,
            )
        )
    coverage_calibration = None
    if boxes and float(coverage_calibration_target) > 0.0:
        boxes, coverage_calibration = _calibrate_proposal_coverage(
            boxes,
            sampled_points,
            target=float(coverage_calibration_target),
            max_scale=float(coverage_calibration_max_scale),
        )
    out_root = Path(output_dir)
    if legacy_layout:
        if not mesh_id:
            mesh_id = Path(mesh_path).parent.name
        bbox_dir = out_root / "result" / "updated0" / str(mesh_id) / "bboxs_steps0"
    else:
        bbox_dir = out_root
    bbox_dir.mkdir(parents=True, exist_ok=True)
    for idx, (_, params) in enumerate(boxes):
        if len(params) == 24:
            write_box_corners_obj(params, bbox_dir / f"bbox{idx}.obj")
        elif len(params) == 12:
            write_box_basis_obj(params, bbox_dir / f"bbox{idx}.obj")
        else:
            write_axis_aligned_box_obj(params, bbox_dir / f"bbox{idx}.obj")
    return {
        "model": str(model_path),
        "mesh_path": str(mesh_path),
        "output_dir": str(bbox_dir),
        "category": category,
        "mesh_id": mesh_id,
        "num_boxes": len(boxes),
        "scores": [score for score, _ in boxes],
        "label_format": _label_format_for_box_dim(box_dim),
        "score_threshold": float(score_threshold),
        "min_boxes": int(min_boxes),
        "nms_iou_threshold": float(nms_iou_threshold),
        "coverage_calibration": coverage_calibration,
    }


def load_box_params_from_dir(path: str | Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for obj in sorted(Path(path).glob("bbox*.obj")):
        try:
            boxes.append(box_params_from_obj(obj))
        except ValueError:
            continue
    return boxes


def load_box_corners_from_dir(path: str | Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for obj in sorted(Path(path).glob("bbox*.obj")):
        vertices = _read_obj_vertices(obj)
        if vertices.size == 0:
            continue
        boxes.append(_box_corners_from_vertices(vertices))
    return boxes


def load_box_basis_from_dir(path: str | Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for obj in sorted(Path(path).glob("bbox*.obj")):
        vertices = _read_obj_vertices(obj)
        if vertices.size == 0:
            continue
        boxes.append(_box_basis_from_vertices(vertices))
    return boxes


def box_params_from_obj(path: str | Path) -> list[float]:
    vertices = _read_obj_vertices(path)
    if vertices.size == 0:
        raise ValueError(f"OBJ has no vertices: {path}")
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    center = (mn + mx) * 0.5
    size = np.maximum(mx - mn, 1.0e-6)
    return [float(center[0]), float(center[1]), float(center[2]), float(size[0]), float(size[1]), float(size[2])]


def write_axis_aligned_box_obj(params: Iterable[float], path: str | Path) -> None:
    cx, cy, cz, sx, sy, sz = [float(value) for value in params]
    hx, hy, hz = max(sx, 1.0e-6) * 0.5, max(sy, 1.0e-6) * 0.5, max(sz, 1.0e-6) * 0.5
    vertices = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    faces = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (4, 8, 5, 1),
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for vertex in vertices:
            file.write("v %.9g %.9g %.9g\n" % vertex)
        for face in faces:
            file.write("f %d %d %d %d\n" % face)


def write_box_corners_obj(params: Iterable[float], path: str | Path) -> None:
    values = [float(value) for value in params]
    if len(values) != 24:
        raise ValueError("corner box params must contain 24 values")
    vertices = np.asarray([values[idx : idx + 3] for idx in range(0, 24, 3)], dtype=np.float64)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import trimesh

        hull = trimesh.Trimesh(vertices=vertices, process=False).convex_hull
        hull.export(out)
        return
    except Exception:
        pass
    faces = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (4, 8, 5, 1),
    ]
    with out.open("w", encoding="utf-8") as file:
        for vertex in vertices:
            file.write("v %.9g %.9g %.9g\n" % tuple(vertex.tolist()))
        for face in faces:
            file.write("f %d %d %d %d\n" % face)


def write_box_basis_obj(params: Iterable[float], path: str | Path) -> None:
    corners = _basis_to_corners_np(_sanitize_basis_np(np.asarray([float(value) for value in params], dtype=np.float64)))
    write_box_corners_obj(corners.reshape(-1).tolist(), path)


def sample_mesh_points(path: str | Path, num_points: int, *, seed: int = 0) -> np.ndarray:
    vertices: np.ndarray
    try:
        import trimesh

        mesh = trimesh.load(path, force="mesh", process=False)
        if getattr(mesh, "faces", None) is not None and len(mesh.faces) > 0:
            points, _ = trimesh.sample.sample_surface(mesh, max(int(num_points), 1), seed=int(seed))
            return np.asarray(points, dtype=np.float32)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
    except Exception:
        vertices = _read_obj_vertices(path).astype(np.float32)
    if vertices.size == 0:
        raise ValueError(f"mesh has no points: {path}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(vertices), size=max(int(num_points), 1), replace=len(vertices) < int(num_points))
    return vertices[indices].astype(np.float32)


def _read_obj_vertices(path: str | Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.asarray(vertices, dtype=np.float64)


def _load_dataset_records(dataset_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(dataset_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _proposal_batch(
    records: list[dict[str, Any]],
    *,
    categories: dict[str, int],
    num_points: int,
    max_boxes: int,
    box_dim: int,
    rng: np.random.Generator,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    points = []
    category_ids = []
    targets = []
    masks = []
    for record in records:
        points.append(sample_mesh_points(record["mesh_path"], num_points, seed=int(rng.integers(0, 2**31 - 1))))
        category_ids.append(categories.get(str(record.get("category", "")), 0))
        boxes = sorted(record.get("boxes", []), key=_box_volume, reverse=True)[:max_boxes]
        target = np.zeros((max_boxes, box_dim), dtype=np.float32)
        mask = np.zeros((max_boxes,), dtype=np.float32)
        for idx, box in enumerate(boxes):
            target[idx, :] = np.asarray(box[:box_dim], dtype=np.float32)
            mask[idx] = 1.0
        targets.append(target)
        masks.append(mask)
    return (
        torch.tensor(np.stack(points, axis=0), dtype=torch.float32, device=device),
        torch.tensor(category_ids, dtype=torch.long, device=device),
        torch.tensor(np.stack(targets, axis=0), dtype=torch.float32, device=device),
        torch.tensor(np.stack(masks, axis=0), dtype=torch.float32, device=device),
    )


def _proposal_losses(
    pred_boxes: Any,
    pred_logits: Any,
    targets: Any,
    mask: Any,
    *,
    points: Any | None = None,
    box_dim: int,
    loss_mode: str,
    coverage_loss_weight: float = 0.0,
    coverage_temperature: float = 0.05,
    compactness_loss_weight: float = 0.0,
    torch: Any,
) -> tuple[Any, Any]:
    if str(loss_mode) == "slot":
        box_loss, objectness_loss = _slot_proposal_loss(
            pred_boxes,
            pred_logits,
            targets,
            mask,
            box_dim=box_dim,
            torch=torch,
        )
    elif str(loss_mode) == "matched":
        box_loss, objectness_loss = _matched_proposal_loss(
            pred_boxes,
            pred_logits,
            targets,
            mask,
            box_dim=box_dim,
            torch=torch,
        )
    else:
        raise ValueError(f"unsupported box proposal loss_mode: {loss_mode!r}")
    if points is not None and float(coverage_loss_weight) > 0.0:
        box_loss = box_loss + float(coverage_loss_weight) * _soft_point_coverage_loss(
            pred_boxes,
            pred_logits,
            points,
            box_dim=box_dim,
            temperature=coverage_temperature,
            torch=torch,
        )
    if float(compactness_loss_weight) > 0.0:
        box_loss = box_loss + float(compactness_loss_weight) * _proposal_compactness_loss(
            pred_boxes,
            pred_logits,
            targets,
            mask,
            box_dim=box_dim,
            torch=torch,
        )
    return box_loss, objectness_loss


def _evaluate_proposal_loss(
    model: Any,
    records: list[dict[str, Any]],
    *,
    categories: dict[str, int],
    num_points: int,
    max_boxes: int,
    box_dim: int,
    batch_size: int,
    loss_mode: str,
    coverage_loss_weight: float,
    coverage_temperature: float,
    compactness_loss_weight: float,
    rng: np.random.Generator,
    torch: Any,
    device: Any,
) -> float:
    was_training = bool(model.training)
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in _batched(records, max(int(batch_size), 1)):
            points, category_ids, targets, mask = _proposal_batch(
                batch,
                categories=categories,
                num_points=num_points,
                max_boxes=max_boxes,
                box_dim=box_dim,
                rng=rng,
                torch=torch,
                device=device,
            )
            pred_boxes, pred_logits = model(points, category_ids)
            box_loss, objectness_loss = _proposal_losses(
                pred_boxes,
                pred_logits,
                targets,
                mask,
                points=points,
                box_dim=box_dim,
                loss_mode=loss_mode,
                coverage_loss_weight=coverage_loss_weight,
                coverage_temperature=coverage_temperature,
                compactness_loss_weight=compactness_loss_weight,
                torch=torch,
            )
            total += float((box_loss + 0.2 * objectness_loss).detach().cpu().item())
            batches += 1
    if was_training:
        model.train()
    return total / max(batches, 1)


def _slot_proposal_loss(
    pred_boxes: Any,
    pred_logits: Any,
    targets: Any,
    mask: Any,
    *,
    box_dim: int,
    torch: Any,
) -> tuple[Any, Any]:
    mask_f = mask.float()
    denom = torch.clamp(mask_f.sum(), min=1.0)
    if box_dim == 6:
        pred_centers = pred_boxes[:, :, :3]
        pred_log_sizes = pred_boxes[:, :, 3:6]
        target_centers = targets[:, :, :3]
        target_log_sizes = torch.log(torch.clamp(targets[:, :, 3:6], min=1.0e-6))
        box_loss = (
            torch.abs(pred_centers - target_centers).sum(dim=2)
            + torch.abs(pred_log_sizes - target_log_sizes).sum(dim=2)
        )
    else:
        box_loss = torch.abs(pred_boxes - targets).sum(dim=2) / float(box_dim)
    box_loss = (box_loss * mask_f).sum() / denom
    objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, mask_f)
    return box_loss, objectness_loss


def _matched_proposal_loss(
    pred_boxes: Any,
    pred_logits: Any,
    targets: Any,
    mask: Any,
    *,
    box_dim: int,
    torch: Any,
) -> tuple[Any, Any]:
    """Permutation-invariant loss over unordered bbox sets.

    SMART boxes do not have a semantic slot order. For corner labels, even the
    eight vertices can arrive in different orders after OBJ round trips. This
    loss matches predicted boxes to target boxes by detached pair costs, then
    applies a differentiable pair loss only to matched pairs.
    """

    batch_size, max_boxes = pred_logits.shape
    matched_mask = torch.zeros_like(mask.float())
    pair_losses = []
    for batch_idx in range(batch_size):
        target_indices = torch.nonzero(mask[batch_idx] > 0.5, as_tuple=False).flatten()
        if int(target_indices.numel()) == 0:
            continue
        candidate_targets = targets[batch_idx, target_indices]
        with torch.no_grad():
            cost = _proposal_cost_matrix(
                pred_boxes[batch_idx],
                candidate_targets,
                box_dim=box_dim,
                torch=torch,
            )
            matches = _linear_assignment(cost.detach().cpu().numpy())
        for pred_idx, local_target_idx in matches[: min(max_boxes, int(target_indices.numel()))]:
            matched_mask[batch_idx, int(pred_idx)] = 1.0
            pair_losses.append(
                _proposal_pair_loss(
                    pred_boxes[batch_idx, int(pred_idx)],
                    candidate_targets[int(local_target_idx)],
                    box_dim=box_dim,
                    torch=torch,
                )
            )
    if pair_losses:
        box_loss = torch.stack(pair_losses).mean()
    else:
        box_loss = pred_boxes.sum() * 0.0
    objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, matched_mask)
    return box_loss, objectness_loss


def _soft_point_coverage_loss(
    pred_boxes: Any,
    pred_logits: Any,
    points: Any,
    *,
    box_dim: int,
    temperature: float,
    torch: Any,
) -> Any:
    inside = _soft_points_inside_boxes(
        points,
        pred_boxes,
        box_dim=box_dim,
        temperature=max(float(temperature), 1.0e-4),
        torch=torch,
    )
    scores = torch.sigmoid(pred_logits).unsqueeze(1).clamp(0.0, 1.0)
    covered_by_box = (inside * scores).clamp(0.0, 1.0)
    uncovered = torch.prod(1.0 - covered_by_box, dim=2)
    return uncovered.mean()


def _proposal_compactness_loss(
    pred_boxes: Any,
    pred_logits: Any,
    targets: Any,
    mask: Any,
    *,
    box_dim: int,
    torch: Any,
) -> Any:
    """Penalize predicted active box volume above pseudo-label volume.

    The coverage loss alone can learn to cover the mesh with oversized boxes.
    This term keeps the learned proposal on the same compactness scale as the
    SMART pseudo-labels while still letting exact guarded refinement decide the
    final output.
    """

    pred_volume = _box_volumes_torch(pred_boxes, box_dim=box_dim, axis_aligned_log_sizes=True, torch=torch)
    target_volume = _box_volumes_torch(targets, box_dim=box_dim, axis_aligned_log_sizes=False, torch=torch)
    active = torch.sigmoid(pred_logits).clamp(0.0, 1.0)
    target_total = (target_volume * mask.float()).sum(dim=1).clamp(min=1.0e-6)
    pred_total = (pred_volume * active).sum(dim=1)
    return torch.relu(pred_total / target_total - 1.0).mean()


def _box_volumes_torch(boxes: Any, *, box_dim: int, axis_aligned_log_sizes: bool, torch: Any) -> Any:
    if int(box_dim) == 6:
        sizes = boxes[:, :, 3:6]
        if bool(axis_aligned_log_sizes):
            sizes = torch.exp(sizes)
        return torch.prod(sizes.clamp(min=1.0e-6), dim=2)
    if int(box_dim) == 12:
        axes = boxes[:, :, 3:12].reshape(boxes.shape[0], boxes.shape[1], 3, 3)
        return torch.prod(torch.linalg.norm(axes, dim=3).clamp(min=1.0e-6), dim=2) * 8.0
    if int(box_dim) == 24:
        corners = boxes.reshape(boxes.shape[0], boxes.shape[1], 8, 3)
        size = (corners.max(dim=2).values - corners.min(dim=2).values).clamp(min=0.0)
        return torch.prod(size, dim=2)
    raise ValueError(f"unsupported proposal box dimension: {box_dim}")


def _soft_points_inside_boxes(points: Any, boxes: Any, *, box_dim: int, temperature: float, torch: Any) -> Any:
    if int(box_dim) == 6:
        centers = boxes[:, :, :3]
        half = torch.exp(boxes[:, :, 3:6]).clamp(min=1.0e-5) * 0.5
        rel = torch.abs(points[:, :, None, :] - centers[:, None, :, :]) / half[:, None, :, :]
        margin = 1.0 - rel.max(dim=3).values
    elif int(box_dim) == 12:
        centers = boxes[:, :, :3]
        axes = boxes[:, :, 3:12].reshape(boxes.shape[0], boxes.shape[1], 3, 3)
        lengths = torch.linalg.norm(axes, dim=3).clamp(min=1.0e-5)
        directions = axes / lengths[:, :, :, None]
        rel = points[:, :, None, :] - centers[:, None, :, :]
        proj = torch.abs(torch.einsum("bnkd,bkad->bnka", rel, directions)) / lengths[:, None, :, :]
        margin = 1.0 - proj.max(dim=3).values
    elif int(box_dim) == 24:
        corners = boxes.reshape(boxes.shape[0], boxes.shape[1], 8, 3)
        mn = corners.min(dim=2).values
        mx = corners.max(dim=2).values
        center = (mn + mx) * 0.5
        half = (mx - mn).clamp(min=1.0e-5) * 0.5
        rel = torch.abs(points[:, :, None, :] - center[:, None, :, :]) / half[:, None, :, :]
        margin = 1.0 - rel.max(dim=3).values
    else:
        raise ValueError(f"unsupported proposal box dimension: {box_dim}")
    return torch.sigmoid(margin / float(temperature))


def _proposal_cost_matrix(pred_boxes: Any, target_boxes: Any, *, box_dim: int, torch: Any) -> Any:
    if box_dim == 6:
        pred = torch.cat(
            [
                pred_boxes[:, :3],
                torch.exp(torch.clamp(pred_boxes[:, 3:6], min=math.log(1.0e-6), max=math.log(4.0))),
            ],
            dim=1,
        )
        target = target_boxes[:, :6]
        return torch.cdist(pred, target, p=1)
    if box_dim == 12:
        pred = _basis_corners_torch(pred_boxes, torch=torch)
        target = _basis_corners_torch(target_boxes, torch=torch)
    else:
        pred = pred_boxes.reshape(pred_boxes.shape[0], 8, 3)
        target = target_boxes.reshape(target_boxes.shape[0], 8, 3)
    diff = torch.abs(pred[:, None, :, None, :] - target[None, :, None, :, :]).sum(dim=4)
    pred_to_target = diff.min(dim=3).values.mean(dim=2)
    target_to_pred = diff.min(dim=2).values.mean(dim=2)
    return 0.5 * (pred_to_target + target_to_pred)


def _proposal_pair_loss(pred_box: Any, target_box: Any, *, box_dim: int, torch: Any) -> Any:
    if box_dim == 6:
        center_loss = torch.abs(pred_box[:3] - target_box[:3]).sum()
        target_log_size = torch.log(torch.clamp(target_box[3:6], min=1.0e-6))
        size_loss = torch.abs(pred_box[3:6] - target_log_size).sum()
        return center_loss + size_loss
    if box_dim == 12:
        pred = _basis_corners_torch(pred_box.reshape(1, 12), torch=torch).reshape(8, 3)
        target = _basis_corners_torch(target_box.reshape(1, 12), torch=torch).reshape(8, 3)
        basis_regularizer = 0.01 * torch.abs(pred_box[3:12] - target_box[3:12]).mean()
    else:
        pred = pred_box.reshape(8, 3)
        target = target_box.reshape(8, 3)
        basis_regularizer = pred_box.sum() * 0.0
    diff = torch.abs(pred[:, None, :] - target[None, :, :]).sum(dim=2)
    pred_to_target = diff.min(dim=1).values.mean()
    target_to_pred = diff.min(dim=0).values.mean()
    return 0.5 * (pred_to_target + target_to_pred) + basis_regularizer


def _linear_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [(int(row), int(col)) for row, col in zip(rows, cols)]
    except Exception:
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        matches: list[tuple[int, int]] = []
        flat = [
            (float(cost[row, col]), int(row), int(col))
            for row in range(cost.shape[0])
            for col in range(cost.shape[1])
        ]
        for _, row, col in sorted(flat, key=lambda item: item[0]):
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            matches.append((row, col))
            if len(matches) >= min(cost.shape):
                break
        return matches


def _proposal_nms(
    boxes: list[tuple[float, list[float]]],
    *,
    iou_threshold: float,
) -> list[tuple[float, list[float]]]:
    kept: list[tuple[float, list[float]]] = []
    for item in boxes:
        if all(_proposal_iou(item[1], previous[1]) <= float(iou_threshold) for previous in kept):
            kept.append(item)
    return kept


def _calibrate_proposal_coverage(
    boxes: list[tuple[float, list[float]]],
    points: np.ndarray,
    *,
    target: float,
    max_scale: float,
) -> tuple[list[tuple[float, list[float]]], dict[str, float]]:
    target = min(max(float(target), 0.0), 1.0)
    max_scale = max(float(max_scale), 1.0)
    initial = _proposal_point_coverage([box for _, box in boxes], points)
    if initial >= target:
        return boxes, {"initial_coverage": initial, "final_coverage": initial, "scale": 1.0, "target": target}
    hi_boxes = [(score, _scale_proposal_box(box, max_scale)) for score, box in boxes]
    hi_coverage = _proposal_point_coverage([box for _, box in hi_boxes], points)
    if hi_coverage < target:
        return hi_boxes, {
            "initial_coverage": initial,
            "final_coverage": hi_coverage,
            "scale": max_scale,
            "target": target,
        }
    lo, hi = 1.0, max_scale
    best = hi
    best_coverage = hi_coverage
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        mid_boxes = [(score, _scale_proposal_box(box, mid)) for score, box in boxes]
        mid_coverage = _proposal_point_coverage([box for _, box in mid_boxes], points)
        if mid_coverage >= target:
            best = mid
            best_coverage = mid_coverage
            hi = mid
        else:
            lo = mid
    scaled = [(score, _scale_proposal_box(box, best)) for score, box in boxes]
    return scaled, {"initial_coverage": initial, "final_coverage": best_coverage, "scale": best, "target": target}


def _proposal_point_coverage(boxes: list[list[float]], points: np.ndarray) -> float:
    if len(boxes) == 0 or points.size == 0:
        return 0.0
    points = np.nan_to_num(np.asarray(points, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    covered = np.zeros((points.shape[0],), dtype=bool)
    for box in boxes:
        covered |= _points_inside_proposal_box(points, box)
    return float(covered.mean())


def _points_inside_proposal_box(points: np.ndarray, box: list[float]) -> np.ndarray:
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.shape[0] == 6:
        values = _sanitize_axis_box_np(values)
        center = values[:3]
        half = np.maximum(values[3:6], 1.0e-6) * 0.5
        return np.all(np.abs(points - center[None, :]) <= half[None, :] + 1.0e-9, axis=1)
    if values.shape[0] == 12:
        values = _sanitize_basis_np(values)
        center = values[:3]
        axes = values[3:12].reshape(3, 3)
        lengths = np.maximum(np.linalg.norm(axes, axis=1), 1.0e-9)
        directions = np.nan_to_num(axes / lengths[:, None], nan=0.0, posinf=0.0, neginf=0.0)
        projections = np.abs((points - center[None, :]) @ directions.T)
        return np.all(projections <= lengths[None, :] + 1.0e-9, axis=1)
    if values.shape[0] == 24:
        mn, mx = values.reshape(8, 3).min(axis=0), values.reshape(8, 3).max(axis=0)
        return np.all((points >= mn[None, :] - 1.0e-9) & (points <= mx[None, :] + 1.0e-9), axis=1)
    raise ValueError(f"unsupported proposal box dimension: {values.shape[0]}")


def _scale_proposal_box(box: list[float], scale: float) -> list[float]:
    values = np.asarray(box, dtype=np.float64).reshape(-1).copy()
    scale = max(float(scale), 1.0e-6)
    if values.shape[0] == 6:
        values = _sanitize_axis_box_np(values)
        values[3:6] *= scale
        return [float(value) for value in values.tolist()]
    if values.shape[0] == 12:
        values = _sanitize_basis_np(values)
        values[3:12] *= scale
        return [float(value) for value in values.tolist()]
    if values.shape[0] == 24:
        corners = values.reshape(8, 3)
        center = corners.mean(axis=0)
        scaled = center[None, :] + (corners - center[None, :]) * scale
        return [float(value) for value in scaled.reshape(-1).tolist()]
    raise ValueError(f"unsupported proposal box dimension: {values.shape[0]}")


def _proposal_iou(a: list[float], b: list[float]) -> float:
    a_min, a_max = _proposal_aabb(a)
    b_min, b_max = _proposal_aabb(b)
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter = float(np.prod(inter_size))
    a_vol = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    b_vol = float(np.prod(np.maximum(b_max - b_min, 0.0)))
    denom = a_vol + b_vol - inter
    if denom <= 1.0e-12:
        return 0.0
    return inter / denom


def _proposal_aabb(box: list[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.shape[0] == 6:
        center = values[:3]
        half = np.maximum(values[3:6], 1.0e-6) * 0.5
        return center - half, center + half
    if values.shape[0] == 12:
        corners = _basis_to_corners_np(_sanitize_basis_np(values))
    elif values.shape[0] == 24:
        corners = values.reshape(8, 3)
    else:
        raise ValueError(f"unsupported proposal box dimension: {values.shape[0]}")
    return corners.min(axis=0), corners.max(axis=0)


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _box_volume(box: Iterable[float]) -> float:
    values = [float(value) for value in box]
    if len(values) == 24:
        pts = np.asarray(values, dtype=float).reshape(8, 3)
        size = np.maximum(pts.max(axis=0) - pts.min(axis=0), 0.0)
        return float(size[0] * size[1] * size[2])
    if len(values) == 12:
        axes = np.asarray(values[3:12], dtype=float).reshape(3, 3)
        return float(abs(np.linalg.det(axes)) * 8.0)
    if len(values) < 6:
        return 0.0
    return max(values[3], 0.0) * max(values[4], 0.0) * max(values[5], 0.0)


def _infer_box_dim(records: list[dict[str, Any]]) -> int:
    for record in records:
        for box in record.get("boxes", []):
            if len(box) >= 24:
                return 24
            if len(box) >= 12:
                return 12
            if len(box) >= 6:
                return 6
    return 6


def _label_format_for_box_dim(box_dim: int) -> str:
    if int(box_dim) == 24:
        return "corners"
    if int(box_dim) == 12:
        return "basis"
    return "axis_aligned"


def _basis_corners_torch(boxes: Any, *, torch: Any) -> Any:
    center = boxes[..., :3]
    axes = boxes[..., 3:12].reshape(*boxes.shape[:-1], 3, 3)
    signs = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    offsets = torch.einsum("ca,...ad->...cd", signs, axes)
    return center[..., None, :] + offsets


def _basis_to_corners_np(box: np.ndarray) -> np.ndarray:
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.shape[0] != 12:
        raise ValueError("basis box params must contain 12 values")
    center = values[:3]
    axes = values[3:12].reshape(3, 3)
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return center[None, :] + signs @ axes


def _sanitize_basis_np(box: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(box, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[0] != 12:
        raise ValueError("basis box params must contain 12 values")
    center = np.clip(values[:3], -2.0, 2.0)
    axes = np.clip(values[3:12], -1.0e6, 1.0e6).reshape(3, 3)
    lengths = np.linalg.norm(axes, axis=1)
    order = np.argsort(-lengths)
    axes = axes[order]
    lengths = np.clip(lengths[order], 1.0e-4, 0.6)
    basis: list[np.ndarray] = []
    for idx in range(3):
        vector = np.asarray(axes[idx], dtype=np.float64)
        for previous in basis:
            vector = vector - previous * float(np.dot(vector, previous))
        norm = float(np.linalg.norm(vector))
        if norm < 1.0e-8:
            fallback = np.zeros(3, dtype=np.float64)
            fallback[idx] = 1.0
            for previous in basis:
                fallback = fallback - previous * float(np.dot(fallback, previous))
            norm = float(np.linalg.norm(fallback))
            vector = fallback if norm >= 1.0e-8 else np.eye(3)[idx]
        direction = vector / max(float(np.linalg.norm(vector)), 1.0e-8)
        nonzero = np.flatnonzero(np.abs(direction) > 1.0e-9)
        if nonzero.size and direction[int(nonzero[0])] < 0:
            direction *= -1.0
        basis.append(direction)
    half_axes = np.stack([direction * length for direction, length in zip(basis, lengths)], axis=0)
    return np.concatenate([center, half_axes.reshape(-1)])


def _box_corners_from_vertices(vertices: np.ndarray) -> list[float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] >= 8:
        unique = np.unique(np.round(vertices, decimals=9), axis=0)
        if unique.shape[0] >= 8:
            vertices = unique[:8]
        else:
            vertices = vertices[:8]
        center = vertices.mean(axis=0)
        order = np.lexsort((vertices[:, 2] >= center[2], vertices[:, 1] >= center[1], vertices[:, 0] >= center[0]))
        ordered = vertices[order[:8]]
        if ordered.shape[0] == 8:
            return [float(value) for value in ordered.reshape(-1).tolist()]
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    cx, cy, cz = (mn + mx) * 0.5
    sx, sy, sz = np.maximum(mx - mn, 1.0e-6)
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    corners = np.asarray(
        [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ],
        dtype=np.float64,
    )
    return [float(value) for value in corners.reshape(-1).tolist()]


def _box_basis_from_vertices(vertices: np.ndarray) -> list[float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError("cannot fit basis box from empty vertices")
    unique = np.unique(np.round(vertices, decimals=9), axis=0)
    vertices = unique if unique.shape[0] >= 4 else vertices
    center = vertices.mean(axis=0)
    centered = vertices - center
    try:
        _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        axes = eigenvectors.T
    except Exception:
        axes = np.eye(3)
    projections = centered @ axes.T
    extents = np.maximum(projections.max(axis=0) - projections.min(axis=0), 1.0e-6)
    order = np.argsort(-extents)
    axes = axes[order]
    extents = extents[order]
    half_axes = axes * (extents[:, None] * 0.5)
    for idx in range(3):
        axis = half_axes[idx]
        nonzero = np.flatnonzero(np.abs(axis) > 1.0e-9)
        if nonzero.size and axis[int(nonzero[0])] < 0:
            half_axes[idx] *= -1.0
    values = np.concatenate([center, half_axes.reshape(-1)])
    return [float(value) for value in values.tolist()]


def _manifest_meshes(cfg: dict[str, Any], stage: str) -> dict[str, list[str]]:
    manifest = workspace_path(cfg, "manifests", f"{stage}.jsonl")
    if not manifest.exists():
        return {}
    latest: dict[tuple[str, str], float] = {}
    order: list[tuple[str, str]] = []
    with manifest.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("stage") != stage or record.get("status") != "success":
                continue
            category = str(record.get("category", ""))
            mesh_id = str(record.get("mesh_id", ""))
            if not category or not mesh_id:
                continue
            key = (category, mesh_id)
            if key not in latest:
                order.append(key)
            latest[key] = max(latest.get(key, -1.0), float(record.get("finished_at", 0.0) or 0.0))
    selected: dict[str, list[str]] = {}
    for category, mesh_id in order:
        if (category, mesh_id) in latest:
            selected.setdefault(category, []).append(mesh_id)
    return selected


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Box proposal learning requires PyTorch. Install smart-bbox[pipeline].") from exc
    return torch


def _select_torch_device(torch: Any, requested: str) -> Any:
    requested = str(requested or "auto")
    if requested == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _format_network_boxes(raw_boxes: Any, *, box_dim: int, structured_basis: bool, torch: Any) -> Any:
    if int(box_dim) == 6:
        return torch.cat(
            [
                torch.tanh(raw_boxes[:, :, :3]) * 1.2,
                raw_boxes[:, :, 3:6].clamp(min=math.log(1.0e-5), max=math.log(4.0)),
            ],
            dim=2,
        )
    if int(box_dim) == 12 and structured_basis:
        return _structured_basis_torch(raw_boxes, torch=torch)
    return raw_boxes


def _network_box_dim(box_dim: int, structured_basis: bool) -> int:
    if int(box_dim) == 12 and bool(structured_basis):
        return 15
    return int(box_dim)


def _structured_basis_torch(raw_boxes: Any, *, torch: Any) -> Any:
    center = torch.tanh(raw_boxes[..., :3]) * 1.2
    if raw_boxes.shape[-1] >= 15:
        lengths = torch.sigmoid(raw_boxes[..., 3:6]) * 0.6
        raw_axes = raw_boxes[..., 6:15].reshape(*raw_boxes.shape[:-1], 3, 3)
    else:
        raw_axes = raw_boxes[..., 3:12].reshape(*raw_boxes.shape[:-1], 3, 3)
        raw_lengths = torch.linalg.norm(raw_axes, dim=-1).clamp(min=1.0e-6)
        lengths = torch.sigmoid(raw_lengths) * 0.6
    unit_axes = []
    for idx in range(3):
        axis = raw_axes[..., idx, :]
        for previous in unit_axes:
            axis = axis - previous * (axis * previous).sum(dim=-1, keepdim=True)
        norm = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1.0e-6)
        unit_axes.append(axis / norm)
    axes = torch.stack(unit_axes, dim=-2) * lengths[..., :, None]
    return torch.cat([center, axes.reshape(*raw_boxes.shape[:-1], 9)], dim=-1)


def _sanitize_axis_box_np(box: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(box, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[0] != 6:
        raise ValueError("axis-aligned box params must contain 6 values")
    center = np.clip(values[:3], -2.0, 2.0)
    size = np.clip(values[3:6], 1.0e-5, 4.0)
    return np.concatenate([center, size])


def _box_model_type_for_architecture(architecture: str) -> str:
    if str(architecture) == "query_pointnet":
        return "query_pointnet_box_proposal_v1"
    if str(architecture) == "global_pointnet":
        return "pointnet_box_proposal_v0"
    raise ValueError(f"unsupported box proposal architecture: {architecture!r}")


def _architecture_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    if checkpoint.get("architecture"):
        return str(checkpoint["architecture"])
    model_type = str(checkpoint.get("model_type", "pointnet_box_proposal_v0"))
    if model_type == "query_pointnet_box_proposal_v1":
        return "query_pointnet"
    return "global_pointnet"


class _PointNetBoxProposal:
    def __new__(
        cls,
        *,
        max_boxes: int,
        box_dim: int,
        network_box_dim: int | None = None,
        num_categories: int,
        hidden_size: int,
        architecture: str = "global_pointnet",
        structured_basis: bool = False,
        torch: Any,
    ):
        nn = torch.nn
        network_box_dim = int(network_box_dim or box_dim)
        if str(architecture) == "query_pointnet":

            class QueryPointNetBoxProposal(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.max_boxes = int(max_boxes)
                    self.box_dim = int(box_dim)
                    self.network_box_dim = int(network_box_dim)
                    self.point_mlp = nn.Sequential(
                        nn.Linear(3, 64),
                        nn.ReLU(),
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, int(hidden_size)),
                        nn.ReLU(),
                    )
                    self.category_embedding = nn.Embedding(max(int(num_categories), 1), 16)
                    self.slot_embedding = nn.Embedding(self.max_boxes, 32)
                    self.decoder = nn.Sequential(
                        nn.Linear(int(hidden_size) + 16 + 32, int(hidden_size)),
                        nn.ReLU(),
                        nn.Linear(int(hidden_size), int(hidden_size)),
                        nn.ReLU(),
                        nn.Linear(int(hidden_size), self.network_box_dim + 1),
                    )

                def forward(self, points: Any, category_ids: Any) -> tuple[Any, Any]:
                    features = self.point_mlp(points)
                    global_feature = features.max(dim=1).values
                    category_feature = self.category_embedding(category_ids)
                    batch_size = points.shape[0]
                    slot_ids = torch.arange(self.max_boxes, device=points.device)
                    slot_feature = self.slot_embedding(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
                    global_feature = global_feature.unsqueeze(1).expand(-1, self.max_boxes, -1)
                    category_feature = category_feature.unsqueeze(1).expand(-1, self.max_boxes, -1)
                    raw = self.decoder(torch.cat([global_feature, category_feature, slot_feature], dim=2))
                    boxes = _format_network_boxes(
                        raw[:, :, : self.network_box_dim],
                        box_dim=self.box_dim,
                        structured_basis=bool(structured_basis),
                        torch=torch,
                    )
                    logits = raw[:, :, self.network_box_dim]
                    return boxes, logits

            return QueryPointNetBoxProposal()
        if str(architecture) != "global_pointnet":
            raise ValueError(f"unsupported box proposal architecture: {architecture!r}")

        class PointNetBoxProposal(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.max_boxes = int(max_boxes)
                self.box_dim = int(box_dim)
                self.network_box_dim = int(network_box_dim)
                self.point_mlp = nn.Sequential(
                    nn.Linear(3, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, int(hidden_size)),
                    nn.ReLU(),
                )
                self.category_embedding = nn.Embedding(max(int(num_categories), 1), 16)
                self.head = nn.Sequential(
                    nn.Linear(int(hidden_size) + 16, int(hidden_size)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_size), self.max_boxes * (self.network_box_dim + 1)),
                )

            def forward(self, points: Any, category_ids: Any) -> tuple[Any, Any, Any]:
                features = self.point_mlp(points)
                global_feature = features.max(dim=1).values
                category_feature = self.category_embedding(category_ids)
                raw = self.head(torch.cat([global_feature, category_feature], dim=1))
                raw = raw.view(points.shape[0], self.max_boxes, self.network_box_dim + 1)
                boxes = _format_network_boxes(
                    raw[:, :, : self.network_box_dim],
                    box_dim=self.box_dim,
                    structured_basis=bool(structured_basis),
                    torch=torch,
                )
                logits = raw[:, :, self.network_box_dim]
                return boxes, logits

        return PointNetBoxProposal()
