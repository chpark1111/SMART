from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import trimesh.repair
import trimesh.sample

from .pipeline.config import REPO_ROOT, workspace_path
from .pipeline.stages import (
    bbox_dir_for_render,
    list_mesh_ids,
    mesh_tetra_dir,
    normalized_mesh_path,
    stage_root,
)


@dataclass
class EvaluationRecord:
    category: str
    mesh_id: str
    stage: str
    status: str
    elapsed_sec: float
    num_box: int = 0
    BVS: float | None = None
    MOV: float | None = None
    TOV: float | None = None
    Covered: float | None = None
    vIoU: float | None = None
    cub_CD: float | None = None
    mesh_path: str | None = None
    surface_path: str | None = None
    bbox_path: str | None = None
    error: str | None = None


def evaluate_config(
    cfg: dict[str, Any],
    *,
    stage: str = "mcts",
    category_name: str | None = None,
    meshes: list[str] | None = None,
    chamfer_points: int = 2048,
    output_path: str | Path | None = None,
    from_manifest: bool = False,
) -> dict[str, Any]:
    records: list[EvaluationRecord] = []
    manifest_meshes = _manifest_meshes(cfg, stage, category_name=category_name, meshes=meshes) if from_manifest else None
    for category in cfg.get("categories", []):
        if category_name and category["name"] != category_name:
            continue
        if manifest_meshes is not None:
            mesh_ids = manifest_meshes.get(str(category["name"]), [])
        else:
            mesh_ids = list_mesh_ids(category, explicit=meshes)
        for mesh_id in mesh_ids:
            records.append(
                evaluate_mesh(
                    cfg,
                    category,
                    mesh_id,
                    stage=stage,
                    chamfer_points=chamfer_points,
                )
            )

    summary = summarize_records(records)
    payload = {
        "stage": stage,
        "chamfer_points": chamfer_points,
        "from_manifest": from_manifest,
        "summary": summary,
        "records": [asdict(record) for record in records],
    }
    out = Path(output_path) if output_path is not None else workspace_path(cfg, "evaluation", f"{stage}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["output_path"] = str(out)
    return payload


def _manifest_meshes(
    cfg: dict[str, Any],
    stage: str,
    *,
    category_name: str | None,
    meshes: list[str] | None,
) -> dict[str, list[str]]:
    manifest = workspace_path(cfg, "manifests", f"{stage}.jsonl")
    if not manifest.exists():
        return {}
    allowed_meshes = set(meshes or [])
    configured = {str(category["name"]) for category in cfg.get("categories", [])}
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
            category = str(record.get("category") or "")
            mesh_id = str(record.get("mesh_id") or "")
            if not category or not mesh_id:
                continue
            if category not in configured:
                continue
            if category_name and category != category_name:
                continue
            if allowed_meshes and mesh_id not in allowed_meshes:
                continue
            output_path = record.get("output_path")
            if not output_path or not Path(str(output_path)).exists():
                continue
            key = (category, mesh_id)
            finished_at = float(record.get("finished_at") or 0.0)
            if key not in latest:
                order.append(key)
            if finished_at >= latest.get(key, -1.0):
                latest[key] = finished_at
    selected: dict[str, list[str]] = {}
    for category, mesh_id in order:
        if (category, mesh_id) in latest:
            selected.setdefault(category, []).append(mesh_id)
    return selected


def evaluate_mesh(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    stage: str,
    chamfer_points: int = 2048,
) -> EvaluationRecord:
    started = time.time()
    source_mesh = normalized_mesh_path(cfg, category, mesh_id)
    if not source_mesh.exists():
        source_mesh = Path(category["mesh_root"]) / mesh_id / "model.obj"
        if not source_mesh.is_absolute():
            source_mesh = REPO_ROOT / source_mesh

    surface_path = mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh__sf.obj"
    bbox_path = bbox_dir_for_render(cfg, category, mesh_id, stage)
    if bbox_path is None:
        bbox_path = latest_evaluation_bbox_dir(cfg, category, mesh_id, stage)

    return evaluate_bbox_dir(
        cfg,
        category,
        mesh_id,
        bbox_path=bbox_path,
        stage=stage,
        chamfer_points=chamfer_points,
    )


def evaluate_bbox_dir(
    cfg: dict[str, Any],
    category: dict[str, Any],
    mesh_id: str,
    *,
    bbox_path: str | Path | None,
    stage: str = "bbox_path",
    chamfer_points: int = 2048,
) -> EvaluationRecord:
    """Evaluate a specific bbox directory with the paper metrics."""

    started = time.time()
    source_mesh = normalized_mesh_path(cfg, category, mesh_id)
    if not source_mesh.exists():
        source_mesh = Path(category["mesh_root"]) / mesh_id / "model.obj"
        if not source_mesh.is_absolute():
            source_mesh = REPO_ROOT / source_mesh

    surface_path = mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh__sf.obj"
    bbox_dir = Path(bbox_path) if bbox_path is not None else None

    record = EvaluationRecord(
        category=str(category["name"]),
        mesh_id=mesh_id,
        stage=stage,
        status="failed",
        elapsed_sec=0.0,
        mesh_path=str(source_mesh),
        surface_path=str(surface_path),
        bbox_path=str(bbox_dir) if bbox_dir is not None else None,
    )

    try:
        if bbox_dir is None or not bbox_dir.exists():
            raise FileNotFoundError(f"No bbox result found for stage={stage} mesh={mesh_id}")
        if not surface_path.exists():
            raise FileNotFoundError(f"Missing tetra surface OBJ: {surface_path}")
        if not source_mesh.exists():
            raise FileNotFoundError(f"Missing source mesh OBJ: {source_mesh}")

        source = _load_mesh(source_mesh)
        surface = _load_mesh(surface_path)
        boxes = _load_bbox_meshes(bbox_dir)
        metrics = EvaluationMetrics(source, surface, boxes).compute(chamfer_points=chamfer_points)

        record.status = "success"
        record.num_box = int(metrics["num_box"])
        record.BVS = float(metrics["BVS"])
        record.MOV = float(metrics["MOV"])
        record.TOV = float(metrics["TOV"])
        record.Covered = float(metrics["Covered"])
        record.vIoU = float(metrics["vIoU"])
        record.cub_CD = float(metrics["cub_CD"])
    except Exception as exc:
        record.error = str(exc)
    finally:
        record.elapsed_sec = time.time() - started
    return record


def latest_evaluation_bbox_dir(
    cfg: dict[str, Any], category: dict[str, Any], mesh_id: str, stage: str
) -> Path | None:
    return bbox_dir_for_render(cfg, category, mesh_id, stage) or bbox_dir_for_render(
        cfg, category, mesh_id, str(stage_root(cfg, stage, category))
    )


def summarize_records(records: list[EvaluationRecord]) -> dict[str, Any]:
    successful = [record for record in records if record.status == "success"]
    summary: dict[str, Any] = {
        "total": len(records),
        "success": len(successful),
        "failed": len(records) - len(successful),
    }
    for key in ["num_box", "BVS", "MOV", "TOV", "Covered", "vIoU", "cub_CD", "elapsed_sec"]:
        values = [getattr(record, key) for record in successful]
        numeric = [float(value) for value in values if value is not None]
        summary[f"Avg_{key}"] = float(np.mean(numeric)) if numeric else None
    return summary


class EvaluationMetrics:
    """Adapter for the metric definitions in past_codes/Evaluation/utils/evaluator.py."""

    def __init__(
        self,
        shapenet_mesh: trimesh.Trimesh,
        surface_mesh: trimesh.Trimesh,
        bbox_meshes: list[trimesh.Trimesh],
    ) -> None:
        _seed_everything()
        if not bbox_meshes:
            raise ValueError("No bbox OBJ files found")

        self.shapenet_mesh = shapenet_mesh
        self.mesh = surface_mesh
        self.bbox_meshes = bbox_meshes

        trimesh.repair.fix_normals(self.mesh)
        for bbox in self.bbox_meshes:
            trimesh.repair.fix_normals(bbox)

        self.volume_sum = float(self.mesh.volume)
        if self.volume_sum <= 0:
            raise ValueError("Surface mesh has non-positive volume")

        pymanifold = _load_pymanifold()
        self.man = _to_manifold(pymanifold, self.mesh)
        self.bbox_manifolds = [_to_manifold(pymanifold, bbox) for bbox in self.bbox_meshes]

    def compute(self, *, chamfer_points: int) -> dict[str, float]:
        covered = self.covered()
        return {
            "num_box": float(len(self.bbox_meshes)),
            "BVS": self.bvs(),
            "MOV": self.mov(),
            "Covered": covered,
            "TOV": self.easy_tov() if covered >= 0.99 else self.tov(),
            "vIoU": self.viou(),
            "cub_CD": self.cub_cd(num_points=chamfer_points),
        }

    def bvs(self) -> float:
        return float(sum(bbox.volume for bbox in self.bbox_meshes) / self.volume_sum)

    def mov(self) -> float:
        ret = 0.0
        for bbox_man in self.bbox_manifolds:
            part_vol = _volume_from_manifold(bbox_man - self.man)
            occ_vol = _volume_from_manifold(bbox_man ^ self.man)
            part_ov = 0.0 if occ_vol < 1e-10 else float(part_vol / occ_vol)
            ret = max(ret, part_ov)
        return float(ret)

    def covered(self) -> float:
        merged = self._merged_bbox_manifold()
        cov_vol = _volume_from_manifold(self.man - merged)
        return float(1 - max(float(cov_vol / self.volume_sum), 0.0))

    def easy_tov(self) -> float:
        merged_vol = _volume_from_manifold(self._merged_bbox_manifold())
        return float((merged_vol - self.volume_sum) / self.volume_sum)

    def tov(self) -> float:
        tov_vol = _volume_from_manifold(self._merged_bbox_manifold() - self.man)
        return float(tov_vol / self.volume_sum)

    def viou(self) -> float:
        merged = self._merged_bbox_manifold()
        int_volume = _volume_from_manifold(merged ^ self.man)
        union_volume = _volume_from_manifold(merged + self.man)
        if union_volume <= 0:
            return 0.0
        return float(int_volume / union_volume)

    def cub_cd(self, *, num_points: int = 2048) -> float:
        if num_points <= 0:
            return 0.0
        np_random_state_raw = np.random.get_state()
        np_random_state = (
            np_random_state_raw[0],
            np_random_state_raw[1].copy(),
            np_random_state_raw[2],
            np_random_state_raw[3],
            np_random_state_raw[4],
        )
        py_random_state = random.getstate()
        try:
            np.random.seed(0)
            random.seed(0)
            bbox_points = []
            per_box = [num_points // len(self.bbox_meshes)] * len(self.bbox_meshes)
            per_box[-1] += num_points - sum(per_box)
            for bbox, count in zip(self.bbox_meshes, per_box):
                if count:
                    bbox_points.append(trimesh.sample.sample_surface(bbox, count)[0])
            source_points = trimesh.sample.sample_surface(self.shapenet_mesh, num_points)[0]
            return _symmetric_chamfer(np.concatenate(bbox_points), source_points)
        finally:
            np.random.set_state(np_random_state)
            random.setstate(py_random_state)

    def _merged_bbox_manifold(self):
        merged = self.bbox_manifolds[0]
        for bbox_man in self.bbox_manifolds[1:]:
            merged = merged + bbox_man
        return merged


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh at {path}, got {type(mesh).__name__}")
    return mesh


def _load_bbox_meshes(path: Path) -> list[trimesh.Trimesh]:
    files = sorted(path.glob("bbox*.obj"), key=_bbox_sort_key)
    return [_load_mesh(file) for file in files]


def _bbox_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem.replace("bbox", "", 1)
    try:
        return int(stem), path.name
    except ValueError:
        return 10**9, path.name


def _load_pymanifold():
    manifold_python = os.environ.get("SMART_MANIFOLD_PYTHON")
    candidates = []
    if manifold_python:
        candidates.append(Path(manifold_python).expanduser())
    candidates.append(REPO_ROOT / "smart" / "pymanifold_runtime")
    candidates.append(REPO_ROOT / "smart" / "vendor" / "manifold" / "build" / "bindings" / "python")
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    import pymanifold  # type: ignore

    return pymanifold


def _to_manifold(pymanifold_module: Any, mesh: trimesh.Trimesh):
    manmesh = pymanifold_module.Mesh(
        vert_pos=np.array(mesh.vertices), tri_verts=np.array(mesh.faces)
    )
    man = pymanifold_module.Manifold()
    return man.from_mesh(manmesh)


def _mesh_from_manifold(manifold: Any) -> trimesh.Trimesh:
    mesh = manifold.to_mesh()
    trimsh = trimesh.Trimesh(vertices=mesh.vert_pos, faces=mesh.tri_verts, process=False)
    trimesh.repair.fix_normals(trimsh)
    return trimsh


def _volume_from_manifold(manifold: Any) -> float:
    mesh = manifold.to_mesh()
    vertices = np.asarray(mesh.vert_pos, dtype=np.float64)
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return _triangle_mesh_volume(vertices, faces)


def _triangle_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(vertices) == 0 or len(faces) == 0:
        return 0.0
    triangles = vertices[faces]
    vectors = triangles[:, 1:, :] - triangles[:, :2, :]
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    f1 = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]
    return float(np.sum(crosses[:, 0] * f1[:, 0]) / 6.0)


def _symmetric_chamfer(left: np.ndarray, right: np.ndarray) -> float:
    try:
        import smart.native as smart_native

        if smart_native.native_core_available():
            return smart_native.symmetric_chamfer(left.tolist(), right.tolist())
    except Exception:
        pass

    try:
        from scipy.spatial import cKDTree

        left_tree = cKDTree(left)
        right_tree = cKDTree(right)
        right_to_left = left_tree.query(right, k=1)[0]
        left_to_right = right_tree.query(left, k=1)[0]
        return float(np.mean(right_to_left**2) + np.mean(left_to_right**2))
    except Exception:
        left_diff = left[:, None, :] - right[None, :, :]
        distances = np.sum(left_diff * left_diff, axis=2)
        return float(np.mean(np.min(distances, axis=0)) + np.mean(np.min(distances, axis=1)))


def _seed_everything() -> None:
    seed = 7777
    random.seed(seed)
    np.random.seed(seed)
