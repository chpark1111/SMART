from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT_ON_DISK = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_ON_DISK) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_ON_DISK))

import numpy as np
import pymesh
import trimesh
import trimesh.repair
from scipy.spatial import ConvexHull

import smart.rust as smart_rust
from smart.evaluation import EvaluationMetrics, _load_bbox_meshes, _load_mesh
from smart.pipeline.config import REPO_ROOT, load_config
from smart.pipeline.stages import bbox_dir_for_render, mesh_tetra_dir, normalized_mesh_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare exact tetrahedron-box clipping volumes against current Manifold metrics."
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument("--category", default="table")
    parser.add_argument("--mesh", default="1692563658149377630047043c6a0c50")
    parser.add_argument("--stage", default="mcts", choices=["merge", "refine", "mcts"])
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=8,
        help="Refuse exact inclusion-exclusion above this many boxes.",
    )
    parser.add_argument(
        "--max-tets",
        type=int,
        default=0,
        help="Debug only: limit tetrahedra. Default 0 uses all tets.",
    )
    parser.add_argument("--chamfer-points", type=int, default=512)
    parser.add_argument("--backend", choices=["auto", "python", "rust"], default="auto")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    started = time.time()
    cfg = load_config(args.config)
    category = _find_category(cfg, args.category)
    source_path = normalized_mesh_path(cfg, category, args.mesh)
    if not source_path.exists():
        source_path = Path(category["mesh_root"]) / args.mesh / "model.obj"
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
    tetra_path = mesh_tetra_dir(cfg, category, args.mesh) / "tetra.msh"
    surface_path = mesh_tetra_dir(cfg, category, args.mesh) / "tetra.msh__sf.obj"
    bbox_path = bbox_dir_for_render(cfg, category, args.mesh, args.stage)
    if bbox_path is None or not bbox_path.exists():
        raise FileNotFoundError(f"No bbox path found for stage={args.stage} mesh={args.mesh}")

    source_mesh = _load_mesh(source_path)
    surface_mesh = _load_mesh(surface_path)
    bbox_meshes = _load_bbox_meshes(bbox_path)
    bbox_volume_diagnostics = _bbox_volume_diagnostics(bbox_meshes)
    manifold_metrics = EvaluationMetrics(source_mesh, surface_mesh, bbox_meshes).compute(
        chamfer_points=args.chamfer_points
    )

    use_rust = args.backend == "rust" or (args.backend == "auto" and smart_rust.using_rust())
    if use_rust:
        clipping_metrics = rust_tet_clipping_metrics(
            tetra_path=tetra_path,
            surface_mesh=surface_mesh,
            bbox_meshes=bbox_meshes,
            max_boxes=args.max_boxes,
            max_tets=args.max_tets,
        )
    else:
        clipping_metrics = tet_clipping_metrics(
            tetra_path=tetra_path,
            surface_mesh=surface_mesh,
            bbox_meshes=bbox_meshes,
            max_boxes=args.max_boxes,
            max_tets=args.max_tets,
        )
    elapsed = time.time() - started
    diffs = {
        key: abs(float(manifold_metrics[key]) - float(clipping_metrics[key]))
        for key in ("BVS", "MOV", "TOV", "Covered", "vIoU")
    }
    payload: dict[str, Any] = {
        "config": args.config,
        "category": args.category,
        "mesh": args.mesh,
        "stage": args.stage,
        "tetra_path": str(tetra_path),
        "surface_path": str(surface_path),
        "bbox_path": str(bbox_path),
        "num_tets": clipping_metrics.pop("_num_tets"),
        "num_tets_used": clipping_metrics.pop("_num_tets_used"),
        "num_boxes": len(bbox_meshes),
        "clipping_backend": "rust" if use_rust else "python",
        "bbox_volume_diagnostics": bbox_volume_diagnostics,
        "max_bbox_mesh_hull_abs_diff": max(
            [row["abs_diff"] for row in bbox_volume_diagnostics], default=0.0
        ),
        "manifold": manifold_metrics,
        "tet_clipping": clipping_metrics,
        "abs_diff": diffs,
        "elapsed_sec": elapsed,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def rust_tet_clipping_metrics(
    *,
    tetra_path: Path,
    surface_mesh: trimesh.Trimesh,
    bbox_meshes: list[trimesh.Trimesh],
    max_boxes: int,
    max_tets: int = 0,
) -> dict[str, float]:
    if max_tets > 0:
        raise ValueError("Rust backend does not support --max-tets debug slicing")
    tetmesh = pymesh.load_mesh(tetra_path)
    metrics = smart_rust.tet_clipping_metrics(
        tetmesh.vertices.tolist(),
        tetmesh.voxels.tolist(),
        [np.asarray(mesh.vertices, dtype=float).tolist() for mesh in bbox_meshes],
        float(surface_mesh.volume),
        max_boxes=max_boxes,
        box_volumes=[float(mesh.volume) for mesh in bbox_meshes],
    )
    metrics["_num_tets"] = float(len(tetmesh.voxels))
    metrics["_num_tets_used"] = float(len(tetmesh.voxels))
    return metrics


def _bbox_volume_diagnostics(bbox_meshes: list[trimesh.Trimesh]) -> list[dict[str, float]]:
    diagnostics = []
    for idx, mesh in enumerate(bbox_meshes):
        mesh_volume = float(mesh.volume)
        try:
            hull_volume = float(ConvexHull(np.asarray(mesh.vertices, dtype=float)).volume)
        except Exception:
            hull_volume = 0.0
        abs_diff = abs(mesh_volume - hull_volume)
        rel_diff = 0.0 if abs(mesh_volume) <= 1e-12 else abs_diff / abs(mesh_volume)
        diagnostics.append(
            {
                "idx": float(idx),
                "mesh_volume": mesh_volume,
                "convex_hull_volume": hull_volume,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )
    return diagnostics


def tet_clipping_metrics(
    *,
    tetra_path: Path,
    surface_mesh: trimesh.Trimesh,
    bbox_meshes: list[trimesh.Trimesh],
    max_boxes: int,
    max_tets: int = 0,
) -> dict[str, float]:
    if not bbox_meshes:
        raise ValueError("No bbox meshes found")
    if len(bbox_meshes) > max_boxes:
        raise ValueError(
            f"Exact box union inclusion-exclusion has {len(bbox_meshes)} boxes; "
            f"increase --max-boxes if this is intentional."
        )

    tetmesh = pymesh.load_mesh(tetra_path)
    vertices = np.asarray(tetmesh.vertices, dtype=float)
    voxels = np.asarray(tetmesh.voxels, dtype=int)
    if max_tets > 0:
        voxels_used = voxels[:max_tets]
    else:
        voxels_used = voxels

    surface_volume = float(surface_mesh.volume)
    tet_volume_sum = float(tetmesh.volume)

    box_infos = [_convex_info(np.asarray(mesh.vertices, dtype=float)) for mesh in bbox_meshes]
    box_volumes = [float(mesh.volume) for mesh in bbox_meshes]
    box_union_volume = _union_volume(box_infos)
    per_box_shape_intersections = [0.0 for _ in box_infos]
    shape_box_union_intersection = 0.0

    for tet_indices in voxels_used:
        tet_points = vertices[tet_indices]
        tet_info = _convex_info(tet_points)
        overlapping = [
            idx
            for idx, box_info in enumerate(box_infos)
            if _aabb_overlap(tet_info["aabb"], box_info["aabb"])
        ]
        if not overlapping:
            continue

        single_intersections: dict[int, float] = {}
        for idx in overlapping:
            volume = _intersection_volume([tet_info, box_infos[idx]])
            single_intersections[idx] = volume
            per_box_shape_intersections[idx] += volume

        local_single_cache = {
            local_idx: single_intersections[box_idx]
            for local_idx, box_idx in enumerate(overlapping)
        }
        shape_box_union_intersection += _union_volume(
            [box_infos[idx] for idx in overlapping],
            base_info=tet_info,
            single_cache=local_single_cache,
        )

    mov = 0.0
    for box_volume, intersection in zip(box_volumes, per_box_shape_intersections):
        if intersection > 1e-10:
            mov = max(mov, (box_volume - intersection) / intersection)

    covered = shape_box_union_intersection / surface_volume
    outside_box_volume = max(box_union_volume - shape_box_union_intersection, 0.0)
    if covered >= 0.99:
        tov = (box_union_volume - surface_volume) / surface_volume
    else:
        tov = outside_box_volume / surface_volume
    union_volume = surface_volume + box_union_volume - shape_box_union_intersection
    viou = 0.0 if union_volume <= 0.0 else shape_box_union_intersection / union_volume

    return {
        "num_box": float(len(bbox_meshes)),
        "BVS": float(sum(box_volumes) / surface_volume),
        "MOV": float(mov),
        "Covered": float(covered),
        "TOV": float(tov),
        "vIoU": float(viou),
        "tet_volume_sum": float(tet_volume_sum),
        "surface_volume": float(surface_volume),
        "box_union_volume": float(box_union_volume),
        "shape_box_union_intersection": float(shape_box_union_intersection),
        "_num_tets": float(len(voxels)),
        "_num_tets_used": float(len(voxels_used)),
    }


def _union_volume(
    infos: list[dict[str, np.ndarray]],
    *,
    base_info: dict[str, np.ndarray] | None = None,
    single_cache: dict[int, float] | None = None,
) -> float:
    total = 0.0
    indexed_infos = list(enumerate(infos))
    for size in range(1, len(indexed_infos) + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for subset in itertools.combinations(indexed_infos, size):
            subset_indices = [idx for idx, _ in subset]
            subset_infos = [info for _, info in subset]
            if base_info is not None:
                if not all(_aabb_overlap(base_info["aabb"], info["aabb"]) for info in subset_infos):
                    continue
                volume = (
                    single_cache[subset_indices[0]]
                    if single_cache is not None and size == 1 and subset_indices[0] in single_cache
                    else _intersection_volume([base_info, *subset_infos])
                )
            else:
                volume = _intersection_volume(subset_infos)
            total += sign * volume
    return max(total, 0.0)


def _intersection_volume(infos: list[dict[str, np.ndarray]]) -> float:
    if not infos:
        return 0.0
    if not _all_aabb_overlap([info["aabb"] for info in infos]):
        return 0.0
    planes = np.concatenate([info["planes"] for info in infos], axis=0)
    points = _halfspace_vertices(planes)
    if len(points) < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 0.0


def _convex_info(points: np.ndarray) -> dict[str, np.ndarray]:
    hull = ConvexHull(points)
    planes = _unique_planes(np.asarray(hull.equations, dtype=float))
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return {"planes": planes, "aabb": np.stack([mins, maxs])}


def _unique_planes(planes: np.ndarray) -> np.ndarray:
    out: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for plane in planes:
        normal = plane[:3]
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            continue
        normalized = plane / norm
        key = tuple(np.round(normalized, 10))
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return np.asarray(out, dtype=float)


def _halfspace_vertices(planes: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    vertices = []
    for i, j, k in itertools.combinations(range(len(planes)), 3):
        a = np.stack([planes[i, :3], planes[j, :3], planes[k, :3]])
        det = float(np.linalg.det(a))
        if abs(det) < 1e-12:
            continue
        b = -np.array([planes[i, 3], planes[j, 3], planes[k, 3]])
        point = np.linalg.solve(a, b)
        if np.all(planes[:, :3] @ point + planes[:, 3] <= tol):
            vertices.append(point)
    if not vertices:
        return np.zeros((0, 3), dtype=float)
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for point in vertices:
        unique[tuple(np.round(point, 10))] = point
    return np.asarray(list(unique.values()), dtype=float)


def _aabb_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left[0] <= right[1] + 1e-12) and np.all(right[0] <= left[1] + 1e-12))


def _all_aabb_overlap(aabbs: list[np.ndarray]) -> bool:
    mins = np.max([aabb[0] for aabb in aabbs], axis=0)
    maxs = np.min([aabb[1] for aabb in aabbs], axis=0)
    return bool(np.all(mins <= maxs + 1e-12))


def _find_category(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for category in cfg.get("categories", []):
        if category["name"] == name:
            return category
    raise KeyError(f"Unknown category: {name}")


if __name__ == "__main__":
    raise SystemExit(main())
