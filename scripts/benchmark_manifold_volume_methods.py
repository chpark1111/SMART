from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import smart.rust as smart_rust  # noqa: E402
from smart.pipeline.config import load_config  # noqa: E402
from smart.pipeline.stages import list_mesh_ids, mesh_tetra_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Manifold residual volume through GetMesh() and "
            "GetProperties().volume without changing SMART rewards."
        )
    )
    parser.add_argument("--config", default="configs/expanded_processed_16.yaml")
    parser.add_argument("--category", default="")
    parser.add_argument("--mesh", action="append", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--output",
        default="runs/bench_exact/manifold_volume_methods.json",
    )
    args = parser.parse_args()

    if not smart_rust.using_rust() or not smart_rust.manifold_bridge_available():
        raise SystemExit("smart._rust Manifold bridge is not available")

    cfg = load_config(args.config)
    targets = _targets(cfg, args.category, set(args.mesh), args.limit)
    results: dict[str, Any] = {
        "config": args.config,
        "repeat": args.repeat,
        "targets": [],
        "summary": {},
    }

    for category, mesh_id, surface_path in targets:
        record = _benchmark_target(category, mesh_id, surface_path, args.repeat)
        results["targets"].append(record)

    results["summary"] = _summary(results["targets"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2, sort_keys=True))
    print(f"wrote {output}")


def _targets(
    cfg: dict[str, Any], category_filter: str, mesh_filter: set[str], limit: int
) -> list[tuple[str, str, Path]]:
    targets: list[tuple[str, str, Path]] = []
    for category in cfg.get("categories", []):
        category_name = str(category["name"])
        if category_filter and category_name != category_filter:
            continue
        for mesh_id in list_mesh_ids(category):
            if mesh_filter and mesh_id not in mesh_filter:
                continue
            surface_path = mesh_tetra_dir(cfg, category, mesh_id) / "tetra.msh__sf.obj"
            if not surface_path.exists():
                continue
            targets.append((category_name, mesh_id, surface_path))
            if limit > 0 and len(targets) >= limit:
                return targets
    return targets


def _benchmark_target(
    category: str, mesh_id: str, surface_path: Path, repeat: int
) -> dict[str, Any]:
    mesh = _load_mesh(surface_path)
    bridge = smart_rust.ManifoldBridgeMesh(
        mesh.vertices.astype(float).tolist(),
        mesh.faces.astype(int).tolist(),
    )
    probes = _probe_bounds(mesh.bounds)

    record: dict[str, Any] = {
        "category": category,
        "mesh_id": mesh_id,
        "surface_path": str(surface_path),
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "probes": [],
    }

    for name, bounds in probes:
        rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in bounds]
        pair_mesh, pair_properties = bridge.residual_volume_for_box_params_pair(
            bounds, rotations
        )
        mesh_times, mesh_values = _time_repeated(
            lambda: bridge.residual_volume_for_box_params(bounds, rotations), repeat
        )
        properties_times, properties_values = _time_repeated(
            lambda: bridge.residual_volume_for_box_params_properties(bounds, rotations),
            repeat,
        )
        mesh_value = mesh_values[-1]
        properties_value = properties_values[-1]
        abs_diff = abs(mesh_value - properties_value)
        rel_diff = abs_diff / max(1.0, abs(mesh_value), abs(properties_value))
        record["probes"].append(
            {
                "name": name,
                "num_boxes": len(bounds),
                "pair_mesh_volume": pair_mesh,
                "pair_properties_volume": pair_properties,
                "mesh_volume": mesh_value,
                "properties_volume": properties_value,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "pair_abs_diff": abs(pair_mesh - pair_properties),
                "mesh_mean_sec": statistics.fmean(mesh_times),
                "properties_mean_sec": statistics.fmean(properties_times),
                "speedup_properties_vs_mesh": (
                    statistics.fmean(mesh_times) / statistics.fmean(properties_times)
                    if statistics.fmean(properties_times) > 0.0
                    else None
                ),
            }
        )
    return record


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), file_type="obj", force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"no mesh geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh for {path}, got {type(loaded)!r}")
    return loaded


def _probe_bounds(mesh_bounds: Any) -> list[tuple[str, list[list[float]]]]:
    minv = [float(value) for value in mesh_bounds[0]]
    maxv = [float(value) for value in mesh_bounds[1]]
    center = [(lo + hi) * 0.5 for lo, hi in zip(minv, maxv)]
    span = [hi - lo for lo, hi in zip(minv, maxv)]
    center_box = [
        center[0] - span[0] * 0.30,
        center[1] - span[1] * 0.30,
        center[2] - span[2] * 0.30,
        center[0] + span[0] * 0.30,
        center[1] + span[1] * 0.30,
        center[2] + span[2] * 0.30,
    ]
    left_x = [minv[0], minv[1], minv[2], center[0], maxv[1], maxv[2]]
    right_x = [center[0], minv[1], minv[2], maxv[0], maxv[1], maxv[2]]
    low_y = [minv[0], minv[1], minv[2], maxv[0], center[1], maxv[2]]
    high_y = [minv[0], center[1], minv[2], maxv[0], maxv[1], maxv[2]]
    full = [minv[0], minv[1], minv[2], maxv[0], maxv[1], maxv[2]]
    return [
        ("full_aabb", [full]),
        ("center_60pct", [center_box]),
        ("left_half_x", [left_x]),
        ("two_x_halves", [left_x, right_x]),
        ("two_y_halves", [low_y, high_y]),
    ]


def _time_repeated(callback: Any, repeat: int) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        values.append(float(callback()))
        times.append(time.perf_counter() - start)
    return times, values


def _summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    probes = [probe for target in targets for probe in target["probes"]]
    if not probes:
        return {"num_targets": len(targets), "num_probes": 0}
    speedups = [
        float(probe["speedup_properties_vs_mesh"])
        for probe in probes
        if probe["speedup_properties_vs_mesh"] is not None
    ]
    return {
        "num_targets": len(targets),
        "num_probes": len(probes),
        "max_abs_diff": max(float(probe["abs_diff"]) for probe in probes),
        "max_rel_diff": max(float(probe["rel_diff"]) for probe in probes),
        "max_pair_abs_diff": max(float(probe["pair_abs_diff"]) for probe in probes),
        "mean_mesh_sec": statistics.fmean(float(probe["mesh_mean_sec"]) for probe in probes),
        "mean_properties_sec": statistics.fmean(
            float(probe["properties_mean_sec"]) for probe in probes
        ),
        "mean_speedup_properties_vs_mesh": statistics.fmean(speedups) if speedups else None,
    }


if __name__ == "__main__":
    main()
