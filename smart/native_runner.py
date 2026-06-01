from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import smart.native as smart_native
from smart.action_prior import load_action_prior
from smart.native_executable import native_executable_path, run_native_command


IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def cpp_native_file_runner_available() -> bool:
    if native_executable_path() is not None:
        return True
    return bool(
        smart_native is not None
        and getattr(smart_native, "native_core_available", lambda: False)()
        and hasattr(smart_native, "native_smart_engine_from_gmsh")
    )


def find_bbox_params_metadata(root: str | Path, mesh_id: str) -> Path | None:
    path = Path(root)
    if path.is_file() and path.name == "bbox_params.json":
        return path
    if path.is_dir() and (path / "bbox_params.json").exists():
        return path / "bbox_params.json"
    candidates = []
    if path.exists():
        candidates.extend(path.glob(f"**/result/updated*/{mesh_id}/bboxs_steps*/bbox_params.json"))
        candidates.extend(path.glob(f"**/result/{mesh_id}/bboxs*/bbox_params.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def load_bbox_params(metadata_path: str | Path) -> tuple[list[list[float]], list[list[float]]]:
    path = Path(metadata_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = data.get("boxes", [])
    bounds: list[list[float]] = []
    rotations: list[list[float]] = []
    for item in sorted(boxes, key=lambda value: int(value.get("index", 0))):
        box = [float(value) for value in item.get("bounds", [])]
        rotation = [float(value) for value in item.get("rotation", [])]
        if len(box) != 6:
            continue
        if len(rotation) != 9:
            rotation = list(IDENTITY_ROTATION)
        bounds.append(box)
        rotations.append(rotation)
    if not bounds:
        raise ValueError(f"no valid bbox metadata in {path}")
    return bounds, rotations


def write_legacy_grd_bbox_params_from_segment(
    segment_path: str | Path,
    msh_path: str | Path,
    output_path: str | Path | None = None,
    *,
    tilted: bool = True,
) -> Path:
    """Write bbox metadata using the legacy grd_merged initialization exactly.

    Legacy SMART does not initialize refine from the merge bbox OBJ metadata. It
    rereads `greedy_segment*.txt`, gathers all tetra vertices in each partition,
    and calls `trimesh.bounds.oriented_bounds(..., angle_digits=3)`. C++ native
    refine/MCTS must consume this metadata to match the original Python path.
    """

    import numpy as np
    import pymesh
    import trimesh

    segment = Path(segment_path)
    if output_path is None:
        output = Path(str(segment) + ".legacy_bbox_params.json")
    else:
        output = Path(output_path)
    mesh = pymesh.load_mesh(msh_path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    voxels = np.asarray(mesh.voxels, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"invalid tetra vertices in {msh_path}")
    if voxels.ndim != 2 or voxels.shape[1] < 4:
        raise ValueError(f"invalid tetra voxels in {msh_path}")

    lines = [
        line.strip()
        for line in segment.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"empty greedy segment file: {segment}")
    num_bbox = int(lines[0])
    if len(lines) < num_bbox + 1:
        raise ValueError(f"greedy segment has {len(lines) - 1} partitions, expected {num_bbox}: {segment}")

    boxes: list[dict[str, Any]] = []
    for index in range(num_bbox):
        partition = [int(value) for value in lines[index + 1].split()]
        if not partition:
            continue
        if min(partition) < 0 or max(partition) >= len(voxels):
            raise ValueError(f"partition {index} contains out-of-range tetra index in {segment}")
        pts = vertices[voxels[np.asarray(partition, dtype=int)].reshape(-1), :3]
        rotation = np.eye(3, dtype=float)
        work_pts = pts
        if tilted:
            to_origin, _ = trimesh.bounds.oriented_bounds(work_pts, angle_digits=3)
            rotation = np.asarray(to_origin[:3, :3], dtype=float)
            if np.all(np.isfinite(rotation)):
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    work_pts = np.matmul(work_pts, np.transpose(rotation))
            else:
                rotation = np.eye(3, dtype=float)
                work_pts = pts
        if not np.all(np.isfinite(work_pts)):
            rotation = np.eye(3, dtype=float)
            work_pts = pts
        mn_pt = np.min(work_pts, axis=0)
        mx_pt = np.max(work_pts, axis=0)
        boxes.append(
            {
                "index": int(index),
                "bounds": [float(value) for value in [*mn_pt.tolist(), *mx_pt.tolist()]],
                "rotation": [float(value) for value in rotation.reshape(-1).tolist()],
                "partition": partition,
            }
        )

    if not boxes:
        raise ValueError(f"no bbox params generated from {segment}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "legacy_grd_merged_bbox_init",
                "segment_path": str(segment),
                "msh_path": str(msh_path),
                "tilted": bool(tilted),
                "boxes": boxes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def effective_native_num_action_scale(configured: Any) -> int:
    return max(1, int(configured or 1)) * 2


def load_partitions(path: str | Path) -> list[list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("partitions", data) if isinstance(data, dict) else data
    partitions = [
        sorted({int(value) for value in partition})
        for partition in raw
        if isinstance(partition, list) and partition
    ]
    if not partitions:
        raise ValueError(f"no valid partitions in {path}")
    return partitions


def find_partition_metadata(tet_dir: str | Path, init_type: str) -> Path | None:
    root = Path(tet_dir)
    candidates = [
        root / f"{init_type}_partitions.json",
        root / "preseg_partitions.json",
        root / "partitions.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_coacd_partition_from_files(
    *,
    msh_path: str | Path,
    coacd_dir: str | Path,
    output_path: str | Path,
    mesh_id: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    command = [
        "smart-cpp-native",
        "partition-coacd",
        "--msh",
        str(msh_path),
        "--coacd_dir",
        str(coacd_dir),
        "--output",
        str(output_path),
    ]
    if mesh_id:
        command.extend(["--mesh_id", str(mesh_id)])
    if dry_run:
        return {
            "status": "dry_run",
            "output_path": Path(output_path),
            "command": command,
            "metadata": {"backend": "smart-cpp-native", "command": "partition-coacd"},
        }

    native_bin = native_executable_path()
    if native_bin is None:
        raise RuntimeError("smart-cpp-native executable is required for CoACD partition metadata")
    args = command[1:]
    started = time.time()
    completed = run_native_command(args)
    if completed.returncode != 0:
        raise RuntimeError(
            "smart-cpp-native partition-coacd failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    elapsed = time.time() - started
    payload = json.loads(completed.stdout or "{}")
    metadata = {
        "backend": "smart-cpp-native",
        "native_executable": str(native_bin),
        "elapsed_sec_wall": elapsed,
        "stdout": payload,
    }
    return {
        "status": "success" if Path(output_path).exists() else "failed",
        "output_path": Path(output_path),
        "command": [str(native_bin), *args],
        "metadata": metadata,
    }


def run_pipeline_from_files(
    *,
    input_mesh: str | Path,
    work_dir: str | Path,
    manifoldplus_bin: str | Path,
    ftetwild_bin: str | Path,
    coacd_bin: str | Path,
    mesh_id: str = "",
    epsilon: float = 0.002,
    edge_length: float = 0.1,
    merge_eps: float = 0.02,
    refine_max_step: int = 2000,
    mcts_iter: int = 3000,
    mcts_max_step: int = 150,
    normalize_mode: str = "bbox_diagonal",
    normalize_target: float = 1.0,
    normalize_center: str = "bbox",
    cover_penalty: float = 100.0,
    refine_action_unit: float = 0.01,
    mcts_action_unit: float = 0.02,
    num_action_scale: int = 2,
    manifold_timeout_sec: float = 600.0,
    ftetwild_timeout_sec: float = 1200.0,
    ftetwild_threads: int | None = None,
    ftetwild_level: int = 2,
    retry_epsilon_scale: float = 2.0,
    retry_edge_length_scale: float = 2.0,
    coacd_timeout_sec: float = 1200.0,
    coacd_threshold: float = 0.05,
    coacd_max_convex_hull: int = 64,
    coacd_preprocess_mode: str = "auto",
    coacd_preprocess_resolution: int = 50,
    coacd_resolution: int = 2000,
    coacd_mcts_nodes: int = 20,
    coacd_mcts_iterations: int = 150,
    coacd_mcts_max_depth: int = 3,
    coacd_seed: int = 7777,
    coacd_pca: bool = False,
    coacd_merge: bool = True,
    coacd_decimate: bool = True,
    merge_tilted: bool = True,
    merge_only_nearby: bool = False,
    final_k: int | None = None,
    exp_w: float = 0.001,
    gamma: float = 1.0,
    cache_capacity: int = 65536,
    volume_method: str = "mesh",
    stateful_union_cache: bool = True,
    transposition_table: bool = False,
    seed: int = 0,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run the full native C++ executable pipeline for one mesh.

    This is the CoACD-style package API: Python only launches the compiled
    executable and returns its JSON; normalization, Mesh2Tet orchestration,
    CoACD CLI splitting/partitioning, merge, refine, and MCTS are all driven
    by `smart-cpp-native`.
    """

    args = pipeline_args_from_files(
        input_mesh=input_mesh,
        work_dir=work_dir,
        manifoldplus_bin=manifoldplus_bin,
        ftetwild_bin=ftetwild_bin,
        coacd_bin=coacd_bin,
        mesh_id=mesh_id,
        epsilon=epsilon,
        edge_length=edge_length,
        merge_eps=merge_eps,
        refine_max_step=refine_max_step,
        mcts_iter=mcts_iter,
        mcts_max_step=mcts_max_step,
        normalize_mode=normalize_mode,
        normalize_target=normalize_target,
        normalize_center=normalize_center,
        cover_penalty=cover_penalty,
        refine_action_unit=refine_action_unit,
        mcts_action_unit=mcts_action_unit,
        num_action_scale=num_action_scale,
        manifold_timeout_sec=manifold_timeout_sec,
        ftetwild_timeout_sec=ftetwild_timeout_sec,
        ftetwild_threads=ftetwild_threads,
        ftetwild_level=ftetwild_level,
        retry_epsilon_scale=retry_epsilon_scale,
        retry_edge_length_scale=retry_edge_length_scale,
        coacd_timeout_sec=coacd_timeout_sec,
        coacd_threshold=coacd_threshold,
        coacd_max_convex_hull=coacd_max_convex_hull,
        coacd_preprocess_mode=coacd_preprocess_mode,
        coacd_preprocess_resolution=coacd_preprocess_resolution,
        coacd_resolution=coacd_resolution,
        coacd_mcts_nodes=coacd_mcts_nodes,
        coacd_mcts_iterations=coacd_mcts_iterations,
        coacd_mcts_max_depth=coacd_mcts_max_depth,
        coacd_seed=coacd_seed,
        coacd_pca=coacd_pca,
        coacd_merge=coacd_merge,
        coacd_decimate=coacd_decimate,
        merge_tilted=merge_tilted,
        merge_only_nearby=merge_only_nearby,
        final_k=final_k,
        exp_w=exp_w,
        gamma=gamma,
        cache_capacity=cache_capacity,
        volume_method=volume_method,
        stateful_union_cache=stateful_union_cache,
        transposition_table=transposition_table,
        seed=seed,
    )
    started = time.time()
    completed = run_native_command(args, timeout=timeout)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "smart-cpp-native run-pipeline failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    payload = json.loads(completed.stdout or "{}")
    output_path = Path(payload.get("mcts_output_dir") or payload.get("work_dir") or work_dir)
    return {
        "status": payload.get("status", "success"),
        "output_path": output_path,
        "command": [str(native_executable_path()), *args],
        "metadata": {
            "backend": "smart-cpp-native",
            "combined": True,
            "elapsed_sec_wall": elapsed,
            "stdout": payload,
            "work_dir": str(work_dir),
        },
    }


def pipeline_args_from_files(
    *,
    input_mesh: str | Path,
    work_dir: str | Path,
    manifoldplus_bin: str | Path,
    ftetwild_bin: str | Path,
    coacd_bin: str | Path,
    mesh_id: str = "",
    epsilon: float = 0.002,
    edge_length: float = 0.1,
    merge_eps: float = 0.02,
    refine_max_step: int = 2000,
    mcts_iter: int = 3000,
    mcts_max_step: int = 150,
    normalize_mode: str = "bbox_diagonal",
    normalize_target: float = 1.0,
    normalize_center: str = "bbox",
    cover_penalty: float = 100.0,
    refine_action_unit: float = 0.01,
    mcts_action_unit: float = 0.02,
    num_action_scale: int = 2,
    manifold_timeout_sec: float = 600.0,
    ftetwild_timeout_sec: float = 1200.0,
    ftetwild_threads: int | None = None,
    ftetwild_level: int = 2,
    retry_epsilon_scale: float = 2.0,
    retry_edge_length_scale: float = 2.0,
    coacd_timeout_sec: float = 1200.0,
    coacd_threshold: float = 0.05,
    coacd_max_convex_hull: int = 64,
    coacd_preprocess_mode: str = "auto",
    coacd_preprocess_resolution: int = 50,
    coacd_resolution: int = 2000,
    coacd_mcts_nodes: int = 20,
    coacd_mcts_iterations: int = 150,
    coacd_mcts_max_depth: int = 3,
    coacd_seed: int = 7777,
    coacd_pca: bool = False,
    coacd_merge: bool = True,
    coacd_decimate: bool = True,
    merge_tilted: bool = True,
    merge_only_nearby: bool = False,
    final_k: int | None = None,
    exp_w: float = 0.001,
    gamma: float = 1.0,
    cache_capacity: int = 65536,
    volume_method: str = "mesh",
    stateful_union_cache: bool = True,
    transposition_table: bool = False,
    seed: int = 0,
) -> list[str]:
    args = [
        "run-pipeline",
        "--input",
        str(input_mesh),
        "--work_dir",
        str(work_dir),
        "--manifoldplus_bin",
        str(manifoldplus_bin),
        "--ftetwild_bin",
        str(ftetwild_bin),
        "--coacd_bin",
        str(coacd_bin),
        "--epsilon",
        str(float(epsilon)),
        "--edge_length",
        str(float(edge_length)),
        "--merge_eps",
        str(float(merge_eps)),
        "--refine_max_step",
        str(int(refine_max_step)),
        "--mcts_iter",
        str(int(mcts_iter)),
        "--mcts_max_step",
        str(int(mcts_max_step)),
        "--normalize_mode",
        str(normalize_mode),
        "--normalize_target",
        str(float(normalize_target)),
        "--normalize_center",
        str(normalize_center),
        "--cover_penalty",
        str(float(cover_penalty)),
        "--refine_action_unit",
        str(float(refine_action_unit)),
        "--mcts_action_unit",
        str(float(mcts_action_unit)),
        "--num_action_scale",
        str(int(num_action_scale)),
        "--manifold_timeout_sec",
        str(float(manifold_timeout_sec)),
        "--ftetwild_timeout_sec",
        str(float(ftetwild_timeout_sec)),
        "--ftetwild_level",
        str(int(ftetwild_level)),
        "--retry_epsilon_scale",
        str(float(retry_epsilon_scale)),
        "--retry_edge_length_scale",
        str(float(retry_edge_length_scale)),
        "--coacd_timeout_sec",
        str(float(coacd_timeout_sec)),
        "--coacd_threshold",
        str(float(coacd_threshold)),
        "--coacd_max_convex_hull",
        str(int(coacd_max_convex_hull)),
        "--coacd_preprocess_mode",
        str(coacd_preprocess_mode),
        "--coacd_preprocess_resolution",
        str(int(coacd_preprocess_resolution)),
        "--coacd_resolution",
        str(int(coacd_resolution)),
        "--coacd_mcts_nodes",
        str(int(coacd_mcts_nodes)),
        "--coacd_mcts_iterations",
        str(int(coacd_mcts_iterations)),
        "--coacd_mcts_max_depth",
        str(int(coacd_mcts_max_depth)),
        "--coacd_seed",
        str(int(coacd_seed)),
        "--exp_w",
        str(float(exp_w)),
        "--gamma",
        str(float(gamma)),
        "--cache_capacity",
        str(int(cache_capacity)),
        "--volume_method",
        str(volume_method),
        "--seed",
        str(int(seed)),
    ]
    if mesh_id:
        args.extend(["--mesh_id", str(mesh_id)])
    if ftetwild_threads is not None:
        args.extend(["--ftetwild_threads", str(int(ftetwild_threads))])
    if coacd_pca:
        args.append("--coacd_pca")
    if not coacd_merge:
        args.append("--coacd_no_merge")
    if not coacd_decimate:
        args.append("--coacd_no_decimate")
    if not merge_tilted:
        args.append("--no_tilted")
    if not merge_only_nearby:
        args.append("--all_pairs")
    if final_k is not None:
        args.extend(["--final_k", str(int(final_k))])
    if not stateful_union_cache:
        args.append("--no_stateful_union_cache")
    if transposition_table:
        args.append("--transposition_table")
    return args


def _write_greedy_segment(path: str | Path, partitions: list[list[int]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(partitions))]
    lines.extend(" ".join(str(int(value)) for value in partition) for partition in partitions)
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_merge_bbox_params(path: str | Path, result: dict[str, Any]) -> None:
    boxes = []
    partitions = result.get("partitions", [])
    for idx, (bounds, rotation) in enumerate(
        zip(result.get("bounds", []), result.get("rotations", []))
    ):
        record = {
            "index": int(idx),
            "bounds": [float(value) for value in bounds],
            "rotation": [float(value) for value in rotation],
        }
        if idx < len(partitions):
            record["partition"] = [int(value) for value in partitions[idx]]
        boxes.append(record)
    metadata = {
        "schema_version": 1,
        "source": "smart.native_runner.run_merge_from_partitions_file",
        "boxes": boxes,
    }
    Path(str(path) + ".bbox_params.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _box_volume(bounds: list[float]) -> float:
    if len(bounds) != 6:
        return 0.0
    return max(0.0, bounds[3] - bounds[0]) * max(0.0, bounds[4] - bounds[1]) * max(
        0.0, bounds[5] - bounds[2]
    )


def _action_prior_context(
    *,
    bounds: list[list[float]],
    category: str,
    mesh_id: str,
    num_action_scale: int,
    action_unit: float,
    max_step: int,
    cover_penalty: float,
    pen_rate: float,
    volume_sum: float,
    reward_backend: str,
    volume_method: str,
) -> dict[str, Any]:
    bbox_volume = sum(_box_volume([float(value) for value in row]) for row in bounds)
    return {
        "category": str(category),
        "mesh": str(mesh_id),
        "step": 0,
        "max_step": int(max_step),
        "num_bbox": len(bounds),
        "num_action_scale": int(num_action_scale),
        "actions_per_bbox": 6 * int(num_action_scale) + 1,
        "action_unit": float(action_unit),
        "bvs": float(bbox_volume / max(float(volume_sum), 1.0e-12)),
        "volume_sum": float(volume_sum),
        "cover_penalty": float(cover_penalty),
        "pen_rate": float(pen_rate),
        "reward_backend": str(reward_backend),
        "manifold_volume_method": str(volume_method),
        "bbox_bounds": [[float(value) for value in row] for row in bounds],
        "mcts_iter": 0,
        "mcts_not_updated": 0,
        "mcts_best_reward": 0.0,
        "mcts_escape_active": False,
        "mcts_escape_policy": False,
        "mcts_action_prior_top_k": 0,
    }


def _native_mcts_prior_logits_values(
    *,
    action_prior_path: str | Path | None,
    action_prior_device: str,
    bounds: list[list[float]],
    category: str,
    mesh_id: str,
    num_action_scale: int,
    action_unit: float,
    max_step: int,
    cover_penalty: float,
    pen_rate: float,
    volume_sum: float,
    volume_method: str,
    need_prior: bool,
    need_value: bool,
) -> tuple[list[float], list[float]]:
    if not action_prior_path or (not need_prior and not need_value):
        return [], []
    prior = load_action_prior(action_prior_path, inference_device=str(action_prior_device or "json"))
    context = _action_prior_context(
        bounds=bounds,
        category=category,
        mesh_id=mesh_id,
        num_action_scale=num_action_scale,
        action_unit=action_unit,
        max_step=max_step,
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        volume_sum=volume_sum,
        reward_backend="manifold",
        volume_method=volume_method,
    )
    num_actions = len(bounds) * (6 * int(num_action_scale) + 1)
    if hasattr(prior, "action_logits_values_for"):
        logits, values = prior.action_logits_values_for(
            range(num_actions),
            num_action_scale=int(num_action_scale),
            context=context,
        )
    else:
        logits = prior.action_logits_for(
            range(num_actions),
            num_action_scale=int(num_action_scale),
            context=context,
        )
        values = (
            prior.action_values_for(
                range(num_actions),
                num_action_scale=int(num_action_scale),
                context=context,
            )
            if hasattr(prior, "action_values_for")
            else []
        )
    return (
        [float(value) for value in logits] if need_prior else [],
        [float(value) for value in values] if need_value else [],
    )


def run_merge_from_partitions_file(
    *,
    msh_path: str | Path,
    partition_metadata_path: str | Path,
    output_segment_path: str | Path,
    category: str,
    merge_eps: float,
    final_k: int,
    tilted: bool,
    only_nearby: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    command = [
        "smart-cpp-native",
        "merge",
        "--msh",
        str(msh_path),
        "--partitions",
        str(partition_metadata_path),
        "--output_segment",
        str(output_segment_path),
        "--merge_eps",
        str(float(merge_eps)),
        "--final_k",
        str(int(final_k)),
    ]
    command.append("--tilted" if tilted else "--no_tilted")
    command.append("--only_nearby" if only_nearby else "--all_pairs")
    if dry_run:
        return {
            "status": "dry_run",
            "output_path": Path(output_segment_path),
            "command": command,
            "metadata": {"backend": "cpp_native_file_runner"},
        }

    native_bin = native_executable_path()
    if native_bin is not None:
        args = [
            "merge",
            "--msh",
            str(msh_path),
            "--partitions",
            str(partition_metadata_path),
            "--output_segment",
            str(output_segment_path),
            "--merge_eps",
            str(float(merge_eps)),
            "--final_k",
            str(int(final_k)),
        ]
        args.append("--tilted" if tilted else "--no_tilted")
        args.append("--only_nearby" if only_nearby else "--all_pairs")
        started = time.time()
        completed = run_native_command(args)
        if completed.returncode != 0:
            raise RuntimeError(
                "smart-cpp-native merge failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        elapsed = time.time() - started
        payload = json.loads(completed.stdout or "{}")
        stats_path = Path(str(output_segment_path) + ".native_stats.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        stats.update(
            {
                "backend": "smart-cpp-native",
                "native_executable": str(native_bin),
                "elapsed_sec_wall": elapsed,
                "stdout": payload,
            }
        )
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "status": "success" if Path(output_segment_path).exists() else "failed",
            "output_path": Path(output_segment_path),
            "command": [str(native_bin), *args],
            "metadata": stats,
        }

    started = time.time()
    partitions = load_partitions(partition_metadata_path)
    vertices, _faces, voxels = smart_native.native_load_gmsh(str(msh_path))
    volumes = smart_native.native_tetra_volumes(vertices, voxels)
    shape_volume = float(sum(float(value) for value in volumes))
    _part_volumes, part_bounds, _part_points = smart_native.partition_summaries(
        vertices,
        voxels,
        volumes,
        partitions,
        unique_points=True,
    )
    rotations = [list(IDENTITY_ROTATION) for _ in part_bounds]
    initial_bvs = sum(_box_volume([float(value) for value in bounds]) for bounds in part_bounds)
    initial_score = -abs((initial_bvs / shape_volume) - 1.0) if shape_volume > 0 else 0.0
    engine = smart_native.native_smart_engine_from_gmsh(
        str(msh_path),
        part_bounds,
        rotations,
        str(category),
        2,
        0.01,
        shape_volume,
        initial_score,
        False,
        65536,
        "mesh",
    )
    result = dict(
        engine.run_partition_merge_auto_adjacency(
            partitions,
            bool(only_nearby),
            float(merge_eps),
            float(shape_volume),
            int(final_k),
            bool(tilted),
        )
    )
    active_partitions = [
        [int(value) for value in partition] for partition in result.get("partitions", [])
    ]
    _write_greedy_segment(output_segment_path, active_partitions)
    _write_merge_bbox_params(output_segment_path, result)
    elapsed = time.time() - started
    stats = dict(engine.stats())
    stats.update(
        {
            "backend": "cpp_native_file_runner",
            "elapsed_sec": elapsed,
            "initial_partition_count": len(partitions),
            "active_partition_count": len(active_partitions),
            "result": result,
        }
    )
    Path(str(output_segment_path) + ".native_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "status": "success" if Path(output_segment_path).exists() else "failed",
        "output_path": Path(output_segment_path),
        "command": command,
        "metadata": stats,
    }


def run_refine_from_files(
    *,
    msh_path: str | Path,
    bbox_metadata_path: str | Path,
    output_root: str | Path,
    exp_name: str,
    mesh_id: str,
    category: str,
    max_step: int,
    cover_penalty: float,
    action_unit: float,
    num_action_scale: int,
    stateful_union_cache: bool,
    cache_capacity: int,
    volume_method: str,
    native_recenter: bool = False,
    learned_router: bool = False,
    learned_router_profile: str = "auto",
    learned_router_policy: str = "default",
    learned_router_overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    learned_router_overrides = dict(learned_router_overrides or {})
    if learned_router:
        command = [
            "smart._cpp",
            "run_builtin_deepset_policy_refine",
            "--msh",
            str(msh_path),
            "--bbox_params",
            str(bbox_metadata_path),
            "--max_step",
            str(int(max_step)),
            "--profile",
            str(learned_router_profile),
            "--policy",
            str(learned_router_policy),
        ]
    else:
        command = [
            "smart-cpp-native",
            "refine",
            "--msh",
            str(msh_path),
            "--bbox_params",
            str(bbox_metadata_path),
            "--max_step",
            str(int(max_step)),
        ]
    mesh_root = Path(output_root) / exp_name / "result" / "updated0" / mesh_id
    bbox_dir = mesh_root / "bboxs_steps0"
    if dry_run:
        return {
            "status": "dry_run",
            "output_path": bbox_dir,
            "command": command,
            "metadata": {
                "backend": "cpp_native_deepset_file_runner"
                if learned_router
                else "cpp_native_file_runner",
                "learned_router": bool(learned_router),
                "learned_router_profile": str(learned_router_profile),
                "learned_router_policy": str(learned_router_policy),
            },
        }

    native_bin = native_executable_path()
    if native_bin is not None and not learned_router:
        args = [
            "refine",
            "--msh",
            str(msh_path),
            "--bbox_params",
            str(bbox_metadata_path),
            "--output_dir",
            str(bbox_dir),
            "--max_step",
            str(int(max_step)),
            "--cover_penalty",
            str(float(cover_penalty)),
            "--action_unit",
            str(float(action_unit)),
            "--num_action_scale",
            str(int(num_action_scale)),
            "--cache_capacity",
            str(int(cache_capacity)),
            "--volume_method",
            str(volume_method),
        ]
        if not stateful_union_cache:
            args.append("--no_stateful_union_cache")
        if native_recenter:
            args.append("--native_recenter")
        started = time.time()
        completed = run_native_command(args)
        if completed.returncode != 0:
            raise RuntimeError(
                "smart-cpp-native refine failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        elapsed = time.time() - started
        payload = json.loads(completed.stdout or "{}")
        mesh_root.mkdir(parents=True, exist_ok=True)
        stats_path = bbox_dir / "native_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        stats.update(
            {
                "backend": "smart-cpp-native",
                "native_executable": str(native_bin),
                "elapsed_sec_wall": elapsed,
                "stdout": payload,
                "native_recenter": bool(native_recenter),
            }
        )
        exported = len(list(bbox_dir.glob("bbox*.obj")))
        (mesh_root / "time.txt").write_text(
            f"{len(load_bbox_params(bbox_metadata_path)[0])}\n{exported}\n{elapsed}\n",
            encoding="utf-8",
        )
        (mesh_root / "native_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
        )
        return {
            "status": "success" if bbox_dir.exists() else "failed",
            "output_path": bbox_dir,
            "command": [str(native_bin), *args],
            "metadata": stats,
        }

    started = time.time()
    bounds, rotations = load_bbox_params(bbox_metadata_path)
    engine = smart_native.native_smart_engine_from_gmsh(
        str(msh_path),
        bounds,
        rotations,
        str(category),
        int(num_action_scale),
        float(action_unit),
        0.0,
        0.0,
        bool(stateful_union_cache),
        int(cache_capacity),
        str(volume_method),
    )
    initial_score = float(engine.recompute_score(float(cover_penalty), 1.0))
    route_diagnostics: dict[str, Any] | None = None
    if learned_router:
        route_diagnostics = dict(
            smart_native.native_deepset_route_diagnostics(
                engine,
                cover_penalty=float(cover_penalty),
                pen_rate=1.0,
                profile=str(learned_router_profile),
                **learned_router_overrides,
            )
        )
        result = dict(
            smart_native.run_builtin_deepset_policy_refine(
                engine,
                max_steps=max(0, int(max_step)),
                policy=str(learned_router_policy),
                cover_penalty=float(cover_penalty),
                pen_rate=1.0,
                profile=str(learned_router_profile),
                **learned_router_overrides,
            )
        )
    else:
        result = dict(engine.run_refine(max(0, int(max_step)), float(cover_penalty), 1.0))
    mesh_root.mkdir(parents=True, exist_ok=True)
    exported = int(engine.export_bbox_dir(str(bbox_dir)))
    elapsed = time.time() - started
    stats = dict(engine.stats())
    stats.update(
        {
            "backend": "cpp_native_deepset_file_runner"
            if learned_router
            else "cpp_native_file_runner",
            "initial_bbox_score": initial_score,
            "elapsed_sec": elapsed,
            "exported_boxes": exported,
            "result": result,
            "learned_router": bool(learned_router),
            "learned_router_profile": str(learned_router_profile),
            "learned_router_policy": str(learned_router_policy),
            "learned_router_overrides": learned_router_overrides,
        }
    )
    if route_diagnostics is not None:
        stats["learned_router_route"] = route_diagnostics
    (mesh_root / "time.txt").write_text(
        f"{len(bounds)}\n{exported}\n{elapsed}\n", encoding="utf-8"
    )
    (bbox_dir / "native_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (mesh_root / "native_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "status": "success" if bbox_dir.exists() else "failed",
        "output_path": bbox_dir,
        "command": command,
        "metadata": stats,
    }


def run_mcts_from_files(
    *,
    msh_path: str | Path,
    bbox_metadata_path: str | Path,
    output_root: str | Path,
    exp_name: str,
    mesh_id: str,
    category: str,
    mcts_iter: int,
    max_step: int,
    cover_penalty: float,
    action_unit: float,
    num_action_scale: int,
    exp_weight: float,
    gamma: float,
    seed: int,
    transposition_table: bool,
    transposition_table_size: int,
    stateful_union_cache: bool,
    cache_capacity: int,
    volume_method: str,
    action_prior_path: str | Path | None = None,
    action_prior_device: str = "json",
    action_prior_weight: float = 0.0,
    puct_prior_weight: float = 0.0,
    action_value_weight: float = 0.0,
    action_prior_top_k: int = 0,
    native_recenter: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    command = [
        "smart-cpp-native",
        "mcts",
        "--msh",
        str(msh_path),
        "--bbox_params",
        str(bbox_metadata_path),
        "--mcts_iter",
        str(int(mcts_iter)),
        "--max_step",
        str(int(max_step)),
    ]
    if action_prior_path:
        command.extend(["--action_prior_path", str(action_prior_path)])
    if action_prior_weight:
        command.extend(["--action_prior_weight", str(float(action_prior_weight))])
    if puct_prior_weight:
        command.extend(["--puct_prior_weight", str(float(puct_prior_weight))])
    if action_value_weight:
        command.extend(["--action_value_weight", str(float(action_value_weight))])
    if action_prior_top_k:
        command.extend(["--action_prior_top_k", str(int(action_prior_top_k))])
    mesh_root = Path(output_root) / exp_name / "result" / "updated0" / mesh_id
    bbox_dir = mesh_root / "bboxs_steps0"
    if dry_run:
        return {
            "status": "dry_run",
            "output_path": bbox_dir,
            "command": command,
            "metadata": {"backend": "cpp_native_file_runner"},
        }

    native_bin = native_executable_path()
    executable_supported = native_bin is not None
    if executable_supported:
        native_prior_weight = (
            float(action_prior_weight)
            if float(action_prior_weight or 0.0) != 0.0
            else float(puct_prior_weight or 0.0)
        )
        need_prior = native_prior_weight != 0.0 or int(action_prior_top_k or 0) > 0
        need_value = float(action_value_weight or 0.0) != 0.0
        prior_logits: list[float] = []
        value_logits: list[float] = []
        prior_logits_path: Path | None = None
        value_logits_path: Path | None = None
        mesh_root.mkdir(parents=True, exist_ok=True)
        if need_prior or need_value:
            bounds, _rotations = load_bbox_params(bbox_metadata_path)
            volume_proxy = sum(_box_volume([float(value) for value in row]) for row in bounds)
            prior_logits, value_logits = _native_mcts_prior_logits_values(
                action_prior_path=action_prior_path,
                action_prior_device=action_prior_device,
                bounds=bounds,
                category=category,
                mesh_id=mesh_id,
                num_action_scale=int(num_action_scale),
                action_unit=float(action_unit),
                max_step=int(max_step),
                cover_penalty=float(cover_penalty),
                pen_rate=1.0,
                volume_sum=max(volume_proxy, 1.0e-12),
                volume_method=str(volume_method),
                need_prior=need_prior,
                need_value=need_value,
            )
            if need_prior and not prior_logits:
                raise RuntimeError("cpp_native executable MCTS requires action-prior logits")
            if need_value and not value_logits:
                raise RuntimeError("cpp_native executable MCTS requires action-value logits")
            if prior_logits:
                prior_logits_path = mesh_root / "mcts_prior_logits.json"
                prior_logits_path.write_text(json.dumps(prior_logits), encoding="utf-8")
            if value_logits:
                value_logits_path = mesh_root / "mcts_value_logits.json"
                value_logits_path.write_text(json.dumps(value_logits), encoding="utf-8")
        args = [
            "mcts",
            "--msh",
            str(msh_path),
            "--bbox_params",
            str(bbox_metadata_path),
            "--output_dir",
            str(bbox_dir),
            "--mcts_iter",
            str(int(mcts_iter)),
            "--max_step",
            str(int(max_step)),
            "--cover_penalty",
            str(float(cover_penalty)),
            "--action_unit",
            str(float(action_unit)),
            "--num_action_scale",
            str(int(num_action_scale)),
            "--exp_w",
            str(float(exp_weight)),
            "--gamma",
            str(float(gamma)),
            "--seed",
            str(int(seed)),
            "--cache_capacity",
            str(int(cache_capacity)),
            "--volume_method",
            str(volume_method),
            "--action_prior_weight",
            str(float(action_prior_weight or 0.0)),
            "--puct_prior_weight",
            str(float(puct_prior_weight or 0.0)),
            "--action_value_weight",
            str(float(action_value_weight or 0.0)),
            "--action_prior_top_k",
            str(int(action_prior_top_k or 0)),
            "--transposition_table_size",
            str(int(transposition_table_size)),
        ]
        if prior_logits_path is not None:
            args.extend(["--prior_logits_file", str(prior_logits_path)])
        if value_logits_path is not None:
            args.extend(["--value_logits_file", str(value_logits_path)])
        if transposition_table:
            args.append("--transposition_table")
        if not stateful_union_cache:
            args.append("--no_stateful_union_cache")
        if native_recenter:
            args.append("--native_recenter")
        started = time.time()
        completed = run_native_command(args)
        if completed.returncode != 0:
            raise RuntimeError(
                "smart-cpp-native mcts failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        elapsed = time.time() - started
        payload = json.loads(completed.stdout or "{}")
        mesh_root.mkdir(parents=True, exist_ok=True)
        stats_path = bbox_dir / "native_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        stats.update(
            {
                "backend": "smart-cpp-native",
                "native_executable": str(native_bin),
                "elapsed_sec_wall": elapsed,
                "stdout": payload,
                "action_prior_logits": len(prior_logits),
                "action_value_logits": len(value_logits),
                "action_prior_weight": float(action_prior_weight or 0.0),
                "puct_prior_weight": float(puct_prior_weight or 0.0),
                "action_value_weight": float(action_value_weight or 0.0),
                "action_prior_top_k": int(action_prior_top_k or 0),
                "native_mcts_action_prior_top_k": float(action_prior_top_k or 0),
                "native_mcts_transposition_table": 1.0 if transposition_table else 0.0,
                "native_recenter": bool(native_recenter),
            }
        )
        exported = len(list(bbox_dir.glob("bbox*.obj")))
        (mesh_root / "time.txt").write_text(
            f"{len(load_bbox_params(bbox_metadata_path)[0])}\n{exported}\n{elapsed}\n",
            encoding="utf-8",
        )
        (mesh_root / "native_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
        )
        return {
            "status": "success" if bbox_dir.exists() else "failed",
            "output_path": bbox_dir,
            "command": [str(native_bin), *args],
            "metadata": stats,
        }

    started = time.time()
    bounds, rotations = load_bbox_params(bbox_metadata_path)
    engine = smart_native.native_smart_engine_from_gmsh(
        str(msh_path),
        bounds,
        rotations,
        str(category),
        int(num_action_scale),
        float(action_unit),
        0.0,
        0.0,
        bool(stateful_union_cache),
        int(cache_capacity),
        str(volume_method),
    )
    initial_score = float(engine.recompute_score(float(cover_penalty), 1.0))
    native_prior_weight = (
        float(action_prior_weight)
        if float(action_prior_weight or 0.0) != 0.0
        else float(puct_prior_weight or 0.0)
    )
    need_prior = native_prior_weight != 0.0
    need_value = float(action_value_weight or 0.0) != 0.0
    prior_logits, value_logits = _native_mcts_prior_logits_values(
        action_prior_path=action_prior_path,
        action_prior_device=action_prior_device,
        bounds=bounds,
        category=category,
        mesh_id=mesh_id,
        num_action_scale=int(num_action_scale),
        action_unit=float(action_unit),
        max_step=int(max_step),
        cover_penalty=float(cover_penalty),
        pen_rate=1.0,
        volume_sum=float(engine.stats().get("volume_sum", 0.0) or 0.0),
        volume_method=str(volume_method),
        need_prior=need_prior,
        need_value=need_value,
    )
    if need_prior and not prior_logits:
        raise RuntimeError("cpp_native MCTS requires action-prior logits for nonzero prior weight")
    if need_value and not value_logits:
        raise RuntimeError("cpp_native MCTS requires action-value logits for nonzero value weight")
    result = dict(
        engine.run_mcts(
            max(0, int(mcts_iter)),
            max(0, int(max_step)),
            float(cover_penalty),
            1.0,
            float(exp_weight),
            float(gamma),
            int(seed),
            prior_logits,
            value_logits,
            native_prior_weight,
            float(action_value_weight or 0.0),
            bool(transposition_table),
            int(transposition_table_size),
            int(action_prior_top_k or 0),
        )
    )
    mesh_root.mkdir(parents=True, exist_ok=True)
    exported = int(engine.export_bbox_dir(str(bbox_dir)))
    elapsed = time.time() - started
    stats = dict(engine.stats())
    stats.update(
        {
            "backend": "cpp_native_file_runner",
            "initial_bbox_score": initial_score,
            "action_prior_path": str(action_prior_path) if action_prior_path else "",
            "action_prior_logits": len(prior_logits),
            "action_value_logits": len(value_logits),
            "action_prior_weight": float(action_prior_weight or 0.0),
            "puct_prior_weight": float(puct_prior_weight or 0.0),
            "action_value_weight": float(action_value_weight or 0.0),
            "action_prior_top_k": int(action_prior_top_k or 0),
            "elapsed_sec": elapsed,
            "exported_boxes": exported,
            "result": result,
        }
    )
    (mesh_root / "time.txt").write_text(
        f"{len(bounds)}\n{exported}\n{elapsed}\n", encoding="utf-8"
    )
    (mesh_root / "native_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "status": "success" if bbox_dir.exists() else "failed",
        "output_path": bbox_dir,
        "command": command,
        "metadata": stats,
    }


def run_refine_mcts_from_files(
    *,
    msh_path: str | Path,
    bbox_metadata_path: str | Path,
    refine_output_dir: str | Path,
    mcts_output_dir: str | Path,
    category: str,
    mesh_id: str,
    refine_max_step: int,
    mcts_iter: int,
    mcts_max_step: int,
    cover_penalty: float,
    refine_action_unit: float,
    mcts_action_unit: float,
    num_action_scale: int,
    exp_weight: float,
    gamma: float,
    seed: int,
    transposition_table: bool,
    transposition_table_size: int,
    stateful_union_cache: bool,
    cache_capacity: int,
    volume_method: str,
    action_prior_path: str | Path | None = None,
    action_prior_device: str = "json",
    action_prior_weight: float = 0.0,
    puct_prior_weight: float = 0.0,
    action_value_weight: float = 0.0,
    action_prior_top_k: int = 0,
    native_recenter: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run greedy refine followed by MCTS inside one smart-cpp-native process."""

    command = [
        "smart-cpp-native",
        "refine-mcts",
        "--msh",
        str(msh_path),
        "--bbox_params",
        str(bbox_metadata_path),
        "--refine_output_dir",
        str(refine_output_dir),
        "--mcts_output_dir",
        str(mcts_output_dir),
    ]
    command.extend(
        [
            "--refine_max_step",
            str(int(refine_max_step)),
            "--mcts_iter",
            str(int(mcts_iter)),
            "--mcts_max_step",
            str(int(mcts_max_step)),
            "--cover_penalty",
            str(float(cover_penalty)),
            "--refine_action_unit",
            str(float(refine_action_unit)),
            "--mcts_action_unit",
            str(float(mcts_action_unit)),
            "--num_action_scale",
            str(int(num_action_scale)),
            "--exp_w",
            str(float(exp_weight)),
            "--gamma",
            str(float(gamma)),
            "--seed",
            str(int(seed)),
            "--cache_capacity",
            str(int(cache_capacity)),
            "--volume_method",
            str(volume_method),
            "--action_prior_weight",
            str(float(action_prior_weight or 0.0)),
            "--puct_prior_weight",
            str(float(puct_prior_weight or 0.0)),
            "--action_value_weight",
            str(float(action_value_weight or 0.0)),
            "--action_prior_top_k",
            str(int(action_prior_top_k or 0)),
            "--transposition_table_size",
            str(int(transposition_table_size)),
        ]
    )
    if action_prior_path is not None:
        command.extend(["--action_prior_path", str(action_prior_path)])
    if transposition_table:
        command.append("--transposition_table")
    if not stateful_union_cache:
        command.append("--no_stateful_union_cache")
    if native_recenter:
        command.append("--native_recenter")
    if dry_run:
        return {
            "status": "dry_run",
            "output_path": Path(mcts_output_dir),
            "command": command,
            "metadata": {
                "backend": "smart-cpp-native",
                "combined": True,
                "single_mesh_load": True,
                "single_state_bridge": True,
            },
        }

    native_bin = native_executable_path()
    if native_bin is None:
        raise RuntimeError("smart-cpp-native executable is required for refine-mcts")

    native_prior_weight = (
        float(action_prior_weight)
        if float(action_prior_weight or 0.0) != 0.0
        else float(puct_prior_weight or 0.0)
    )
    need_prior = native_prior_weight != 0.0 or int(action_prior_top_k or 0) > 0
    need_value = float(action_value_weight or 0.0) != 0.0
    prior_logits: list[float] = []
    value_logits: list[float] = []
    prior_logits_path: Path | None = None
    value_logits_path: Path | None = None
    Path(mcts_output_dir).mkdir(parents=True, exist_ok=True)
    if need_prior or need_value:
        bounds, _rotations = load_bbox_params(bbox_metadata_path)
        volume_proxy = sum(_box_volume([float(value) for value in row]) for row in bounds)
        prior_logits, value_logits = _native_mcts_prior_logits_values(
            action_prior_path=action_prior_path,
            action_prior_device=action_prior_device,
            bounds=bounds,
            category=category,
            mesh_id=mesh_id,
            num_action_scale=int(num_action_scale),
            action_unit=float(mcts_action_unit),
            max_step=int(mcts_max_step),
            cover_penalty=float(cover_penalty),
            pen_rate=1.0,
            volume_sum=max(volume_proxy, 1.0e-12),
            volume_method=str(volume_method),
            need_prior=need_prior,
            need_value=need_value,
        )
        if need_prior and not prior_logits:
            raise RuntimeError("cpp_native refine-mcts requires action-prior logits")
        if need_value and not value_logits:
            raise RuntimeError("cpp_native refine-mcts requires action-value logits")
        if prior_logits:
            prior_logits_path = Path(mcts_output_dir) / "mcts_prior_logits.json"
            prior_logits_path.write_text(json.dumps(prior_logits), encoding="utf-8")
        if value_logits:
            value_logits_path = Path(mcts_output_dir) / "mcts_value_logits.json"
            value_logits_path.write_text(json.dumps(value_logits), encoding="utf-8")

    args = [
        "refine-mcts",
        "--msh",
        str(msh_path),
        "--bbox_params",
        str(bbox_metadata_path),
        "--refine_output_dir",
        str(refine_output_dir),
        "--mcts_output_dir",
        str(mcts_output_dir),
        "--refine_max_step",
        str(int(refine_max_step)),
        "--mcts_iter",
        str(int(mcts_iter)),
        "--mcts_max_step",
        str(int(mcts_max_step)),
        "--cover_penalty",
        str(float(cover_penalty)),
        "--refine_action_unit",
        str(float(refine_action_unit)),
        "--mcts_action_unit",
        str(float(mcts_action_unit)),
        "--num_action_scale",
        str(int(num_action_scale)),
        "--exp_w",
        str(float(exp_weight)),
        "--gamma",
        str(float(gamma)),
        "--seed",
        str(int(seed)),
        "--cache_capacity",
        str(int(cache_capacity)),
        "--volume_method",
        str(volume_method),
        "--action_prior_weight",
        str(float(action_prior_weight or 0.0)),
        "--puct_prior_weight",
        str(float(puct_prior_weight or 0.0)),
        "--action_value_weight",
        str(float(action_value_weight or 0.0)),
        "--action_prior_top_k",
        str(int(action_prior_top_k or 0)),
        "--transposition_table_size",
        str(int(transposition_table_size)),
    ]
    if prior_logits_path is not None:
        args.extend(["--prior_logits_file", str(prior_logits_path)])
    if value_logits_path is not None:
        args.extend(["--value_logits_file", str(value_logits_path)])
    if transposition_table:
        args.append("--transposition_table")
    if not stateful_union_cache:
        args.append("--no_stateful_union_cache")
    if native_recenter:
        args.append("--native_recenter")

    started = time.time()
    completed = run_native_command(args)
    if completed.returncode != 0:
        raise RuntimeError(
            "smart-cpp-native refine-mcts failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    elapsed = time.time() - started
    payload = json.loads(completed.stdout or "{}")
    stats = {
        "backend": "smart-cpp-native",
        "native_executable": str(native_bin),
        "elapsed_sec_wall": elapsed,
        "stdout": payload,
        "combined": True,
        "single_mesh_load": bool(payload.get("single_mesh_load", False)),
        "single_state_bridge": bool(payload.get("single_state_bridge", False)),
        "action_prior_logits": len(prior_logits),
        "action_value_logits": len(value_logits),
        "action_prior_weight": float(action_prior_weight or 0.0),
        "puct_prior_weight": float(puct_prior_weight or 0.0),
        "action_value_weight": float(action_value_weight or 0.0),
        "action_prior_top_k": int(action_prior_top_k or 0),
        "native_mcts_transposition_table": 1.0 if transposition_table else 0.0,
        "native_recenter": bool(native_recenter),
    }
    stats_path = Path(mcts_output_dir) / "native_stats.json"
    if stats_path.exists():
        stats.update(json.loads(stats_path.read_text(encoding="utf-8")))
    combined_stats_path = Path(mcts_output_dir) / "refine_mcts_native_stats.json"
    if combined_stats_path.exists():
        combined_stats = json.loads(combined_stats_path.read_text(encoding="utf-8"))
        stats["combined_stats_path"] = str(combined_stats_path)
        stats["combined_stats"] = combined_stats
        stats["single_mesh_load"] = bool(combined_stats.get("single_mesh_load", stats["single_mesh_load"]))
        stats["single_state_bridge"] = bool(
            combined_stats.get("single_state_bridge", stats["single_state_bridge"])
        )
        stats["refine_output_path"] = combined_stats.get("refine_output_path")
        stats["mcts_output_path"] = combined_stats.get("mcts_output_path")
    return {
        "status": "success" if Path(mcts_output_dir).exists() else "failed",
        "output_path": Path(mcts_output_dir),
        "command": [str(native_bin), *args],
        "metadata": stats,
    }
