import json
import os
import time

import numpy as np
import smart.pymesh_compat as pymesh
import trimesh
from tqdm import tqdm

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.tetmesh_environment import TetInfo, TetMeshEnv
from src.utils.utils import calculate_reward

try:
    import smart.native as smart_native
except ImportError:
    smart_native = None


def _dataset_name(path):
    return os.path.basename(os.path.normpath(path))


def _greedy_segment_path(args, env):
    return os.path.join(
        os.path.join(args.path_to_msh_file, env.name),
        "greedy_segment%d%s_mgeps%.5g%s.txt"
        % (
            args.final_k,
            ("_" + args.init_type) if args.init_type != "bsp" else "",
            args.merge_eps,
            "_fm" if args.fast_merge else "",
        ),
    )


def _save_greedy_segment(args, env):
    save_preseg = []
    for i in range(len(env.left_part)):
        save_preseg.append([])
        nwi = env.left_part[i]
        for j in range(len(env.partition[nwi])):
            save_preseg[i].append(env.partition[nwi][j])

    with open(_greedy_segment_path(args, env), "w") as f:
        f.writelines(str(len(env.left_part)))
        f.write("\n")

        for i in range(len(env.left_part)):
            for j in range(len(save_preseg[i])):
                f.write(
                    str(save_preseg[i][j])
                    + (" " if j != len(save_preseg[i]) - 1 else "")
                )

            if i != len(env.left_part) - 1:
                f.write("\n")


def _save_native_merge_bbox_params(args, env, result):
    bounds = result.get("bounds", [])
    rotations = result.get("rotations", [])
    if not bounds or not rotations or len(bounds) != len(rotations):
        return
    boxes = []
    partitions = result.get("partitions", [])
    for idx, (box, rotation) in enumerate(zip(bounds, rotations)):
        record = {
            "index": int(idx),
            "bounds": [float(value) for value in box],
            "rotation": [float(value) for value in rotation],
        }
        if idx < len(partitions):
            record["partition"] = [int(value) for value in partitions[idx]]
        boxes.append(record)
    metadata = {
        "schema_version": 1,
        "source": "smart._cpp.NativeSmartEngine.run_partition_merge",
        "boxes": boxes,
    }
    with open(_greedy_segment_path(args, env) + ".bbox_params.json", "w") as file:
        json.dump(metadata, file, indent=2)


def _native_merge_adjacency_pairs(env):
    pairs = []
    for first, second in env._iter_greedy_candidate_keys():
        first = int(first)
        second = int(second)
        if first != second:
            pairs.append([min(first, second), max(first, second)])
    return pairs


def _partition_points(env, partition):
    voxels = np.asarray(env.tetmsh.voxels, dtype=int)
    vertices = np.asarray(env.tetmsh.vertices, dtype=float)
    if len(partition) == 0:
        return np.empty((0, 3), dtype=float)
    points = vertices[voxels[np.asarray(partition, dtype=int)].reshape(-1)]
    return np.unique(points, axis=0)


def _axis_bounds(points):
    if len(points) == 0:
        return [0.0] * 6
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return [float(mins[0]), float(mins[1]), float(mins[2]), float(maxs[0]), float(maxs[1]), float(maxs[2])]


def _sync_env_from_native_merge(env, result):
    active_indices = [int(value) for value in result["active_indices"]]
    active_partitions = [
        np.asarray([int(item) for item in partition], dtype=int)
        for partition in result["partitions"]
    ]
    active_bounds = [
        [float(value) for value in bounds]
        for bounds in result.get("bounds", [])
    ]
    active_rotations = [
        [float(value) for value in rotation]
        for rotation in result.get("rotations", [])
    ]
    if len(active_indices) != len(active_partitions):
        raise RuntimeError("cpp_native merge returned inconsistent active partition data")
    if active_bounds and len(active_bounds) != len(active_indices):
        raise RuntimeError("cpp_native merge returned inconsistent active bounds data")
    if active_rotations and len(active_rotations) != len(active_indices):
        raise RuntimeError("cpp_native merge returned inconsistent active rotation data")

    partition_by_idx = dict(zip(active_indices, active_partitions))
    point_by_idx = {idx: _partition_points(env, part) for idx, part in partition_by_idx.items()}

    voxel_to_part = {}
    if env.args.only_nearby:
        for idx, part in partition_by_idx.items():
            for voxel_idx in part:
                voxel_to_part[int(voxel_idx)] = idx

    nearby_by_idx = {idx: set() for idx in active_indices}
    if env.args.only_nearby:
        for idx, part in partition_by_idx.items():
            for voxel_idx in part:
                for adjacent in env.tetmsh.get_voxel_adjacent_voxels(int(voxel_idx)):
                    other = voxel_to_part.get(int(adjacent))
                    if other is not None and other != idx:
                        nearby_by_idx[idx].add(other)

    env.left_part = active_indices
    env.left_part_set = set(active_indices)
    env.partition = [np.array([], dtype=int) for _ in range(len(env.partition))]
    env.part_pts = [np.empty((0, 3), dtype=float) for _ in range(len(env.part_pts))]
    env.part_bmesh = [None for _ in range(len(env.part_bmesh))]
    env.part_bman = [None for _ in range(len(env.part_bman))]
    env.part_ov = [0 for _ in range(len(env.part_ov))]
    env.tetinf = [None for _ in range(len(env.tetinf))]
    env.native_part_box_bounds = [None for _ in range(len(env.partition))]
    env.native_part_box_rotations = [None for _ in range(len(env.partition))]

    for pos, idx in enumerate(active_indices):
        partition = partition_by_idx[idx]
        points = point_by_idx[idx]
        env.partition[idx] = partition
        env.part_pts[idx] = points
        volume = float(np.sum(np.asarray(env.volume, dtype=float)[partition])) if len(partition) else 0.0
        env.tetinf[idx] = TetInfo(volume, _axis_bounds(points), nearby_by_idx[idx])
        if active_bounds:
            env.native_part_box_bounds[idx] = active_bounds[pos]
        if active_rotations:
            env.native_part_box_rotations[idx] = active_rotations[pos]

    env.done = int(len(env.left_part) <= 1)
    env.greedy_map = {}
    env.greedy_heap = []
    env.greedy_heap_order = 0
    env.greedy_cache_complete = False
    env._bvs_cache = None


def _run_cpp_native_merge(args, env):
    if (
        smart_native is None
        or not getattr(smart_native, "native_core_available", lambda: False)()
        or getattr(smart_native, "NativeSmartEngine", None) is None
    ):
        raise RuntimeError("merge_backend=cpp_native requires smart._cpp NativeSmartEngine")
    bounds = []
    rotations = []
    partitions = []
    for idx in range(len(env.partition)):
        if len(env.partition[idx]) == 0:
            raise RuntimeError("cpp_native merge requires dense initial partitions")
        bounds.append([float(value) for value in env.tetinf[idx].box])
        rotations.append(np.eye(3, dtype=float).reshape(-1).tolist())
        partitions.append([int(value) for value in env.partition[idx]])

    engine = smart_native.NativeSmartEngine(
        np.asarray(env.trimsh.vertices, dtype=float).tolist(),
        np.asarray(env.trimsh.faces, dtype=int).tolist(),
        np.asarray(env.tetmsh.voxels, dtype=int).tolist(),
        np.asarray(env.volume, dtype=float).reshape(-1).tolist(),
        np.asarray(env.centroid, dtype=float).reshape((-1, 3)).tolist(),
        bounds,
        rotations,
        str(getattr(args, "category", "")),
        2,
        0.01,
        float(env.volume_sum),
        -abs(float(env.BVS()) - 1.0),
        False,
        65536,
        "mesh",
    )
    if hasattr(engine, "run_partition_merge_auto_adjacency"):
        result = engine.run_partition_merge_auto_adjacency(
            partitions,
            bool(args.only_nearby),
            float(args.merge_eps),
            float(env.volume_sum),
            int(args.final_k),
            bool(args.tilted),
        )
    else:
        adjacency_pairs = _native_merge_adjacency_pairs(env)
        result = engine.run_partition_merge(
            partitions,
            adjacency_pairs,
            float(args.merge_eps),
            float(env.volume_sum),
            int(args.final_k),
            bool(args.tilted),
    )
    rewards = [float(value) for value in result["rewards"]]
    _sync_env_from_native_merge(env, result)
    _save_native_merge_bbox_params(args, env, result)
    stats = dict(engine.stats())
    stats.update(
        {
            "merge_backend": "cpp_native",
            "native_merge_count": len(result["merges"]),
            "applied_merge_count": len(rewards),
            "native_direct_sync": True,
            "ordered_delta_queue": bool(result["ordered_delta_queue"]),
            "candidate_inserts": int(result["candidate_inserts"]),
            "candidate_erases": int(result["candidate_erases"]),
            "candidate_queries": int(result["candidate_queries"]),
            "native_adjacency_pair_count": int(result.get("adjacency_pair_count", -1)),
            "native_adjacency_only_nearby": bool(
                result.get("adjacency_only_nearby", bool(args.only_nearby))
            ),
            "tilted_partition_pca": bool(result.get("tilted", False)),
        }
    )
    return rewards, stats


def greedy(args):

    dataset = STM_DataLoader(False, args)

    print("Dataloader loaded %d meshes" % (len(dataset)))

    def identity(x):
        return x

    cover = tqdm if args.print_off else identity

    for batch in cover(dataset):
        vertices, faces, voxels, name = batch
        if not args.print_off:
            print("Greedy search on %s mesh" % (name[0][:10]))
        start = time.time()
        env = TetMeshEnv(vertices, faces, voxels, args, name[0])

        if args.final_k > env.max_partition:
            assert 0, "Target final partition is bigger that initial partition"

        env.exp_name = (
            "greedy_%s%s_fastmerge%d_mgeps%.5g_final_k%dtilted%d_mov%.5gtov%.5g"
            % (
                _dataset_name(args.path_to_msh_file),
                ("_" + args.init_type if args.init_type != "bsp" else ""),
                1 if args.fast_merge else 0,
                args.merge_eps,
                args.final_k,
                (1 if args.tilted else 0),
                (args.mov_alpha if args.mov else 0),
                (args.tov_beta if args.tov else 0),
            )
        )
        env.render()

        native_stats = None
        if args.merge_backend == "cpp_native":
            rewards, native_stats = _run_cpp_native_merge(args, env)
        else:
            rewards = []
            greedy_idx=0
            while 1:
                action = env.greedy_sample(until_k=args.final_k)
                if action[0] == action[1]:
                    break

                reward = env.step(action)[0]
                rewards.append(reward)
                # env.render()
                env.render(bboxs=True, index=greedy_idx)
                greedy_idx += 1

                if not args.print_off:
                    print("Action: (%d, %d)" % (action[0], action[1]))
                    print("Reward: %f\n" % (reward))

        path_to_result = os.path.join(
            os.path.join(os.path.join(args.result_path, env.exp_name), "result"),
            name[0],
        )
        with open(
            os.path.join(
                path_to_result,
                "time.txt",
            ),
            "w",
        ) as f:
            f.write(str(env.max_partition))
            f.write("\n")
            f.write(str(len(env.left_part)))
            f.write("\n")
            f.write(str(time.time() - start))
        if native_stats is not None:
            with open(os.path.join(path_to_result, "native_merge_stats.json"), "w") as f:
                json.dump(native_stats, f, indent=2, sort_keys=True)

        sum_reward = calculate_reward(rewards, args.gamma)
        final_occ = env.compute_current_occupancy()
        occ, mov, tov, bvs = env.current_state_summary()
        if not args.print_off:
            print(
                "Shape %s (%d -> %s %d), Final metric: %g, Improved metric: %g, Elapsed time: %g"
                % (
                    name[0][:10],
                    env.max_partition,
                    "Target partition" if args.final_k else "Find partition",
                    len(env.left_part),
                    final_occ,
                    sum_reward,
                    time.time() - start,
                )
            )
            print(
                "Metrics after merging, OCC: %g, MOV: %g, TOV: %g, BVS: %g"
                % (occ, mov, tov, bvs)
            )
        env.render(bboxs=True)

        # Save greedy merged result on the mesh file path
        _save_greedy_segment(args, env)

        # env.save_color_info(path_to_fd)
