from __future__ import annotations

import os
from typing import Iterable


if os.environ.get("SMART_DISABLE_CPP", "").lower() in {"1", "true", "yes", "on"}:
    _backend = None
else:
    try:
        from . import _cpp as _backend  # type: ignore
    except ImportError:
        try:
            import _cpp as _backend  # type: ignore
        except ImportError:
            _backend = None


def using_cpp() -> bool:
    return _backend is not None


def backend_path() -> str | None:
    if _backend is None:
        return None
    return getattr(_backend, "__file__", None)


def _require_backend():
    if _backend is None:
        raise RuntimeError("smart._cpp is not built. Run: smart build-cpp")
    return _backend


ManifoldState = _backend.ManifoldState if _backend is not None and hasattr(_backend, "ManifoldState") else None  # type: ignore
BBoxState = _backend.BBoxState if _backend is not None and hasattr(_backend, "BBoxState") else None  # type: ignore
ManifoldBridgeMesh = _backend.ManifoldBridgeMesh if _backend is not None and hasattr(_backend, "ManifoldBridgeMesh") else None  # type: ignore
CandidateBitsetState = _backend.CandidateBitsetState if _backend is not None and hasattr(_backend, "CandidateBitsetState") else None  # type: ignore
TetClippingState = _backend.TetClippingState if _backend is not None and hasattr(_backend, "TetClippingState") else None  # type: ignore
ActionMlpPolicy = _backend.ActionMlpPolicy if _backend is not None and hasattr(_backend, "ActionMlpPolicy") else None  # type: ignore
NativeSmartEngine = _backend.NativeSmartEngine if _backend is not None and hasattr(_backend, "NativeSmartEngine") else None  # type: ignore


def native_core_available() -> bool:
    if _backend is None:
        return False
    return bool(_backend.native_core_available())


def manifold_bridge_available() -> bool:
    if _backend is None:
        return False
    return bool(_backend.manifold_bridge_available())


def manifold_cube_volume(x: float, y: float, z: float) -> float:
    return float(_require_backend().manifold_cube_volume(float(x), float(y), float(z)))


def manifold_mesh_volume(vertices: Iterable[Iterable[float]], faces: Iterable[Iterable[int]]) -> float:
    return float(
        _require_backend().manifold_mesh_volume(
            [[float(v) for v in row] for row in vertices],
            [[int(v) for v in row] for row in faces],
        )
    )


def manifold_axis_box_intersection_volume(
    vertices: Iterable[Iterable[float]],
    faces: Iterable[Iterable[int]],
    bounds: Iterable[float],
) -> float:
    return float(
        _require_backend().manifold_axis_box_intersection_volume(
            [[float(v) for v in row] for row in vertices],
            [[int(v) for v in row] for row in faces],
            [float(value) for value in bounds],
        )
    )


def native_action_count(num_bbox: int, num_action_scale: int) -> int:
    return int(_require_backend().native_action_count(int(num_bbox), int(num_action_scale)))


def native_action_scales(num_action_scale: int) -> list[float]:
    return list(_require_backend().native_action_scales(int(num_action_scale)))


def native_action_indices(num_bbox: int, num_action_scale: int) -> list[list[int]]:
    return [
        [int(value) for value in row]
        for row in _require_backend().native_action_indices(int(num_bbox), int(num_action_scale))
    ]


def native_opposite_actions(num_bbox: int, num_action_scale: int) -> list[int]:
    return [
        int(value)
        for value in _require_backend().native_opposite_actions(int(num_bbox), int(num_action_scale))
    ]


def native_child_action_mask(
    total_actions: int,
    action: int,
    num_action_scale: int,
    parent_mask: Iterable[bool] | None = None,
) -> list[bool]:
    parent = None if parent_mask is None else [bool(value) for value in parent_mask]
    return [
        bool(value)
        for value in _require_backend().native_child_action_mask(
            int(total_actions),
            int(action),
            int(num_action_scale),
            parent,
        )
    ]


def native_discounted_reward(rewards: Iterable[float], gamma: float) -> float:
    return float(_require_backend().native_discounted_reward([float(value) for value in rewards], float(gamma)))


def native_ucb_best_count(
    parent_visits: int,
    child_qs: Iterable[float],
    child_visits: Iterable[int],
    exp_weight: float,
) -> int:
    return int(
        _require_backend().native_ucb_best_count(
            int(parent_visits),
            [float(value) for value in child_qs],
            [int(value) for value in child_visits],
            float(exp_weight),
        )
    )


def native_best_ucb_child(
    parent_visits: int,
    child_qs: Iterable[float],
    child_visits: Iterable[int],
    exp_weight: float,
    tie_pick: int,
) -> int:
    return int(
        _require_backend().native_best_ucb_child(
            int(parent_visits),
            [float(value) for value in child_qs],
            [int(value) for value in child_visits],
            float(exp_weight),
            int(tie_pick),
        )
    )


def native_prob_skip_exploration(
    parent_reward: float,
    child_rewards: Iterable[float],
    child_qs: Iterable[float],
    best_reward: float,
    skip_rate: float,
) -> float:
    return float(
        _require_backend().native_prob_skip_exploration(
            float(parent_reward),
            [float(value) for value in child_rewards],
            [float(value) for value in child_qs],
            float(best_reward),
            float(skip_rate),
        )
    )


def native_softmax_scaled(values: Iterable[float], scale: float = 100.0) -> list[float]:
    return [
        float(value)
        for value in _require_backend().native_softmax_scaled(
            [float(value) for value in values],
            float(scale),
        )
    ]


def native_weighted_action_scores(
    base_rewards: Iterable[float],
    prior_logits: Iterable[float],
    value_logits: Iterable[float],
    base_scale: float,
    prior_weight: float,
    value_weight: float,
) -> list[float]:
    return [
        float(value)
        for value in _require_backend().native_weighted_action_scores(
            [float(value) for value in base_rewards],
            [float(value) for value in prior_logits],
            [float(value) for value in value_logits],
            float(base_scale),
            float(prior_weight),
            float(value_weight),
        )
    ]


def native_top_k_actions(
    actions: Iterable[int],
    scores: Iterable[float],
    top_k: int,
) -> list[int]:
    return [
        int(value)
        for value in _require_backend().native_top_k_actions(
            [int(value) for value in actions],
            [float(value) for value in scores],
            int(top_k),
        )
    ]


def native_best_score_action(
    actions: Iterable[int],
    scores: Iterable[float],
    tie_pick: int,
) -> int:
    return int(
        _require_backend().native_best_score_action(
            [int(value) for value in actions],
            [float(value) for value in scores],
            int(tie_pick),
        )
    )


def native_diverse_escape_actions(
    actions: Iterable[int],
    scores: Iterable[float],
    primary_keep: Iterable[int],
    num_action_scale: int,
    escape_top_k: int,
) -> list[int]:
    return [
        int(value)
        for value in _require_backend().native_diverse_escape_actions(
            [int(value) for value in actions],
            [float(value) for value in scores],
            [int(value) for value in primary_keep],
            int(num_action_scale),
            int(escape_top_k),
        )
    ]


def native_add_puct_prior(
    uct_scores: Iterable[float],
    prior_logits: Iterable[float],
    child_visits: Iterable[int],
    parent_visits: int,
    prior_weight: float,
) -> list[float]:
    return [
        float(value)
        for value in _require_backend().native_add_puct_prior(
            [float(value) for value in uct_scores],
            [float(value) for value in prior_logits],
            [int(value) for value in child_visits],
            int(parent_visits),
            float(prior_weight),
        )
    ]


def native_action_mlp_logits_values(
    actions: Iterable[int],
    *,
    action_num_action_scale: int,
    model_num_action_scale: int,
    context: dict,
    categories: Iterable[str],
    action_input_weights: Iterable[Iterable[float]],
    action_hidden_bias: Iterable[float],
    action_output_weights: Iterable[float],
    action_output_bias: float,
    action_value_output_weights: Iterable[float] | None = None,
    action_value_output_bias: float = 0.0,
) -> tuple[list[float], list[float]]:
    logits, values = _require_backend().native_action_mlp_logits_values(
        [int(value) for value in actions],
        int(action_num_action_scale),
        int(model_num_action_scale),
        dict(context),
        [str(value) for value in categories],
        [[float(v) for v in row] for row in action_input_weights],
        [float(value) for value in action_hidden_bias],
        [float(value) for value in action_output_weights],
        float(action_output_bias),
        [] if action_value_output_weights is None else [float(value) for value in action_value_output_weights],
        float(action_value_output_bias),
    )
    return [float(value) for value in logits], [float(value) for value in values]


def opposite_action(action: int, num_action_scale: int) -> int:
    return int(_require_backend().opposite_action(int(action), int(num_action_scale)))


def opposite_action_mask(action: int, num_bbox: int, num_action_scale: int) -> list[bool]:
    return [
        bool(value)
        for value in _require_backend().opposite_action_mask(
            int(action),
            int(num_bbox),
            int(num_action_scale),
        )
    ]


def untried_actions(action_mask: Iterable[bool]) -> list[int]:
    return [int(value) for value in _require_backend().untried_actions([bool(value) for value in action_mask])]


def single_untried_action_mask(total_actions: int, action: int) -> list[bool]:
    return [
        bool(value)
        for value in _require_backend().single_untried_action_mask(
            int(total_actions),
            int(action),
        )
    ]


def ucb_scores(
    parent_visits: int,
    child_qs: Iterable[float],
    child_visits: Iterable[int],
    exp_weight: float,
) -> list[float]:
    return [
        float(value)
        for value in _require_backend().ucb_scores(
            int(parent_visits),
            [float(value) for value in child_qs],
            [int(value) for value in child_visits],
            float(exp_weight),
        )
    ]


def ucb_best_indices(
    parent_visits: int,
    child_qs: Iterable[float],
    child_visits: Iterable[int],
    exp_weight: float,
) -> list[int]:
    return [
        int(value)
        for value in _require_backend().ucb_best_indices(
            int(parent_visits),
            [float(value) for value in child_qs],
            [int(value) for value in child_visits],
            float(exp_weight),
        )
    ]


def native_bbox_volumes(bounds: Iterable[Iterable[float]]) -> list[float]:
    return [float(value) for value in _require_backend().native_bbox_volumes([[float(v) for v in row] for row in bounds])]


def native_bbox_valid_mask(bounds: Iterable[Iterable[float]]) -> list[bool]:
    return [bool(value) for value in _require_backend().native_bbox_valid_mask([[float(v) for v in row] for row in bounds])]


def native_total_bbox_volume(bounds: Iterable[Iterable[float]]) -> float:
    return float(_require_backend().native_total_bbox_volume([[float(v) for v in row] for row in bounds]))


def native_bbox_union_bounds(bounds: Iterable[Iterable[float]]) -> list[float]:
    return [float(value) for value in _require_backend().native_bbox_union_bounds([[float(v) for v in row] for row in bounds])]


def native_bbox_union_volume(bounds: Iterable[Iterable[float]]) -> float:
    return float(_require_backend().native_bbox_union_volume([[float(v) for v in row] for row in bounds]))


def native_box_mesh(
    x: float,
    y: float,
    z: float,
    lx: float,
    ly: float,
    lz: float,
    rotation: Iterable[float],
) -> tuple[list[list[float]], list[list[int]]]:
    vertices, faces = _require_backend().native_box_mesh(
        float(x),
        float(y),
        float(z),
        float(lx),
        float(ly),
        float(lz),
        [float(value) for value in rotation],
    )
    return (
        [[float(value) for value in row] for row in vertices],
        [[int(value) for value in row] for row in faces],
    )


def native_coverage_mask(points: Iterable[Iterable[float]], bounds: Iterable[float]) -> list[bool]:
    return [
        bool(value)
        for value in _require_backend().native_coverage_mask(
            [[float(v) for v in row] for row in points],
            [float(value) for value in bounds],
        )
    ]


def native_recenter_points_for_box(vertices, voxels, centroids, bounds: Iterable[float], rotation: Iterable[float]):
    return _require_backend().native_recenter_points_for_box(
        vertices,
        voxels,
        centroids,
        [float(value) for value in bounds],
        [float(value) for value in rotation],
    )


def native_apply_axis_action(
    bounds: Iterable[Iterable[float]],
    action: int,
    num_action_scale: int,
    action_unit: float,
) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in _require_backend().native_apply_axis_action(
            [[float(v) for v in row] for row in bounds],
            int(action),
            int(num_action_scale),
            float(action_unit),
        )
    ]


def native_action_upper_rewards(
    bounds: Iterable[Iterable[float]],
    num_action_scale: int,
    action_unit: float,
    volume_sum: float,
    last_bbox_score: float,
) -> list[float]:
    return [
        float(value)
        for value in _require_backend().native_action_upper_rewards(
            [[float(v) for v in row] for row in bounds],
            int(num_action_scale),
            float(action_unit),
            float(volume_sum),
            float(last_bbox_score),
        )
    ]


def native_bbox_action_upper_rewards(
    bounds: Iterable[Iterable[float]],
    bbox_idx: int,
    num_action_scale: int,
    action_unit: float,
    volume_sum: float,
    last_bbox_score: float,
) -> list[float]:
    return [
        float(value)
        for value in _require_backend().native_bbox_action_upper_rewards(
            [[float(v) for v in row] for row in bounds],
            int(bbox_idx),
            int(num_action_scale),
            float(action_unit),
            float(volume_sum),
            float(last_bbox_score),
        )
    ]


def native_bavf_scores(
    part_volumes: Iterable[float],
    bbox_volumes: Iterable[float],
    alpha: float = 100.0,
) -> list[float]:
    return [
        float(value)
        for value in _require_backend().native_bavf_scores(
            [float(value) for value in part_volumes],
            [float(value) for value in bbox_volumes],
            float(alpha),
        )
    ]


def native_merge_bavf_reward(
    prev_bvs: float,
    left_bbox_volume: float,
    right_bbox_volume: float,
    merged_bbox_volume: float,
    shape_volume: float,
) -> float:
    return float(
        _require_backend().native_merge_bavf_reward(
            float(prev_bvs),
            float(left_bbox_volume),
            float(right_bbox_volume),
            float(merged_bbox_volume),
            float(shape_volume),
        )
    )


def native_incremental_average(previous: float, count: int, value: float) -> float:
    return float(_require_backend().native_incremental_average(float(previous), int(count), float(value)))


def native_normalize_vertices_raw(
    vertices: Iterable[Iterable[float]],
    mode: str,
    center: str,
    target: float,
) -> tuple[list[list[float]], list[float]]:
    normalized, stats = _require_backend().native_normalize_vertices_raw(
        [[float(v) for v in row] for row in vertices],
        str(mode),
        str(center),
        float(target),
    )
    return [[float(value) for value in row] for row in normalized], [float(value) for value in stats]


def _native_vertex_stats(raw: list[float], offset: int) -> dict[str, object]:
    return {
        "vertex_count": int(round(raw[offset])),
        "bbox_min": [raw[offset + 1], raw[offset + 2], raw[offset + 3]],
        "bbox_max": [raw[offset + 4], raw[offset + 5], raw[offset + 6]],
        "bbox_extent": [raw[offset + 7], raw[offset + 8], raw[offset + 9]],
        "bbox_diagonal": raw[offset + 10],
        "bbox_center": [raw[offset + 11], raw[offset + 12], raw[offset + 13]],
        "sphere_radius": raw[offset + 14],
    }


def _native_normalization_stats(raw: list[float]) -> dict[str, object]:
    if len(raw) < 34:
        raise ValueError("native normalization stats are incomplete")
    return {
        "before": _native_vertex_stats(raw, 0),
        "center": [raw[15], raw[16], raw[17]],
        "scale": raw[18],
        "after": _native_vertex_stats(raw, 19),
    }


def native_normalize_vertices(
    vertices: Iterable[Iterable[float]],
    *,
    mode: str,
    center: str,
    target: float,
) -> tuple[list[tuple[float, float, float]], dict[str, object]]:
    normalized, raw_stats = native_normalize_vertices_raw(
        vertices,
        str(mode),
        str(center),
        float(target),
    )
    return [
        tuple(float(value) for value in row)
        for row in normalized
    ], _native_normalization_stats(list(raw_stats))


def native_normalize_obj_file(
    input_path: str,
    output_path: str,
    *,
    mode: str,
    center: str,
    target: float,
) -> dict[str, object]:
    return dict(
        _require_backend().native_normalize_obj_file(
            str(input_path),
            str(output_path),
            str(mode),
            str(center),
            float(target),
        )
    )


def native_load_obj_mesh(input_path: str) -> tuple[list[list[float]], list[list[int]]]:
    vertices, faces = _require_backend().native_load_obj_mesh(str(input_path))
    return (
        [[float(value) for value in row] for row in vertices],
        [[int(value) for value in row] for row in faces],
    )


def native_save_obj_mesh(
    output_path: str,
    vertices: Iterable[Iterable[float]],
    faces: Iterable[Iterable[int]],
) -> int:
    return int(
        _require_backend().native_save_obj_mesh(
            str(output_path),
            [[float(value) for value in row] for row in vertices],
            [[int(value) for value in row] for row in faces],
        )
    )


def native_symmetric_chamfer(left: Iterable[Iterable[float]], right: Iterable[Iterable[float]]) -> float:
    return float(
        _require_backend().native_symmetric_chamfer(
            [[float(v) for v in row] for row in left],
            [[float(v) for v in row] for row in right],
        )
    )


def native_tetra_volumes(vertices: Iterable[Iterable[float]], voxels: Iterable[Iterable[int]]) -> list[float]:
    return [
        float(value)
        for value in _require_backend().native_tetra_volumes(
            [[float(v) for v in row] for row in vertices],
            [[int(v) for v in row] for row in voxels],
        )
    ]


def native_tetra_centroids(vertices: Iterable[Iterable[float]], voxels: Iterable[Iterable[int]]) -> list[float]:
    return [
        float(value)
        for value in _require_backend().native_tetra_centroids(
            [[float(v) for v in row] for row in vertices],
            [[int(v) for v in row] for row in voxels],
        )
    ]


def native_tetra_surface_faces(voxels: Iterable[Iterable[int]]) -> list[list[int]]:
    return [[int(value) for value in row] for row in _require_backend().native_tetra_surface_faces([[int(v) for v in row] for row in voxels])]


def native_tetra_adjacency(voxels: Iterable[Iterable[int]]) -> list[list[int]]:
    return [[int(value) for value in row] for row in _require_backend().native_tetra_adjacency([[int(v) for v in row] for row in voxels])]


def native_load_gmsh(filename: str) -> tuple[list[list[float]], list[list[int]], list[list[int]]]:
    vertices, faces, voxels = _require_backend().native_load_gmsh(str(filename))
    return (
        [[float(value) for value in row] for row in vertices],
        [[int(value) for value in row] for row in faces],
        [[int(value) for value in row] for row in voxels],
    )


def native_smart_engine_from_gmsh(
    filename: str,
    bounds: Iterable[Iterable[float]],
    rotations: Iterable[Iterable[float]],
    category: str = "",
    num_action_scale: int = 2,
    action_unit: float = 0.01,
    volume_sum: float = 0.0,
    last_bbox_score: float = 0.0,
    stateful_union_cache: bool = True,
    cache_capacity: int = 65536,
    volume_method: str = "mesh",
):
    return _require_backend().native_smart_engine_from_gmsh(
        str(filename),
        [[float(v) for v in row] for row in bounds],
        [[float(v) for v in row] for row in rotations],
        str(category),
        int(num_action_scale),
        float(action_unit),
        float(volume_sum),
        float(last_bbox_score),
        bool(stateful_union_cache),
        int(cache_capacity),
        str(volume_method),
    )


def native_save_gmsh(
    filename: str,
    vertices: Iterable[Iterable[float]],
    faces: Iterable[Iterable[int]],
    voxels: Iterable[Iterable[int]],
) -> None:
    _require_backend().native_save_gmsh(
        str(filename),
        [[float(v) for v in row] for row in vertices],
        [[int(v) for v in row] for row in faces],
        [[int(v) for v in row] for row in voxels],
    )


def native_centroid_proxy_axis_rewards(
    centroids: Iterable[Iterable[float]],
    volumes: Iterable[float],
    bounds: Iterable[Iterable[float]],
    rotations: Iterable[Iterable[float]],
    num_action_scale: int,
    action_unit: float,
    volume_sum: float,
    last_bbox_score: float,
    cover_penalty: float,
    pen_rate: float,
) -> list[tuple[int, float]]:
    return [
        (int(action), float(reward))
        for action, reward in _require_backend().native_centroid_proxy_axis_rewards(
            [[float(v) for v in row] for row in centroids],
            [float(value) for value in volumes],
            [[float(v) for v in row] for row in bounds],
            [[float(v) for v in row] for row in rotations],
            int(num_action_scale),
            float(action_unit),
            float(volume_sum),
            float(last_bbox_score),
            float(cover_penalty),
            float(pen_rate),
        )
    ]


def native_partition_summaries(
    vertices: Iterable[Iterable[float]],
    voxels: Iterable[Iterable[int]],
    volumes: Iterable[float],
    partitions: Iterable[Iterable[int]],
    unique_points: bool = False,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    part_volumes, part_bounds, part_points = _require_backend().native_partition_summaries(
        [[float(v) for v in row] for row in vertices],
        [[int(v) for v in row] for row in voxels],
        [float(value) for value in volumes],
        [[int(v) for v in row] for row in partitions],
        bool(unique_points),
    )
    return (
        [float(value) for value in part_volumes],
        [[float(value) for value in row] for row in part_bounds],
        [[float(value) for value in row] for row in part_points],
    )


def tet_clipping_metrics(
    vertices: Iterable[Iterable[float]],
    voxels: Iterable[Iterable[int]],
    box_vertices: Iterable[Iterable[Iterable[float]]],
    surface_volume: float,
    max_boxes: int = 8,
    box_volumes: Iterable[float] | None = None,
) -> dict[str, float]:
    volumes = None if box_volumes is None else [float(value) for value in box_volumes]
    return {
        str(key): float(value)
        for key, value in _require_backend()
        .tet_clipping_metrics(
            [[float(v) for v in row] for row in vertices],
            [[int(v) for v in row] for row in voxels],
            [
                [[float(v) for v in row] for row in box]
                for box in box_vertices
            ],
            float(surface_volume),
            int(max_boxes),
            volumes,
        )
        .items()
    }


def bbox_rot_state_key(
    bounds: Iterable[Iterable[float]],
    rotations: Iterable[Iterable[float]],
) -> str:
    return str(
        _require_backend().bbox_rot_state_key(
            [[float(v) for v in row] for row in bounds],
            [[float(v) for v in row] for row in rotations],
        )
    )


def run_mcts_callbacks(
    args,
    env,
    num_iter: int,
    action_prior_logits: Iterable[float] | None = None,
    action_value_logits: Iterable[float] | None = None,
) -> dict[str, float]:
    prior_logits = [] if action_prior_logits is None else [float(value) for value in action_prior_logits]
    value_logits = [] if action_value_logits is None else [float(value) for value in action_value_logits]
    return {
        str(key): float(value)
        for key, value in _require_backend()
        .run_mcts_callbacks(args, env, int(num_iter), prior_logits, value_logits)
        .items()
    }


def run_greedy_refine_callbacks(args, env) -> tuple[list[float], int]:
    rewards, count = _require_backend().run_greedy_refine_callbacks(args, env)
    return [float(value) for value in rewards], int(count)


# Compatibility names used by legacy Python modules. The official native
# implementation is C++, and smart.native resolves directly to this module.
action_scales = native_action_scales
action_indices = native_action_indices
action_upper_rewards = native_action_upper_rewards
bbox_action_upper_rewards = native_bbox_action_upper_rewards
total_bbox_volume = native_total_bbox_volume
centroid_proxy_axis_rewards = native_centroid_proxy_axis_rewards
opposite_actions = native_opposite_actions
mcts_child_action_mask = native_child_action_mask
discounted_reward = native_discounted_reward
softmax_scaled = native_softmax_scaled
bbox_union_volume = native_bbox_union_volume
merge_bavf_reward = native_merge_bavf_reward
symmetric_chamfer = native_symmetric_chamfer
tetra_volumes = native_tetra_volumes
tetra_centroids = native_tetra_centroids
tetra_surface_faces = native_tetra_surface_faces
tetra_adjacency = native_tetra_adjacency
load_gmsh = native_load_gmsh
save_gmsh = native_save_gmsh
partition_summaries = native_partition_summaries
