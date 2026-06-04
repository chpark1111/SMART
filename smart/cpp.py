from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable


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


_BUILTIN_DEEPSET_POLICIES = {
    "deepset_setaware_v2_h128_v1": "assets/policies/deepset_setaware_v2_h128_v1.smartmlp",
    "h128_v1": "assets/policies/deepset_setaware_v2_h128_v1.smartmlp",
    "default": "assets/policies/deepset_setaware_v2_h128_v1.smartmlp",
}
_BUILTIN_DEEPSET_POLICY_CACHE: dict[str, Any] = {}


def builtin_deepset_policy_path(name: str = "default") -> str:
    """Return a packaged opt-in DeepSets router checkpoint.

    The bundled policy is research/acceleration metadata only.  It orders
    candidate actions; exact SMART/Manifold scoring still chooses among the
    checked candidates.
    """

    key = str(name or "default")
    if key not in _BUILTIN_DEEPSET_POLICIES:
        valid = ", ".join(sorted(_BUILTIN_DEEPSET_POLICIES))
        raise ValueError(f"unknown built-in DeepSets policy {name!r}; expected one of: {valid}")
    path = Path(__file__).resolve().parent / _BUILTIN_DEEPSET_POLICIES[key]
    if not path.exists():
        raise FileNotFoundError(f"built-in DeepSets policy is missing from the package: {path}")
    return str(path)


def load_builtin_deepset_policy(name: str = "default", *, cache: bool = True) -> Any:
    """Load a packaged DeepSets candidate router for C++ native refine.

    The returned scorer is a C++ inference object.  It is safe to reuse across
    multiple ``NativeSmartEngine`` instances because engine state is passed into
    each scoring call.
    """

    if NativeDeepSetCandidateScorer is None:
        raise RuntimeError("smart._cpp NativeDeepSetCandidateScorer is unavailable")
    path = builtin_deepset_policy_path(name)
    if cache and path in _BUILTIN_DEEPSET_POLICY_CACHE:
        return _BUILTIN_DEEPSET_POLICY_CACHE[path]
    policy = NativeDeepSetCandidateScorer(path)
    if cache:
        _BUILTIN_DEEPSET_POLICY_CACHE[path] = policy
    return policy


def run_builtin_macro_skill_controller(engine: Any, *, category: str, **kwargs: Any) -> dict[str, Any]:
    """Run the packaged experimental macro-skill controller on a native engine.

    This is a convenience wrapper around :func:`smart.run_builtin_macro_skill_controller`.
    The exact SMART/Manifold backend still validates the final accepted update.
    """

    from .macro_skills import run_builtin_macro_skill_controller as _run

    return _run(engine, category=category, **kwargs)


def run_builtin_macro_skill_controller_from_files(**kwargs: Any) -> dict[str, Any]:
    """Load a native engine from files and run the macro-skill controller."""

    from .macro_skills import run_builtin_macro_skill_controller_from_files as _run

    return _run(**kwargs)


def run_builtin_macro_skill_planner(engine: Any, *, category: str, **kwargs: Any) -> dict[str, Any]:
    """Run the packaged multi-round macro-skill planner on a native engine."""

    from .macro_skills import run_builtin_macro_skill_planner as _run

    return _run(engine, category=category, **kwargs)


def run_builtin_macro_skill_planner_from_files(**kwargs: Any) -> dict[str, Any]:
    """Load a native engine from files and run the macro-skill planner."""

    from .macro_skills import run_builtin_macro_skill_planner_from_files as _run

    return _run(**kwargs)


ManifoldState = _backend.ManifoldState if _backend is not None and hasattr(_backend, "ManifoldState") else None  # type: ignore
BBoxState = _backend.BBoxState if _backend is not None and hasattr(_backend, "BBoxState") else None  # type: ignore
ManifoldBridgeMesh = _backend.ManifoldBridgeMesh if _backend is not None and hasattr(_backend, "ManifoldBridgeMesh") else None  # type: ignore
CandidateBitsetState = _backend.CandidateBitsetState if _backend is not None and hasattr(_backend, "CandidateBitsetState") else None  # type: ignore
TetClippingState = _backend.TetClippingState if _backend is not None and hasattr(_backend, "TetClippingState") else None  # type: ignore
ActionMlpPolicy = _backend.ActionMlpPolicy if _backend is not None and hasattr(_backend, "ActionMlpPolicy") else None  # type: ignore
NativeFastGeometryMlpPolicy = _backend.NativeFastGeometryMlpPolicy if _backend is not None and hasattr(_backend, "NativeFastGeometryMlpPolicy") else None  # type: ignore
NativeDeepSetCandidateScorer = _backend.NativeDeepSetCandidateScorer if _backend is not None and hasattr(_backend, "NativeDeepSetCandidateScorer") else None  # type: ignore
NativeScalarMlpScorer = _backend.NativeScalarMlpScorer if _backend is not None and hasattr(_backend, "NativeScalarMlpScorer") else None  # type: ignore
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


def native_centroid_proxy_axis_metrics(
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
) -> list[tuple[int, float, float, float]]:
    return [
        (int(action), float(reward), float(coverage), float(bvs))
        for action, reward, coverage, bvs in _require_backend().native_centroid_proxy_axis_metrics(
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


class NativeFastGeometryMlpPolicy:
    """C++ fast_v1 geometry-candidate MLP used by research evaluators.

    The policy keeps weights in the native extension and ranks centroid-proxy
    axis candidates without constructing per-candidate feature rows in Python.
    Exact SMART reward is still evaluated separately by the caller.
    """

    def __init__(
        self,
        mean: Iterable[float],
        std: Iterable[float],
        weights: Iterable[Iterable[Iterable[float]]],
        biases: Iterable[Iterable[float]],
    ) -> None:
        self._impl = _require_backend().NativeFastGeometryMlpPolicy(
            [float(value) for value in mean],
            [float(value) for value in std],
            [
                [[float(value) for value in row] for row in layer]
                for layer in weights
            ],
            [[float(value) for value in row] for row in biases],
        )

    @property
    def impl(self):
        return self._impl

    def rank_centroid_proxy_axis_metrics(
        self,
        centroids: Iterable[Iterable[float]],
        volumes: Iterable[float],
        bounds: Iterable[Iterable[float]],
        rotations: Iterable[Iterable[float]],
        category: str,
        turn: int,
        num_action_scale: int,
        action_unit: float,
        volume_sum: float,
        last_bbox_score: float,
        cover_penalty: float,
        pen_rate: float,
        covered_before: float,
        bvs_before: float,
        candidate_count: int,
    ) -> list[tuple[int, float, float, float, float, int]]:
        return [
            (
                int(action),
                float(reward),
                float(coverage),
                float(bvs),
                float(score),
                int(proxy_rank),
            )
            for action, reward, coverage, bvs, score, proxy_rank in self._impl.rank_centroid_proxy_axis_metrics(
                [[float(v) for v in row] for row in centroids],
                [float(value) for value in volumes],
                [[float(v) for v in row] for row in bounds],
                [[float(v) for v in row] for row in rotations],
                str(category),
                int(turn),
                int(num_action_scale),
                float(action_unit),
                float(volume_sum),
                float(last_bbox_score),
                float(cover_penalty),
                float(pen_rate),
                float(covered_before),
                float(bvs_before),
                int(candidate_count),
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


_DEEPSET_REFINE_PRESETS: dict[str, dict[str, Any]] = {
    "auto": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 6,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        "auto_exact_max_boxes": 1,
    },
    "hard": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 6,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 0,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
    },
    "mixed": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 6,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 32,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
    },
    "fast": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 8,
        "fallback_budget": 8,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 0,
        "adaptive_margin_threshold": 0.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 0,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
    },
    "hard_risk_v2": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 6,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        # Experimental structural hard-risk gate mined from the 2026-06-03
        # token-state stress test.  It is a default-candidate profile only,
        # not a default profile: exact SMART still validates every selected
        # action, and held-out zero-loss validation is required before promotion.
        "structural_high_budget": 128,
        "structural_category": "airplane",
        "structural_action_unit_min": 0.0049,
        "structural_action_unit_max": 0.0201,
        "structural_min_boxes": 6,
        "structural_min_turn": 1,
        "structural_min_coverage": 0.0,
        "structural_max_coverage": 1.0,
        "structural_max_bvs": 4.0,
        "structural_min_aspect_mean": 18.31,
        "structural_max_aspect_mean": 9.0e10,
        "structural_min_proxy_gap": 0.648,
        "structural_initial_high_budget": 128,
        "structural_initial_max_turn": 0,
        "structural_initial_min_coverage": 0.95,
        "structural_initial_max_coverage": 1.0,
        "structural_initial_min_bvs": 5.0,
        "structural_initial_max_bvs": 6.0,
        "structural_initial_min_aspect_mean": 0.0,
        "structural_initial_max_aspect_mean": 18.31,
        "structural_initial_min_proxy_gap": 0.0,
    },
    "hard_risk_v3_safe": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 32,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        # Quality-safe opt-in profile from the 2026-06-03 500 token-state
        # stress run.  It reaches zero regret on that stress set by opening a
        # broader exact fallback for structural airplane ambiguity.  It is not
        # the default because the wider fallback reduces speedup.
        "structural_high_budget": 128,
        "structural_category": "airplane",
        "structural_action_unit_min": 0.0049,
        "structural_action_unit_max": 0.0201,
        "structural_min_boxes": 6,
        "structural_min_turn": 0,
        "structural_min_coverage": 0.0,
        "structural_max_coverage": 1.0,
        "structural_max_bvs": 4.0,
        "structural_min_aspect_mean": 10.0,
        "structural_max_aspect_mean": 9.0e10,
        "structural_min_proxy_gap": 0.0,
        "structural_initial_high_budget": 128,
        "structural_initial_max_turn": 3,
        "structural_initial_min_coverage": 0.99,
        "structural_initial_max_coverage": 1.0,
        "structural_initial_min_bvs": 4.7,
        "structural_initial_max_bvs": 5.0,
        "structural_initial_min_aspect_mean": 0.0,
        "structural_initial_max_aspect_mean": 16.0,
        "structural_initial_min_proxy_gap": 0.0,
    },
    "hard_risk_v4_candidate": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 32,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        # Narrower default-candidate gate for the two remaining 500-state
        # stress-test failure families:
        #   1. low-coverage, low-BVS, high-aspect airplane ambiguity;
        #   2. nearly full-coverage, BVS 4.7-5.0, moderate-aspect airplane
        #      unit005 ambiguity.
        # It is still opt-in until validated on a larger strict held-out set.
        "structural_high_budget": 128,
        "structural_category": "airplane",
        "structural_action_unit_min": 0.0049,
        "structural_action_unit_max": 0.0201,
        "structural_min_boxes": 6,
        "structural_min_turn": 0,
        "structural_min_coverage": 0.0,
        "structural_max_coverage": 0.55,
        "structural_max_bvs": 1.35,
        "structural_min_aspect_mean": 20.0,
        "structural_max_aspect_mean": 9.0e10,
        "structural_min_proxy_gap": 0.0,
        "structural_initial_high_budget": 128,
        "structural_initial_max_turn": 3,
        "structural_initial_min_coverage": 0.99,
        "structural_initial_max_coverage": 1.0,
        "structural_initial_min_bvs": 4.7,
        "structural_initial_max_bvs": 5.0,
        "structural_initial_min_aspect_mean": 12.5,
        "structural_initial_max_aspect_mean": 16.0,
        "structural_initial_min_proxy_gap": 0.0,
    },
    "hard_risk_v5_candidate": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 32,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        # Broader high-aspect ambiguity guard than v4.  It keeps the v4
        # full-coverage unit005 guard, but also opens high budget for
        # high-aspect airplane states up to BVS 2.65.  This is intended to
        # cover the remaining turn6/unit005 hard cases without the very broad
        # v3 fallback.
        "structural_high_budget": 128,
        "structural_category": "airplane",
        "structural_action_unit_min": 0.0049,
        "structural_action_unit_max": 0.0201,
        "structural_min_boxes": 6,
        "structural_min_turn": 0,
        "structural_min_coverage": 0.0,
        "structural_max_coverage": 1.0,
        "structural_max_bvs": 2.65,
        "structural_min_aspect_mean": 20.0,
        "structural_max_aspect_mean": 9.0e10,
        "structural_min_proxy_gap": 0.0,
        "structural_initial_high_budget": 128,
        "structural_initial_max_turn": 3,
        "structural_initial_min_coverage": 0.99,
        "structural_initial_max_coverage": 1.0,
        "structural_initial_min_bvs": 4.7,
        "structural_initial_max_bvs": 5.0,
        "structural_initial_min_aspect_mean": 12.5,
        "structural_initial_max_aspect_mean": 16.0,
        "structural_initial_min_proxy_gap": 0.0,
    },
    "hard_risk_v6_candidate": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 32,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        # v6 keeps v4's narrow low-coverage guard and adds a separate covered
        # high-aspect/high-BVS guard.  The split avoids v5's broad fallback
        # while still catching the two remaining broad-500 failure buckets.
        "structural_high_budget": 128,
        "structural_category": "airplane",
        "structural_action_unit_min": 0.0049,
        "structural_action_unit_max": 0.0201,
        "structural_min_boxes": 6,
        "structural_min_turn": 0,
        "structural_min_coverage": 0.0,
        "structural_max_coverage": 0.55,
        "structural_max_bvs": 2.65,
        "structural_min_aspect_mean": 20.0,
        "structural_max_aspect_mean": 9.0e10,
        "structural_min_proxy_gap": 0.0,
        "structural_initial_high_budget": 128,
        "structural_initial_max_turn": 3,
        "structural_initial_min_coverage": 0.99,
        "structural_initial_max_coverage": 1.0,
        "structural_initial_min_bvs": 4.7,
        "structural_initial_max_bvs": 5.0,
        "structural_initial_min_aspect_mean": 12.5,
        "structural_initial_max_aspect_mean": 16.0,
        "structural_initial_min_proxy_gap": 0.0,
        "structural_secondary_high_budget": 128,
        "structural_secondary_category": "airplane",
        "structural_secondary_min_turn": 1,
        "structural_secondary_max_turn": 3,
        "structural_secondary_min_coverage": 0.95,
        "structural_secondary_max_coverage": 1.0,
        "structural_secondary_min_bvs": 2.3,
        "structural_secondary_max_bvs": 2.7,
        "structural_secondary_min_aspect_mean": 30.0,
        "structural_secondary_max_aspect_mean": 9.0e10,
        "structural_secondary_min_proxy_gap": 0.5,
        "structural_secondary_max_proxy_gap": 9.0e10,
    },
    "hard_risk_v7_candidate": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 32,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        # v7 is the first 1000-state strict candidate: v6's airplane guard plus
        # a narrow chair turn-3 guard for the high-coverage, moderate-BVS,
        # low-proxy-gap hard bucket.
        "structural_high_budget": 128,
        "structural_category": "airplane",
        "structural_action_unit_min": 0.0049,
        "structural_action_unit_max": 0.0201,
        "structural_min_boxes": 6,
        "structural_min_turn": 0,
        "structural_min_coverage": 0.0,
        "structural_max_coverage": 0.55,
        "structural_max_bvs": 2.65,
        "structural_min_aspect_mean": 20.0,
        "structural_max_aspect_mean": 9.0e10,
        "structural_min_proxy_gap": 0.0,
        "structural_initial_high_budget": 128,
        "structural_initial_max_turn": 3,
        "structural_initial_min_coverage": 0.99,
        "structural_initial_max_coverage": 1.0,
        "structural_initial_min_bvs": 4.7,
        "structural_initial_max_bvs": 5.0,
        "structural_initial_min_aspect_mean": 12.5,
        "structural_initial_max_aspect_mean": 16.0,
        "structural_initial_min_proxy_gap": 0.0,
        "structural_secondary_high_budget": 128,
        "structural_secondary_category": "chair",
        "structural_secondary_min_turn": 3,
        "structural_secondary_max_turn": 3,
        "structural_secondary_min_coverage": 0.99,
        "structural_secondary_max_coverage": 1.0,
        "structural_secondary_min_bvs": 2.2,
        "structural_secondary_max_bvs": 2.4,
        "structural_secondary_min_aspect_mean": 20.0,
        "structural_secondary_max_aspect_mean": 24.0,
        "structural_secondary_min_proxy_gap": -9.0e10,
        "structural_secondary_max_proxy_gap": 0.05,
    },
    "hard_risk_v8_candidate": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 32,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        # v8 keeps the airplane low-coverage guard and uses a generic turn-3
        # secondary guard for the common "mostly covered, moderate-BVS,
        # high-aspect" ambiguity shared by airplane unit005 and chair unit02.
        "structural_high_budget": 128,
        "structural_category": "airplane",
        "structural_action_unit_min": 0.0049,
        "structural_action_unit_max": 0.0201,
        "structural_min_boxes": 6,
        "structural_min_turn": 0,
        "structural_min_coverage": 0.0,
        "structural_max_coverage": 0.55,
        "structural_max_bvs": 2.65,
        "structural_min_aspect_mean": 20.0,
        "structural_max_aspect_mean": 9.0e10,
        "structural_min_proxy_gap": 0.0,
        "structural_initial_high_budget": 128,
        "structural_initial_max_turn": 3,
        "structural_initial_min_coverage": 0.99,
        "structural_initial_max_coverage": 1.0,
        "structural_initial_min_bvs": 4.7,
        "structural_initial_max_bvs": 5.0,
        "structural_initial_min_aspect_mean": 12.5,
        "structural_initial_max_aspect_mean": 16.0,
        "structural_initial_min_proxy_gap": 0.0,
        "structural_secondary_high_budget": 128,
        "structural_secondary_category": "",
        "structural_secondary_min_turn": 3,
        "structural_secondary_max_turn": 3,
        "structural_secondary_min_coverage": 0.985,
        "structural_secondary_max_coverage": 1.0,
        "structural_secondary_min_bvs": 2.2,
        "structural_secondary_max_bvs": 2.7,
        "structural_secondary_min_aspect_mean": 20.0,
        "structural_secondary_max_aspect_mean": 40.0,
        "structural_secondary_min_proxy_gap": -9.0e10,
        "structural_secondary_max_proxy_gap": 9.0e10,
    },
    "hard_risk_v9_candidate": {
        "candidate_count": 128,
        "multibox_min_boxes": 2,
        "budget": 24,
        "fallback_budget": 20,
        "tie_eps": 0.0,
        "tie_break": "action_id",
        "hist_bins": 3,
        "adaptive_high_budget": 72,
        "adaptive_margin_threshold": 2.0,
        "adaptive_margin_rank": 8,
        "small_pool_exact_threshold": 64,
        "nonfinite_proxy_rescue_budget": 4,
        "proxy_rescue_budget": 0,
        # v9 targets the 1000-state speed/quality frontier: lower base budget
        # plus three narrow exact rescue families.
        "structural_high_budget": 128,
        "structural_category": "airplane",
        "structural_action_unit_min": 0.0049,
        "structural_action_unit_max": 0.0201,
        "structural_min_boxes": 6,
        "structural_min_turn": 0,
        "structural_min_coverage": 0.0,
        "structural_max_coverage": 0.55,
        "structural_max_bvs": 2.65,
        "structural_min_aspect_mean": 20.0,
        "structural_max_aspect_mean": 9.0e10,
        "structural_min_proxy_gap": 0.0,
        "structural_initial_high_budget": 128,
        "structural_initial_max_turn": 3,
        "structural_initial_min_coverage": 0.99,
        "structural_initial_max_coverage": 1.0,
        "structural_initial_min_bvs": 4.7,
        "structural_initial_max_bvs": 5.0,
        "structural_initial_min_aspect_mean": 12.5,
        "structural_initial_max_aspect_mean": 16.0,
        "structural_initial_min_proxy_gap": 0.0,
        "structural_secondary_high_budget": 128,
        "structural_secondary_category": "",
        "structural_secondary_min_turn": 3,
        "structural_secondary_max_turn": 3,
        "structural_secondary_min_coverage": 0.985,
        "structural_secondary_max_coverage": 1.0,
        "structural_secondary_min_bvs": 2.2,
        "structural_secondary_max_bvs": 2.7,
        "structural_secondary_min_aspect_mean": 20.0,
        "structural_secondary_max_aspect_mean": 40.0,
        "structural_secondary_min_proxy_gap": -9.0e10,
        "structural_secondary_max_proxy_gap": 9.0e10,
        "structural_tertiary_high_budget": 128,
        "structural_tertiary_category": "airplane",
        "structural_tertiary_min_turn": 0,
        "structural_tertiary_max_turn": 3,
        "structural_tertiary_min_coverage": 0.7,
        "structural_tertiary_max_coverage": 0.9,
        "structural_tertiary_min_bvs": 1.55,
        "structural_tertiary_max_bvs": 2.1,
        "structural_tertiary_min_aspect_mean": 1.0e6,
        "structural_tertiary_max_aspect_mean": 9.0e12,
        "structural_tertiary_min_proxy_gap": 0.6,
        "structural_tertiary_max_proxy_gap": 4.0,
    },
}

_DEEPSET_REFINE_PRESETS["auto_safe"] = dict(
    _DEEPSET_REFINE_PRESETS["hard_risk_v9_candidate"]
)
_DEEPSET_REFINE_PRESETS["production_candidate"] = dict(
    _DEEPSET_REFINE_PRESETS["hard_risk_v9_candidate"]
)
_DEEPSET_REFINE_PRESETS["learned_auto_safe"] = dict(
    _DEEPSET_REFINE_PRESETS["hard_risk_v9_candidate"]
)
_DEEPSET_REFINE_PRESETS["auto"] = dict(
    _DEEPSET_REFINE_PRESETS["hard_risk_v9_candidate"]
)
_DEEPSET_REFINE_PRESETS["auto"]["auto_exact_max_boxes"] = 1


def native_deepset_refine_defaults(profile: str = "mixed") -> dict[str, Any]:
    """Return recommended C++ DeepSets refine routing settings.

    The learned policy is a candidate router only: selected candidates are still
    exact-scored by the native SMART/Manifold reward before applying an action.
    """

    key = str(profile or "mixed").lower()
    if key == "balanced":
        key = "mixed"
    if key not in _DEEPSET_REFINE_PRESETS:
        valid = ", ".join(sorted(_DEEPSET_REFINE_PRESETS))
        raise ValueError(f"unknown DeepSets refine profile {profile!r}; expected one of: {valid}")
    return dict(_DEEPSET_REFINE_PRESETS[key])


def run_native_deepset_policy_refine(
    engine: Any,
    checkpoint_or_policy: Any,
    *,
    max_steps: int,
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    profile: str = "mixed",
    **overrides: Any,
) -> dict[str, Any]:
    """Run native exact refine with a DeepSets candidate-ordering policy.

    ``checkpoint_or_policy`` may be either a ``.smartmlp`` path or an already
    constructed ``NativeDeepSetCandidateScorer``.  This helper keeps the long
    C++ signature stable for experiments while preserving exact final scoring.
    """

    settings = native_deepset_refine_defaults(profile)
    settings.update(overrides)
    auto_exact_max_boxes = int(settings.pop("auto_exact_max_boxes", -1))
    if auto_exact_max_boxes >= 0:
        try:
            stats = dict(engine.stats())
            num_boxes = int(stats.get("num_boxes", 0))
        except Exception:
            num_boxes = 0
        if num_boxes <= auto_exact_max_boxes:
            out = dict(engine.run_refine(int(max_steps), float(cover_penalty), float(pen_rate)))
            if "total_reward" not in out:
                out["total_reward"] = float(sum(float(value) for value in out.get("rewards", [])))
            out.setdefault("exact_checks", 0)
            out.setdefault("exact_errors", 0)
            out.setdefault("adaptive_high_budget_uses", 0)
            out.setdefault("small_pool_exact_uses", 0)
            out.setdefault("nonfinite_proxy_rescue_checks", 0)
            out.setdefault("proxy_rescue_checks", 0)
            out.setdefault("structural_high_budget_uses", 0)
            out.setdefault("structural_initial_high_budget_uses", 0)
            out.setdefault("structural_secondary_high_budget_uses", 0)
            out.setdefault("structural_tertiary_high_budget_uses", 0)
            out["router_profile"] = "auto_exact"
            out["learned_router_used"] = False
            out["auto_exact_max_boxes"] = auto_exact_max_boxes
            return out

    if NativeDeepSetCandidateScorer is None:
        raise RuntimeError("smart._cpp NativeDeepSetCandidateScorer is unavailable")
    policy = (
        checkpoint_or_policy
        if hasattr(checkpoint_or_policy, "score_setaware_axis_candidates")
        else NativeDeepSetCandidateScorer(str(checkpoint_or_policy))
    )
    route_profile = str(profile or "mixed")
    out = dict(
        engine.run_deepset_policy_refine(
            policy,
            int(max_steps),
            float(cover_penalty),
            float(pen_rate),
            int(settings["candidate_count"]),
            int(settings["multibox_min_boxes"]),
            int(settings["budget"]),
            int(settings["fallback_budget"]),
            float(settings["tie_eps"]),
            str(settings["tie_break"]),
            int(settings["hist_bins"]),
            int(settings["adaptive_high_budget"]),
            float(settings["adaptive_margin_threshold"]),
            int(settings["adaptive_margin_rank"]),
            int(settings["small_pool_exact_threshold"]),
            int(settings["nonfinite_proxy_rescue_budget"]),
            int(settings["proxy_rescue_budget"]),
            int(settings.get("structural_high_budget", 0)),
            str(settings.get("structural_category", "")),
            float(settings.get("structural_action_unit_min", float("-inf"))),
            float(settings.get("structural_action_unit_max", float("inf"))),
            int(settings.get("structural_min_boxes", 0)),
            int(settings.get("structural_min_turn", 0)),
            float(settings.get("structural_min_coverage", float("-inf"))),
            float(settings.get("structural_max_coverage", float("inf"))),
            float(settings.get("structural_max_bvs", float("inf"))),
            float(settings.get("structural_min_aspect_mean", float("-inf"))),
            float(settings.get("structural_max_aspect_mean", float("inf"))),
            float(settings.get("structural_min_proxy_gap", float("-inf"))),
            int(settings.get("structural_initial_high_budget", 0)),
            int(settings.get("structural_initial_max_turn", 2**63 - 1)),
            float(settings.get("structural_initial_min_coverage", float("-inf"))),
            float(settings.get("structural_initial_max_coverage", float("inf"))),
            float(settings.get("structural_initial_min_bvs", float("-inf"))),
            float(settings.get("structural_initial_max_bvs", float("inf"))),
            float(settings.get("structural_initial_min_aspect_mean", float("-inf"))),
            float(settings.get("structural_initial_max_aspect_mean", float("inf"))),
            float(settings.get("structural_initial_min_proxy_gap", float("-inf"))),
            int(settings.get("structural_secondary_high_budget", 0)),
            str(settings.get("structural_secondary_category", "")),
            int(settings.get("structural_secondary_min_turn", 0)),
            int(settings.get("structural_secondary_max_turn", 2**63 - 1)),
            float(settings.get("structural_secondary_min_coverage", float("-inf"))),
            float(settings.get("structural_secondary_max_coverage", float("inf"))),
            float(settings.get("structural_secondary_min_bvs", float("-inf"))),
            float(settings.get("structural_secondary_max_bvs", float("inf"))),
            float(settings.get("structural_secondary_min_aspect_mean", float("-inf"))),
            float(settings.get("structural_secondary_max_aspect_mean", float("inf"))),
            float(settings.get("structural_secondary_min_proxy_gap", float("-inf"))),
            float(settings.get("structural_secondary_max_proxy_gap", float("inf"))),
            int(settings.get("structural_tertiary_high_budget", 0)),
            str(settings.get("structural_tertiary_category", "")),
            int(settings.get("structural_tertiary_min_turn", 0)),
            int(settings.get("structural_tertiary_max_turn", 2**63 - 1)),
            float(settings.get("structural_tertiary_min_coverage", float("-inf"))),
            float(settings.get("structural_tertiary_max_coverage", float("inf"))),
            float(settings.get("structural_tertiary_min_bvs", float("-inf"))),
            float(settings.get("structural_tertiary_max_bvs", float("inf"))),
            float(settings.get("structural_tertiary_min_aspect_mean", float("-inf"))),
            float(settings.get("structural_tertiary_max_aspect_mean", float("inf"))),
            float(settings.get("structural_tertiary_min_proxy_gap", float("-inf"))),
            float(settings.get("structural_tertiary_max_proxy_gap", float("inf"))),
        )
    )
    out["router_profile"] = route_profile
    out["learned_router_used"] = True
    return out


def native_deepset_route_diagnostics(
    engine: Any,
    *,
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    profile: str = "auto",
    **overrides: Any,
) -> dict[str, Any]:
    """Explain which native refine route the DeepSets helper will prefer.

    This is intentionally cheap: it uses native engine stats and, only when a
    small-pool exact threshold is configured, the centroid-proxy candidate pool
    size.  It does not run Manifold exact action scoring.
    """

    settings = native_deepset_refine_defaults(profile)
    settings.update(overrides)
    try:
        stats = dict(engine.stats())
    except Exception:
        stats = {}
    num_boxes = int(stats.get("num_boxes", 0) or 0)
    auto_exact_max_boxes = int(settings.get("auto_exact_max_boxes", -1))
    if auto_exact_max_boxes >= 0 and num_boxes <= auto_exact_max_boxes:
        return {
            "route": "exact_native",
            "reason": "auto_exact_max_boxes",
            "num_boxes": num_boxes,
            "auto_exact_max_boxes": auto_exact_max_boxes,
            "learned_router_used": False,
        }

    candidate_pool_size: int | None = None
    small_pool_exact_threshold = int(settings.get("small_pool_exact_threshold", 0) or 0)
    if small_pool_exact_threshold > 0 and hasattr(engine, "centroid_proxy_axis_metrics"):
        try:
            candidate_pool_size = len(
                engine.centroid_proxy_axis_metrics(
                    float(cover_penalty),
                    float(pen_rate),
                    int(settings["candidate_count"]),
                )
            )
        except Exception:
            candidate_pool_size = None
        if candidate_pool_size is not None and 0 < candidate_pool_size <= small_pool_exact_threshold:
            return {
                "route": "deepset_router",
                "reason": "small_candidate_pool_exact_scoring",
                "num_boxes": num_boxes,
                "candidate_pool_size": candidate_pool_size,
                "small_pool_exact_threshold": small_pool_exact_threshold,
                "candidate_count": int(settings["candidate_count"]),
                "budget": int(settings["budget"]),
                "fallback_budget": int(settings["fallback_budget"]),
                "proxy_rescue_budget": int(settings.get("proxy_rescue_budget", 0)),
                "structural_high_budget": int(settings.get("structural_high_budget", 0)),
                "structural_category": str(settings.get("structural_category", "")),
                "structural_initial_high_budget": int(
                    settings.get("structural_initial_high_budget", 0)
                ),
                "structural_secondary_high_budget": int(
                    settings.get("structural_secondary_high_budget", 0)
                ),
                "structural_secondary_category": str(
                    settings.get("structural_secondary_category", "")
                ),
                "structural_tertiary_high_budget": int(
                    settings.get("structural_tertiary_high_budget", 0)
                ),
                "structural_tertiary_category": str(
                    settings.get("structural_tertiary_category", "")
                ),
                "learned_router_used": True,
                "small_pool_exact_scoring": True,
            }

    return {
        "route": "deepset_router",
        "reason": "multibox_candidate_routing",
        "num_boxes": num_boxes,
        "candidate_pool_size": candidate_pool_size,
        "candidate_count": int(settings["candidate_count"]),
        "budget": int(settings["budget"]),
        "fallback_budget": int(settings["fallback_budget"]),
        "proxy_rescue_budget": int(settings.get("proxy_rescue_budget", 0)),
        "structural_high_budget": int(settings.get("structural_high_budget", 0)),
        "structural_category": str(settings.get("structural_category", "")),
        "structural_initial_high_budget": int(
            settings.get("structural_initial_high_budget", 0)
        ),
        "structural_secondary_high_budget": int(
            settings.get("structural_secondary_high_budget", 0)
        ),
        "structural_secondary_category": str(
            settings.get("structural_secondary_category", "")
        ),
        "structural_tertiary_high_budget": int(
            settings.get("structural_tertiary_high_budget", 0)
        ),
        "structural_tertiary_category": str(
            settings.get("structural_tertiary_category", "")
        ),
        "learned_router_used": True,
    }


def deepset_axis_prior_logits(
    engine: Any,
    checkpoint_or_policy: Any,
    *,
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    candidate_count: int = 128,
    turn: int = 0,
    hist_bins: int = 3,
    num_action_scale: int = 2,
    floor_margin: float = 1.0,
) -> list[float]:
    """Create full action prior logits for native MCTS from DeepSets ranking.

    The DeepSets model scores axis-edit candidates for the current state.  This
    helper expands those sparse candidate scores into the full SMART action
    vector expected by ``NativeSmartEngine.run_mcts``.  Unranked actions receive
    a conservative floor below the worst ranked action, so they are still
    available unless MCTS is explicitly run with ``action_prior_top_k``.
    """

    try:
        stats = dict(engine.stats())
        num_boxes = int(stats.get("num_boxes", 0) or 0)
    except Exception:
        num_boxes = 0
    num_actions = native_action_count(num_boxes, int(num_action_scale))
    if num_actions <= 0:
        return []
    if NativeDeepSetCandidateScorer is None:
        raise RuntimeError("smart._cpp NativeDeepSetCandidateScorer is unavailable")
    policy = (
        checkpoint_or_policy
        if hasattr(checkpoint_or_policy, "score_setaware_axis_candidates")
        else NativeDeepSetCandidateScorer(str(checkpoint_or_policy))
    )
    rows = list(
        engine.rank_deepset_policy_axis_metrics(
            policy,
            float(cover_penalty),
            float(pen_rate),
            int(candidate_count),
            int(turn),
            int(hist_bins),
        )
    )
    if not rows:
        return [0.0] * num_actions
    scores = [float(row[4]) for row in rows]
    span = max(scores) - min(scores)
    floor = min(scores) - max(float(floor_margin), span)
    logits = [float(floor)] * num_actions
    for row in rows:
        action = int(row[0])
        if 0 <= action < num_actions:
            logits[action] = float(row[4])
    return logits


def builtin_deepset_axis_prior_logits(
    engine: Any,
    *,
    policy: str = "default",
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    candidate_count: int = 128,
    turn: int = 0,
    hist_bins: int = 3,
    num_action_scale: int = 2,
    floor_margin: float = 1.0,
) -> list[float]:
    """Create native MCTS prior logits from the packaged DeepSets policy."""

    return deepset_axis_prior_logits(
        engine,
        load_builtin_deepset_policy(policy),
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        candidate_count=candidate_count,
        turn=turn,
        hist_bins=hist_bins,
        num_action_scale=num_action_scale,
        floor_margin=floor_margin,
    )


_DEEPSET_MCTS_PRIOR_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "recommended_num_iter": 100,
        "candidate_count": 128,
        "hist_bins": 3,
        "prior_weight": 0.05,
        "multibox_top_k": 6,
        "single_box_top_k": 15,
        "multibox_max_step": 3,
        "single_box_max_step": 2,
        "floor_margin": 1.0,
    },
    "speed": {
        "recommended_num_iter": 50,
        "candidate_count": 128,
        "hist_bins": 3,
        "prior_weight": 0.05,
        "multibox_top_k": 6,
        "single_box_top_k": 15,
        "multibox_max_step": 2,
        "single_box_max_step": 2,
        "floor_margin": 1.0,
    },
    "quality": {
        "recommended_num_iter": 50,
        "candidate_count": 128,
        "hist_bins": 3,
        "prior_weight": 0.05,
        "multibox_top_k": 4,
        "single_box_top_k": 15,
        "multibox_max_step": 4,
        "single_box_max_step": 4,
        "floor_margin": 1.0,
    },
    "frontier": {
        "recommended_num_iter": 25,
        "candidate_count": 128,
        "hist_bins": 3,
        "prior_weight": 0.05,
        "multibox_top_k": 1,
        "single_box_top_k": 15,
        "multibox_max_step": 4,
        "single_box_max_step": 4,
        "floor_margin": 1.0,
    },
    "guarded": {
        "recommended_num_iter": 25,
        "candidate_count": 128,
        "hist_bins": 3,
        "prior_weight": 0.05,
        "multibox_top_k": 1,
        "single_box_top_k": 15,
        "multibox_max_step": 4,
        "single_box_max_step": 4,
        "floor_margin": 1.0,
        "guard_multibox_score_gt": -0.5,
        # Guarded fallback disables learned priors/top-k pruning.  Keep the
        # baseline MCTS budget so the safety path cannot lose quality merely
        # because it ran fewer exact MCTS iterations.
        "guard_num_iter": 50,
        "guard_max_step": 2,
        "guard_top_k": 0,
        "guard_prior_weight": 0.0,
        "guard_fast_score_gt": -0.05,
        "guard_fast_num_iter": 50,
    },
}
_DEEPSET_MCTS_PRIOR_PRESETS["auto_safe"] = dict(_DEEPSET_MCTS_PRIOR_PRESETS["guarded"])
_DEEPSET_MCTS_PRIOR_PRESETS["production_candidate"] = dict(_DEEPSET_MCTS_PRIOR_PRESETS["guarded"])


def native_deepset_mcts_prior_defaults(mode: str = "balanced") -> dict[str, Any]:
    """Return current experimental DeepSets MCTS-prior defaults."""

    key = str(mode or "balanced").lower()
    if key not in _DEEPSET_MCTS_PRIOR_PRESETS:
        valid = ", ".join(sorted(_DEEPSET_MCTS_PRIOR_PRESETS))
        raise ValueError(f"unknown DeepSets MCTS prior mode {mode!r}; expected one of: {valid}")
    return dict(_DEEPSET_MCTS_PRIOR_PRESETS[key])


def learned_router_profile_summary() -> dict[str, Any]:
    """Return the packaged learned-router release profile and promotion gate.

    This is intentionally data-bearing metadata, not a benchmark runner.  It
    lets source checkouts and installed wheels report exactly what the bundled
    learned router is allowed to do: prune/order candidates, then let native
    exact SMART/Manifold scoring choose the accepted action.
    """

    policy_path = ""
    policy_error = ""
    try:
        policy_path = builtin_deepset_policy_path()
    except Exception as exc:
        policy_error = str(exc)
    refine_defaults = native_deepset_refine_defaults("production_candidate")
    mcts_defaults = native_deepset_mcts_prior_defaults("production_candidate")
    return {
        "status": "release_candidate_opt_in",
        "default_smart_path": "unchanged_exact_cpp_native",
        "packaged_policy": Path(policy_path).name if policy_path else "deepset_setaware_v2_h128_v1.smartmlp",
        "policy_path": policy_path,
        "policy_error": policy_error,
        "feature_schema": "setaware_v2",
        "refine_profile": "production_candidate",
        "refine_profile_aliases": ["auto", "auto_safe", "learned_auto_safe"],
        "mcts_prior_mode": "production_candidate",
        "exact_reward_contract": [
            "learned policy only ranks or prunes candidate actions",
            "selected candidates are exact-scored by native SMART/Manifold",
            "one-box, small-pool, and hard-risk states use exact fallback or larger exact budgets",
            "accepted updates remain exact-reward selected and rollback-safe",
        ],
        "release_gate": {
            "can_ship_opt_in": True,
            "can_be_default": False,
            "default_blockers": [
                "fresh full-pipeline mesh-level validation across categories",
                "CI-sized replay benchmark artifact with zero quality-loss gate",
                "macro-skill controller evidence on 500+ replay-ready held-out states",
            ],
            "promotion_requirements": {
                "min_state_checks": 1000,
                "max_quality_loss_cases": 0,
                "min_exact_call_reduction": 0.20,
                "fresh_pipeline_cases": 50,
                "fallback_contract_required": True,
            },
        },
        "validation_snapshot": {
            "refine_full_token_split": {
                "cases": 1015,
                "quality_losses": 0,
                "exact_call_reduction": 0.30542,
                "speedup_vs_oracle_pool": 1.204,
                "profile": "production_candidate",
            },
            "refine_replay_states": {
                "cases": 1000,
                "quality_losses": 0,
                "exact_call_reduction": 0.307,
                "speedup_vs_oracle_pool": 1.203,
                "profile": "production_candidate",
            },
            "refine_heldout_test": {
                "cases": 264,
                "quality_losses": 0,
                "exact_call_reduction": 0.387,
                "speedup_vs_oracle_pool": 1.361,
                "profile": "production_candidate",
            },
            "mcts_guarded_weighted_state_checks": {
                "weighted_checks": 9670,
                "quality_losses": 0,
                "status": "validation_only_not_default",
            },
            "macro_skill_replay": {
                "status": "release_candidate_opt_in_post_refine",
                "heldout_cases": 173,
                "quality_losses_vs_conditional_budget_v1": 0,
                "default_ready": False,
            },
        },
        "runtime_requirements": {
            "native_cpp_imported": using_cpp(),
            "native_core_available": native_core_available(),
            "deepset_scorer_available": NativeDeepSetCandidateScorer is not None,
            "policy_asset_exists": bool(policy_path and Path(policy_path).exists()),
            "dynamic_mcts_prior_available": bool(
                NativeSmartEngine is not None and hasattr(NativeSmartEngine, "run_deepset_prior_mcts")
            ),
        },
        "refine_defaults": refine_defaults,
        "mcts_prior_defaults": mcts_defaults,
        "recommended_configs": [
            "configs/learned_auto_safe.yaml",
            "configs/learned_macro_safe.yaml",
            "configs/learned_macro_refine_only.yaml",
            "configs/learned_frontier.yaml",
        ],
        "recommended_commands": [
            "smart learned-router-summary --json",
            "smart assets --kind policies --json",
            "smart --config configs/smoke_5.yaml refine --set refine.learned_router.enabled=true --set refine.learned_router.profile=auto",
        ],
    }


def run_builtin_deepset_prior_mcts(
    engine: Any,
    *,
    num_iter: int | None = None,
    max_step: int | None = None,
    mode: str = "balanced",
    policy: str = "default",
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    exp_weight: float = 0.001,
    gamma: float = 1.0,
    seed: int = 7777,
    transposition_table: bool = False,
    transposition_table_size: int = 8192,
    num_action_scale: int = 2,
    **overrides: Any,
) -> dict[str, Any]:
    """Run native MCTS with packaged DeepSets action-prior logits.

    This is an opt-in research helper.  MCTS reward evaluation remains exact;
    the DeepSets policy only biases/prunes action expansion.
    """

    settings = native_deepset_mcts_prior_defaults(mode)
    settings.update(overrides)
    try:
        stats = dict(engine.stats())
        num_boxes = int(stats.get("num_boxes", 0) or 0)
        has_score = "last_bbox_score" in stats or "best_bbox_score" in stats
        current_score = float(stats.get("last_bbox_score", stats.get("best_bbox_score", 0.0)) or 0.0)
        if not has_score and hasattr(engine, "recompute_score"):
            current_score = float(engine.recompute_score(float(cover_penalty), float(pen_rate)))
            stats = dict(engine.stats())
            num_boxes = int(stats.get("num_boxes", num_boxes) or num_boxes)
    except Exception:
        num_boxes = 0
        current_score = 0.0
    scheduled_num_iter = int(settings.get("recommended_num_iter", 50) if num_iter is None else num_iter)
    guard_threshold = settings.get("guard_multibox_score_gt")
    guard_triggered = (
        num_boxes > 1
        and guard_threshold is not None
        and current_score > float(guard_threshold)
    )
    top_k = (
        int(settings["single_box_top_k"])
        if num_boxes <= 1
        else int(settings["multibox_top_k"])
    )
    scheduled_max_step = (
        int(settings["single_box_max_step"])
        if num_boxes <= 1
        else int(settings["multibox_max_step"])
    )
    if max_step is not None:
        scheduled_max_step = int(max_step)
    prior_weight = float(settings["prior_weight"])
    guard_fast_triggered = False
    if guard_triggered and num_iter is None:
        scheduled_num_iter = int(settings.get("guard_num_iter", scheduled_num_iter))
        fast_guard_score = settings.get("guard_fast_score_gt")
        if fast_guard_score is not None and current_score > float(fast_guard_score):
            scheduled_num_iter = int(settings.get("guard_fast_num_iter", scheduled_num_iter))
            guard_fast_triggered = True
    if guard_triggered and max_step is None:
        scheduled_max_step = int(settings.get("guard_max_step", scheduled_max_step))
    if guard_triggered:
        top_k = int(settings.get("guard_top_k", top_k))
        prior_weight = float(settings.get("guard_prior_weight", prior_weight))
    logits = [] if prior_weight == 0.0 and top_k == 0 else builtin_deepset_axis_prior_logits(
        engine,
        policy=policy,
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        candidate_count=int(settings["candidate_count"]),
        turn=0,
        hist_bins=int(settings["hist_bins"]),
        num_action_scale=int(num_action_scale),
        floor_margin=float(settings["floor_margin"]),
    )
    out = dict(
        engine.run_mcts(
            int(scheduled_num_iter),
            int(scheduled_max_step),
            float(cover_penalty),
            float(pen_rate),
            float(exp_weight),
            float(gamma),
            int(seed),
            logits,
            [],
            float(prior_weight),
            0.0,
            bool(transposition_table),
            int(transposition_table_size),
            int(top_k),
        )
    )
    out["learned_mcts_prior_used"] = True
    out["mcts_prior_mode"] = str(mode or "balanced")
    out["mcts_prior_top_k"] = int(top_k)
    out["mcts_prior_max_step"] = int(scheduled_max_step)
    out["mcts_prior_weight"] = float(prior_weight)
    out["mcts_prior_num_iter"] = int(scheduled_num_iter)
    out["mcts_prior_guarded"] = bool(guard_triggered)
    out["mcts_prior_guard_fast"] = bool(guard_fast_triggered)
    out["mcts_prior_initial_score"] = float(current_score)
    out["num_boxes"] = int(num_boxes)
    return out


def run_builtin_deepset_dynamic_prior_mcts(
    engine: Any,
    *,
    num_iter: int | None = None,
    max_step: int | None = None,
    mode: str = "balanced",
    policy: str = "default",
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    exp_weight: float = 0.001,
    gamma: float = 1.0,
    seed: int = 7777,
    transposition_table: bool = False,
    transposition_table_size: int = 8192,
    **overrides: Any,
) -> dict[str, Any]:
    """Run native MCTS with state-refreshed DeepSets priors.

    Unlike :func:`run_builtin_deepset_prior_mcts`, which computes one prior at
    the root state, this opt-in research helper asks the C++ engine to refresh
    DeepSets candidate scores when each MCTS node is created.  Exact SMART
    reward evaluation remains unchanged; the learned model only prunes/orders
    action expansion.
    """

    if not hasattr(engine, "run_deepset_prior_mcts"):
        raise RuntimeError("smart._cpp was built without dynamic DeepSets MCTS support")
    settings = native_deepset_mcts_prior_defaults(mode)
    settings.update(overrides)
    try:
        stats = dict(engine.stats())
        num_boxes = int(stats.get("num_boxes", 0) or 0)
    except Exception:
        num_boxes = 0
    scheduled_num_iter = int(settings.get("recommended_num_iter", 50) if num_iter is None else num_iter)
    top_k = (
        int(settings["single_box_top_k"])
        if num_boxes <= 1
        else int(settings["multibox_top_k"])
    )
    scheduled_max_step = (
        int(settings["single_box_max_step"])
        if num_boxes <= 1
        else int(settings["multibox_max_step"])
    )
    if max_step is not None:
        scheduled_max_step = int(max_step)
    out = dict(
        engine.run_deepset_prior_mcts(
            load_builtin_deepset_policy(policy),
            int(scheduled_num_iter),
            int(scheduled_max_step),
            float(cover_penalty),
            float(pen_rate),
            float(exp_weight),
            float(gamma),
            int(seed),
            float(settings["prior_weight"]),
            int(settings["candidate_count"]),
            int(settings["hist_bins"]),
            int(top_k),
            float(settings["floor_margin"]),
            bool(transposition_table),
            int(transposition_table_size),
        )
    )
    out["learned_mcts_prior_used"] = True
    out["dynamic_mcts_prior_used"] = True
    out["mcts_prior_mode"] = str(mode or "balanced")
    out["mcts_prior_top_k"] = int(top_k)
    out["mcts_prior_max_step"] = int(scheduled_max_step)
    out["mcts_prior_weight"] = float(settings["prior_weight"])
    out["mcts_prior_num_iter"] = int(scheduled_num_iter)
    out["num_boxes"] = int(num_boxes)
    return out


_DEEPSET_PORTFOLIO_PRESETS: dict[str, dict[str, Any]] = {
    "speed": {
        "single_box_steps": 5,
        "multibox_steps": 5,
        "multibox_profile": "auto",
    },
    "balanced": {
        "single_box_steps": 6,
        "multibox_steps": 6,
        "multibox_profile": "hard",
    },
    "quality": {
        "single_box_steps": 6,
        "multibox_steps": 6,
        "multibox_profile": "hard",
    },
}


def native_deepset_portfolio_defaults(mode: str = "balanced") -> dict[str, Any]:
    """Return the current research portfolio settings.

    The portfolio is deliberately simple: exact native refine is used for
    one-box states, where learned routing has not helped, while multibox states
    use the packaged DeepSets router.  This helper is opt-in research code, not
    the default SMART release path.
    """

    key = str(mode or "balanced").lower()
    if key not in _DEEPSET_PORTFOLIO_PRESETS:
        valid = ", ".join(sorted(_DEEPSET_PORTFOLIO_PRESETS))
        raise ValueError(f"unknown DeepSets portfolio mode {mode!r}; expected one of: {valid}")
    return dict(_DEEPSET_PORTFOLIO_PRESETS[key])


def run_native_deepset_portfolio_refine(
    engine: Any,
    checkpoint_or_policy: Any,
    *,
    mode: str = "balanced",
    max_steps: int | None = None,
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Run the current best-known opt-in refine portfolio.

    ``mode="speed"`` prefers lower wall time on multibox states.  ``balanced``
    and ``quality`` spend more router turns for a higher score.  The accepted
    action is always exact-scored by SMART/Manifold; the portfolio only chooses
    whether to use native exact refine or the learned candidate router.
    """

    settings = native_deepset_portfolio_defaults(mode)
    settings.update(overrides.pop("portfolio", {}) or {})
    try:
        stats = dict(engine.stats())
        num_boxes = int(stats.get("num_boxes", 0) or 0)
    except Exception:
        num_boxes = 0

    if num_boxes <= 1:
        steps = int(max_steps or settings["single_box_steps"])
        out = dict(engine.run_refine(steps, float(cover_penalty), float(pen_rate)))
        if "total_reward" not in out:
            out["total_reward"] = float(sum(float(value) for value in out.get("rewards", [])))
        out.setdefault("exact_checks", 0)
        out.setdefault("exact_errors", 0)
        out.setdefault("adaptive_high_budget_uses", 0)
        out.setdefault("small_pool_exact_uses", 0)
        out.setdefault("nonfinite_proxy_rescue_checks", 0)
        out["router_profile"] = "portfolio_exact_native"
        out["portfolio_mode"] = str(mode or "balanced")
        out["learned_router_used"] = False
        out["num_boxes"] = num_boxes
        return out

    profile = str(settings["multibox_profile"])
    steps = int(max_steps or settings["multibox_steps"])
    out = run_native_deepset_policy_refine(
        engine,
        checkpoint_or_policy,
        max_steps=steps,
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        profile=profile,
        **overrides,
    )
    out["portfolio_mode"] = str(mode or "balanced")
    out["portfolio_multibox_profile"] = profile
    out["num_boxes"] = num_boxes
    return out


def run_builtin_deepset_policy_refine(
    engine: Any,
    *,
    max_steps: int,
    policy: str = "default",
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    profile: str = "auto",
    **overrides: Any,
) -> dict[str, Any]:
    """Run C++ native refine with the packaged opt-in DeepSets router.

    This is the public convenience entry point for the current research
    controller.  The router only reduces the exact candidate set; the final
    action choice still uses the exact native SMART/Manifold score.  With the
    default ``profile="auto"``, small one-box states use the exact native path
    directly because the learned router does not help there.
    """

    out = run_native_deepset_policy_refine(
        engine,
        load_builtin_deepset_policy(policy),
        max_steps=max_steps,
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        profile=profile,
        **overrides,
    )
    out.setdefault("builtin_policy", str(policy or "default"))
    return out


def run_builtin_deepset_portfolio_refine(
    engine: Any,
    *,
    mode: str = "balanced",
    max_steps: int | None = None,
    policy: str = "default",
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Run the packaged DeepSets portfolio with the built-in checkpoint."""

    out = run_native_deepset_portfolio_refine(
        engine,
        load_builtin_deepset_policy(policy),
        mode=mode,
        max_steps=max_steps,
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        **overrides,
    )
    out.setdefault("builtin_policy", str(policy or "default"))
    return out


# Compatibility names used by legacy Python modules. The official native
# implementation is C++, and smart.native resolves directly to this module.
action_scales = native_action_scales
action_indices = native_action_indices
action_upper_rewards = native_action_upper_rewards
bbox_action_upper_rewards = native_bbox_action_upper_rewards
total_bbox_volume = native_total_bbox_volume
centroid_proxy_axis_rewards = native_centroid_proxy_axis_rewards
centroid_proxy_axis_metrics = native_centroid_proxy_axis_metrics
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
