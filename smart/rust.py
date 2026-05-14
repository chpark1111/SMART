from __future__ import annotations

import math
import os
from typing import Iterable


if os.environ.get("SMART_DISABLE_RUST", "").lower() in {"1", "true", "yes", "on"}:
    _backend = None
else:
    try:
        from . import _rust as _backend  # type: ignore
    except ImportError:
        try:
            import _rust as _backend  # type: ignore
        except ImportError:
            _backend = None

# SMART_DISABLE_RUST lets parity/perf checks compare Python fallback kernels
# with Rust kernels without uninstalling the extension.


RUST_KERNELS = (
    "BBoxState",
    "TetClippingState",
    "ManifoldBridgeMesh",
    "ManifoldState",
    "CandidateBitsetState",
    "manifold_bridge_available",
    "manifold_cube_volume",
    "manifold_mesh_volume",
    "manifold_axis_box_intersection_volume",
    "bbox_volumes",
    "coverage_mask",
    "centroid_proxy_axis_rewards",
    "action_count",
    "action_scales",
    "action_indices",
    "opposite_action",
    "opposite_actions",
    "opposite_action_mask",
    "untried_actions",
    "single_untried_action_mask",
    "mcts_child_action_mask",
    "apply_axis_action",
    "action_upper_rewards",
    "bbox_action_upper_rewards",
    "bbox_valid_mask",
    "total_bbox_volume",
    "bbox_union_bounds",
    "bbox_union_volume",
    "bbox_rot_state_key",
    "bavf_scores",
    "merge_bavf_reward",
    "softmax_scaled",
    "ucb_scores",
    "ucb_best_indices",
    "incremental_average",
    "discounted_reward",
    "symmetric_chamfer",
    "tetra_volumes",
    "tetra_centroids",
    "tetra_surface_faces",
    "tetra_adjacency",
    "load_gmsh",
    "save_gmsh",
    "partition_summaries",
    "tet_clipping_metrics",
    "run_mcts_callbacks",
    "run_greedy_refine_callbacks",
)

_BBOX_STATE_METHODS = (
    "num_bbox",
    "num_actions",
    "bounds",
    "volumes",
    "total_volume",
    "bvs",
    "valid_mask",
    "valid_count",
    "last_bbox_score",
    "set_last_bbox_score",
    "with_last_bbox_score",
    "state_key",
    "action_upper_rewards",
    "bbox_action_upper_rewards",
    "apply_axis_action",
    "after_axis_action",
    "apply_axis_action_in_place",
)


def using_rust() -> bool:
    return _backend is not None


def backend_path() -> str | None:
    if _backend is None:
        return None
    return getattr(_backend, "__file__", None)


def backend_features() -> dict[str, bool]:
    return {
        name: _has_backend_bbox_state() if name == "BBoxState" else _has_backend_function(name)
        for name in RUST_KERNELS
    }


def _has_backend_function(name: str) -> bool:
    return _backend is not None and hasattr(_backend, name)


def _has_backend_bbox_state() -> bool:
    if not _has_backend_function("BBoxState"):
        return False
    bbox_state = _backend.BBoxState
    return all(hasattr(bbox_state, name) for name in _BBOX_STATE_METHODS)


if _has_backend_bbox_state():
    BBoxState = _backend.BBoxState  # type: ignore
else:

    class BBoxState:
        def __init__(
            self,
            bounds: Iterable[Iterable[float]],
            num_action_scale: int,
            action_unit: float,
            volume_sum: float,
            last_bbox_score: float,
        ) -> None:
            if volume_sum <= 0.0:
                raise ValueError("volume_sum must be positive")
            self._bounds = [list(row) for row in bounds]
            self._num_action_scale = int(num_action_scale)
            self._action_unit = float(action_unit)
            self._volume_sum = float(volume_sum)
            self._last_bbox_score = float(last_bbox_score)
            self._refresh_volumes()

        def num_bbox(self) -> int:
            return len(self._bounds)

        def num_actions(self) -> int:
            return action_count(len(self._bounds), self._num_action_scale)

        def bounds(self) -> list[list[float]]:
            return [list(row) for row in self._bounds]

        def volumes(self) -> list[float]:
            return list(self._bbox_volumes)

        def total_volume(self) -> float:
            return float(self._total_bbox_volume)

        def bvs(self) -> float:
            return self._total_bbox_volume / self._volume_sum

        def valid_mask(self) -> list[bool]:
            return bbox_valid_mask(self._bounds)

        def valid_count(self) -> int:
            return sum(1 for is_valid in self.valid_mask() if is_valid)

        def last_bbox_score(self) -> float:
            return self._last_bbox_score

        def set_last_bbox_score(self, last_bbox_score: float) -> None:
            self._last_bbox_score = float(last_bbox_score)

        def with_last_bbox_score(self, last_bbox_score: float) -> "BBoxState":
            return BBoxState(
                self._bounds,
                self._num_action_scale,
                self._action_unit,
                self._volume_sum,
                float(last_bbox_score),
            )

        def state_key(self) -> str:
            return "|".join(
                ",".join(float(value).hex() for value in row) for row in self._bounds
            )

        def action_upper_rewards(self) -> list[float]:
            return action_upper_rewards(
                self._bounds,
                self._num_action_scale,
                self._action_unit,
                self._volume_sum,
                self._last_bbox_score,
            )

        def bbox_action_upper_rewards(self, bbox_idx: int) -> list[float]:
            return bbox_action_upper_rewards(
                self._bounds,
                bbox_idx,
                self._num_action_scale,
                self._action_unit,
                self._volume_sum,
                self._last_bbox_score,
            )

        def apply_axis_action(self, action: int) -> list[list[float]]:
            return apply_axis_action(
                self._bounds,
                action,
                self._num_action_scale,
                self._action_unit,
            )

        def after_axis_action(self, action: int) -> "BBoxState":
            return BBoxState(
                self.apply_axis_action(action),
                self._num_action_scale,
                self._action_unit,
                self._volume_sum,
                self._last_bbox_score,
            )

        def apply_axis_action_in_place(self, action: int) -> None:
            self._bounds = self.apply_axis_action(action)
            self._refresh_volumes()

        def _refresh_volumes(self) -> None:
            self._bbox_volumes = [_bbox_volume(row) for row in self._bounds]
            self._total_bbox_volume = sum(self._bbox_volumes)


TetClippingState = _backend.TetClippingState if _has_backend_function("TetClippingState") else None  # type: ignore
ManifoldBridgeMesh = _backend.ManifoldBridgeMesh if _has_backend_function("ManifoldBridgeMesh") else None  # type: ignore
ManifoldState = _backend.ManifoldState if _has_backend_function("ManifoldState") else None  # type: ignore
CandidateBitsetState = _backend.CandidateBitsetState if _has_backend_function("CandidateBitsetState") else None  # type: ignore


def manifold_bridge_available() -> bool:
    if _has_backend_function("manifold_bridge_available"):
        return bool(_backend.manifold_bridge_available())
    return False


def manifold_cube_volume(x: float, y: float, z: float) -> float:
    if _has_backend_function("manifold_cube_volume"):
        return float(_backend.manifold_cube_volume(float(x), float(y), float(z)))
    raise RuntimeError("Manifold C++ bridge is not available")


def manifold_mesh_volume(
    vertices: Iterable[Iterable[float]], faces: Iterable[Iterable[int]]
) -> float:
    if _has_backend_function("manifold_mesh_volume"):
        verts = [[float(value) for value in row] for row in vertices]
        face_rows = [[int(value) for value in row] for row in faces]
        return float(_backend.manifold_mesh_volume(verts, face_rows))
    raise RuntimeError("Manifold C++ bridge is not available")


def manifold_axis_box_intersection_volume(
    vertices: Iterable[Iterable[float]],
    faces: Iterable[Iterable[int]],
    bounds: Iterable[float],
) -> float:
    if _has_backend_function("manifold_axis_box_intersection_volume"):
        verts = [[float(value) for value in row] for row in vertices]
        face_rows = [[int(value) for value in row] for row in faces]
        box = [float(value) for value in bounds]
        return float(_backend.manifold_axis_box_intersection_volume(verts, face_rows, box))
    raise RuntimeError("Manifold C++ bridge is not available")


def bbox_volumes(bounds: Iterable[Iterable[float]]) -> list[float]:
    data = [list(row) for row in bounds]
    if _has_backend_function("bbox_volumes"):
        return list(_backend.bbox_volumes(data))
    return [_bbox_volume(row) for row in data]


def coverage_mask(points: Iterable[Iterable[float]], bounds: Iterable[float]) -> list[bool]:
    pts = [list(row) for row in points]
    box = list(bounds)
    if _has_backend_function("coverage_mask"):
        return list(_backend.coverage_mask(pts, box))
    _check_bounds(box)
    return [
        box[0] <= point[0] <= box[3]
        and box[1] <= point[1] <= box[4]
        and box[2] <= point[2] <= box[5]
        for point in pts
    ]


def centroid_proxy_axis_rewards(
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
    pts = [[float(value) for value in row] for row in centroids]
    vols = [float(value) for value in volumes]
    box_rows = [[float(value) for value in row] for row in bounds]
    rot_rows = [[float(value) for value in row] for row in rotations]
    if _has_backend_function("centroid_proxy_axis_rewards"):
        return [
            (int(action), float(reward))
            for action, reward in _backend.centroid_proxy_axis_rewards(
                pts,
                vols,
                box_rows,
                rot_rows,
                int(num_action_scale),
                float(action_unit),
                float(volume_sum),
                float(last_bbox_score),
                float(cover_penalty),
                float(pen_rate),
            )
        ]

    if len(pts) != len(vols):
        raise ValueError("centroids and volumes must have the same length")
    if len(box_rows) != len(rot_rows):
        raise ValueError("bounds and rotations must have the same length")
    if volume_sum <= 0.0:
        raise ValueError("volume_sum must be positive")
    scales = _action_scales(int(num_action_scale))
    actions_per_bbox = 6 * int(num_action_scale) + 1
    base_masks = [
        _oriented_centroid_mask(pts, row, rot_rows[idx])
        if _bbox_is_valid(row)
        else [False] * len(pts)
        for idx, row in enumerate(box_rows)
    ]
    bbox_vols = [_bbox_volume(row) if _bbox_is_valid(row) else 0.0 for row in box_rows]
    total_bbox_volume = sum(bbox_vols)
    out: list[tuple[int, float]] = []
    for bbox_idx, row in enumerate(box_rows):
        for coord_idx in range(6):
            for scale_idx, scale in enumerate(scales):
                action = bbox_idx * actions_per_bbox + coord_idx * int(num_action_scale) + scale_idx
                candidate = list(row)
                candidate[coord_idx] += scale * float(action_unit)
                if not _bbox_is_valid(candidate):
                    out.append((action, float("-inf")))
                    continue
                candidate_mask = _oriented_centroid_mask(pts, candidate, rot_rows[bbox_idx])
                union = list(candidate_mask)
                for other_idx, mask in enumerate(base_masks):
                    if other_idx == bbox_idx:
                        continue
                    union = [left or right for left, right in zip(union, mask)]
                covered = sum(volume for volume, masked in zip(vols, union) if masked) / float(volume_sum)
                new_total = total_bbox_volume - bbox_vols[bbox_idx] + _bbox_volume(candidate)
                bvs = new_total / float(volume_sum)
                proxy_score = -abs(bvs - 1.0) - (1.0 - covered) * float(pen_rate) * float(cover_penalty)
                out.append((action, proxy_score - float(last_bbox_score)))
    return out


def action_count(num_bbox: int, num_action_scale: int) -> int:
    if _has_backend_function("action_count"):
        return int(_backend.action_count(num_bbox, num_action_scale))
    return int(num_bbox) * (6 * int(num_action_scale) + 1)


def action_scales(num_action_scale: int) -> list[float]:
    if _has_backend_function("action_scales"):
        return list(_backend.action_scales(num_action_scale))
    return _action_scales(int(num_action_scale))


def action_indices(num_bbox: int, num_action_scale: int) -> list[list[int]]:
    if _has_backend_function("action_indices"):
        return [list(row) for row in _backend.action_indices(num_bbox, num_action_scale)]
    _check_action_scale(int(num_action_scale))
    out: list[list[int]] = []
    for bbox_idx in range(int(num_bbox)):
        for coord_idx in range(6):
            for scale_idx in range(int(num_action_scale)):
                out.append([bbox_idx, coord_idx, scale_idx])
        out.append([bbox_idx, 6, 0])
    return out


def opposite_action(action: int, num_action_scale: int) -> int:
    if _has_backend_function("opposite_action"):
        return int(_backend.opposite_action(action, num_action_scale))
    bbox_idx, coord_idx, scale_idx = _decode_action(int(action), int(num_action_scale))
    if coord_idx == 6:
        return int(action)
    return _encode_action(bbox_idx, coord_idx, int(num_action_scale) - 1 - scale_idx, int(num_action_scale))


def opposite_actions(num_bbox: int, num_action_scale: int) -> list[int]:
    if _has_backend_function("opposite_actions"):
        return list(_backend.opposite_actions(num_bbox, num_action_scale))
    return [opposite_action(action, int(num_action_scale)) for action in range(action_count(num_bbox, num_action_scale))]


def opposite_action_mask(action: int, num_bbox: int, num_action_scale: int) -> list[bool]:
    if _has_backend_function("opposite_action_mask"):
        return list(_backend.opposite_action_mask(int(action), int(num_bbox), int(num_action_scale)))
    total = action_count(num_bbox, num_action_scale)
    if action < 0 or action >= total:
        raise ValueError("action is out of range")
    mask = [False] * total
    mask[opposite_action(action, num_action_scale)] = True
    return mask


def untried_actions(action_mask: Iterable[bool]) -> list[int]:
    data = [bool(value) for value in action_mask]
    if _has_backend_function("untried_actions"):
        return list(_backend.untried_actions(data))
    return [idx for idx, masked in enumerate(data) if not masked]


def single_untried_action_mask(total_actions: int, action: int) -> list[bool]:
    if _has_backend_function("single_untried_action_mask"):
        return list(_backend.single_untried_action_mask(int(total_actions), int(action)))
    if action < 0 or action >= total_actions:
        raise ValueError("action is out of range")
    mask = [True] * int(total_actions)
    mask[int(action)] = False
    return mask


def mcts_child_action_mask(
    total_actions: int,
    action: int,
    num_action_scale: int,
    parent_mask: Iterable[bool] | None = None,
) -> list[bool]:
    total_actions = int(total_actions)
    action = int(action)
    num_action_scale = int(num_action_scale)
    parent = None if parent_mask is None else [bool(value) for value in parent_mask]
    if _has_backend_function("mcts_child_action_mask"):
        return list(
            _backend.mcts_child_action_mask(
                total_actions,
                action,
                num_action_scale,
                parent,
            )
        )
    _check_action_scale(num_action_scale)
    per_bbox = 6 * num_action_scale + 1
    if total_actions <= 0 or total_actions % per_bbox != 0:
        raise ValueError("total_actions must match the legacy bbox action space")
    if action < 0 or action >= total_actions:
        raise ValueError("action is out of range")
    if parent is None:
        mask = [False] * total_actions
    else:
        if len(parent) != total_actions:
            raise ValueError("parent_mask length must match total_actions")
        mask = parent
    mask[opposite_action(action, num_action_scale)] = True
    return mask


def apply_axis_action(
    bounds: Iterable[Iterable[float]],
    action: int,
    num_action_scale: int,
    action_unit: float,
) -> list[list[float]]:
    data = [list(row) for row in bounds]
    if _has_backend_function("apply_axis_action"):
        return [list(row) for row in _backend.apply_axis_action(data, action, num_action_scale, float(action_unit))]

    scales = _action_scales(int(num_action_scale))
    bbox_idx, coord_idx, scale_idx = _decode_action(int(action), int(num_action_scale))
    if bbox_idx >= len(data):
        raise ValueError("action bbox index is out of range")
    for row in data:
        _check_bounds(row)
    if coord_idx < 6:
        data[bbox_idx][coord_idx] += scales[scale_idx] * float(action_unit)
    return data


def action_upper_rewards(
    bounds: Iterable[Iterable[float]],
    num_action_scale: int,
    action_unit: float,
    volume_sum: float,
    last_bbox_score: float,
) -> list[float]:
    data = [list(row) for row in bounds]
    if _has_backend_function("action_upper_rewards"):
        return list(
            _backend.action_upper_rewards(
                data,
                int(num_action_scale),
                float(action_unit),
                float(volume_sum),
                float(last_bbox_score),
            )
        )

    if volume_sum <= 0.0:
        raise ValueError("volume_sum must be positive")
    scales = _action_scales(int(num_action_scale))
    old_volumes = [_bbox_volume(row) for row in data]
    total_volume = sum(old_volumes)
    out: list[float] = []
    for bbox_idx, row in enumerate(data):
        _check_bounds(row)
        for coord_idx in range(6):
            for scale in scales:
                candidate = list(row)
                candidate[coord_idx] += scale * float(action_unit)
                new_volume = _bbox_volume(candidate) if _bbox_is_valid(candidate) else 0.0
                new_total = total_volume - old_volumes[bbox_idx] + new_volume
                bvs = new_total / float(volume_sum)
                out.append(-abs(bvs - 1.0) - float(last_bbox_score))
        bvs = total_volume / float(volume_sum)
        out.append(-abs(bvs - 1.0) - float(last_bbox_score))
    return out


def bbox_action_upper_rewards(
    bounds: Iterable[Iterable[float]],
    bbox_idx: int,
    num_action_scale: int,
    action_unit: float,
    volume_sum: float,
    last_bbox_score: float,
) -> list[float]:
    data = [list(row) for row in bounds]
    if _has_backend_function("bbox_action_upper_rewards"):
        return list(
            _backend.bbox_action_upper_rewards(
                data,
                int(bbox_idx),
                int(num_action_scale),
                float(action_unit),
                float(volume_sum),
                float(last_bbox_score),
            )
        )

    if bbox_idx < 0 or bbox_idx >= len(data):
        raise ValueError("bbox_idx is out of range")
    if volume_sum <= 0.0:
        raise ValueError("volume_sum must be positive")
    scales = _action_scales(int(num_action_scale))
    old_volumes = [_bbox_volume(row) for row in data]
    total_volume = sum(old_volumes)
    row = data[int(bbox_idx)]
    _check_bounds(row)
    out: list[float] = []
    for coord_idx in range(6):
        for scale in scales:
            candidate = list(row)
            candidate[coord_idx] += scale * float(action_unit)
            new_volume = _bbox_volume(candidate) if _bbox_is_valid(candidate) else 0.0
            new_total = total_volume - old_volumes[int(bbox_idx)] + new_volume
            bvs = new_total / float(volume_sum)
            out.append(-abs(bvs - 1.0) - float(last_bbox_score))
    bvs = total_volume / float(volume_sum)
    out.append(-abs(bvs - 1.0) - float(last_bbox_score))
    return out


def bbox_valid_mask(bounds: Iterable[Iterable[float]]) -> list[bool]:
    data = [list(row) for row in bounds]
    if _has_backend_function("bbox_valid_mask"):
        return list(_backend.bbox_valid_mask(data))
    return [_bbox_is_valid(row) for row in data]


def total_bbox_volume(bounds: Iterable[Iterable[float]]) -> float:
    data = [list(row) for row in bounds]
    if _has_backend_function("total_bbox_volume"):
        return float(_backend.total_bbox_volume(data))
    return float(sum(_bbox_volume(row) for row in data))


def bbox_union_bounds(bounds: Iterable[Iterable[float]]) -> list[float]:
    data = [list(row) for row in bounds]
    if _has_backend_function("bbox_union_bounds"):
        return list(_backend.bbox_union_bounds(data))
    if not data:
        raise ValueError("bounds must not be empty")
    out = list(data[0])
    _check_bounds(out)
    for row in data[1:]:
        _check_bounds(row)
        out[0] = min(out[0], row[0])
        out[1] = min(out[1], row[1])
        out[2] = min(out[2], row[2])
        out[3] = max(out[3], row[3])
        out[4] = max(out[4], row[4])
        out[5] = max(out[5], row[5])
    return out


def bbox_union_volume(bounds: Iterable[Iterable[float]]) -> float:
    data = [list(row) for row in bounds]
    if _has_backend_function("bbox_union_volume"):
        return float(_backend.bbox_union_volume(data))
    return _bbox_volume(bbox_union_bounds(data))


def bbox_rot_state_key(
    bounds: Iterable[Iterable[float]],
    rotations: Iterable[Iterable[float]],
) -> str:
    boxes = [list(row) for row in bounds]
    rots = [list(row) for row in rotations]
    if _has_backend_function("bbox_rot_state_key"):
        return str(_backend.bbox_rot_state_key(boxes, rots))
    if len(boxes) != len(rots):
        raise ValueError("bounds and rotations must have the same length")
    parts = []
    for box, rot in zip(boxes, rots):
        _check_bounds(box)
        if len(rot) != 9:
            raise ValueError("rotations must contain 3x3 rows")
        parts.append(
            "b"
            + ",".join(float(value).hex() for value in box)
            + "r"
            + ",".join(float(value).hex() for value in rot)
        )
    return "|".join(parts)


def bavf_scores(part_volumes: Iterable[float], bbox_volumes_: Iterable[float], alpha: float = 100.0) -> list[float]:
    parts = list(part_volumes)
    boxes = list(bbox_volumes_)
    if _has_backend_function("bavf_scores"):
        return list(_backend.bavf_scores(parts, boxes, float(alpha)))
    if len(parts) != len(boxes):
        raise ValueError("part_volumes and bbox_volumes must have the same length")
    return [float(alpha) * (part / box if box > 0 else 0.0) for part, box in zip(parts, boxes)]


def merge_bavf_reward(
    prev_bvs: float,
    left_bbox_volume: float,
    right_bbox_volume: float,
    merged_bbox_volume: float,
    shape_volume: float,
) -> float:
    if _has_backend_function("merge_bavf_reward"):
        return float(
            _backend.merge_bavf_reward(
                float(prev_bvs),
                float(left_bbox_volume),
                float(right_bbox_volume),
                float(merged_bbox_volume),
                float(shape_volume),
            )
        )
    if shape_volume <= 0.0:
        raise ValueError("shape_volume must be positive")
    new_bvs = (
        float(prev_bvs) * float(shape_volume)
        - float(left_bbox_volume)
        - float(right_bbox_volume)
        + float(merged_bbox_volume)
    ) / float(shape_volume)
    return -abs(new_bvs - 1.0) + abs(float(prev_bvs) - 1.0)


def softmax_scaled(values: Iterable[float], scale: float = 100.0) -> list[float]:
    data = [float(value) for value in values]
    if _has_backend_function("softmax_scaled"):
        return list(_backend.softmax_scaled(data, float(scale)))
    if not data:
        return []
    scaled = [value * float(scale) for value in data]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(data) for _ in data]
    return [value / total for value in exps]


def ucb_scores(
    parent_visits: int,
    child_qs: Iterable[float],
    child_visits: Iterable[int],
    exp_weight: float,
) -> list[float]:
    qs = [float(value) for value in child_qs]
    visits = [int(value) for value in child_visits]
    if _has_backend_function("ucb_scores"):
        return list(_backend.ucb_scores(int(parent_visits), qs, visits, float(exp_weight)))
    if len(qs) != len(visits):
        raise ValueError("child_qs and child_visits must have the same length")
    if parent_visits <= 0:
        return [float("inf") for _ in qs]
    log_parent = math.log(float(parent_visits))
    out = []
    for q_value, visit_count in zip(qs, visits):
        if visit_count <= 0:
            out.append(float("inf"))
        else:
            out.append(q_value + float(exp_weight) * math.sqrt(2.0 * log_parent / visit_count))
    return out


def ucb_best_indices(
    parent_visits: int,
    child_qs: Iterable[float],
    child_visits: Iterable[int],
    exp_weight: float,
) -> list[int]:
    qs = [float(value) for value in child_qs]
    visits = [int(value) for value in child_visits]
    if _has_backend_function("ucb_best_indices"):
        return list(
            _backend.ucb_best_indices(
                int(parent_visits),
                qs,
                visits,
                float(exp_weight),
            )
        )
    scores = ucb_scores(int(parent_visits), qs, visits, float(exp_weight))
    if not scores:
        return []
    max_score = max(scores)
    return [idx for idx, score in enumerate(scores) if score == max_score]


def incremental_average(previous: float, count: int, value: float) -> float:
    if _has_backend_function("incremental_average"):
        return float(_backend.incremental_average(float(previous), int(count), float(value)))
    return float(previous) / (int(count) + 1) * int(count) + float(value) / (int(count) + 1)


def discounted_reward(rewards: Iterable[float], gamma: float) -> float:
    data = [float(reward) for reward in rewards]
    if _has_backend_function("discounted_reward"):
        return float(_backend.discounted_reward(data, float(gamma)))
    out = 0.0
    for reward in reversed(data):
        out = out * float(gamma) + reward
    return out


def symmetric_chamfer(left: Iterable[Iterable[float]], right: Iterable[Iterable[float]]) -> float:
    left_points = [list(row) for row in left]
    right_points = [list(row) for row in right]
    if _has_backend_function("symmetric_chamfer"):
        return float(_backend.symmetric_chamfer(left_points, right_points))
    if not left_points or not right_points:
        raise ValueError("point sets must not be empty")
    _check_points(left_points)
    _check_points(right_points)
    return _mean_nearest_squared_distance(right_points, left_points) + _mean_nearest_squared_distance(
        left_points, right_points
    )


def tetra_volumes(vertices: Iterable[Iterable[float]], voxels: Iterable[Iterable[int]]) -> list[float]:
    verts = [[float(value) for value in row] for row in vertices]
    tets = [[int(value) for value in row] for row in voxels]
    if _has_backend_function("tetra_volumes"):
        return list(_backend.tetra_volumes(verts, tets))
    _check_vertices(verts)
    _check_voxels(tets, len(verts))
    out = []
    for voxel in tets:
        p0 = verts[voxel[0]]
        p1 = verts[voxel[1]]
        p2 = verts[voxel[2]]
        p3 = verts[voxel[3]]
        a = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
        b = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]]
        c = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]]
        out.append(abs(_dot3(a, _cross3(b, c))) / 6.0)
    return out


def tetra_centroids(vertices: Iterable[Iterable[float]], voxels: Iterable[Iterable[int]]) -> list[float]:
    verts = [[float(value) for value in row] for row in vertices]
    tets = [[int(value) for value in row] for row in voxels]
    if _has_backend_function("tetra_centroids"):
        return list(_backend.tetra_centroids(verts, tets))
    _check_vertices(verts)
    _check_voxels(tets, len(verts))
    out = []
    for voxel in tets:
        for axis in range(3):
            out.append(sum(verts[index][axis] for index in voxel) / 4.0)
    return out


def tetra_surface_faces(voxels: Iterable[Iterable[int]]) -> list[list[int]]:
    tets = [[int(value) for value in row] for row in voxels]
    if _has_backend_function("tetra_surface_faces"):
        return [list(row) for row in _backend.tetra_surface_faces(tets)]
    _check_voxels_shape(tets)
    counts: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for voxel in tets:
        for face in _tet_faces(voxel):
            key = tuple(sorted(face))
            counts[key] = None if key in counts else face
    return [list(face) for face in counts.values() if face is not None]


def tetra_adjacency(voxels: Iterable[Iterable[int]]) -> list[list[int]]:
    tets = [[int(value) for value in row] for row in voxels]
    if _has_backend_function("tetra_adjacency"):
        return [list(row) for row in _backend.tetra_adjacency(tets)]
    _check_voxels_shape(tets)
    face_to_voxels: dict[tuple[int, int, int], list[int]] = {}
    for index, voxel in enumerate(tets):
        for face in _tet_faces(voxel):
            key = tuple(sorted(face))
            face_to_voxels.setdefault(key, []).append(index)
    adjacency = [set() for _ in range(len(tets))]
    for owners in face_to_voxels.values():
        if len(owners) < 2:
            continue
        for owner in owners:
            adjacency[owner].update(other for other in owners if other != owner)
    return [sorted(values) for values in adjacency]


def load_gmsh(filename: str) -> tuple[list[list[float]], list[list[int]], list[list[int]]]:
    path = str(filename)
    if _has_backend_function("load_gmsh"):
        vertices, faces, voxels = _backend.load_gmsh(path)
        return (
            [list(row) for row in vertices],
            [list(row) for row in faces],
            [list(row) for row in voxels],
        )

    lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    if "$Nodes" not in lines or "$Elements" not in lines:
        raise ValueError(f"Unsupported or invalid Gmsh file: {path}")
    nodes_start = lines.index("$Nodes") + 1
    node_count = int(lines[nodes_start].split()[0])
    node_id_to_index: dict[int, int] = {}
    vertices = []
    for offset in range(node_count):
        parts = lines[nodes_start + 1 + offset].split()
        node_id = int(parts[0])
        node_id_to_index[node_id] = offset
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    elements_start = lines.index("$Elements") + 1
    element_count = int(lines[elements_start].split()[0])
    faces = []
    voxels = []
    for offset in range(element_count):
        parts = lines[elements_start + 1 + offset].split()
        element_type = int(parts[1])
        tag_count = int(parts[2])
        ids = [node_id_to_index[int(value)] for value in parts[3 + tag_count :]]
        if element_type == 2 and len(ids) >= 3:
            faces.append(ids[:3])
        elif element_type == 4 and len(ids) >= 4:
            voxels.append(ids[:4])
    if not faces and voxels:
        faces = tetra_surface_faces(voxels)
    return vertices, faces, voxels


def save_gmsh(
    filename: str,
    vertices: Iterable[Iterable[float]],
    faces: Iterable[Iterable[int]],
    voxels: Iterable[Iterable[int]],
) -> None:
    path = str(filename)
    verts = [[float(value) for value in row] for row in vertices]
    tris = [[int(value) for value in row] for row in faces]
    tets = [[int(value) for value in row] for row in voxels]
    if _has_backend_function("save_gmsh"):
        _backend.save_gmsh(path, verts, tris, tets)
        return

    _check_vertices(verts)
    _check_faces(tris, len(verts))
    _check_voxels(tets, len(verts))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    elements = [(2, [int(v) + 1 for v in face[:3]]) for face in tris]
    elements.extend((4, [int(v) + 1 for v in voxel[:4]]) for voxel in tets)
    with open(path, "w", encoding="utf-8") as file:
        file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        file.write("$Nodes\n%d\n" % len(verts))
        for index, vertex in enumerate(verts, start=1):
            file.write("%d %.17g %.17g %.17g\n" % (index, vertex[0], vertex[1], vertex[2]))
        file.write("$EndNodes\n")
        file.write("$Elements\n%d\n" % len(elements))
        for index, (element_type, ids) in enumerate(elements, start=1):
            file.write("%d %d 0 %s\n" % (index, element_type, " ".join(str(value) for value in ids)))
        file.write("$EndElements\n")


def partition_summaries(
    vertices: Iterable[Iterable[float]],
    voxels: Iterable[Iterable[int]],
    volumes: Iterable[float],
    partitions: Iterable[Iterable[int]],
    unique_points: bool = False,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    verts = [[float(value) for value in row] for row in vertices]
    tets = [[int(value) for value in row] for row in voxels]
    vols = [float(value) for value in volumes]
    parts = [[int(value) for value in row] for row in partitions]
    if _has_backend_function("partition_summaries"):
        part_volumes, part_bounds, part_points = _backend.partition_summaries(
            verts, tets, vols, parts, bool(unique_points)
        )
        return (
            list(part_volumes),
            [list(row) for row in part_bounds],
            [list(row) for row in part_points],
        )

    _check_vertices(verts)
    _check_voxels(tets, len(verts))
    if len(vols) != len(tets):
        raise ValueError("volumes must have the same length as voxels")
    out_volumes = []
    out_bounds = []
    out_points = []
    for partition in parts:
        if not partition:
            raise ValueError("partition must not be empty")
        first_vertex = verts[tets[partition[0]][0]]
        min_x, min_y, min_z = first_vertex
        max_x, max_y, max_z = first_vertex
        volume = 0.0
        points = []
        for tet_idx in partition:
            if tet_idx < 0 or tet_idx >= len(tets):
                raise ValueError("partition voxel index is out of range")
            volume += vols[tet_idx]
            for vertex_idx in tets[tet_idx]:
                vertex = verts[vertex_idx]
                min_x = min(min_x, vertex[0])
                min_y = min(min_y, vertex[1])
                min_z = min(min_z, vertex[2])
                max_x = max(max_x, vertex[0])
                max_y = max(max_y, vertex[1])
                max_z = max(max_z, vertex[2])
                points.extend(vertex)
        out_volumes.append(volume)
        out_bounds.append([min_x, min_y, min_z, max_x, max_y, max_z])
        out_points.append(points)
    return out_volumes, out_bounds, out_points


def tet_clipping_metrics(
    vertices: Iterable[Iterable[float]],
    voxels: Iterable[Iterable[int]],
    box_vertices: Iterable[Iterable[Iterable[float]]],
    surface_volume: float,
    max_boxes: int = 8,
    box_volumes: Iterable[float] | None = None,
) -> dict[str, float]:
    verts = [[float(value) for value in row] for row in vertices]
    tets = [[int(value) for value in row] for row in voxels]
    boxes = [
        [[float(value) for value in row] for row in box]
        for box in box_vertices
    ]
    if not _has_backend_function("tet_clipping_metrics"):
        raise RuntimeError("Rust tet_clipping_metrics kernel is not available")
    volumes = None
    if box_volumes is not None:
        volumes = [float(value) for value in box_volumes]
    return {
        str(key): float(value)
        for key, value in _backend.tet_clipping_metrics(
            verts,
            tets,
            boxes,
            float(surface_volume),
            int(max_boxes),
            volumes,
        ).items()
    }


def run_mcts_callbacks(args, env, num_iter: int) -> dict[str, float]:
    if not _has_backend_function("run_mcts_callbacks"):
        raise RuntimeError("Rust MCTS callback runner is not available")
    return dict(_backend.run_mcts_callbacks(args, env, int(num_iter)))


def run_greedy_refine_callbacks(args, env) -> tuple[list[float], int]:
    if not _has_backend_function("run_greedy_refine_callbacks"):
        raise RuntimeError("Rust greedy refine callback runner is not available")
    rewards, count = _backend.run_greedy_refine_callbacks(args, env)
    return list(rewards), int(count)


def _oriented_centroid_mask(
    points: list[list[float]], bounds: list[float], rotation: list[float]
) -> list[bool]:
    _check_bounds(bounds)
    if len(rotation) != 9:
        raise ValueError("rotation must be a flattened 3x3 row-major matrix")
    out = []
    for point in points:
        if len(point) != 3:
            raise ValueError("centroids must be [x, y, z] rows")
        x = point[0] * rotation[0] + point[1] * rotation[1] + point[2] * rotation[2]
        y = point[0] * rotation[3] + point[1] * rotation[4] + point[2] * rotation[5]
        z = point[0] * rotation[6] + point[1] * rotation[7] + point[2] * rotation[8]
        out.append(
            bounds[0] <= x <= bounds[3]
            and bounds[1] <= y <= bounds[4]
            and bounds[2] <= z <= bounds[5]
        )
    return out


def _bbox_volume(row: list[float]) -> float:
    _check_bounds(row)
    return max(0.0, row[3] - row[0]) * max(0.0, row[4] - row[1]) * max(0.0, row[5] - row[2])


def _bbox_is_valid(row: list[float]) -> bool:
    _check_bounds(row)
    return row[0] < row[3] and row[1] < row[4] and row[2] < row[5]


def _check_bounds(row: list[float]) -> None:
    if len(row) != 6:
        raise ValueError("bounds must contain six values: min_x min_y min_z max_x max_y max_z")


def _check_points(points: list[list[float]]) -> None:
    for point in points:
        if len(point) != 3:
            raise ValueError("points must be [x, y, z] rows")


def _mean_nearest_squared_distance(source: list[list[float]], target: list[list[float]]) -> float:
    total = 0.0
    for point in source:
        best = math.inf
        for candidate in target:
            dx = point[0] - candidate[0]
            dy = point[1] - candidate[1]
            dz = point[2] - candidate[2]
            best = min(best, dx * dx + dy * dy + dz * dz)
        total += best
    return total / len(source)


def _check_vertices(vertices: list[list[float]]) -> None:
    for vertex in vertices:
        if len(vertex) != 3:
            raise ValueError("vertices must be [x, y, z] rows")


def _check_faces(faces: list[list[int]], vertex_count: int) -> None:
    for face in faces:
        if len(face) != 3:
            raise ValueError("faces must be triangle index rows")
        for index in face:
            if index < 0 or index >= vertex_count:
                raise ValueError("face index is out of range")


def _check_voxels_shape(voxels: list[list[int]]) -> None:
    for voxel in voxels:
        if len(voxel) != 4:
            raise ValueError("voxels must be tetrahedron index rows")


def _check_voxels(voxels: list[list[int]], vertex_count: int) -> None:
    _check_voxels_shape(voxels)
    for voxel in voxels:
        for index in voxel:
            if index < 0 or index >= vertex_count:
                raise ValueError("voxel index is out of range")


def _dot3(left: list[float], right: list[float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross3(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _tet_faces(voxel: list[int]) -> list[tuple[int, int, int]]:
    a, b, c, d = voxel
    return [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]


def _check_action_scale(num_action_scale: int) -> None:
    if num_action_scale <= 0:
        raise ValueError("num_action_scale must be positive")


def _action_scales(num_action_scale: int) -> list[float]:
    _check_action_scale(num_action_scale)
    if num_action_scale % 2 != 0:
        raise ValueError("num_action_scale must be the expanded even legacy value")
    half = num_action_scale // 2
    return [-(2.0**idx) for idx in range(half - 1, -1, -1)] + [2.0**idx for idx in range(half)]


def _decode_action(action: int, num_action_scale: int) -> tuple[int, int, int]:
    _check_action_scale(num_action_scale)
    per_bbox = 6 * num_action_scale + 1
    bbox_idx = action // per_bbox
    local_idx = action % per_bbox
    if local_idx == 6 * num_action_scale:
        return bbox_idx, 6, 0
    return bbox_idx, local_idx // num_action_scale, local_idx % num_action_scale


def _encode_action(bbox_idx: int, coord_idx: int, scale_idx: int, num_action_scale: int) -> int:
    return bbox_idx * (6 * num_action_scale + 1) + coord_idx * num_action_scale + scale_idx
