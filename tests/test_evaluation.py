from __future__ import annotations

import random

import numpy as np
import trimesh

from smart.evaluation import EvaluationMetrics


def test_cub_cd_sampling_is_deterministic_and_restores_rng() -> None:
    metrics = object.__new__(EvaluationMetrics)
    metrics.bbox_meshes = [trimesh.creation.box(extents=(1.0, 1.0, 1.0))]
    metrics.shapenet_mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    np.random.seed(123)
    random.seed(123)
    expected_np = np.random.random()
    expected_py = random.random()

    np.random.seed(123)
    random.seed(123)
    first = metrics.cub_cd(num_points=64)
    assert np.random.random() == expected_np
    assert random.random() == expected_py

    np.random.seed(999)
    random.seed(999)
    second = metrics.cub_cd(num_points=64)
    assert second == first
