from __future__ import annotations

import smart.rust as sr
import pymesh


def test_python_fallback_kernel_shapes() -> None:
    assert sr.bbox_volumes([[0, 0, 0, 1, 2, 3]]) == [6]
    assert sr.total_bbox_volume([[0, 0, 0, 1, 2, 3], [0, 0, 0, 2, 2, 2]]) == 14
    assert sr.bbox_union_bounds([[0, 0, 0, 1, 1, 1], [-1, 0, 0, 0.5, 2, 1]]) == [-1, 0, 0, 1, 2, 1]
    assert sr.bbox_union_volume([[0, 0, 0, 1, 1, 1], [-1, 0, 0, 0.5, 2, 1]]) == 4
    assert sr.bbox_rot_state_key(
        [[0, 0, 0, 1, 1, 1]],
        [[1, 0, 0, 0, 1, 0, 0, 0, 1]],
    )
    assert sr.bbox_valid_mask([[0, 0, 0, 1, 2, 3], [1, 0, 0, 1, 2, 3]]) == [True, False]
    assert sr.coverage_mask([[0, 0, 0], [2, 0, 0]], [-1, -1, -1, 1, 1, 1]) == [True, False]
    proxy_rewards = dict(
        sr.centroid_proxy_axis_rewards(
            [[0.5, 0.5, 0.5], [1.2, 0.5, 0.5]],
            [0.5, 0.5],
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            num_action_scale=2,
            action_unit=0.3,
            volume_sum=1.0,
            last_bbox_score=0.0,
            cover_penalty=100.0,
            pen_rate=1.0,
        )
    )
    assert len(proxy_rewards) == 12
    assert abs(proxy_rewards[7] + 0.3) < 1e-12
    assert proxy_rewards[0] < -50.0
    if sr.CandidateBitsetState is not None:
        bitset_state = sr.CandidateBitsetState(
            [[0.5, 0.5, 0.5], [1.2, 0.5, 0.5]],
            [0.5, 0.5],
            1.0,
        )
        state_rewards = dict(
            bitset_state.axis_rewards(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
                num_action_scale=2,
                action_unit=0.3,
                last_bbox_score=0.0,
                cover_penalty=100.0,
                pen_rate=1.0,
            )
        )
        assert state_rewards == proxy_rewards
        topk_actions = bitset_state.topk_axis_actions(
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            num_action_scale=2,
            action_unit=0.3,
            last_bbox_score=0.0,
            cover_penalty=100.0,
            pen_rate=1.0,
            bbox_idx=-1,
            top_k=3,
        )
        assert [action for action, _ in topk_actions] == [7, 0, 1]
        assert abs(topk_actions[0][1] + 0.3) < 1e-12
        assert all(abs(reward + 50.3) < 1e-12 for _, reward in topk_actions[1:])
    assert sr.action_count(3, 2) == 39
    assert sr.action_scales(4) == [-2.0, -1.0, 1.0, 2.0]
    assert sr.bavf_scores([1.0, 2.0], [2.0, 4.0], alpha=100.0) == [50.0, 50.0]
    assert abs(sr.merge_bavf_reward(
        prev_bvs=1.8,
        left_bbox_volume=0.5,
        right_bbox_volume=0.4,
        merged_bbox_volume=0.7,
        shape_volume=1.0,
    ) - 0.2) < 1e-12
    assert sr.untried_actions([False, True, False]) == [0, 2]
    assert sr.single_untried_action_mask(4, 2) == [True, True, False, True]


def test_action_mapping_matches_legacy_order() -> None:
    assert sr.action_indices(1, 2) == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 2, 0],
        [0, 2, 1],
        [0, 3, 0],
        [0, 3, 1],
        [0, 4, 0],
        [0, 4, 1],
        [0, 5, 0],
        [0, 5, 1],
        [0, 6, 0],
    ]
    assert sr.opposite_actions(1, 2) == [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12]
    assert sr.opposite_action_mask(0, 1, 2) == [False, True, False, False, False, False, False, False, False, False, False, False, False]
    assert sr.mcts_child_action_mask(13, 0, 2) == sr.opposite_action_mask(0, 1, 2)

    parent_mask = [False] * 13
    parent_mask[3] = True
    child_mask = sr.mcts_child_action_mask(13, 0, 2, parent_mask)
    assert child_mask == [False, True, False, True, False, False, False, False, False, False, False, False, False]
    assert parent_mask == [False, False, False, True, False, False, False, False, False, False, False, False, False]


def test_apply_axis_action_uses_legacy_scale_order() -> None:
    bounds = [[0, 0, 0, 1, 1, 1]]

    assert sr.apply_axis_action(bounds, action=0, num_action_scale=2, action_unit=0.1) == [
        [-0.1, 0, 0, 1, 1, 1]
    ]
    assert sr.apply_axis_action(bounds, action=1, num_action_scale=2, action_unit=0.1) == [
        [0.1, 0, 0, 1, 1, 1]
    ]
    assert sr.apply_axis_action(bounds, action=12, num_action_scale=2, action_unit=0.1) == bounds


def test_manifold_state_apply_and_rollback_parity() -> None:
    if not sr.manifold_bridge_available() or sr.ManifoldState is None:
        return

    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    faces = [
        [1, 3, 0],
        [1, 5, 7],
        [4, 6, 7],
        [0, 2, 6],
        [2, 3, 7],
        [0, 5, 1],
        [3, 2, 0],
        [1, 7, 3],
        [4, 7, 5],
        [0, 6, 4],
        [2, 7, 6],
        [0, 4, 5],
    ]
    rotation = [[1, 0, 0, 0, 1, 0, 0, 0, 1]]
    state = sr.ManifoldState(
        vertices,
        faces,
        [[0, 0, 0, 1, 1, 1]],
        rotation,
        2,
        0.1,
        1.0,
        0.0,
        True,
        1024,
    )

    action = 7
    score = state.score_axis_action(action, cover_penalty=100.0, pen_rate=1.0)
    reward, bbox_idx, bounds, delta_rotation, last_score = state.apply_axis_action_delta(
        action, cover_penalty=100.0, pen_rate=1.0
    )
    assert abs(score - reward) < 1e-12
    assert bbox_idx == 0
    assert bounds == [0.0, 0.0, 0.0, 1.1, 1.0, 1.0]
    assert delta_rotation == rotation[0]
    assert abs(last_score - score) < 1e-12
    assert abs(state.last_bbox_score() - score) < 1e-12
    assert state.bbox_params(0) == (bounds, delta_rotation)
    assert state.bounds() == [[0.0, 0.0, 0.0, 1.1, 1.0, 1.0]]

    state.rollback()
    assert state.bounds() == [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
    assert abs(state.last_bbox_score()) < 1e-12


def test_action_upper_rewards_are_bvs_reward_bounds() -> None:
    rewards = sr.action_upper_rewards(
        [[0, 0, 0, 1, 1, 1]],
        num_action_scale=2,
        action_unit=0.1,
        volume_sum=1.0,
        last_bbox_score=0.0,
    )

    assert len(rewards) == 13
    assert abs(rewards[0] + 0.1) < 1e-12
    assert abs(rewards[1] + 0.1) < 1e-12
    assert rewards[-1] == 0.0
    assert sr.bbox_action_upper_rewards(
        [[0, 0, 0, 1, 1, 1]],
        bbox_idx=0,
        num_action_scale=2,
        action_unit=0.1,
        volume_sum=1.0,
        last_bbox_score=0.0,
    ) == rewards


def test_bbox_state_tracks_action_space_and_updates() -> None:
    state = sr.BBoxState(
        [[0, 0, 0, 1, 1, 1], [0, 0, 0, 2, 1, 1]],
        num_action_scale=2,
        action_unit=0.1,
        volume_sum=3.0,
        last_bbox_score=-0.2,
    )

    assert state.num_bbox() == 2
    assert state.num_actions() == 26
    assert state.volumes() == [1.0, 2.0]
    assert state.total_volume() == 3.0
    assert state.bvs() == 1.0
    assert state.valid_mask() == [True, True]
    assert state.valid_count() == 2
    assert state.last_bbox_score() == -0.2
    assert len(state.state_key()) > 0
    assert state.action_upper_rewards() == sr.action_upper_rewards(
        state.bounds(),
        num_action_scale=2,
        action_unit=0.1,
        volume_sum=3.0,
        last_bbox_score=-0.2,
    )
    assert state.bbox_action_upper_rewards(0) == sr.bbox_action_upper_rewards(
        state.bounds(),
        bbox_idx=0,
        num_action_scale=2,
        action_unit=0.1,
        volume_sum=3.0,
        last_bbox_score=-0.2,
    )

    next_state = state.after_axis_action(0)
    assert next_state.bounds()[0] == [-0.1, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert state.bounds()[0] == [0, 0, 0, 1, 1, 1]

    state.apply_axis_action_in_place(0)
    assert state.bounds()[0] == [-0.1, 0.0, 0.0, 1.0, 1.0, 1.0]
    state.set_last_bbox_score(-0.4)
    assert state.last_bbox_score() == -0.4
    assert state.with_last_bbox_score(-0.6).last_bbox_score() == -0.6
    assert state.last_bbox_score() == -0.4

    invalid_state = sr.BBoxState(
        [[0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1]],
        num_action_scale=2,
        action_unit=0.1,
        volume_sum=1.0,
        last_bbox_score=0.0,
    )
    assert invalid_state.valid_mask() == [True, False]
    assert invalid_state.valid_count() == 1


def test_manifold_state_matches_exact_cube_bridge_when_available() -> None:
    if not sr.using_rust() or not sr.manifold_bridge_available() or sr.ManifoldState is None:
        return

    vertices = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    faces = [
        [1, 3, 0],
        [1, 5, 7],
        [4, 6, 7],
        [0, 2, 6],
        [2, 3, 7],
        [0, 5, 1],
        [3, 2, 0],
        [1, 7, 3],
        [4, 7, 5],
        [0, 6, 4],
        [2, 7, 6],
        [0, 4, 5],
    ]
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    state = sr.ManifoldState(
        vertices,
        faces,
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        rotations,
        2,
        0.1,
        1.0,
        0.0,
    )

    assert state.valid_count() == 1
    assert abs(state.covered() - 1.0) < 1e-12
    assert abs(state.score(cover_penalty=100.0, pen_rate=1.0)) < 1e-12
    stats0 = state.cache_stats()
    assert stats0["reward_cache_size"] == 0
    assert abs(state.score_axis_action(0, 100.0, 1.0) + 0.1) < 1e-12
    stats1 = state.cache_stats()
    assert stats1["reward_cache_misses"] == stats0["reward_cache_misses"] + 1
    assert abs(state.score_axis_action(0, 100.0, 1.0) + 0.1) < 1e-12
    stats2 = state.cache_stats()
    assert stats2["reward_cache_hits"] == stats1["reward_cache_hits"] + 1

    no_union_cache_state = sr.ManifoldState(
        vertices,
        faces,
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        rotations,
        2,
        0.1,
        1.0,
        0.0,
        False,
        1024,
    )
    assert abs(no_union_cache_state.score_axis_action(0, 100.0, 1.0) + 0.1) < 1e-12
    assert abs(no_union_cache_state.score_axis_action(0, 100.0, 1.0) + 0.1) < 1e-12
    no_union_stats = no_union_cache_state.cache_stats()
    assert no_union_stats["reward_cache_size"] == 1
    assert no_union_stats["reward_cache_hits"] == 1
    assert no_union_stats["reward_cache_misses"] == 1
    assert no_union_stats["except_union_builds"] == 0
    assert no_union_stats["except_union_cache_hits"] == 0

    replacement_score = state.score_replacement(
        0,
        [-0.1, 0.0, 0.0, 1.0, 1.0, 1.0],
        rotations[0],
        100.0,
        1.0,
    )
    assert abs(replacement_score - state.score_axis_action(0, 100.0, 1.0)) < 1e-12
    actions, rewards = state.score_action_batch([True], 100.0, 1.0, -1e9)
    assert actions == [0]
    assert abs(rewards[0] + 0.1) < 1e-12
    assert abs(state.apply_axis_action(0, 100.0, 1.0) + 0.1) < 1e-12
    assert state.bounds()[0] == [-0.1, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert abs(state.last_bbox_score() + 0.1) < 1e-12
    state.rollback()
    assert state.bounds()[0] == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert state.last_bbox_score() == 0.0


def test_mcts_math_helpers_are_stable() -> None:
    probs = sr.softmax_scaled([100.0, 101.0], scale=100.0)
    assert len(probs) == 2
    assert abs(sum(probs) - 1.0) < 1e-12
    assert probs[1] > probs[0]

    scores = sr.ucb_scores(parent_visits=10, child_qs=[1.0, 2.0], child_visits=[2, 5], exp_weight=0.5)
    assert len(scores) == 2
    assert scores[1] > scores[0]
    assert sr.ucb_best_indices(parent_visits=10, child_qs=[1.0, 2.0], child_visits=[2, 5], exp_weight=0.5) == [1]
    assert sr.ucb_best_indices(parent_visits=10, child_qs=[1.0, 1.0], child_visits=[0, 0], exp_weight=0.5) == [0, 1]
    assert sr.incremental_average(previous=2.0, count=3, value=6.0) == 3.0
    assert sr.discounted_reward([1.0, 2.0, 3.0], gamma=0.5) == 2.75


def test_symmetric_chamfer_matches_expected_small_case() -> None:
    left = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    right = [[0.0, 0.0, 0.0]]

    assert sr.symmetric_chamfer(left, right) == 2.0


def test_tetmesh_helpers_match_legacy_shapes_and_order() -> None:
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    voxels = [
        [0, 1, 2, 3],
        [0, 1, 2, 4],
    ]

    assert sr.tetra_volumes(vertices, voxels) == [1.0 / 6.0, 1.0 / 6.0]
    assert sr.tetra_centroids(vertices, voxels) == [
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
    ]
    assert sr.tetra_surface_faces(voxels) == [
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1, 4],
        [0, 2, 4],
        [1, 2, 4],
    ]
    assert sr.tetra_adjacency(voxels) == [[1], [0]]
    part_volumes, part_bounds, part_points = sr.partition_summaries(
        vertices,
        voxels,
        [1.0 / 6.0, 1.0 / 6.0],
        [[0], [1]],
    )
    assert part_volumes == [1.0 / 6.0, 1.0 / 6.0]
    assert part_bounds == [
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, -1.0, 1.0, 1.0, 0.0],
    ]
    assert part_points[0] == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    _, _, unique_part_points = sr.partition_summaries(
        vertices,
        voxels,
        [1.0 / 6.0, 1.0 / 6.0],
        [[0, 1]],
        unique_points=True,
    )
    assert len(unique_part_points[0]) == 15


def test_pymesh_shim_uses_tetmesh_helpers_without_changing_values() -> None:
    mesh = pymesh.form_mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        voxels=[[0, 1, 2, 3]],
    )

    assert mesh.get_attribute("voxel_volume").tolist() == [1.0 / 6.0]
    assert mesh.get_attribute("voxel_centroid").tolist() == [0.25, 0.25, 0.25]
    assert mesh.get_voxel_adjacent_voxels(0).tolist() == []


def test_gmsh_loader_preserves_legacy_indexing_and_surface_order(tmp_path) -> None:
    msh = tmp_path / "tetra.msh"
    msh.write_text(
        "\n".join(
            [
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$Nodes",
                "5",
                "10 0 0 0",
                "20 1 0 0",
                "30 0 1 0",
                "40 0 0 1",
                "50 0 0 -1",
                "$EndNodes",
                "$Elements",
                "2",
                "1 4 0 10 20 30 40",
                "2 4 0 10 20 30 50",
                "$EndElements",
            ]
        ),
        encoding="utf-8",
    )

    vertices, faces, voxels = sr.load_gmsh(str(msh))

    assert vertices[1] == [1.0, 0.0, 0.0]
    assert voxels == [[0, 1, 2, 3], [0, 1, 2, 4]]
    assert faces == [
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1, 4],
        [0, 2, 4],
        [1, 2, 4],
    ]

    mesh = pymesh.load_mesh(msh)
    assert mesh.vertices.tolist() == vertices
    assert mesh.faces.tolist() == faces
    assert mesh.voxels.tolist() == voxels


def test_gmsh_writer_roundtrips_through_pymesh_shim(tmp_path) -> None:
    path = tmp_path / "written" / "tetra.msh"
    mesh = pymesh.form_mesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        faces=[[0, 1, 2]],
        voxels=[[0, 1, 2, 3]],
    )

    pymesh.save_mesh(path, mesh, "voxel_partition", ascii=True)

    loaded = pymesh.load_mesh(path)
    assert loaded.vertices.tolist() == mesh.vertices.tolist()
    assert loaded.faces.tolist() == mesh.faces.tolist()
    assert loaded.voxels.tolist() == mesh.voxels.tolist()


def test_tet_clipping_metrics_match_simple_tetrahedron() -> None:
    if not sr.using_rust():
        return

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    voxels = [[0, 1, 2, 3]]
    metrics = sr.tet_clipping_metrics(
        vertices,
        voxels,
        [vertices],
        surface_volume=1.0 / 6.0,
        max_boxes=1,
    )

    assert abs(metrics["BVS"] - 1.0) < 1e-12
    assert abs(metrics["Covered"] - 1.0) < 1e-12
    assert abs(metrics["MOV"]) < 1e-12
    assert abs(metrics["TOV"]) < 1e-12
    assert abs(metrics["vIoU"] - 1.0) < 1e-12

    assert sr.TetClippingState is not None
    state = sr.TetClippingState(vertices, voxels, 1.0 / 6.0)
    state_metrics = state.metrics([vertices], max_boxes=1)
    for key in ("BVS", "Covered", "MOV", "TOV", "vIoU"):
        assert abs(state_metrics[key] - metrics[key]) < 1e-12

    box_metrics = state.metrics_for_boxes(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
        max_boxes=1,
    )
    assert box_metrics["BVS"] == 6.0
    assert abs(box_metrics["Covered"] - 1.0) < 1e-12


def test_optional_manifold_cpp_bridge_cube_volume() -> None:
    if not sr.using_rust() or not sr.manifold_bridge_available():
        return
    assert abs(sr.manifold_cube_volume(1.25, 2.0, 0.5) - 1.25) < 1e-6

    vertices = [[x, y, z] for x in [0.0, 1.0] for y in [0.0, 1.0] for z in [0.0, 1.0]]
    faces = [
        [1, 3, 0],
        [1, 5, 7],
        [4, 6, 7],
        [0, 2, 6],
        [2, 3, 7],
        [0, 5, 1],
        [3, 2, 0],
        [1, 7, 3],
        [4, 7, 5],
        [0, 6, 4],
        [2, 7, 6],
        [0, 4, 5],
    ]
    assert abs(sr.manifold_mesh_volume(vertices, faces) - 1.0) < 1e-6
    assert (
        abs(sr.manifold_axis_box_intersection_volume(vertices, faces, [0, 0, 0, 0.5, 1, 1]) - 0.5)
        < 1e-6
    )

    mesh = sr.ManifoldBridgeMesh(vertices, faces)
    half_box = [[x, y, z] for x in [0.0, 0.5] for y in [0.0, 1.0] for z in [0.0, 1.0]]
    assert abs(mesh.volume() - 1.0) < 1e-6
    assert abs(mesh.residual_volume_for_boxes([]) - 1.0) < 1e-6
    assert abs(mesh.residual_volume_for_boxes([half_box]) - 0.5) < 1e-6
    assert (
        abs(
            mesh.residual_volume_for_box_params(
                [[0, 0, 0, 0.5, 1, 1]],
                [[1, 0, 0, 0, 1, 0, 0, 0, 1]],
            )
            - 0.5
        )
        < 1e-6
    )
    action, reward = mesh.best_axis_action(
        [[0, 0, 0, 1.2, 1, 1]],
        [[1, 0, 0, 0, 1, 0, 0, 0, 1]],
        0,
        2,
        0.1,
        1.0,
        -0.2,
        100.0,
        1.0,
        -1e100,
    )
    assert action == 6
    assert abs(reward - 0.1) < 1e-5

    actions, rewards = mesh.best_axis_actions_for_mask(
        [[0, 0, 0, 1.2, 1, 1], [0, 0, 0, 0, 1, 1]],
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
        ],
        [True, False],
        2,
        0.1,
        1.0,
        -0.2,
        100.0,
        1.0,
        -1e100,
    )
    assert actions == [6, -1]
    assert abs(rewards[0] - 0.1) < 1e-5
    assert rewards[1] == -1e100
