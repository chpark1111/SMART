from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import smart.cpp as sc
import smart
import smart.native as sn
from smart import native_runner
import smart.native_compat as sr


pytestmark = pytest.mark.skipif(not sc.using_cpp(), reason="smart._cpp is not built")


def test_cpp_native_core_matches_existing_native_helpers() -> None:
    assert sc.native_core_available()
    assert smart.cpp_native_available()
    assert smart.NativeSmartEngine is sc.NativeSmartEngine
    assert sn.native_core_available()
    assert sn.using_cpp()
    assert sn.backend_path() == sc.backend_path()
    assert sn.native_action_count(3, 2) == sc.native_action_count(3, 2)
    assert sc.manifold_bridge_available()
    assert abs(sc.manifold_cube_volume(1.0, 2.0, 3.0) - 6.0) < 1e-5

    assert sc.native_action_count(3, 2) == sr.native_action_count(3, 2)
    assert sc.native_action_scales(6) == sr.native_action_scales(6)
    assert sc.native_action_indices(2, 2) == sr.native_action_indices(2, 2)
    assert sc.native_opposite_actions(2, 2) == sr.native_opposite_actions(2, 2)

    parent_mask = [False] * sc.native_action_count(1, 2)
    parent_mask[3] = True
    assert sc.native_child_action_mask(13, 0, 2) == sr.native_child_action_mask(13, 0, 2)
    assert sc.native_child_action_mask(13, 0, 2, parent_mask) == sr.native_child_action_mask(
        13,
        0,
        2,
        parent_mask,
    )

    rewards = [1.0, 2.0, 3.0]
    assert abs(sc.native_discounted_reward(rewards, 0.5) - sr.native_discounted_reward(rewards, 0.5)) < 1e-12

    child_qs = [1.0, 1.0, 0.0]
    child_visits = [2, 2, 1]
    assert sc.native_ucb_best_count(10, child_qs, child_visits, 0.1) == sr.native_ucb_best_count(
        10,
        child_qs,
        child_visits,
        0.1,
    )
    assert sc.native_best_ucb_child(10, child_qs, child_visits, 0.1, 1) == sr.native_best_ucb_child(
        10,
        child_qs,
        child_visits,
        0.1,
        1,
    )
    assert abs(
        sc.native_prob_skip_exploration(0.5, [0.4, 0.6, 0.7], [0.9, 1.5, 0.3], 3.0, 0.7)
        - sr.native_prob_skip_exploration(0.5, [0.4, 0.6, 0.7], [0.9, 1.5, 0.3], 3.0, 0.7)
    ) < 1e-12

    softmax = sc.native_softmax_scaled([1.0, 2.0, 3.0], 0.5)
    expected = sr.native_softmax_scaled([1.0, 2.0, 3.0], 0.5)
    assert all(abs(left - right) < 1e-12 for left, right in zip(softmax, expected))
    weighted = sc.native_weighted_action_scores(
        [0.1, 0.2, 0.3],
        [1.0, 0.0, 2.0],
        [0.5, 0.5, -1.0],
        100.0,
        0.25,
        0.5,
    )
    assert weighted == [10.5, 20.25, 30.0]
    assert sc.native_top_k_actions([5, 2, 7, 1], [0.5, 0.5, 0.4, 1.0], 3) == [1, 2, 5]
    assert sc.native_best_score_action([5, 2, 7], [0.5, 0.5, 0.4], 1) == 2
    assert sc.native_diverse_escape_actions(
        [1, 5, 3, 2],
        [4.0, 3.0, 2.0, 1.0],
        [1],
        1,
        2,
    ) == [5, 3]
    puct = sc.native_add_puct_prior([1.0, 1.0], [0.0, 0.0], [0, 1], 4, 0.5)
    assert puct == [1.5, 1.25]
    assert callable(sc.run_mcts_callbacks)
    assert callable(sc.run_greedy_refine_callbacks)


def test_cpp_native_normalize_obj_file_keeps_obj_payload(tmp_path) -> None:
    source = tmp_path / "model.obj"
    output = tmp_path / "normalized.obj"
    source.write_text(
        "\n".join(
            [
                "o sample",
                "v 0 0 0 1 0 0",
                "v 2 0 0",
                "v 0 2 0",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = sc.native_normalize_obj_file(
        str(source),
        str(output),
        mode="bbox_diagonal",
        center="bbox",
        target=1.0,
    )

    text = output.read_text(encoding="utf-8")
    assert "o sample" in text
    assert "f 1 2 3" in text
    assert "1 0 0" in text
    assert stats["before"]["vertex_count"] == 3
    assert stats["after"]["bbox_diagonal"] == pytest.approx(1.0)


def test_cpp_native_obj_load_save_triangulates_faces(tmp_path) -> None:
    source = tmp_path / "quad.obj"
    output = tmp_path / "roundtrip.obj"
    source.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "f 1/1/1 2/2/2 3/3/3 4/4/4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    vertices, faces = sc.native_load_obj_mesh(str(source))
    written = sc.native_save_obj_mesh(str(output), vertices, faces)

    assert written == 4
    assert len(vertices) == 4
    assert faces == [[0, 1, 2], [0, 2, 3]]
    assert output.read_text(encoding="utf-8").count("\nf ") == 2


def test_smart_cpp_native_executable_normalizes_without_python(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")
    source = tmp_path / "model.obj"
    output = tmp_path / "normalized.obj"
    source.write_text(
        "v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            str(binary),
            "normalize",
            "--input",
            str(source),
            "--output",
            str(output),
            "--target",
            "1.0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert payload["backend"] == "smart-cpp-native"
    assert payload["after"]["vertex_count"] == 3
    assert payload["after"]["bbox_diagonal"] == pytest.approx(1.0)
    assert output.exists()


def test_smart_cpp_native_executable_splits_coacd_obj_parts(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")
    source = tmp_path / "coacd.obj"
    source.write_text(
        "\n".join(
            [
                "o convex_0",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
                "o convex_1",
                "v 0 0 1",
                "v 1 0 1",
                "v 0 1 1",
                "f 4 5 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "coacd"

    completed = subprocess.run(
        [
            str(binary),
            "split-obj-parts",
            "--input",
            str(source),
            "--output_dir",
            str(out_dir),
            "--prefix",
            "part",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["parts"] == 2
    assert (out_dir / "part_0000.obj").read_text(encoding="utf-8").count("\nf ") == 1
    assert (out_dir / "part_0001.obj").read_text(encoding="utf-8").count("\nf ") == 1


def test_smart_cpp_native_executable_splits_bsp_usemtl_parts(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")
    source = tmp_path / "bsp_seg.obj"
    source.write_text(
        "\n".join(
            [
                "mtllib labels.mtl",
                "usemtl part_0",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
                "usemtl part_1",
                "v 0 0 1",
                "v 1 0 1",
                "v 0 1 1",
                "f 4 5 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "bsp_parts"

    completed = subprocess.run(
        [
            str(binary),
            "split-obj-parts",
            "--input",
            str(source),
            "--output_dir",
            str(out_dir),
            "--split_on_usemtl",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["split_on_usemtl"] is True
    assert payload["parts"] == 2
    assert (out_dir / "part_0000.obj").exists()
    assert (out_dir / "part_0001.obj").exists()


def test_smart_cpp_native_executable_writes_coacd_partitions(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    msh = tmp_path / "tetra.msh"
    sc.native_save_gmsh(str(msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    coacd_dir = tmp_path / "coacd"
    coacd_dir.mkdir()
    (coacd_dir / "part0.obj").write_text(
        "\n".join(
            [
                "v -0.1 -0.1 -0.1",
                "v 0.8 -0.1 -0.1",
                "v 0.8 0.8 -0.1",
                "v -0.1 0.8 -0.1",
                "v -0.1 -0.1 0.8",
                "v 0.8 -0.1 0.8",
                "v 0.8 0.8 0.8",
                "v -0.1 0.8 0.8",
                "f 1 2 3",
                "f 1 3 4",
                "f 5 7 6",
                "f 5 8 7",
                "f 1 5 6",
                "f 1 6 2",
                "f 2 6 7",
                "f 2 7 3",
                "f 3 7 8",
                "f 3 8 4",
                "f 4 8 5",
                "f 4 5 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "coacd_partitions.json"

    completed = subprocess.run(
        [
            str(binary),
            "partition-coacd",
            "--msh",
            str(msh),
            "--coacd_dir",
            str(coacd_dir),
            "--output",
            str(out),
            "--mesh_id",
            "one-tet",
            "--partition_threads",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"] == "smart-cpp-native partition-coacd"
    assert payload["mesh_id"] == "one-tet"
    assert payload["partitions"] == [[0]]
    stdout = json.loads(completed.stdout)
    assert stdout["partition_count"] == 1
    assert stdout["partition_threads"] == 1


def test_smart_cpp_native_executable_writes_bsp_partitions_from_usemtl_obj(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    msh = tmp_path / "tetra.msh"
    sc.native_save_gmsh(str(msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    bsp_obj = tmp_path / "bsp_seg.obj"
    bsp_obj.write_text(
        "\n".join(
            [
                "mtllib labels.mtl",
                "usemtl bsp_0",
                "v -1 -1 -1",
                "v 1 -1 -1",
                "v 1 1 -1",
                "v -1 1 -1",
                "v -1 -1 1",
                "v 1 -1 1",
                "v 1 1 1",
                "v -1 1 1",
                "f 1 2 3",
                "f 1 3 4",
                "f 5 7 6",
                "f 5 8 7",
                "f 1 5 6",
                "f 1 6 2",
                "f 2 6 7",
                "f 2 7 3",
                "f 3 7 8",
                "f 3 8 4",
                "f 4 8 5",
                "f 4 5 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "bsp_partitions.json"

    completed = subprocess.run(
        [
            str(binary),
            "partition-bsp",
            "--msh",
            str(msh),
            "--bsp_obj",
            str(bsp_obj),
            "--output",
            str(out),
            "--mesh_id",
            "one-tet-bsp",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"] == "smart-cpp-native partition-bsp"
    assert payload["init_type"] == "bsp"
    assert payload["mesh_id"] == "one-tet-bsp"
    assert payload["partitions"] == [[0]]
    stdout = json.loads(completed.stdout)
    assert stdout["command"] == "partition-bsp"
    assert stdout["rotate_y_minus_90"] is True


def test_cpp_mcts_callback_runner_applies_puct_prior() -> None:
    class Env:
        num_bbox = 1
        num_action_scale = 2

        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            self.step_cnt = 0
            self.done = 0

        def step(self, action: int):
            self.step_cnt += 1
            self.done = 1
            return float(action + 1), None, self.done

        def render(self, *args, **kwargs) -> None:
            return None

        def current_state_summary(self) -> dict:
            return {}

        def _bridge_apply_cached_action(self, action: int):
            return None

        def _bridge_apply_unscored_action(self, action: int):
            return None

        def _bridge_apply_scored_action(self, action: int, expected_reward: float):
            return None

    args = Namespace(
        exp_w=0.1,
        skip_rate=0.7,
        gamma=1.0,
        max_step=1,
        pns=False,
        grdexp=False,
        mask_prun=False,
        skip_summary_metrics=True,
        stateful_unscored_apply=False,
        mcts_fused_rollout_step=False,
        mcts_native_axis_rollout_step=False,
        mcts_native_axis_rollout_segment=False,
        transposition_table=True,
        transposition_table_size=128,
        action_prior_weight=0.0,
        puct_prior_weight=1.0,
        action_prior_top_k=0,
    )
    logits = [0.0] * sc.native_action_count(1, 2)
    logits[1] = 5.0

    stats = sc.run_mcts_callbacks(args, Env(), 15, logits)

    assert stats["mcts_runner_cpp"] == 1.0
    assert stats["puct_prior_selections"] > 0


def test_cpp_mcts_callback_runner_accepts_value_and_escape_policy() -> None:
    class Env:
        num_bbox = 1
        num_action_scale = 2

        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            self.step_cnt = 0
            self.done = 0

        def step(self, action: int):
            self.step_cnt += 1
            self.done = 1
            return float(action + 1), None, self.done

        def render(self, *args, **kwargs) -> None:
            return None

        def current_state_summary(self) -> dict:
            return {}

        def _bridge_apply_cached_action(self, action: int):
            return None

        def _bridge_apply_unscored_action(self, action: int):
            return None

        def _bridge_apply_scored_action(self, action: int, expected_reward: float):
            return None

    def args(**overrides):
        payload = dict(
            exp_w=0.1,
            skip_rate=0.7,
            gamma=1.0,
            max_step=1,
            pns=False,
            grdexp=False,
            mask_prun=False,
            skip_summary_metrics=True,
            stateful_unscored_apply=False,
            mcts_fused_rollout_step=False,
            mcts_native_axis_rollout_step=False,
            mcts_native_axis_rollout_segment=False,
            mcts_cpp_rng=False,
            mcts_cpp_rng_seed=7777,
            transposition_table=False,
            transposition_table_size=128,
            action_prior_weight=0.0,
            puct_prior_weight=0.0,
            action_value_weight=0.0,
            action_prior_top_k=0,
            action_prior_select="legacy",
            action_prior_select_temperature=1.0,
            escape_policy=False,
            escape_after_no_update=0,
            escape_action_top_k=0,
            escape_probability=0.0,
        )
        payload.update(overrides)
        return Namespace(**payload)

    n_actions = sc.native_action_count(1, 2)
    value_logits = [0.0] * n_actions
    value_logits[1] = 10.0
    value_stats = sc.run_mcts_callbacks(
        args(action_value_weight=1.0, action_prior_select="best"),
        Env(),
        15,
        [0.0] * n_actions,
        value_logits,
    )
    assert value_stats["mcts_runner_cpp"] == 1.0

    prior_logits = [0.0] * n_actions
    prior_logits[1] = 10.0
    prior_logits[5] = 9.0
    prior_logits[3] = 8.0
    escape_stats = sc.run_mcts_callbacks(
        args(
            action_prior_weight=1.0,
            action_prior_top_k=1,
            escape_policy=True,
            escape_action_top_k=2,
            escape_probability=1.0,
        ),
        Env(),
        15,
        prior_logits,
    )
    assert escape_stats["escape_pruned_nodes"] > 0
    assert escape_stats["escape_choices"] > 0

    rng_stats = sc.run_mcts_callbacks(
        args(mcts_cpp_rng=True, mcts_cpp_rng_seed=123),
        Env(),
        15,
        [0.0] * n_actions,
        [0.0] * n_actions,
    )
    assert rng_stats["cpp_rng_enabled"] == 1.0


def test_cpp_mcts_runner_calls_stateful_segment_directly() -> None:
    class State:
        def __init__(self) -> None:
            self.calls = []

        def greedy_axis_rollout_segment_delta(
            self,
            mask,
            cover_penalty: float,
            pen_rate: float,
            remaining: int,
        ):
            self.calls.append((list(mask), cover_penalty, pen_rate, remaining))
            return (
                [2],
                [0.75],
                [0.75],
                [True],
                [0],
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
                0.75,
            )

    class Env:
        num_bbox = 1
        num_action_scale = 2
        max_step = 4

        def __init__(self) -> None:
            self.state = State()
            self.sync_calls = []
            self.bridge_segment_calls = 0
            self.reset_calls = 0
            self.fast_reset_calls = 0
            self.reset()

        def reset(self) -> None:
            self.reset_calls += 1
            self.step_cnt = 0
            self.done = 0
            self.pen_rate = 1.0

        def _bridge_reset_for_cpp_mcts(self) -> bool:
            self.fast_reset_calls += 1
            self.step_cnt = 0
            self.done = 0
            self.pen_rate = 1.0
            return True

        def step(self, action: int):
            self.step_cnt += 1
            return 0.1, None, 0

        def render(self, *args, **kwargs) -> None:
            return None

        def current_state_summary(self) -> dict:
            return {}

        def _ensure_manifold_stateful_state(self):
            return self.state

        def _bridge_sync_axis_deltas(
            self,
            touched_indices,
            bounds,
            rotations,
            last_bbox_score,
            step_count,
            state_already_current=False,
        ) -> None:
            self.sync_calls.append(
                (
                    list(touched_indices),
                    bounds,
                    rotations,
                    last_bbox_score,
                    step_count,
                    state_already_current,
                )
            )
            self.step_cnt += int(step_count)

        def _bridge_mcts_greedy_rollout_segment(self, mask, remaining):
            self.bridge_segment_calls += 1
            return None

        def _bridge_apply_cached_action(self, action: int):
            return None

        def _bridge_apply_unscored_action(self, action: int):
            return None

        def _bridge_apply_scored_action(self, action: int, expected_reward: float):
            return None

    env = Env()
    stats = sc.run_mcts_callbacks(
        Namespace(
            exp_w=0.1,
            skip_rate=0.7,
            gamma=1.0,
            cover_penalty=100.0,
            max_step=4,
            pns=False,
            grdexp=False,
            mask_prun=False,
            skip_summary_metrics=True,
            stateful_unscored_apply=False,
            mcts_fused_rollout_step=False,
            mcts_native_axis_rollout_step=False,
            mcts_native_axis_rollout_segment=True,
            mcts_cpp_rng=True,
            mcts_cpp_rng_seed=123,
            transposition_table=False,
            transposition_table_size=128,
            action_prior_weight=0.0,
            puct_prior_weight=0.0,
            action_value_weight=0.0,
            action_prior_top_k=0,
            action_prior_select="legacy",
            action_prior_select_temperature=1.0,
            escape_policy=False,
            escape_after_no_update=0,
            escape_action_top_k=0,
            escape_probability=0.0,
            trace_actions_path="",
        ),
        env,
        3,
        [0.0] * sc.native_action_count(1, 2),
        [0.0] * sc.native_action_count(1, 2),
    )

    assert stats["direct_stateful_segments"] > 0
    assert stats["direct_stateful_segment_steps"] > 0
    assert stats["cpp_fast_resets"] > 0
    assert stats["python_resets"] == 0
    assert env.state.calls
    assert env.sync_calls
    assert env.sync_calls[0][0] == [0]
    assert env.sync_calls[0][-1] is True
    assert env.bridge_segment_calls == 0
    assert env.fast_reset_calls > 0
    assert env.reset_calls == 1


def test_cpp_mcts_runner_applies_stateful_axis_action_directly() -> None:
    class State:
        def __init__(self) -> None:
            self.calls = []

        def apply_axis_action_delta(
            self,
            action: int,
            cover_penalty: float,
            pen_rate: float,
        ):
            self.calls.append((action, cover_penalty, pen_rate))
            return (
                0.5,
                0,
                [0.0, 0.0, 0.0, 1.1, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                0.5,
            )

    class Env:
        num_bbox = 1
        num_action_scale = 2

        def __init__(self) -> None:
            self.state = State()
            self.sync_calls = []
            self.unscored_calls = 0
            self.step_calls = 0
            self.reset()

        def reset(self) -> None:
            self.step_cnt = 0
            self.done = 0
            self.pen_rate = 1.0

        def step(self, action: int):
            self.step_calls += 1
            return 0.0, None, 1

        def render(self, *args, **kwargs) -> None:
            return None

        def current_state_summary(self) -> dict:
            return {}

        def _ensure_manifold_stateful_state(self):
            return self.state

        def _bridge_sync_axis_deltas(
            self,
            touched_indices,
            bounds,
            rotations,
            last_bbox_score,
            step_count,
            state_already_current=False,
        ) -> None:
            self.sync_calls.append(
                (
                    list(touched_indices),
                    bounds,
                    rotations,
                    last_bbox_score,
                    step_count,
                    state_already_current,
                )
            )
            self.step_cnt += int(step_count)
            self.done = 1

        def _bridge_apply_cached_action(self, action: int):
            return None

        def _bridge_apply_unscored_action(self, action: int):
            self.unscored_calls += 1
            return None

        def _bridge_apply_scored_action(self, action: int, expected_reward: float):
            return None

    n_actions = sc.native_action_count(1, 2)
    priors = [0.0] * n_actions
    priors[0] = 10.0
    env = Env()
    stats = sc.run_mcts_callbacks(
        Namespace(
            exp_w=0.1,
            skip_rate=0.7,
            gamma=1.0,
            cover_penalty=100.0,
            max_step=1,
            pns=False,
            grdexp=False,
            mask_prun=False,
            skip_summary_metrics=True,
            stateful_unscored_apply=True,
            mcts_fused_rollout_step=False,
            mcts_native_axis_rollout_step=False,
            mcts_native_axis_rollout_segment=False,
            mcts_cpp_rng=True,
            mcts_cpp_rng_seed=123,
            transposition_table=False,
            transposition_table_size=128,
            action_prior_weight=1.0,
            puct_prior_weight=0.0,
            action_value_weight=0.0,
            action_prior_top_k=1,
            action_prior_select="best",
            action_prior_select_temperature=1.0,
            escape_policy=False,
            escape_after_no_update=0,
            escape_action_top_k=0,
            escape_probability=0.0,
            trace_actions_path="",
        ),
        env,
        1,
        priors,
        [0.0] * n_actions,
    )

    assert stats["direct_stateful_axis_applies"] == 1
    assert env.state.calls == [(0, 100.0, 1.0)]
    assert env.sync_calls
    assert env.sync_calls[0][0] == [0]
    assert env.sync_calls[0][-1] is True
    assert env.unscored_calls == 0
    assert env.step_calls == 0


def test_cpp_mcts_runner_applies_stateful_recenter_action_directly() -> None:
    class State:
        def __init__(self) -> None:
            self.calls = []

        def apply_replacement_delta(
            self,
            bbox_idx: int,
            candidate_bounds,
            candidate_rotation,
            cover_penalty: float,
            pen_rate: float,
        ):
            self.calls.append(
                (
                    bbox_idx,
                    list(candidate_bounds),
                    list(candidate_rotation),
                    cover_penalty,
                    pen_rate,
                )
            )
            return (
                0.75,
                bbox_idx,
                list(candidate_bounds),
                list(candidate_rotation),
                0.75,
            )

    class Env:
        num_bbox = 1
        num_action_scale = 2

        def __init__(self) -> None:
            self.state = State()
            self.recenter_calls = []
            self.sync_calls = []
            self.unscored_calls = 0
            self.step_calls = 0
            self.reset()

        def reset(self) -> None:
            self.step_cnt = 0
            self.done = 0
            self.pen_rate = 1.0

        def step(self, action: int):
            self.step_calls += 1
            return 0.0, None, 1

        def render(self, *args, **kwargs) -> None:
            return None

        def current_state_summary(self) -> dict:
            return {}

        def _ensure_manifold_stateful_state(self):
            return self.state

        def _bridge_recenter_bbox_params(self, bbox_idx: int):
            self.recenter_calls.append(bbox_idx)
            return (
                [0.0, 0.0, 0.0, 0.8, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            )

        def _bridge_sync_axis_deltas(
            self,
            touched_indices,
            bounds,
            rotations,
            last_bbox_score,
            step_count,
            state_already_current=False,
        ) -> None:
            self.sync_calls.append(
                (
                    list(touched_indices),
                    bounds,
                    rotations,
                    last_bbox_score,
                    step_count,
                    state_already_current,
                )
            )
            self.step_cnt += int(step_count)
            self.done = 1

        def _bridge_apply_cached_action(self, action: int):
            return None

        def _bridge_apply_unscored_action(self, action: int):
            self.unscored_calls += 1
            return None

        def _bridge_apply_scored_action(self, action: int, expected_reward: float):
            return None

    n_actions = sc.native_action_count(1, 2)
    recenter_action = n_actions - 1
    priors = [0.0] * n_actions
    priors[recenter_action] = 10.0
    env = Env()
    stats = sc.run_mcts_callbacks(
        Namespace(
            exp_w=0.1,
            skip_rate=0.7,
            gamma=1.0,
            cover_penalty=100.0,
            max_step=1,
            pns=False,
            grdexp=False,
            mask_prun=False,
            skip_summary_metrics=True,
            stateful_unscored_apply=True,
            mcts_fused_rollout_step=False,
            mcts_native_axis_rollout_step=False,
            mcts_native_axis_rollout_segment=False,
            mcts_cpp_rng=True,
            mcts_cpp_rng_seed=123,
            transposition_table=False,
            transposition_table_size=128,
            action_prior_weight=1.0,
            puct_prior_weight=0.0,
            action_value_weight=0.0,
            action_prior_top_k=1,
            action_prior_select="best",
            action_prior_select_temperature=1.0,
            escape_policy=False,
            escape_after_no_update=0,
            escape_action_top_k=0,
            escape_probability=0.0,
            trace_actions_path="",
        ),
        env,
        1,
        priors,
        [0.0] * n_actions,
    )

    assert stats["direct_stateful_axis_applies"] == 0
    assert stats["direct_stateful_recenter_applies"] == 1
    assert env.recenter_calls == [0]
    assert env.state.calls == [
        (
            0,
            [0.0, 0.0, 0.0, 0.8, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            100.0,
            1.0,
        )
    ]
    assert env.sync_calls
    assert env.sync_calls[0][0] == [0]
    assert env.sync_calls[0][-1] is True
    assert env.unscored_calls == 0
    assert env.step_calls == 0


def test_cpp_greedy_refine_callback_runner_caches_loop_state() -> None:
    class Env:
        max_step = 10

        def __init__(self) -> None:
            self.reset_calls = 0
            self.segment_calls = 0
            self.greedy_calls = 0
            self.step_calls = 0
            self.scored_calls = 0
            self.remaining_seen = []

        def reset(self) -> None:
            self.reset_calls += 1

        def _bridge_axis_refine_segment(self, remaining: int):
            self.segment_calls += 1
            self.remaining_seen.append(remaining)
            if self.segment_calls == 1:
                return [1.5, 0.5], [2, 3], 0
            return [], [], 0

        def greedy_sample(self, ret_reward: bool):
            self.greedy_calls += 1
            assert ret_reward is True
            return 4, 0.25

        def _bridge_apply_scored_action(self, action: int, candidate_reward: float):
            self.scored_calls += 1
            assert action == 4
            assert candidate_reward == 0.25
            return 1.25, 1

        def _bridge_apply_cached_action(self, action: int):
            return None

        def step(self, action: int):
            self.step_calls += 1
            return 0.0, None, 1

    env = Env()
    rewards, count = sc.run_greedy_refine_callbacks(Namespace(print_off=True), env)

    assert rewards == [1.5, 0.5, 1.25]
    assert count == 3
    assert env.reset_calls == 1
    assert env.remaining_seen == [9, 7]
    assert env.greedy_calls == 1
    assert env.scored_calls == 1
    assert env.step_calls == 0


def test_cpp_exposes_bbox_tet_and_merge_helpers(tmp_path) -> None:
    bounds = [[0, 0, 0, 1, 2, 3], [0, 0, 0, 2, 2, 2]]
    assert sc.native_bbox_volumes(bounds) == [6.0, 8.0]
    assert sc.native_bbox_valid_mask(bounds) == [True, True]
    assert sc.native_total_bbox_volume(bounds) == 14.0
    assert sc.native_bbox_union_bounds(bounds) == [0.0, 0.0, 0.0, 2.0, 2.0, 3.0]
    assert sc.native_bbox_union_volume(bounds) == 12.0
    box_vertices, box_faces = sc.native_box_mesh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, [1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert box_vertices == [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 9.0],
        [1.0, 7.0, 3.0],
        [1.0, 7.0, 9.0],
        [5.0, 2.0, 3.0],
        [5.0, 2.0, 9.0],
        [5.0, 7.0, 3.0],
        [5.0, 7.0, 9.0],
    ]
    assert box_faces == [
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
    assert sc.native_coverage_mask([[0.5, 0.5, 0.5], [2, 2, 2]], [0, 0, 0, 1, 1, 1]) == [
        True,
        False,
    ]
    assert sc.native_apply_axis_action([[0, 0, 0, 1, 1, 1]], 0, 2, 0.1) == [
        [-0.1, 0.0, 0.0, 1.0, 1.0, 1.0]
    ]
    assert len(sc.native_action_upper_rewards([[0, 0, 0, 1, 1, 1]], 2, 0.1, 1.0, 0.0)) == 13
    assert len(sc.native_bbox_action_upper_rewards([[0, 0, 0, 1, 1, 1]], 0, 2, 0.1, 1.0, 0.0)) == 13
    assert sc.native_bavf_scores([1.0, 2.0], [2.0, 4.0]) == [50.0, 50.0]
    assert abs(sc.native_merge_bavf_reward(1.2, 1.0, 1.0, 1.5, 10.0) - 0.05) < 1e-12
    assert abs(sc.native_incremental_average(2.0, 3, 6.0) - 3.0) < 1e-12

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    recenter_points = sc.native_recenter_points_for_box(
        np.asarray(vertices, dtype=float),
        np.asarray(voxels, dtype=np.int64),
        np.asarray([[0.25, 0.25, 0.25]], dtype=float),
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
    )
    assert np.asarray(recenter_points).tolist() == vertices
    assert sc.native_tetra_volumes(vertices, voxels) == [1.0 / 6.0]
    assert sc.native_tetra_centroids(vertices, voxels) == [0.25, 0.25, 0.25]
    assert sc.native_tetra_surface_faces(voxels) == [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    assert sc.native_tetra_adjacency(voxels) == [[]]
    part_volumes, part_bounds, part_points = sc.native_partition_summaries(
        vertices,
        voxels,
        [1.0],
        [[0]],
        False,
    )
    assert part_volumes == [1.0]
    assert part_bounds == [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
    assert len(part_points[0]) == 12

    path = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(path), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    loaded_vertices, loaded_faces, loaded_voxels = sc.native_load_gmsh(str(path))
    assert loaded_vertices == vertices
    assert loaded_faces
    assert loaded_voxels == voxels
    engine = sc.native_smart_engine_from_gmsh(
        str(path),
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
        "tet",
        2,
        0.1,
        1.0 / 6.0,
        0.0,
        True,
        128,
        "mesh",
    )
    initial_score = engine.recompute_score(100.0, 1.0)
    assert initial_score == engine.stats()["last_bbox_score"]
    assert engine.stats()["num_boxes"] == 1.0
    assert engine.stats()["native_recenter_enabled"] == 1.0


def test_cpp_native_file_runner_refine_exports_legacy_bbox_layout(tmp_path) -> None:
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    path = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(path), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    metadata = tmp_path / "bbox_params.json"
    metadata.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = native_runner.run_refine_from_files(
        msh_path=path,
        bbox_metadata_path=metadata,
        output_root=tmp_path / "refine",
        exp_name="native",
        mesh_id="mesh-a",
        category="tet",
        max_step=1,
        cover_penalty=100.0,
        action_unit=0.1,
        num_action_scale=2,
        stateful_union_cache=True,
        cache_capacity=128,
        volume_method="mesh",
    )

    assert result["status"] == "success"
    assert result["metadata"]["backend"] == "smart-cpp-native"
    assert result["metadata"]["stdout"]["command"] == "refine"
    output = tmp_path / "refine" / "native" / "result" / "updated0" / "mesh-a" / "bboxs_steps0"
    assert output == result["output_path"]
    assert (output / "bbox0.obj").exists()
    assert (output / "bbox_params.json").exists()
    assert (output.parent / "native_stats.json").exists()


def test_legacy_grd_bbox_params_are_generated_from_segment_txt(tmp_path) -> None:
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    msh = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    segment = tmp_path / "greedy_segment0_coacd_mgeps0.02_fm.txt"
    segment.write_text("1\n0\n", encoding="utf-8")

    metadata_path = native_runner.write_legacy_grd_bbox_params_from_segment(segment, msh)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path.name.endswith(".legacy_bbox_params.json")
    assert data["source"] == "legacy_grd_merged_bbox_init"
    assert data["boxes"][0]["partition"] == [0]
    assert len(data["boxes"][0]["bounds"]) == 6
    assert len(data["boxes"][0]["rotation"]) == 9


def test_cpp_native_file_runner_mcts_uses_standalone_executable_without_prior(tmp_path) -> None:
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    path = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(path), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    metadata = tmp_path / "bbox_params.json"
    metadata.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = native_runner.run_mcts_from_files(
        msh_path=path,
        bbox_metadata_path=metadata,
        output_root=tmp_path / "mcts",
        exp_name="native",
        mesh_id="mesh-a",
        category="tet",
        mcts_iter=2,
        max_step=1,
        cover_penalty=100.0,
        action_unit=0.1,
        num_action_scale=2,
        exp_weight=0.001,
        gamma=1.0,
        seed=7,
        transposition_table=True,
        transposition_table_size=128,
        stateful_union_cache=True,
        cache_capacity=128,
        volume_method="mesh",
    )

    assert result["status"] == "success"
    assert result["metadata"]["backend"] == "smart-cpp-native"
    assert result["metadata"]["stdout"]["command"] == "mcts"
    output = tmp_path / "mcts" / "native" / "result" / "updated0" / "mesh-a" / "bboxs_steps0"
    assert (output / "bbox0.obj").exists()
    assert (output / "native_stats.json").exists()


def test_cpp_native_file_runner_mcts_accepts_static_prior(tmp_path) -> None:
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    path = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(path), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    metadata = tmp_path / "bbox_params.json"
    metadata.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "policy_type": "coord_scale_count_prior",
                "coord_scale_logits": {"6:0": 0.0},
                "default_logit": -1.0,
                "num_action_scale": 2,
            }
        ),
        encoding="utf-8",
    )

    result = native_runner.run_mcts_from_files(
        msh_path=path,
        bbox_metadata_path=metadata,
        output_root=tmp_path / "mcts",
        exp_name="native",
        mesh_id="mesh-a",
        category="tet",
        mcts_iter=2,
        max_step=1,
        cover_penalty=100.0,
        action_unit=0.1,
        num_action_scale=2,
        exp_weight=0.001,
        gamma=1.0,
        seed=7,
        transposition_table=True,
        transposition_table_size=128,
        stateful_union_cache=True,
        cache_capacity=128,
        volume_method="mesh",
        action_prior_path=prior,
        action_prior_weight=0.1,
        action_prior_top_k=1,
    )

    assert result["status"] == "success"
    assert result["metadata"]["action_prior_logits"] == 13
    assert result["metadata"]["action_prior_weight"] == 0.1
    assert result["metadata"]["action_prior_top_k"] == 1
    assert result["metadata"]["native_mcts_action_prior_top_k"] == 1.0
    assert result["metadata"]["native_mcts_transposition_table"] == 1.0
    assert result["metadata"]["stdout"]["transposition_table_size"] >= 1
    output = tmp_path / "mcts" / "native" / "result" / "updated0" / "mesh-a" / "bboxs_steps0"
    assert (output / "bbox0.obj").exists()


def test_cpp_native_file_runner_mcts_can_apply_native_recenter_action(tmp_path) -> None:
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    path = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(path), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    metadata = tmp_path / "bbox_params.json"
    metadata.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [-0.3, -0.3, -0.3, 0.7, 0.7, 0.7],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prior = tmp_path / "recenter_prior.json"
    prior.write_text(
        json.dumps(
            {
                "policy_type": "coord_scale_count_prior",
                "coord_scale_logits": {"6:0": 10.0},
                "default_logit": -10.0,
                "num_action_scale": 2,
            }
        ),
        encoding="utf-8",
    )

    result = native_runner.run_mcts_from_files(
        msh_path=path,
        bbox_metadata_path=metadata,
        output_root=tmp_path / "mcts",
        exp_name="native",
        mesh_id="mesh-a",
        category="tet",
        mcts_iter=1,
        max_step=1,
        cover_penalty=100.0,
        action_unit=0.1,
        num_action_scale=2,
        exp_weight=0.001,
        gamma=1.0,
        seed=7,
        transposition_table=False,
        transposition_table_size=128,
        stateful_union_cache=True,
        cache_capacity=128,
        volume_method="mesh",
        action_prior_path=prior,
        action_prior_weight=1.0,
        action_prior_top_k=1,
        native_recenter=True,
    )

    assert result["status"] == "success"
    assert result["metadata"]["native_recenter"] is True
    assert result["metadata"]["stdout"]["axis_only"] is False
    assert result["metadata"]["stdout"]["recenter_applies"] == 1
    output = tmp_path / "mcts" / "native" / "result" / "updated0" / "mesh-a" / "bboxs_steps0"
    assert (output / "bbox0.obj").exists()
    assert (output / "native_stats.json").exists()


def test_smart_cpp_native_executable_refine_mcts_runs_in_one_process(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    path = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(path), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    metadata = tmp_path / "bbox_params.json"
    metadata.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [-0.3, -0.3, -0.3, 0.7, 0.7, 0.7],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prior_logits = tmp_path / "prior_logits.json"
    logits = [-10.0] * 13
    logits[12] = 10.0
    prior_logits.write_text(json.dumps(logits), encoding="utf-8")

    refine_dir = tmp_path / "refine_bboxs_steps0"
    mcts_dir = tmp_path / "mcts_bboxs_steps0"
    completed = subprocess.run(
        [
            str(binary),
            "refine-mcts",
            "--msh",
            str(path),
            "--bbox_params",
            str(metadata),
            "--refine_output_dir",
            str(refine_dir),
            "--mcts_output_dir",
            str(mcts_dir),
            "--refine_max_step",
            "0",
            "--mcts_iter",
            "1",
            "--mcts_max_step",
            "1",
            "--num_action_scale",
            "2",
            "--prior_logits_file",
            str(prior_logits),
            "--action_prior_weight",
            "1.0",
            "--action_prior_top_k",
            "1",
            "--native_recenter",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["command"] == "refine-mcts"
    assert payload["single_mesh_load"] is True
    assert payload["single_state_bridge"] is True
    assert payload["refine"]["command"] == "refine"
    assert payload["mcts"]["command"] == "mcts"
    assert payload["refine"]["output_path"] == str(refine_dir)
    assert payload["mcts"]["output_path"] == str(mcts_dir)
    assert payload["mcts"]["axis_only"] is False
    assert payload["mcts"]["recenter_applies"] == 1
    assert (refine_dir / "bbox_params.json").exists()
    assert (mcts_dir / "bbox0.obj").exists()
    combined_stats = json.loads((mcts_dir / "refine_mcts_native_stats.json").read_text(encoding="utf-8"))
    assert combined_stats["command"] == "refine-mcts"
    assert combined_stats["single_mesh_load"] is True
    assert combined_stats["single_state_bridge"] is True
    assert combined_stats["refine_output_path"] == str(refine_dir)
    assert combined_stats["mcts_output_path"] == str(mcts_dir)


def test_cpp_native_runner_refine_mcts_uses_standalone_executable(tmp_path) -> None:
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    path = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(path), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    metadata = tmp_path / "bbox_params.json"
    metadata.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [-0.3, -0.3, -0.3, 0.7, 0.7, 0.7],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prior = tmp_path / "recenter_prior.json"
    prior.write_text(
        json.dumps(
            {
                "policy_type": "coord_scale_count_prior",
                "coord_scale_logits": {"6:0": 10.0},
                "default_logit": -10.0,
                "num_action_scale": 2,
            }
        ),
        encoding="utf-8",
    )

    result = native_runner.run_refine_mcts_from_files(
        msh_path=path,
        bbox_metadata_path=metadata,
        refine_output_dir=tmp_path / "refine_bboxs_steps0",
        mcts_output_dir=tmp_path / "mcts_bboxs_steps0",
        category="tet",
        mesh_id="mesh-a",
        refine_max_step=0,
        mcts_iter=1,
        mcts_max_step=1,
        cover_penalty=100.0,
        refine_action_unit=0.1,
        mcts_action_unit=0.1,
        num_action_scale=2,
        exp_weight=0.001,
        gamma=1.0,
        seed=7,
        transposition_table=False,
        transposition_table_size=128,
        stateful_union_cache=True,
        cache_capacity=128,
        volume_method="mesh",
        action_prior_path=prior,
        action_prior_weight=1.0,
        action_prior_top_k=1,
        native_recenter=True,
    )

    assert result["status"] == "success"
    assert result["metadata"]["combined"] is True
    assert result["metadata"]["single_mesh_load"] is True
    assert result["metadata"]["single_state_bridge"] is True
    assert result["metadata"]["stdout"]["command"] == "refine-mcts"
    assert result["metadata"]["stdout"]["single_mesh_load"] is True
    assert result["metadata"]["stdout"]["single_state_bridge"] is True
    assert result["metadata"]["stdout"]["mcts"]["output_path"] == str(result["output_path"])
    assert result["metadata"]["stdout"]["mcts"]["recenter_applies"] == 1
    assert (result["output_path"] / "bbox0.obj").exists()
    assert (result["output_path"] / "refine_mcts_native_stats.json").exists()
    assert result["metadata"]["combined_stats_path"] == str(
        result["output_path"] / "refine_mcts_native_stats.json"
    )
    assert result["metadata"]["combined_stats"]["command"] == "refine-mcts"
    assert result["metadata"]["mcts_output_path"] == str(result["output_path"])


def test_smart_cpp_native_executable_runs_full_native_pipeline_with_external_tools(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")

    def write_cube_obj(path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "o part0",
                    "v 0 0 0",
                    "v 1 0 0",
                    "v 0 1 0",
                    "v 1 1 0",
                    "v 0 0 1",
                    "v 1 0 1",
                    "v 0 1 1",
                    "v 1 1 1",
                    "f 1 3 4",
                    "f 1 4 2",
                    "f 5 6 8",
                    "f 5 8 7",
                    "f 1 2 6",
                    "f 1 6 5",
                    "f 3 7 8",
                    "f 3 8 4",
                    "f 1 5 7",
                    "f 1 7 3",
                    "f 2 4 8",
                    "f 2 8 6",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    input_obj = tmp_path / "model.obj"
    write_cube_obj(input_obj)
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    fixture_msh = tmp_path / "fixture.msh"
    sc.native_save_gmsh(str(fixture_msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    fixture_surface = tmp_path / "fixture_surface.obj"
    write_cube_obj(fixture_surface)

    tools = tmp_path / "tools"
    tools.mkdir()
    manifold = tools / "fake_manifoldplus.py"
    manifold.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "inp = sys.argv[sys.argv.index('--input') + 1]\n"
        "shutil.copyfile(inp, out)\n",
        encoding="utf-8",
    )
    ftetwild = tools / "fake_ftetwild.py"
    ftetwild.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, shutil, sys\n"
        f"fixture_msh = {str(fixture_msh)!r}\n"
        f"fixture_surface = {str(fixture_surface)!r}\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "marker = pathlib.Path(out).with_suffix('.first_failed')\n"
        "if not marker.exists():\n"
        "    marker.write_text('failed once', encoding='utf-8')\n"
        "    raise SystemExit(3)\n"
        "shutil.copyfile(fixture_msh, out)\n"
        "shutil.copyfile(fixture_surface, out + '__sf.obj')\n",
        encoding="utf-8",
    )
    coacd = tools / "fake_coacd.py"
    coacd.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, shutil, sys\n"
        f"fixture_surface = {str(fixture_surface)!r}\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "shutil.copyfile(fixture_surface, out)\n",
        encoding="utf-8",
    )
    for tool in (manifold, ftetwild, coacd):
        tool.chmod(0o755)

    work_dir = tmp_path / "native_pipeline"
    completed = subprocess.run(
        [
            str(binary),
            "run-pipeline",
            "--input",
            str(input_obj),
            "--work_dir",
            str(work_dir),
            "--manifoldplus_bin",
            str(manifold),
            "--ftetwild_bin",
            str(ftetwild),
            "--coacd_bin",
            str(coacd),
            "--refine_max_step",
            "0",
            "--mcts_iter",
            "1",
            "--mcts_max_step",
            "1",
            "--num_action_scale",
            "2",
            "--mesh_id",
            "mesh-a",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["command"] == "run-pipeline"
    assert payload["status"] == "success"
    stats = json.loads(Path(payload["stats_path"]).read_text(encoding="utf-8"))
    assert stats["command"] == "run-pipeline"
    assert stats["elapsed_sec"] >= 0
    stage_names = {stage["name"] for stage in stats["stages"]}
    assert {"normalize", "manifoldplus", "ftetwild", "coacd", "merge", "refine_mcts"} <= stage_names
    assert Path(payload["normalized_obj"]).exists()
    assert Path(payload["tetra_msh"]).exists()
    assert Path(payload["partitions"]).exists()
    assert Path(payload["bbox_params"]).exists()
    assert (work_dir / "coacd" / "part_0000.obj").exists()
    assert (work_dir / "refine_bboxs_steps0" / "bbox_params.json").exists()
    assert (work_dir / "mcts_bboxs_steps0" / "bbox0.obj").exists()
    assert (work_dir / "logs" / "coacd.log").exists()
    assert (work_dir / "logs" / "ftetwild_retry.log").exists()

    reused = subprocess.run(
        [
            str(binary),
            "run-pipeline",
            "--input",
            str(input_obj),
            "--work_dir",
            str(work_dir),
            "--manifoldplus_bin",
            str(manifold),
            "--ftetwild_bin",
            str(ftetwild),
            "--coacd_bin",
            str(coacd),
            "--refine_max_step",
            "0",
            "--mcts_iter",
            "1",
            "--mcts_max_step",
            "1",
            "--num_action_scale",
            "2",
            "--mesh_id",
            "mesh-a",
            "--reuse_existing",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert reused.returncode == 0, reused.stderr
    reused_payload = json.loads(reused.stdout)
    reused_stats = json.loads(Path(reused_payload["stats_path"]).read_text(encoding="utf-8"))
    assert reused_stats["reuse_existing"] is True
    reused_stage_names = {stage["name"] for stage in reused_stats["stages"]}
    assert {
        "normalize_reuse",
        "manifoldplus_reuse",
        "ftetwild_reuse",
        "preseg_reuse",
        "merge_reuse",
        "refine_mcts_reuse",
    } <= reused_stage_names


def test_smart_cpp_native_run_pipeline_can_use_original_bsp_init(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")

    def write_cube_obj(path: Path, header: str = "o part0") -> None:
        path.write_text(
            "\n".join(
                [
                    header,
                    "v -1 -1 -1",
                    "v 1 -1 -1",
                    "v 1 1 -1",
                    "v -1 1 -1",
                    "v -1 -1 1",
                    "v 1 -1 1",
                    "v 1 1 1",
                    "v -1 1 1",
                    "f 1 2 3",
                    "f 1 3 4",
                    "f 5 7 6",
                    "f 5 8 7",
                    "f 1 5 6",
                    "f 1 6 2",
                    "f 2 6 7",
                    "f 2 7 3",
                    "f 3 7 8",
                    "f 3 8 4",
                    "f 4 8 5",
                    "f 4 5 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    input_obj = tmp_path / "model.obj"
    write_cube_obj(input_obj)
    bsp_obj = tmp_path / "bsp_seg.obj"
    write_cube_obj(bsp_obj, "usemtl bsp_part0")
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    fixture_msh = tmp_path / "fixture.msh"
    sc.native_save_gmsh(str(fixture_msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    fixture_surface = tmp_path / "fixture_surface.obj"
    write_cube_obj(fixture_surface)

    tools = tmp_path / "tools"
    tools.mkdir()
    manifold = tools / "fake_manifoldplus.py"
    manifold.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "shutil.copyfile(sys.argv[sys.argv.index('--input') + 1], sys.argv[sys.argv.index('--output') + 1])\n",
        encoding="utf-8",
    )
    ftetwild = tools / "fake_ftetwild.py"
    ftetwild.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        f"fixture_msh = {str(fixture_msh)!r}\n"
        f"fixture_surface = {str(fixture_surface)!r}\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "shutil.copyfile(fixture_msh, out)\n"
        "shutil.copyfile(fixture_surface, out + '__sf.obj')\n",
        encoding="utf-8",
    )
    for tool in (manifold, ftetwild):
        tool.chmod(0o755)

    work_dir = tmp_path / "native_pipeline_bsp"
    completed = subprocess.run(
        [
            str(binary),
            "run-pipeline",
            "--input",
            str(input_obj),
            "--work_dir",
            str(work_dir),
            "--manifoldplus_bin",
            str(manifold),
            "--ftetwild_bin",
            str(ftetwild),
            "--init_type",
            "bsp",
            "--bsp_obj",
            str(bsp_obj),
            "--refine_max_step",
            "0",
            "--mcts_iter",
            "1",
            "--mcts_max_step",
            "1",
            "--num_action_scale",
            "2",
            "--mesh_id",
            "mesh-bsp",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["command"] == "run-pipeline"
    assert payload["init_type"] == "bsp"
    assert Path(payload["partitions"]).name == "bsp_partitions.json"
    assert Path(payload["bbox_params"]).name == "greedy_segment0_mgeps0.02_fm.txt.bbox_params.json"
    assert (work_dir / "bsp_parts" / "part_0000.obj").exists()
    assert (work_dir / "mcts_bboxs_steps0" / "bbox0.obj").exists()
    partitions = json.loads(Path(payload["partitions"]).read_text(encoding="utf-8"))
    assert partitions["init_type"] == "bsp"
    assert partitions["source"] == "smart-cpp-native partition-bsp"


def test_smart_cpp_native_run_batch_can_use_original_bsp_init(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")

    def write_cube_obj(path: Path, header: str = "o part0") -> None:
        path.write_text(
            "\n".join(
                [
                    header,
                    "v -1 -1 -1",
                    "v 1 -1 -1",
                    "v 1 1 -1",
                    "v -1 1 -1",
                    "v -1 -1 1",
                    "v 1 -1 1",
                    "v 1 1 1",
                    "v -1 1 1",
                    "f 1 2 3",
                    "f 1 3 4",
                    "f 5 7 6",
                    "f 5 8 7",
                    "f 1 5 6",
                    "f 1 6 2",
                    "f 2 6 7",
                    "f 2 7 3",
                    "f 3 7 8",
                    "f 3 8 4",
                    "f 4 8 5",
                    "f 4 5 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    input_obj = tmp_path / "model.obj"
    bsp_obj = tmp_path / "bsp_seg.obj"
    write_cube_obj(input_obj)
    write_cube_obj(bsp_obj, "usemtl bsp_part0")

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    fixture_msh = tmp_path / "fixture.msh"
    sc.native_save_gmsh(str(fixture_msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    fixture_surface = tmp_path / "fixture_surface.obj"
    write_cube_obj(fixture_surface)

    tools = tmp_path / "tools"
    tools.mkdir()
    manifold = tools / "fake_manifoldplus.py"
    manifold.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "shutil.copyfile(sys.argv[sys.argv.index('--input') + 1], sys.argv[sys.argv.index('--output') + 1])\n",
        encoding="utf-8",
    )
    ftetwild = tools / "fake_ftetwild.py"
    ftetwild.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        f"fixture_msh = {str(fixture_msh)!r}\n"
        f"fixture_surface = {str(fixture_surface)!r}\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "shutil.copyfile(fixture_msh, out)\n"
        "shutil.copyfile(fixture_surface, out + '__sf.obj')\n",
        encoding="utf-8",
    )
    for tool in (manifold, ftetwild):
        tool.chmod(0o755)

    mesh_list = tmp_path / "meshes.tsv"
    mesh_list.write_text(
        f"mesh-bsp\t{input_obj}\t{bsp_obj}\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "native_batch"
    completed = subprocess.run(
        [
            str(binary),
            "run-batch",
            "--mesh_list",
            str(mesh_list),
            "--output_root",
            str(output_root),
            "--manifoldplus_bin",
            str(manifold),
            "--ftetwild_bin",
            str(ftetwild),
            "--init_type",
            "bsp",
            "--refine_max_step",
            "0",
            "--mcts_iter",
            "1",
            "--mcts_max_step",
            "1",
            "--num_action_scale",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["command"] == "run-batch"
    assert payload["mesh_count"] == 1
    assert payload["success"] == 1
    assert payload["failed"] == 0
    assert payload["pipeline_execution"] == "in_process"
    manifest = Path(payload["manifest"])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["mesh_id"] == "mesh-bsp"
    assert rows[0]["status"] == "success"
    assert rows[0]["pipeline_execution"] == "in_process"
    assert Path(rows[0]["stats_path"]).exists()
    work_dir = output_root / "mesh-bsp"
    assert (work_dir / "tetra" / "bsp_partitions.json").exists()
    assert (work_dir / "bsp_parts" / "part_0000.obj").exists()
    assert (work_dir / "mcts_bboxs_steps0" / "bbox0.obj").exists()
    partitions = json.loads((work_dir / "tetra" / "bsp_partitions.json").read_text(encoding="utf-8"))
    assert partitions["init_type"] == "bsp"

    resumed = subprocess.run(
        [
            str(binary),
            "run-batch",
            "--mesh_list",
            str(mesh_list),
            "--output_root",
            str(output_root),
            "--manifoldplus_bin",
            str(tmp_path / "missing_manifold"),
            "--ftetwild_bin",
            str(tmp_path / "missing_ftetwild"),
            "--init_type",
            "bsp",
            "--resume_success",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    resumed_payload = json.loads(resumed.stdout)
    assert resumed_payload["success"] == 1
    resumed_rows = [
        json.loads(line)
        for line in (output_root / "native_pipeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert resumed_rows[0]["pipeline_execution"] == "resume_success"
    assert resumed_rows[0]["elapsed_sec"] == 0

    data_root = tmp_path / "discovered_data"
    discovered_mesh = data_root / "shapenet_airplane" / "mesh_0"
    discovered_mesh.mkdir(parents=True)
    write_cube_obj(discovered_mesh / "model.obj")
    write_cube_obj(discovered_mesh / "bsp_seg.obj", "usemtl bsp_part0")
    discovered_mesh_1 = data_root / "shapenet_airplane" / "mesh_1"
    discovered_mesh_1.mkdir(parents=True)
    write_cube_obj(discovered_mesh_1 / "model.obj")
    write_cube_obj(discovered_mesh_1 / "bsp_seg.obj", "usemtl bsp_part0")
    discovered_output_root = tmp_path / "native_batch_discovered"
    discovered = subprocess.run(
        [
            str(binary),
            "run-batch",
            "--data_root",
            str(data_root),
            "--output_root",
            str(discovered_output_root),
            "--manifoldplus_bin",
            str(manifold),
            "--ftetwild_bin",
            str(ftetwild),
            "--init_type",
            "bsp",
            "--limit_per_category",
            "2",
            "--jobs",
            "auto",
            "--refine_max_step",
            "0",
            "--mcts_iter",
            "1",
            "--mcts_max_step",
            "1",
            "--num_action_scale",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert discovered.returncode == 0, discovered.stderr
    discovered_payload = json.loads(discovered.stdout)
    assert discovered_payload["pipeline_execution"] == "subprocess_parallel"
    assert discovered_payload["jobs"] == 2
    assert discovered_payload["mesh_count"] == 2
    assert discovered_payload["attempted"] == 2
    assert discovered_payload["success"] == 2
    discovered_work_dir = discovered_output_root / "shapenet_airplane__mesh_0"
    discovered_work_dir_1 = discovered_output_root / "shapenet_airplane__mesh_1"
    assert (discovered_output_root / "native_meshes.tsv").exists()
    assert (discovered_work_dir / "mcts_bboxs_steps0" / "bbox0.obj").exists()
    assert (discovered_work_dir_1 / "mcts_bboxs_steps0" / "bbox0.obj").exists()
    discovered_rows = [
        json.loads(line)
        for line in (discovered_output_root / "native_pipeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["pipeline_execution"] for row in discovered_rows} == {"subprocess_parallel"}


def test_smart_cpp_native_discover_meshes_writes_batch_list(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")

    root = tmp_path / "data"
    for category in ("shapenet_airplane", "shapenet_chair"):
        for idx in range(2):
            mesh_dir = root / category / f"mesh_{idx}"
            mesh_dir.mkdir(parents=True)
            (mesh_dir / "model.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            if idx == 0:
                (mesh_dir / "bsp_seg.obj").write_text(
                    "usemtl part0\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                    encoding="utf-8",
                )

    output = tmp_path / "meshes.tsv"
    completed = subprocess.run(
        [
            str(binary),
            "discover-meshes",
            "--data_root",
            str(root),
            "--output",
            str(output),
            "--categories",
            "shapenet_airplane,shapenet_chair",
            "--limit_per_category",
            "1",
            "--require_bsp",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["command"] == "discover-meshes"
    assert payload["mesh_count"] == 2
    assert payload["categories"] == {"shapenet_airplane": 1, "shapenet_chair": 1}
    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert rows[0].startswith("shapenet_airplane__mesh_0\t")
    assert rows[1].startswith("shapenet_chair__mesh_0\t")
    assert all("bsp_seg.obj" in row for row in rows)


def test_smart_cpp_native_batch_summary_reports_stage_bottlenecks(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")

    stats_a = tmp_path / "a_stats.json"
    output_b = tmp_path / "mesh_b"
    output_b.mkdir()
    stats_b = output_b / "native_pipeline_stats.json"
    stats_a.write_text(
        json.dumps(
            {
                "status": "success",
                "backend": "smart-cpp-native",
                "command": "run-pipeline",
                "elapsed_sec": 4.0,
                "reuse_existing": False,
                "stages": [
                    {"name": "normalize", "elapsed_sec": 0.2},
                    {"name": "ftetwild", "elapsed_sec": 3.1},
                    {"name": "refine_mcts", "elapsed_sec": 0.7},
                ],
            }
        ),
        encoding="utf-8",
    )
    stats_b.write_text(
        json.dumps(
            {
                "status": "success",
                "backend": "smart-cpp-native",
                "command": "run-pipeline",
                "elapsed_sec": 1.0,
                "reuse_existing": True,
                "stages": [
                    {"name": "normalize_reuse", "elapsed_sec": 0.0},
                    {"name": "ftetwild_reuse", "elapsed_sec": 0.0},
                    {"name": "refine_mcts", "elapsed_sec": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "native_pipeline.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "stage": "native_pipeline",
                        "backend": "smart-cpp-native",
                        "command": "run-batch",
                        "mesh_id": "shapenet_airplane__mesh_a",
                        "status": "success",
                        "pipeline_execution": "in_process",
                        "stats_path": str(stats_a),
                        "elapsed_sec": 4.0,
                    }
                ),
                json.dumps(
                    {
                        "stage": "native_pipeline",
                        "backend": "smart-cpp-native",
                        "command": "run-pipeline",
                        "category": "chair",
                        "mesh_id": "mesh_b",
                        "status": "success",
                        "pipeline_execution": "subprocess_parallel",
                        "output_path": str(output_b),
                        "elapsed_sec": 1.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            str(binary),
            "batch-summary",
            "--manifest",
            str(manifest),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["command"] == "batch-summary"
    assert payload["records"] == 2
    assert payload["success"] == 2
    assert payload["failed"] == 0
    assert payload["reuse_stage_count"] == 2
    assert payload["slowest_mesh"] == "shapenet_airplane__mesh_a"
    assert payload["slowest_stage_by_total"]["name"] == "ftetwild"
    assert payload["stages"]["refine_mcts"]["count"] == 2
    assert payload["pipeline_execution_counts"] == {
        "in_process": 1,
        "subprocess_parallel": 1,
    }
    assert payload["category_counts"] == {
        "chair": 1,
        "shapenet_airplane": 1,
    }


def test_smart_cpp_native_run_batch_applies_smart_category_tetra_defaults(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")

    def write_obj(path: Path) -> None:
        path.write_text(
            "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\n"
            "v -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
            "f 1 2 3\nf 1 3 4\nf 5 7 6\nf 5 8 7\n",
            encoding="utf-8",
        )

    data_root = tmp_path / "data"
    mesh_dir = data_root / "shapenet_chair" / "mesh_0"
    mesh_dir.mkdir(parents=True)
    write_obj(mesh_dir / "model.obj")

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    fixture_msh = tmp_path / "fixture.msh"
    sc.native_save_gmsh(str(fixture_msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    fixture_surface = tmp_path / "fixture_surface.obj"
    write_obj(fixture_surface)

    tools = tmp_path / "tools"
    tools.mkdir()
    manifold = tools / "fake_manifoldplus.py"
    manifold.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "shutil.copyfile(sys.argv[sys.argv.index('--input') + 1], sys.argv[sys.argv.index('--output') + 1])\n",
        encoding="utf-8",
    )
    ftetwild = tools / "fake_ftetwild.py"
    seen_args = tmp_path / "ftetwild_args.txt"
    ftetwild.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, shutil, sys\n"
        f"fixture_msh = {str(fixture_msh)!r}\n"
        f"fixture_surface = {str(fixture_surface)!r}\n"
        f"seen_args = pathlib.Path({str(seen_args)!r})\n"
        "seen_args.write_text('\\n'.join(sys.argv), encoding='utf-8')\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "shutil.copyfile(fixture_msh, out)\n"
        "shutil.copyfile(fixture_surface, out + '__sf.obj')\n",
        encoding="utf-8",
    )
    coacd = tools / "fake_coacd.py"
    coacd.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, shutil, sys\n"
        f"fixture_surface = {str(fixture_surface)!r}\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "shutil.copyfile(fixture_surface, out)\n",
        encoding="utf-8",
    )
    for tool in (manifold, ftetwild, coacd):
        tool.chmod(0o755)

    output_root = tmp_path / "native_batch_defaults"
    completed = subprocess.run(
        [
            str(binary),
            "run-batch",
            "--data_root",
            str(data_root),
            "--output_root",
            str(output_root),
            "--manifoldplus_bin",
            str(manifold),
            "--ftetwild_bin",
            str(ftetwild),
            "--coacd_bin",
            str(coacd),
            "--refine_max_step",
            "0",
            "--mcts_iter",
            "1",
            "--mcts_max_step",
            "1",
            "--num_action_scale",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["success"] == 1
    args = seen_args.read_text(encoding="utf-8").splitlines()
    assert args[args.index("-e") + 1] == "0.004"
    assert args[args.index("-l") + 1] == "0.2"
    manifest_rows = [
        json.loads(line)
        for line in (output_root / "native_pipeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows[0]["tetra_default_source"] == "smart_chair_table"
    assert manifest_rows[0]["tetra_epsilon"] == "0.004"
    assert manifest_rows[0]["tetra_edge_length"] == "0.2"


def test_cpp_native_runner_full_pipeline_wrapper_invokes_executable(tmp_path) -> None:
    binary = Path("build/smart-cpp-native")
    if not binary.exists():
        pytest.skip("smart-cpp-native executable is not built")
    input_obj = tmp_path / "model.obj"
    input_obj.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
        encoding="utf-8",
    )
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    fixture_msh = tmp_path / "fixture.msh"
    sc.native_save_gmsh(str(fixture_msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    fixture_obj = tmp_path / "fixture.obj"
    fixture_obj.write_text(
        "o part0\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
        encoding="utf-8",
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    manifold = tools / "manifold.py"
    ftetwild = tools / "ftetwild.py"
    coacd = tools / "coacd.py"
    manifold.write_text(
        "#!/usr/bin/env python3\nimport shutil, sys\nshutil.copyfile(sys.argv[sys.argv.index('--input') + 1], sys.argv[sys.argv.index('--output') + 1])\n",
        encoding="utf-8",
    )
    ftetwild.write_text(
        "#!/usr/bin/env python3\nimport shutil, sys\n"
        f"msh={str(fixture_msh)!r}; obj={str(fixture_obj)!r}\n"
        "out=sys.argv[sys.argv.index('--output')+1]\n"
        "shutil.copyfile(msh, out); shutil.copyfile(obj, out+'__sf.obj')\n",
        encoding="utf-8",
    )
    coacd.write_text(
        "#!/usr/bin/env python3\nimport shutil, sys\n"
        f"obj={str(fixture_obj)!r}\n"
        "out=sys.argv[sys.argv.index('-o')+1]\n"
        "shutil.copyfile(obj, out)\n",
        encoding="utf-8",
    )
    for tool in (manifold, ftetwild, coacd):
        tool.chmod(0o755)

    result = native_runner.run_pipeline_from_files(
        input_mesh=input_obj,
        work_dir=tmp_path / "wrapped_pipeline",
        manifoldplus_bin=manifold,
        ftetwild_bin=ftetwild,
        coacd_bin=coacd,
        mesh_id="mesh-a",
        refine_max_step=0,
        mcts_iter=1,
        mcts_max_step=1,
        num_action_scale=2,
    )

    assert result["status"] == "success"
    assert result["metadata"]["stdout"]["command"] == "run-pipeline"
    assert (result["output_path"] / "bbox0.obj").exists()


def test_cpp_native_file_runner_merge_writes_segment_and_bbox_metadata(tmp_path) -> None:
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ]
    voxels = [[0, 1, 2, 3], [1, 2, 3, 4]]
    path = tmp_path / "two_tet.msh"
    sc.native_save_gmsh(str(path), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    partitions = tmp_path / "partitions.json"
    partitions.write_text(json.dumps({"partitions": [[0], [1]]}), encoding="utf-8")
    segment = tmp_path / "greedy_segment0_coacd_mgeps0.02_fm.txt"

    result = native_runner.run_merge_from_partitions_file(
        msh_path=path,
        partition_metadata_path=partitions,
        output_segment_path=segment,
        category="tet",
        merge_eps=0.02,
        final_k=1,
        tilted=False,
        only_nearby=True,
    )

    assert result["status"] == "success"
    assert segment.exists()
    assert Path(str(segment) + ".bbox_params.json").exists()
    assert Path(str(segment) + ".native_stats.json").exists()
    assert result["metadata"]["active_partition_count"] == 1


def test_cpp_manifold_bridge_mesh_matches_cube_volume() -> None:
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
    bridge = sc.ManifoldBridgeMesh(vertices, faces)
    assert abs(bridge.volume() - 1.0) < 1e-6
    bounds = [[0, 0, 0, 0.5, 1, 1]]
    rotations = [[1, 0, 0, 0, 1, 0, 0, 0, 1]]
    assert abs(bridge.covered_for_bounds(bounds, rotations, 1.0) - 0.5) < 1e-6
    assert abs(bridge.residual_volume_for_box_params(bounds, rotations) - 0.5) < 1e-6


def test_cpp_manifold_state_segment_delta_matches_full_state_copy() -> None:
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
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    bounds = [[-0.1, 0.0, 0.0, 1.0, 1.0, 1.0]]

    full_state = sc.ManifoldState(
        vertices,
        faces,
        bounds,
        rotations,
        2,
        0.1,
        1.0,
        -0.1,
        True,
        1024,
        "mesh",
    )
    delta_state = sc.ManifoldState(
        vertices,
        faces,
        bounds,
        rotations,
        2,
        0.1,
        1.0,
        -0.1,
        True,
        1024,
        "mesh",
    )

    full_bounds, full_rotations, rewards, actions, last_score = full_state.greedy_axis_refine_segment(
        100.0,
        1.0,
        1,
    )
    delta_actions, delta_rewards, touched, touched_bounds, touched_rotations, delta_score = (
        delta_state.greedy_axis_refine_segment_delta(100.0, 1.0, 1)
    )

    assert delta_actions == actions
    assert delta_rewards == rewards
    assert touched == [0]
    assert touched_bounds == [full_bounds[0]]
    assert touched_rotations == [full_rotations[0]]
    assert delta_score == last_score
    assert delta_state.state_key() == full_state.state_key()


def test_cpp_manifold_state_can_reset_to_initial() -> None:
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
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    bounds = [[0.0, 0.0, 0.0, 0.5, 1.0, 1.0]]
    state = sc.ManifoldState(
        vertices,
        faces,
        bounds,
        rotations,
        2,
        0.1,
        1.0,
        -0.5,
        True,
        1024,
        "mesh",
    )
    initial_key = state.state_key()
    initial_score = state.last_bbox_score()
    reward, bbox_idx, new_bounds, _rotation, next_score = state.apply_axis_action_delta(
        0,
        100.0,
        1.0,
    )
    assert bbox_idx == 0
    assert new_bounds != bounds[0]
    assert state.state_key() != initial_key
    assert next_score == state.last_bbox_score()
    assert reward == next_score - initial_score

    state.reset_to_initial()

    assert state.state_key() == initial_key
    assert state.bounds() == bounds
    assert state.rotations() == rotations
    assert state.last_bbox_score() == initial_score


def test_cpp_native_smart_engine_runs_without_python_env_callbacks(tmp_path) -> None:
    assert sc.NativeSmartEngine is not None
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
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    bounds = [[-0.1, 0.0, 0.0, 1.0, 1.0, 1.0]]
    engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [],
        [],
        [],
        bounds,
        rotations,
        "cube",
        2,
        0.1,
        1.0,
        -0.1,
        True,
        1024,
        "mesh",
    )

    refine = engine.run_refine(1, 100.0, 1.0)
    assert refine["axis_only"] is True
    assert refine["actions"]
    assert refine["last_bbox_score"] > -0.1

    mcts_engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [],
        [],
        [],
        bounds,
        rotations,
        "cube",
        2,
        0.1,
        1.0,
        -0.1,
        True,
        1024,
        "mesh",
    )
    mcts = mcts_engine.run_mcts(4, 2, 100.0, 1.0, 0.001, 1.0, 123, [], [], 0.0, 0.0)
    assert mcts["axis_only"] is True
    assert mcts["tree"] is True
    assert mcts["iterations_run"] == 4
    assert mcts["node_count"] >= 2
    assert mcts_engine.stats()["native_mcts_axis_only"] == 1.0
    assert mcts_engine.stats()["native_mcts_tree"] == 1.0

    tt_engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [],
        [],
        [],
        bounds,
        rotations,
        "cube",
        2,
        0.1,
        1.0,
        -0.1,
        True,
        1024,
        "mesh",
    )
    tt_mcts = tt_engine.run_mcts(
        4, 2, 100.0, 1.0, 0.001, 1.0, 123, [], [], 0.0, 0.0, True, 16
    )
    assert tt_mcts["tree"] is True
    assert tt_engine.stats()["native_mcts_transposition_table"] == 1.0
    assert "transposition_hits" in tt_mcts

    output = tmp_path / "native_boxes.obj"
    assert mcts_engine.export_obj(str(output)) == 1
    assert output.read_text().count("\nv ") == 8
    assert output.read_text().startswith("o bbox_0\nv ")
    bbox_dir = tmp_path / "bboxs_steps0"
    assert mcts_engine.export_bbox_dir(str(bbox_dir)) == 1
    assert (bbox_dir / "bbox0.obj").exists()
    assert (bbox_dir / "bbox_params.json").exists()
    assert '"bounds"' in (bbox_dir / "bbox_params.json").read_text()

    combined_engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [],
        [],
        [],
        bounds,
        rotations,
        "cube",
        2,
        0.1,
        1.0,
        -0.1,
        True,
        1024,
        "mesh",
    )
    combined = combined_engine.run_refine_then_mcts(
        1, 2, 2, 100.0, 1.0, 0.001, 1.0, 123, [], [], 0.0, 0.0, True, 16
    )
    assert combined["command"] == "refine-mcts"
    assert combined["single_state_bridge"] is True
    assert combined["single_engine_state"] is True
    assert combined["refine"]["actions"]
    assert combined["mcts"]["tree"] is True
    assert combined["stats"]["native_refine_then_mcts_runs"] == 1.0
    assert combined["stats"]["native_refine_then_mcts_single_state_bridge"] == 1.0
    assert combined["stats"]["native_mcts_transposition_table"] == 1.0


def test_cpp_native_smart_engine_can_apply_recenter_action_in_mcts() -> None:
    assert sc.NativeSmartEngine is not None
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
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    bounds = [[-0.3, -0.3, -0.3, 0.7, 0.7, 0.7]]
    engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [[0, 1, 2, 4]],
        [1.0 / 6.0],
        [[0.25, 0.25, 0.25]],
        bounds,
        rotations,
        "cube",
        2,
        0.1,
        1.0,
        -10.0,
        True,
        1024,
        "mesh",
    )

    recenter_action = 12
    prior_logits = [0.0] * 13
    prior_logits[recenter_action] = 10.0
    mcts = engine.run_mcts(
        1,
        1,
        100.0,
        1.0,
        0.001,
        1.0,
        123,
        prior_logits,
        [],
        1.0,
        0.0,
    )

    assert mcts["recenter_enabled"] is True
    assert mcts["axis_only"] is False
    assert mcts["actions"] == [recenter_action]
    assert engine.stats()["native_recenter_applies"] == 1.0


def test_cpp_native_smart_engine_heap_merge() -> None:
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
    rotations = [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ]
    bounds = [
        [0.0, 0.0, 0.0, 0.4, 1.0, 1.0],
        [0.6, 0.0, 0.0, 1.0, 1.0, 1.0],
    ]
    engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [],
        [],
        [],
        bounds,
        rotations,
        "cube",
        2,
        0.1,
        1.0,
        -0.2,
        True,
        1024,
        "mesh",
    )
    merged = engine.run_merge([[0, 1]], 0.0, 1.0)
    assert merged["heap"] is True
    assert merged["ordered_delta_queue"] is True
    assert merged["merges"] == [(0, 1)]
    assert merged["active_indices"] == [0]
    assert merged["bounds"] == [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
    assert engine.stats()["native_merge_ordered_delta"] == 1.0


def test_cpp_native_smart_engine_ordered_delta_merge_updates_only_affected_edges() -> None:
    vertices = [[x, y, z] for x in [0.0, 2.0] for y in [0.0, 1.0] for z in [0.0, 1.0]]
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
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(4)]
    bounds = [
        [0.0, 0.0, 0.0, 0.2, 1.0, 1.0],
        [0.25, 0.0, 0.0, 0.45, 1.0, 1.0],
        [0.5, 0.0, 0.0, 0.7, 1.0, 1.0],
        [0.75, 0.0, 0.0, 0.95, 1.0, 1.0],
    ]
    engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [],
        [],
        [],
        bounds,
        rotations,
        "chain",
        2,
        0.1,
        1.0,
        -0.2,
        True,
        1024,
        "mesh",
    )
    merged = engine.run_merge([[0, 1], [1, 2], [2, 3]], -1.0, 1.0)

    assert merged["ordered_delta_queue"] is True
    assert len(merged["merges"]) == 3
    assert merged["active_indices"] == [0]
    assert merged["bounds"] == [[0.0, 0.0, 0.0, 0.95, 1.0, 1.0]]
    assert merged["candidate_inserts"] < 9


def test_cpp_native_smart_engine_merge_uses_legacy_negative_eps_threshold() -> None:
    vertices = [[x, y, z] for x in [0.0, 2.0] for y in [0.0, 1.0] for z in [0.0, 1.0]]
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
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(2)]
    bounds = [
        [0.0, 0.0, 0.0, 0.4, 1.0, 1.0],
        [0.6, 0.0, 0.0, 1.0, 1.0, 1.0],
    ]
    engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [],
        [],
        [],
        bounds,
        rotations,
        "gap",
        2,
        0.1,
        1.0,
        -0.2,
        True,
        1024,
        "mesh",
    )

    merged = engine.run_merge([[0, 1]], 0.21, 1.0)
    assert merged["merges"] == [(0, 1)]

    engine = sc.NativeSmartEngine(
        vertices,
        faces,
        [],
        [],
        [],
        bounds,
        rotations,
        "gap",
        2,
        0.1,
        1.0,
        -0.2,
        True,
        1024,
        "mesh",
    )
    forced = engine.run_merge([[0, 1]], 0.0, 1.0, 1)
    assert forced["merges"] == [(0, 1)]


def test_cpp_native_smart_engine_partition_merge_supports_tilted_scoring() -> None:
    vertices = [[x, y, z] for x in [0.0, 2.0] for y in [0.0, 1.0] for z in [0.0, 1.0]]
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
    voxels = [[0, 1, 2, 4], [3, 5, 6, 7]]
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(2)]
    bounds = [
        [0.0, 0.0, 0.0, 0.4, 1.0, 1.0],
        [0.6, 0.0, 0.0, 1.0, 1.0, 1.0],
    ]
    engine = sc.NativeSmartEngine(
        vertices,
        faces,
        voxels,
        [1.0, 1.0],
        [[0.25, 0.25, 0.25], [1.75, 0.75, 0.75]],
        bounds,
        rotations,
        "partition-merge",
        2,
        0.1,
        2.0,
        -0.2,
        True,
        1024,
        "mesh",
    )

    merged = engine.run_partition_merge([[0], [1]], [[0, 1]], 0.0, 2.0, 1, True)

    assert merged["tilted"] is True
    assert merged["ordered_delta_queue"] is True
    assert merged["merges"] == [(0, 1)]
    assert merged["active_indices"] == [0]
    assert merged["partitions"] == [[0, 1]]
    assert len(merged["bounds"]) == 1
    assert len(merged["rotations"][0]) == 9
    assert engine.stats()["native_partition_merge_tilted"] == 1.0


def test_cpp_native_partition_merge_builds_adjacency_in_cpp() -> None:
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
    voxels = [[0, 1, 2, 4], [1, 2, 4, 7], [3, 5, 6, 7]]
    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(3)]
    bounds = [
        [0.0, 0.0, 0.0, 0.4, 0.4, 0.4],
        [0.5, 0.0, 0.0, 0.8, 0.4, 0.4],
        [0.0, 0.5, 0.5, 0.4, 0.9, 0.9],
    ]
    engine = sc.NativeSmartEngine(
        vertices,
        faces,
        voxels,
        [1.0, 1.0, 1.0],
        [[0.25, 0.25, 0.25], [0.75, 0.25, 0.25], [0.25, 0.75, 0.75]],
        bounds,
        rotations,
        "adjacency",
        2,
        0.1,
        3.0,
        -0.2,
        True,
        1024,
        "mesh",
    )

    assert engine.partition_adjacency_pairs([[0], [1], [2]], True) == [[0, 1]]
    assert engine.partition_adjacency_pairs([[0], [1], [2]], False) == [
        [0, 1],
        [0, 2],
        [1, 2],
    ]
    merged = engine.run_partition_merge_auto_adjacency(
        [[0], [1], [2]], True, 0.0, 3.0, 2, False
    )
    assert merged["adjacency_only_nearby"] is True
    assert merged["adjacency_pair_count"] == 1
    assert merged["merges"] == [(0, 1)]


def test_cpp_tet_clipping_state_matches_simple_tetrahedron() -> None:
    assert sc.TetClippingState is not None

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    voxels = [[0, 1, 2, 3]]
    metrics = sc.tet_clipping_metrics(
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

    state = sc.TetClippingState(vertices, voxels, 1.0 / 6.0)
    state_metrics = state.metrics([vertices], max_boxes=1)
    for key in ("BVS", "Covered", "MOV", "TOV", "vIoU"):
        assert abs(state_metrics[key] - metrics[key]) < 1e-12

    rotations = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    box_metrics = state.metrics_for_boxes(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        rotations,
        max_boxes=1,
    )
    assert box_metrics["BVS"] == 6.0
    assert abs(box_metrics["Covered"] - 1.0) < 1e-12
    assert (
        abs(
            state.covered_for_boxes(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
                rotations,
                max_boxes=1,
            )
            - box_metrics["Covered"]
        )
        < 1e-12
    )
