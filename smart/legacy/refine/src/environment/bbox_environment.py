import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pymanifold
import pymesh
import trimesh
import trimesh.repair

try:
    import smart.rust as smart_rust
except ImportError:
    smart_rust = None

_RUST_ACTION_HELPER_MIN = 128


class BBox:
    def __init__(self, box: List[float], rot=None) -> None:
        # l_x, l_y ,l_z, r_x, r_y, r_z
        assert len(box) == 6, "BBox must have length 6 list"

        if box[0] >= box[3] or box[1] >= box[4] or box[2] >= box[5]:
            assert 0, "Invalid bbox initialized"

        self.box: List[float] = box
        if rot is None:
            rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.rot = rot

    def valid_bbox(self) -> bool:
        assert len(self.box) == 6, "BBox must have length 6 list"

        if self.box[0] >= self.box[3]:
            return False
        if self.box[1] >= self.box[4]:
            return False
        if self.box[2] >= self.box[5]:
            return False
        return True

    def get_obs_bbox(self):
        nw_mn = list(np.matmul(self.box[:3], self.rot))
        nw_mx = list(np.matmul(self.box[3:], self.rot))
        nw_rot = list(self.rot.reshape((-1)))

        return nw_mn + nw_mx + nw_rot


class MeshBBoxEnv:
    metadata = {"render.modes": ["file", "views", "video"]}

    def __init__(self, vertices, faces, voxels, args, name) -> None:
        super(MeshBBoxEnv, self).__init__()
        random.seed(args.seed)
        np.random.seed(args.seed)

        self.name = name
        self.exp_name = None
        self.args = args
        self.max_points = 0
        self.max_bboxs = 0
        self.num_bbox = 0
        self.num_meshes = 1

        self.num_actions = 0
        self.action_unit = args.action_unit
        self.num_action_scale = args.num_action_scale * 2
        self._actions_per_bbox = 6 * self.num_action_scale + 1
        self.action_scale = _action_scales(self.num_action_scale)
        self.score_cache_size = max(0, int(getattr(args, "score_cache_size", 4096)))
        self.score_cache = OrderedDict()
        self.covered_cache = OrderedDict()
        self.step_reward_cache = OrderedDict()
        self.bbox_geometry_cache = OrderedDict()
        self.recenter_params_cache = OrderedDict()
        self.recenter_candidate_cache = OrderedDict()
        self._cached_state_key = None
        self._rust_bbox_state = None
        self.reward_backend = str(getattr(args, "reward_backend", "manifold"))
        self.manifold_volume_method = str(getattr(args, "manifold_volume_method", "mesh"))
        if self.manifold_volume_method not in {"mesh", "properties"}:
            raise ValueError(
                "manifold_volume_method must be one of {'mesh', 'properties'}"
            )
        self.tet_clipping_max_boxes = int(getattr(args, "tet_clipping_max_boxes", 12))
        self._tet_clipping_state = None
        self._manifold_bridge_mesh = None
        self._manifold_stateful = None
        self._manifold_stateful_key = None
        self._manifold_stateful_score = None
        self._bridge_apply_cache = {}
        self._initial_bbox_state = None
        self._initial_bbox_cache_hits = 0
        self._initial_bbox_cache_misses = 0
        self.candidate_backend = str(getattr(args, "candidate_backend", "exact"))
        self.candidate_top_k = max(0, int(getattr(args, "candidate_top_k", 8)))
        self.candidate_prefilter_stats = {
            "calls": 0,
            "actions_total": 0,
            "proxy_candidates": 0,
            "proxy_exact": 0,
            "fallback_exact": 0,
            "upper_pruned": 0,
            "selected_from_proxy": 0,
            "selected_from_fallback": 0,
            "no_action": 0,
        }
        self._candidate_bitset_state = None

        self.max_step = args.max_step

        self.step_vec = None
        self.action_mask = None

        self.pen_rate = 1.0

        vertices = _as_numpy(vertices)
        faces = _as_numpy(faces)
        voxels = _as_numpy(voxels)

        self.tetmsh = pymesh.form_mesh(vertices, faces, voxels)
        self.tetmsh.enable_connectivity()
        self.tetmsh.add_attribute("voxel_volume")
        self.tetmsh.add_attribute("voxel_centroid")
        self.tetmsh.add_attribute("voxel_partition")

        self.volume = self.tetmsh.get_attribute("voxel_volume")
        self.volume_sum = np.sum(self.volume)
        if self.reward_backend == "tet_clipping":
            if smart_rust is None or not smart_rust.using_rust() or smart_rust.TetClippingState is None:
                raise RuntimeError(
                    "reward_backend=tet_clipping requires the smart._rust TetClippingState backend"
                )
            self._tet_clipping_state = smart_rust.TetClippingState(
                self.tetmsh.vertices.tolist(),
                self.tetmsh.voxels.astype(int).tolist(),
                float(self.volume_sum),
            )

        tmp = self.tetmsh.get_attribute("voxel_centroid")
        self.centroid = np.asarray(tmp, dtype=float).reshape((-1, 3))

        data_path = os.path.join(args.path_to_msh_file, name)
        if self.args.baseline == "":
            self.args.path_to_bbox = os.path.join(self.args.path_to_bbox, "result")

        self.trimsh = trimesh.exchange.load.load(
            os.path.join(data_path, "tetra.msh__sf.obj"), file_type="obj", process=False
        )
        trimesh.repair.fix_normals(self.trimsh)

        mesh = pymanifold.Mesh(
            vert_pos=np.array(self.trimsh.vertices), tri_verts=np.array(self.trimsh.faces)
        )
        self.manmsh = pymanifold.Manifold()
        self.manmsh = self.manmsh.from_mesh(mesh)
        if self.reward_backend in {"manifold_bridge", "manifold_stateful"}:
            if (
                smart_rust is None
                or not smart_rust.using_rust()
                or not smart_rust.manifold_bridge_available()
                or smart_rust.ManifoldBridgeMesh is None
            ):
                raise RuntimeError(
                    "reward_backend=manifold_bridge/manifold_stateful requires smart._rust with the fixed C++ Manifold bridge"
                )
            self._manifold_bridge_mesh = smart_rust.ManifoldBridgeMesh(
                np.asarray(self.trimsh.vertices, dtype=float).tolist(),
                np.asarray(self.trimsh.faces, dtype=int).tolist(),
            )
            if self.reward_backend == "manifold_stateful" and (
                not hasattr(smart_rust, "ManifoldState")
                or smart_rust.ManifoldState is None
            ):
                raise RuntimeError(
                    "reward_backend=manifold_stateful requires smart._rust.ManifoldState"
                )

        if self.args.run_type == "train":
            self.global_obs = []
            for i in range(len(self.tetmsh.voxels)):
                tet_info = []
                for j in range(4):
                    tet_info.append(list(self.tetmsh.vertices[self.tetmsh.voxels[i][j]]))
                tet_info.sort()

                tet_info = np.array(tet_info).reshape(-1)
                tet_info = np.concatenate((tet_info, np.array([self.volume[i]])))
                self.global_obs.append(tet_info)

            self.global_obs = np.array(self.global_obs)
            self.max_points = len(self.global_obs)
        else:
            self.global_obs = None
            self.max_points = len(self.tetmsh.voxels)

        self.reset()

        self.num_actions = self.max_bboxs * self._actions_per_bbox
        self.action_mask = np.zeros(self.num_actions)
        self.action_mask[self.num_bbox * self._actions_per_bbox :] = 1

        self.action2idx = np.zeros(
            (self.max_bboxs * self._actions_per_bbox, 3), dtype=int
        )
        self.idx2action = np.zeros(
            ((self.max_bboxs, 7, self.num_action_scale)), dtype=int
        )

        for action_id, (i, j, k) in enumerate(
            _action_indices(self.max_bboxs, self.num_action_scale)
        ):
            self.idx2action[i][j][k] = action_id
            self.action2idx[action_id] = np.array([i, j, k])

        print("Environment initialization done")

    def reset(self, pen_rate=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Reset the state of the environment to an initial state
        self.done = 0
        self.step_cnt = 0
        self.last_bbox_score = 0

        self.bbox_list: List[BBox] = []
        self.bbox_man = []
        self.bbox_mesh: List[trimesh.Trimesh] = []
        self.bbox_part_vol: List[float] = []
        self.bbox_part_ov = []
        self.bbox_part_occ = []

        self.action_mask = np.array([])

        self.step_cache = None
        self._cached_state_key = None
        self._rust_bbox_state = None
        if not self._use_manifold_stateful_reward():
            self._manifold_stateful = None
        self._manifold_stateful_key = None
        self._manifold_stateful_score = None
        self._bridge_apply_cache = {}

        if self._restore_cached_initial_bbox_state():
            self._initial_bbox_cache_hits += 1
        else:
            self._initial_bbox_cache_misses += 1
            if self.args.bbox_init == "random":
                self.random_bbox_init(self.args.num_bbox)
            elif self.args.bbox_init == "bsp_preseg":
                self.bsp_bbox_init(self.args.num_bbox)
            elif self.args.bbox_init == "grd_merged":
                self.grd_merged_bbox_init(self.args.num_bbox)
            elif self.args.bbox_init == "bbox_direct":
                self.bbox_direct_init()
            else:
                assert 0, "Bounding box initialization not selected"
            self._store_initial_bbox_state()

        self.max_bboxs = len(self.bbox_list)
        self.num_bbox = len(self.bbox_list)

        for i in range(self.num_bbox):
            self.bbox_mesh.append(None)
            self.bbox_man.append(None)

            # Exact bridge/stateful reward backends evaluate from explicit
            # bounds/rotations in the Rust/C++ bridge. Building Python
            # pymanifold bbox objects during every MCTS reset is pure overhead
            # unless TOV/legacy metrics or rendering later request them.
            if not (self._use_manifold_bridge_reward() and not bool(self.args.tov)):
                self.bbox_mesh[i], self.bbox_man[i] = self.get_bbox_bmesh_bman(i)

            self.bbox_part_vol.append(-1)
            self.bbox_part_ov.append(-1)
            self.bbox_part_occ.append(-1)

        self._refresh_step_vec()

        self.action_mask = np.zeros(self.num_actions)
        self.action_mask[self.num_bbox * self._actions_per_bbox :] = 1

        self.pen_rate = pen_rate
        self.last_bbox_score = self.evaluate_bbox_score()
        self._rust_bbox_state = None
        self._ensure_manifold_stateful_state()

        return self.current_observation()

    def random_bbox_init(self, num_bbox: int) -> None:
        assert (
            num_bbox != 0
        ), "random bounding box initialization does not support auto number inference of bounding boxes"
        assert not self.args.tilted, "Tilting is not supported in random bbox init"

        for i in range(num_bbox):
            lx, ly, lz, rx, ry, rz = np.random.normal(0, 0.3, 6)
            lx, rx = min(lx, rx), max(lx, rx)
            ly, ry = min(ly, ry), max(ly, ry)
            lz, rz = min(lz, rz), max(lz, rz)

            self.bbox_list.append(BBox([lx, ly, lz, rx, ry, rz]))

    def bsp_bbox_init(
        self,
        num_bbox: int,
    ) -> None:
        """
        Initialize bounding boxes with BSP-Net output pre-segment
        """
        assert not self.args.tilted, "Tilting is not supported in random bbox init"

        from ..utils.bbox_bsp_preseg import bbox_bsp_preseg

        self.bbox_list = bbox_bsp_preseg(
            os.path.join(self.args.path_to_msh_file, self.name), self.tetmsh, num_bbox
        )
        for i in range(len(self.bbox_list)):
            self.bbox_list[i] = BBox(self.bbox_list[i])

    def grd_merged_bbox_init(self, num_bbox) -> None:
        assert (
            num_bbox == 0
        ), "Greedy merged bounding box initialization does not support selecting parts yet"

        grd_txt = os.path.join(
            os.path.join(self.args.path_to_msh_file, self.name),
            "greedy_segment%d_%smgeps%g%s.txt"
            % (
                num_bbox,
                "coacd_" if self.args.init_type == "coacd" else "",
                self.args.merge_eps,
                "_fm" if self.args.fast_merge else "",
            ),
        )
        assert os.path.exists(grd_txt), "Initial greedy merged result file does not exist"

        num_bbox = 0
        part_pts = []
        with open(grd_txt, "r") as f:
            num_bbox = int(f.readline().strip())

            for i in range(num_bbox):
                pts_idx = list(map(int, f.readline().strip().split(" ")))
                part_pts.append(pts_idx)

        for i in range(num_bbox):
            pts = []
            for j in range(len(part_pts[i])):
                for k in range(4):
                    pts.append(
                        self.tetmsh.vertices[self.tetmsh.voxels[part_pts[i][j]][k]]
                    )

            rot_mat = None
            if self.args.tilted:
                to_origin, _ = trimesh.bounds.oriented_bounds(pts, angle_digits=3)
                rot_mat = to_origin[:3, :3]

                pts = np.matmul(pts, np.transpose(rot_mat))

            mn_pt, mx_pt = list(np.min(pts, axis=0)), list(np.max(pts, axis=0))
            self.bbox_list.append(BBox(mn_pt + mx_pt, rot=rot_mat))

    def bbox_direct_init(
        self,
    ) -> None:
        if self.args.baseline == "mcts":
            result_path = os.path.join(
                os.path.join(self.args.path_to_bbox, self.name), "result"
            )

            updates = [int(file[7:]) for file in os.listdir(result_path)]
            updates.sort()

            assert len(updates), "rl does not have any results"

            best_update = updates[-1]

            result_path = os.path.join(result_path, "updated%d" % (best_update))

            bbox_update_path = os.path.join(result_path, self.name)
            steps = [int(file[11:]) for file in os.listdir(bbox_update_path)]
            steps.sort()
            best_step = steps[-1]

            mesh_path = os.path.join(bbox_update_path, "bboxs_steps%d" % (best_step))
            bbox_dir = os.listdir(mesh_path)
            bbox_list = []
            for i in range(len(bbox_dir)):
                if bbox_dir[i][:4] == "bbox":
                    bbox_list.append(
                        trimesh.load(
                            os.path.join(mesh_path, bbox_dir[i]),
                            file_type="obj",
                            process=False,
                        )
                    )
            for i in range(len(bbox_list)):
                to_origin, _ = trimesh.bounds.oriented_bounds(
                    bbox_list[i].vertices, angle_digits=3
                )
                rot_mat = to_origin[:3, :3]

                pts = np.matmul(bbox_list[i].vertices, np.transpose(rot_mat))

                mn_pt, mx_pt = list(np.min(pts, axis=0)), list(np.max(pts, axis=0))
                self.bbox_list.append(BBox(mn_pt + mx_pt, rot=rot_mat))
            return

        assert os.path.exists(self.args.path_to_bbox), "result path does not exist"
        if self.args.baseline != "":

            bbox_path = os.path.join(self.args.path_to_bbox, self.name)
            if self.args.baseline == "cubseg":
                bbox_path = os.path.join(bbox_path, "cube_masked")
            bbox_list = []
            bbox_dir = os.listdir(bbox_path)
            for i in range(len(bbox_dir)):
                if bbox_dir[i][:4] == "bbox":
                    bbox_list.append(
                        trimesh.load(
                            os.path.join(bbox_path, bbox_dir[i]),
                            file_type="obj",
                            process=False,
                        )
                    )

            for i in range(len(bbox_list)):
                to_origin, _ = trimesh.bounds.oriented_bounds(
                    bbox_list[i].vertices, angle_digits=3
                )
                rot_mat = to_origin[:3, :3]

                pts = np.matmul(bbox_list[i].vertices, np.transpose(rot_mat))

                mn_pt, mx_pt = list(np.min(pts, axis=0)), list(np.max(pts, axis=0))
                self.bbox_list.append(BBox(mn_pt + mx_pt, rot=rot_mat))
        else:
            updates = [int(file[7:]) for file in os.listdir(self.args.path_to_bbox)]
            updates.sort()

            assert len(updates), "rl does not have any results"

            best_update = updates[-1]

            bbox_update_path = os.path.join(
                os.path.join(self.args.path_to_bbox, "updated%d" % (best_update)),
                self.name,
            )

            steps = [
                int(file[11:]) if file[11:].isdigit() else 0
                for file in os.listdir(bbox_update_path)
            ]
            steps.sort()
            best_step = steps[-1]

            bbox_path = os.path.join(bbox_update_path, "bboxs_steps%d" % (best_step))

            bbox_list = []
            bbox_dir = os.listdir(bbox_path)
            for i in range(len(bbox_dir)):
                if bbox_dir[i][:4] == "bbox":
                    bbox_list.append(
                        trimesh.load(
                            os.path.join(bbox_path, bbox_dir[i]),
                            file_type="obj",
                            process=False,
                        )
                    )

            for i in range(len(bbox_list)):
                to_origin, _ = trimesh.bounds.oriented_bounds(
                    bbox_list[i].vertices, angle_digits=3
                )
                rot_mat = to_origin[:3, :3]

                pts = np.matmul(bbox_list[i].vertices, np.transpose(rot_mat))

                mn_pt, mx_pt = list(np.min(pts, axis=0)), list(np.max(pts, axis=0))
                self.bbox_list.append(BBox(mn_pt + mx_pt, rot=rot_mat))

    def _can_cache_initial_bbox_state(self):
        # MCTS repeatedly resets to the same deterministic init. Avoid re-reading
        # bbox OBJ files and recomputing oriented bounds on every rollout.
        return bool(getattr(self.args, "cache_initial_bbox_state", True)) and (
            self.args.bbox_init in {"bsp_preseg", "grd_merged", "bbox_direct"}
        )

    def _store_initial_bbox_state(self):
        if not self._can_cache_initial_bbox_state():
            return
        self._initial_bbox_state = [
            (
                list(map(float, bbox.box)),
                np.asarray(bbox.rot, dtype=float).copy(),
            )
            for bbox in self.bbox_list
        ]

    def _restore_cached_initial_bbox_state(self):
        if not self._can_cache_initial_bbox_state():
            return False
        if self._initial_bbox_state is None:
            return False
        self.bbox_list = [
            BBox(list(box), rot=np.asarray(rot, dtype=float).copy())
            for box, rot in self._initial_bbox_state
        ]
        return True

    def render(self, num_update=0, mode="file") -> None:
        def render_axis_bbox(box_idx) -> None:
            self._ensure_bbox_mesh(box_idx)
            trimesh.exchange.export.export_mesh(
                self.bbox_mesh[box_idx],
                os.path.join(
                    os.path.join(f, "bboxs_steps%d" % (self.step_cnt)),
                    "bbox%d.obj" % (box_idx),
                ),
                "obj",
            )

        if self.args.debug:
            return

        # Render the tetmesh into .msh file
        if mode == "file":
            if self.exp_name is None:
                assert 0, "experiment name not set"
            f = os.path.join(
                os.path.join(self.args.result_path, self.exp_name),
                os.path.join(
                    "result", os.path.join("updated%d" % (num_update), self.name)
                ),
            )
            os.makedirs(f, exist_ok=True)

            filename = "%s" % (self.name[0:10],)

            os.makedirs(os.path.join(f, "bboxs_steps%d" % (self.step_cnt)), exist_ok=True)

            if not getattr(self.args, "skip_render_partition", False):
                pt_sdf = [-1e30 for i in range(len(self.volume))]
                part = [-1 for i in range(len(self.volume))]

                for i in range(self.num_bbox):
                    if not self.bbox_list[i].valid_bbox():
                        continue

                    self._ensure_bbox_mesh(i)
                    sdf_query = trimesh.proximity.ProximityQuery(self.bbox_mesh[i])
                    sdf = sdf_query.signed_distance(self.centroid)

                    for j in range(len(sdf)):
                        if sdf[j] > pt_sdf[j]:
                            pt_sdf[j] = sdf[j]
                            part[j] = i + 1

                part = np.array(part)
                self.tetmsh.set_attribute("voxel_partition", part)

                pymesh.save_mesh(
                    os.path.join(
                        os.path.join(f, "bboxs_steps%d" % (self.step_cnt)), filename + ".msh"
                    ),
                    self.tetmsh,
                    "voxel_partition",
                    ascii=True,
                )
                trimesh.exchange.export.export_mesh(
                    self.trimsh,
                    os.path.join(
                        os.path.join(f, "bboxs_steps%d" % (self.step_cnt)), filename + ".obj"
                    ),
                    "obj",
                )

            for i in range(self.num_bbox):
                if self.bbox_list[i].valid_bbox():
                    render_axis_bbox(i)

    def current_observation(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return observation which consist below information
        N = self.max_points
        N_B = self.max_bboxs
        Per tetrahedral info: N x [(x, y, z) x 4, volume] (4 points of tetrahedral mesh)
        Per bounding box info: N_B x [x_min, y_min, z_min, x_max, y_max, z_max]
        Step vector: max_steps
        Action mask: num_actions
        """
        if self.args.run_type != "train":
            return None
        else:
            assert len(self.volume) == len(self.global_obs)

            tet_obs = []
            for i in range(len(self.global_obs)):
                tet_obs.append(self.global_obs[i])

            while len(tet_obs) < self.max_points:
                tet_obs.append(np.zeros_like(self.global_obs[0]))

            assert len(tet_obs) == self.max_points

            bbox_obs = []
            for bbox in self.bbox_list:
                bbox_obs.append(bbox.get_obs_bbox())

            while len(bbox_obs) < self.max_bboxs:
                bbox_obs.append(np.zeros_like(bbox_obs[0]))

            assert len(bbox_obs) == self.max_bboxs

            return (
                np.array(tet_obs),
                np.array(bbox_obs),
                np.array(self.step_vec),
                np.array(self.action_mask),
            )

    def environment_info(self):
        # N, d1, N_B, d2, num_actions
        return (
            *self.current_observation()[0].shape,
            *self.current_observation()[1].shape,
            self.num_actions,
        )

    def bbox_greedy_sample(self) -> int:
        prefiltered = self._candidate_prefilter_greedy_sample(self.step_cnt % self.num_bbox)
        if prefiltered is not None and prefiltered[0] is not None:
            return prefiltered[0]

        bridge = self._bridge_greedy_sample(self.step_cnt % self.num_bbox)
        if bridge is not None and bridge[0] is not None:
            return bridge[0]

        action = None
        max_reward = -sys.float_info.max

        nw_box = self.step_cnt % self.num_bbox

        upper_rewards = self._action_upper_rewards(nw_box)
        for i in range(6 * self.num_action_scale + 1):
            cur_action = nw_box * (6 * self.num_action_scale + 1) + i
            if upper_rewards[i] <= max_reward:
                continue
            reward = self.step(cur_action, apply=0)
            if max_reward < reward:
                action = cur_action
                max_reward = reward

        return action

    def greedy_sample(self, ret_reward=False) -> int:
        prefiltered = self._candidate_prefilter_greedy_sample(-1)
        if prefiltered is not None and prefiltered[0] is not None:
            return prefiltered if ret_reward else prefiltered[0]

        bridge = self._bridge_greedy_sample(-1)
        if bridge is not None and bridge[0] is not None:
            return bridge if ret_reward else bridge[0]

        action = None
        max_reward = -sys.float_info.max

        upper_rewards = self._action_upper_rewards()
        for i in range(self.num_bbox * (6 * self.num_action_scale + 1)):
            if upper_rewards[i] <= max_reward:
                continue
            reward = self.step(i, apply=0)
            if max_reward < reward:
                action = i
                max_reward = reward

        if ret_reward:
            return action, max_reward
        else:
            return action

    def ith_bbox_greedy_sample(self, nw_box) -> int:
        prefiltered = self._candidate_prefilter_greedy_sample(nw_box)
        if prefiltered is not None and prefiltered[0] is not None:
            return prefiltered

        bridge = self._bridge_greedy_sample(nw_box)
        if bridge is not None and bridge[0] is not None:
            return bridge

        action = None
        max_reward = -sys.float_info.max

        upper_rewards = self._action_upper_rewards(nw_box)
        for i in range(6 * self.num_action_scale + 1):
            cur_action = nw_box * (6 * self.num_action_scale + 1) + i
            if upper_rewards[i] <= max_reward:
                continue
            reward = self.step(cur_action, apply=0)
            if max_reward < reward:
                action = cur_action
                max_reward = reward

        return action, max_reward

    def _candidate_prefilter_greedy_sample(self, nw_box=-1):
        if not self._use_candidate_prefilter():
            return None
        try:
            return self._bitset_topk_exact_greedy_sample(nw_box)
        except Exception as exc:
            if not bool(getattr(self.args, "print_off", False)):
                print("candidate prefilter fallback: %s" % exc)
            return None

    def _use_candidate_prefilter(self):
        return (
            self.candidate_backend == "bitset_topk"
            and self.candidate_top_k > 0
            and not bool(self.args.tov)
            and smart_rust is not None
            and smart_rust.using_rust()
            and hasattr(smart_rust, "centroid_proxy_axis_rewards")
        )

    def _bitset_topk_exact_greedy_sample(self, nw_box=-1):
        self.candidate_prefilter_stats["calls"] += 1
        actions, upper_rewards = self._candidate_action_order_and_upper(nw_box)
        if not actions:
            self.candidate_prefilter_stats["no_action"] += 1
            return None

        proxy_actions = self._proxy_topk_axis_actions(nw_box, actions)
        if not proxy_actions:
            self.candidate_prefilter_stats["no_action"] += 1
            return None
        self.candidate_prefilter_stats["actions_total"] += len(actions)
        self.candidate_prefilter_stats["proxy_candidates"] += len(proxy_actions)
        proxy_action_ids = {int(action) for action, _ in proxy_actions}

        box_bounds, box_rotations = self._bridge_current_bounds_rotations()
        exact_rewards = {}
        prefilter_floor = -sys.float_info.max
        for action, _ in proxy_actions:
            if upper_rewards[action] < prefilter_floor:
                continue
            reward = self._exact_candidate_reward(action, box_bounds, box_rotations)
            exact_rewards[action] = reward
            self.candidate_prefilter_stats["proxy_exact"] += 1
            if prefilter_floor < reward:
                prefilter_floor = reward

        action = None
        max_reward = -sys.float_info.max
        for candidate_action in actions:
            upper_reward = upper_rewards[candidate_action]
            if upper_reward < max(max_reward, prefilter_floor):
                self.candidate_prefilter_stats["upper_pruned"] += 1
                continue
            reward = exact_rewards.get(candidate_action)
            if reward is None:
                reward = self._exact_candidate_reward(
                    candidate_action, box_bounds, box_rotations
                )
                exact_rewards[candidate_action] = reward
                self.candidate_prefilter_stats["fallback_exact"] += 1
            if max_reward < reward:
                action = candidate_action
                max_reward = reward
        if action is None:
            self.candidate_prefilter_stats["no_action"] += 1
        elif int(action) in proxy_action_ids:
            self.candidate_prefilter_stats["selected_from_proxy"] += 1
        else:
            self.candidate_prefilter_stats["selected_from_fallback"] += 1
        return action, max_reward

    def _proxy_topk_axis_actions(self, nw_box, actions):
        bitset_state = self._ensure_candidate_bitset_state()
        box_bounds, box_rotations = self._bridge_current_bounds_rotations()
        if bitset_state is not None and hasattr(bitset_state, "topk_axis_actions"):
            return [
                (int(action), float(proxy_reward))
                for action, proxy_reward in bitset_state.topk_axis_actions(
                    box_bounds,
                    box_rotations,
                    int(self.num_action_scale),
                    float(self.action_unit),
                    float(self.last_bbox_score),
                    float(self.args.cover_penalty),
                    float(self.pen_rate),
                    int(nw_box),
                    int(self.candidate_top_k),
                )
            ]

        selected = set(actions)
        proxy_actions = [
            (int(action), float(proxy_reward))
            for action, proxy_reward in self._proxy_axis_action_rewards()
            if int(action) in selected
        ]
        proxy_actions.sort(key=lambda row: row[1], reverse=True)
        return proxy_actions[: self.candidate_top_k]

    def _candidate_action_order_and_upper(self, nw_box=-1):
        if int(nw_box) >= 0:
            bbox_idx = int(nw_box)
            local_upper = self._action_upper_rewards(bbox_idx)
            start = bbox_idx * self._actions_per_bbox
            actions = [start + local for local in range(self._actions_per_bbox)]
            upper_rewards = {
                start + local: float(local_upper[local])
                for local in range(self._actions_per_bbox)
            }
            return actions, upper_rewards

        upper = self._action_upper_rewards()
        actions = list(range(self.num_bbox * self._actions_per_bbox))
        upper_rewards = {action: float(upper[action]) for action in actions}
        return actions, upper_rewards

    def _proxy_axis_action_rewards(self):
        box_bounds, box_rotations = self._bridge_current_bounds_rotations()
        bitset_state = self._ensure_candidate_bitset_state()
        if bitset_state is not None:
            return bitset_state.axis_rewards(
                box_bounds,
                box_rotations,
                int(self.num_action_scale),
                float(self.action_unit),
                float(self.last_bbox_score),
                float(self.args.cover_penalty),
                float(self.pen_rate),
            )
        return smart_rust.centroid_proxy_axis_rewards(
            self.centroid.tolist(),
            np.asarray(self.volume, dtype=float).tolist(),
            box_bounds,
            box_rotations,
            int(self.num_action_scale),
            float(self.action_unit),
            float(self.volume_sum),
            float(self.last_bbox_score),
            float(self.args.cover_penalty),
            float(self.pen_rate),
        )

    def _ensure_candidate_bitset_state(self):
        if (
            self._candidate_bitset_state is None
            and smart_rust is not None
            and hasattr(smart_rust, "CandidateBitsetState")
            and smart_rust.CandidateBitsetState is not None
        ):
            self._candidate_bitset_state = smart_rust.CandidateBitsetState(
                self.centroid.tolist(),
                np.asarray(self.volume, dtype=float).tolist(),
                float(self.volume_sum),
            )
        return self._candidate_bitset_state

    def _exact_candidate_reward(self, action, box_bounds, box_rotations):
        action = int(action)
        bbox_idx, coord_idx, scale_idx = self._decode_action(action)
        if self._use_manifold_stateful_reward() and coord_idx < 6:
            state = self._ensure_manifold_stateful_state()
            return float(
                state.score_axis_action_reward(
                    int(action),
                    float(self.args.cover_penalty),
                    float(self.pen_rate),
                )
            )

        if self._use_manifold_bridge_reward():
            if coord_idx < 6:
                candidate_bounds = [list(row) for row in box_bounds]
                candidate_rotations = [list(row) for row in box_rotations]
                candidate_bounds[bbox_idx][coord_idx] += (
                    self.action_scale[scale_idx] * self.action_unit
                )
                score = self._bridge_score_for_bounds(
                    candidate_bounds, candidate_rotations
                )
                reward = score - self.last_bbox_score
                self._bridge_cache_apply_candidate(
                    action,
                    candidate_bounds,
                    candidate_rotations,
                    score,
                    reward,
                    {bbox_idx},
                )
                return float(reward)

            (
                recenter_bounds,
                recenter_rotations,
                recenter_score,
                reward,
            ) = self._bridge_recenter_candidate_score(
                bbox_idx, box_bounds, box_rotations
            )
            self._bridge_cache_apply_candidate(
                action,
                recenter_bounds,
                recenter_rotations,
                recenter_score,
                reward,
                {bbox_idx},
            )
            return float(reward)

        return float(self.step(action, apply=0))

    def _bridge_greedy_sample(self, nw_box=-1):
        if not self._use_manifold_bridge_reward() or bool(self.args.tov):
            return None

        if self._use_manifold_stateful_reward():
            mask_bbox = [True] * self.num_bbox
            if int(nw_box) >= 0:
                mask_bbox = [False] * self.num_bbox
                mask_bbox[int(nw_box)] = True
            actions, rewards = self._bridge_greedy_samples_for_mask(mask_bbox)
            action = None
            max_reward = -sys.float_info.max
            for enabled, candidate_action, reward in zip(mask_bbox, actions, rewards):
                if enabled and candidate_action >= 0 and max_reward < reward:
                    action = candidate_action
                    max_reward = reward
            return action, max_reward

        box_bounds, box_rotations = self._bridge_current_bounds_rotations()

        max_reward = -sys.float_info.max
        action = None
        bbox_indices = range(self.num_bbox) if int(nw_box) < 0 else [int(nw_box)]
        current_bvs_reward = -abs(
            sum(_box_volume(bbox.box) for bbox in self.bbox_list if bbox.valid_bbox())
            / self.volume_sum
            - 1
        ) - self.last_bbox_score

        for bbox_idx in bbox_indices:
            bridge_action, bridge_reward = self._manifold_bridge_mesh.best_axis_action(
                box_bounds,
                box_rotations,
                int(bbox_idx),
                int(self.num_action_scale),
                float(self.action_unit),
                float(self.volume_sum),
                float(self.last_bbox_score),
                float(self.args.cover_penalty),
                float(self.pen_rate),
                float(max_reward),
                self.manifold_volume_method,
            )
            if bridge_action >= 0 and max_reward < bridge_reward:
                action = int(bridge_action)
                max_reward = float(bridge_reward)

            recenter_action = bbox_idx * self._actions_per_bbox + (self._actions_per_bbox - 1)
            if current_bvs_reward <= max_reward:
                continue
            (
                recenter_bounds,
                recenter_rotations,
                recenter_score,
                reward,
            ) = self._bridge_recenter_candidate_score(
                bbox_idx, box_bounds, box_rotations
            )
            if max_reward < reward:
                action = recenter_action
                max_reward = reward

        return action, max_reward

    def _bridge_greedy_samples_for_mask(self, mask_bbox):
        if not self._use_manifold_bridge_reward() or bool(self.args.tov):
            return None
        if len(mask_bbox) != self.num_bbox:
            raise RuntimeError("mask_bbox length does not match bbox count")

        self._bridge_apply_cache = {}
        box_bounds = None
        box_rotations = None

        if self._use_manifold_stateful_reward():
            state = self._ensure_manifold_stateful_state()
            actions, rewards = state.score_action_batch(
                [bool(value) for value in mask_bbox],
                float(self.args.cover_penalty),
                float(self.pen_rate),
                -sys.float_info.max,
            )
        else:
            box_bounds, box_rotations = self._bridge_current_bounds_rotations()
            actions, rewards = self._manifold_bridge_mesh.best_axis_actions_for_mask(
                box_bounds,
                box_rotations,
                [bool(value) for value in mask_bbox],
                int(self.num_action_scale),
                float(self.action_unit),
                float(self.volume_sum),
                float(self.last_bbox_score),
                float(self.args.cover_penalty),
                float(self.pen_rate),
                -sys.float_info.max,
                self.manifold_volume_method,
            )
        actions = [int(action) for action in actions]
        rewards = [float(reward) for reward in rewards]

        if self._use_manifold_stateful_reward():
            current_bvs_reward = -abs(self._manifold_stateful.bvs() - 1) - self.last_bbox_score
        else:
            current_bvs_reward = -abs(
                sum(_box_volume(bbox.box) for bbox in self.bbox_list if bbox.valid_bbox())
                / self.volume_sum
                - 1
            ) - self.last_bbox_score

        for bbox_idx, enabled in enumerate(mask_bbox):
            if not enabled:
                continue
            recenter_action = bbox_idx * self._actions_per_bbox + (self._actions_per_bbox - 1)
            if current_bvs_reward <= rewards[bbox_idx]:
                continue
            if box_bounds is None or box_rotations is None:
                box_bounds, box_rotations = self._bridge_current_bounds_rotations()
            (
                recenter_bounds,
                recenter_rotations,
                recenter_score,
                reward,
            ) = self._bridge_recenter_candidate_score(
                bbox_idx, box_bounds, box_rotations
            )
            if rewards[bbox_idx] < reward:
                actions[bbox_idx] = recenter_action
                rewards[bbox_idx] = reward
                self._bridge_cache_apply_candidate(
                    recenter_action,
                    recenter_bounds,
                    recenter_rotations,
                    recenter_score,
                    reward,
                    {bbox_idx},
                )

        return actions, rewards

    def _bridge_mcts_greedy_rollout_step(self, mask_bbox):
        if not self._use_manifold_bridge_reward() or bool(self.args.tov):
            return None
        if len(mask_bbox) != self.num_bbox:
            raise RuntimeError("mask_bbox length does not match bbox count")

        actions_rewards = self._bridge_greedy_samples_for_mask(mask_bbox)
        if actions_rewards is None:
            return None
        actions, rewards = actions_rewards

        next_mask = [bool(value) for value in mask_bbox]
        mx_action = None
        mx_reward = -sys.float_info.max
        for idx, enabled in enumerate(mask_bbox):
            if not enabled:
                continue
            reward = float(rewards[idx])
            action = int(actions[idx]) if int(actions[idx]) >= 0 else None
            if mx_reward < reward:
                mx_reward = reward
                mx_action = action
            if reward < 0.0:
                next_mask[idx] = False

        if mx_reward <= 0.0 or mx_action is None:
            return None, mx_reward, 0.0, self.done, next_mask

        applied = self._bridge_apply_cached_action(mx_action)
        if applied is None:
            applied = self._bridge_apply_scored_action(mx_action, mx_reward)
        if applied is None:
            return None

        applied_reward, done = applied
        applied_reward = float(applied_reward)
        if not math.isfinite(applied_reward) or applied_reward <= 0.0:
            return int(mx_action), mx_reward, applied_reward, done, next_mask

        reward_to_store = (
            mx_reward
            if math.isclose(applied_reward, mx_reward, rel_tol=1e-9, abs_tol=1e-12)
            else applied_reward
        )
        return int(mx_action), mx_reward, reward_to_store, done, next_mask

    def _bridge_current_bounds_rotations(self):
        box_bounds = []
        box_rotations = []
        for bbox in self.bbox_list:
            box_bounds.append(list(bbox.box))
            if bool(self.args.tilted):
                box_rotations.append(bbox.rot.reshape(-1).tolist())
            else:
                box_rotations.append(np.eye(3).reshape(-1).tolist())
        return box_bounds, box_rotations

    def _bridge_recenter_bbox_params(self, bbox_idx):
        bbox = self.bbox_list[bbox_idx]
        box = np.asarray(bbox.box, dtype=float)
        rot = np.asarray(bbox.rot, dtype=float)

        if not bool(self.args.tilted):
            return box.tolist(), np.eye(3, dtype=float).reshape(-1).tolist()

        cache_key = None
        if self.score_cache_size:
            cache_key = (
                int(bbox_idx),
                tuple(float(value).hex() for value in box),
                tuple(float(value).hex() for value in rot.reshape(-1)),
                bool(self.args.tilted),
            )
            cached = self._cache_get(self.recenter_params_cache, cache_key)
            if cached is not None:
                cached_box, cached_rotation = cached
                return list(cached_box), list(cached_rotation)

        pts = self.centroid
        rot_pts = np.matmul(pts, np.transpose(rot))

        min_x, min_y, min_z = box[:3]
        max_x, max_y, max_z = box[3:]

        mask_x = (rot_pts[:, 0] >= min_x) & (rot_pts[:, 0] <= max_x)
        mask_y = (rot_pts[:, 1] >= min_y) & (rot_pts[:, 1] <= max_y)
        mask_z = (rot_pts[:, 2] >= min_z) & (rot_pts[:, 2] <= max_z)
        mask = mask_x & mask_y & mask_z

        masked_voxels = self.tetmsh.voxels[mask]
        if len(masked_voxels) == 0:
            return list(map(float, box)), np.asarray(rot, dtype=float).reshape(-1).tolist()
        nw_pts = self.tetmsh.vertices[np.asarray(masked_voxels, dtype=int).reshape(-1)]
        try:
            to_origin, _ = trimesh.bounds.oriented_bounds(nw_pts, angle_digits=3)
            rot_mat = to_origin[:3, :3]

            rot_pts = np.matmul(nw_pts, np.transpose(rot_mat))

            mn = np.min(rot_pts, axis=0)
            mx = np.max(rot_pts, axis=0)

            cen = (mn + mx) / 2
            cur_cen = (box[:3] + box[3:]) / 2

            new_box = list(box[:3] + cen - cur_cen) + list(box[3:] + cen - cur_cen)
        except:
            rot_mat = rot
            new_box = list(box)
        result_box = list(map(float, new_box))
        result_rotation = np.asarray(rot_mat, dtype=float).reshape(-1).tolist()
        if cache_key is not None:
            self._cache_set(
                self.recenter_params_cache,
                cache_key,
                (list(result_box), list(result_rotation)),
            )
        return result_box, result_rotation

    def _bridge_recenter_candidate_score(self, bbox_idx, box_bounds, box_rotations):
        cache_key = None
        if self.score_cache_size:
            cache_key = (
                self._state_cache_key(),
                int(bbox_idx),
                float(self.volume_sum).hex(),
                float(self.pen_rate).hex(),
                float(self.args.cover_penalty).hex(),
                bool(self.args.tilted),
            )
            cached = self._cache_get(self.recenter_candidate_cache, cache_key)
            if cached is not None:
                recenter_bounds, recenter_rotations, score = cached
                bounds = [list(row) for row in recenter_bounds]
                rotations = [list(row) for row in recenter_rotations]
                return bounds, rotations, float(score), float(score) - self.last_bbox_score

        recenter_box, recenter_rotation = self._bridge_recenter_bbox_params(bbox_idx)
        candidate_bounds = [list(row) for row in box_bounds]
        candidate_rotations = [list(row) for row in box_rotations]
        candidate_bounds[bbox_idx] = recenter_box
        candidate_rotations[bbox_idx] = recenter_rotation
        if (
            recenter_box == list(box_bounds[bbox_idx])
            and recenter_rotation == list(box_rotations[bbox_idx])
        ):
            score = float(self.last_bbox_score)
        elif self._use_manifold_stateful_reward():
            state = self._ensure_manifold_stateful_state()
            score = float(
                state.score_replacement(
                    int(bbox_idx),
                    recenter_box,
                    recenter_rotation,
                    float(self.args.cover_penalty),
                    float(self.pen_rate),
                )
            )
        else:
            score = self._bridge_score_for_bounds(candidate_bounds, candidate_rotations)
        if cache_key is not None:
            self._cache_set(
                self.recenter_candidate_cache,
                cache_key,
                (
                    [list(row) for row in candidate_bounds],
                    [list(row) for row in candidate_rotations],
                    float(score),
                ),
            )
        return candidate_bounds, candidate_rotations, score, score - self.last_bbox_score

    def _bridge_score_for_bounds(self, box_bounds, box_rotations):
        bvs = (
            sum(_box_volume(box) for box in box_bounds if _box_valid(box))
            / self.volume_sum
        )
        covered = self._manifold_bridge_mesh.covered_for_bounds(
            box_bounds,
            box_rotations,
            float(self.volume_sum),
            self.manifold_volume_method,
        )
        return -abs(bvs - 1) - (1 - covered) * self.pen_rate * self.args.cover_penalty

    def _bridge_cache_apply_candidate(
        self, action, bounds, rotations, score, reward, touched_indices
    ):
        self._bridge_apply_cache[int(action)] = (
            self._state_cache_key(),
            bounds,
            rotations,
            float(score),
            float(reward),
            set(touched_indices),
        )

    def _bridge_cache_axis_candidate(self, action, box_bounds, box_rotations, reward):
        action = int(action)
        bbox_idx, coord_idx, scale_idx = self._decode_action(action)
        if coord_idx >= 6:
            return
        candidate_bounds = [list(row) for row in box_bounds]
        candidate_rotations = [list(row) for row in box_rotations]
        candidate_bounds[bbox_idx][coord_idx] += (
            self.action_scale[scale_idx] * self.action_unit
        )
        self._bridge_cache_apply_candidate(
            action,
            candidate_bounds,
            candidate_rotations,
            self.last_bbox_score + float(reward),
            float(reward),
            {bbox_idx},
        )

    def _bridge_apply_scored_action(self, action, reward):
        if not self._use_manifold_bridge_reward() or bool(self.args.tov):
            return None
        action = int(action)
        reward = float(reward)
        bbox_idx, coord_idx, scale_idx = self._decode_action(action)

        if self._use_manifold_stateful_reward() and coord_idx < 6:
            state = self._ensure_manifold_stateful_state()
            (
                reward,
                updated_bbox_idx,
                updated_bounds,
                updated_rotation,
                next_score,
            ) = state.apply_axis_action_delta(
                int(action),
                float(self.args.cover_penalty),
                float(self.pen_rate),
            )
            reward = float(reward)
            self._bridge_sync_axis_delta(
                updated_bbox_idx,
                updated_bounds,
                updated_rotation,
                next_score,
                1,
                state_already_current=True,
            )
            self._trace_action("manifold_stateful", action, reward)
            return reward, self.done

        bounds, rotations = self._bridge_current_bounds_rotations()

        if coord_idx < 6:
            bounds[bbox_idx][coord_idx] += self.action_scale[scale_idx] * self.action_unit
        else:
            bounds[bbox_idx], rotations[bbox_idx] = self._bridge_recenter_bbox_params(bbox_idx)

        self._bridge_sync_axis_state(
            bounds,
            rotations,
            self.last_bbox_score + reward,
            1,
            touched_indices={bbox_idx},
        )
        self._trace_action("manifold_bridge", action, reward)
        return reward, self.done

    def _bridge_apply_cached_action(self, action):
        if not self._use_manifold_bridge_reward() or bool(self.args.tov):
            return None
        cached = self._bridge_apply_cache.get(int(action))
        if cached is None:
            return None
        state_key, bounds, rotations, score, reward, touched_indices = cached
        if state_key != self._state_cache_key():
            self._bridge_apply_cache = {}
            return None
        self._bridge_sync_axis_state(
            bounds,
            rotations,
            score,
            1,
            touched_indices=touched_indices,
        )
        return reward, self.done

    def _bridge_apply_unscored_action(self, action):
        if not self._use_manifold_stateful_reward() or bool(self.args.tov):
            return None
        action = int(action)
        bbox_idx, coord_idx, _ = self._decode_action(action)

        if coord_idx < 6:
            state = self._ensure_manifold_stateful_state()
            (
                reward,
                updated_bbox_idx,
                updated_bounds,
                updated_rotation,
                next_score,
            ) = state.apply_axis_action_delta(
                int(action),
                float(self.args.cover_penalty),
                float(self.pen_rate),
            )
            reward = float(reward)
            self._bridge_sync_axis_delta(
                updated_bbox_idx,
                updated_bounds,
                updated_rotation,
                next_score,
                1,
                state_already_current=True,
            )
            self._trace_action("manifold_stateful_unscored_axis", action, reward)
            return reward, self.done

        box_bounds, box_rotations = self._bridge_current_bounds_rotations()
        recenter_bounds, recenter_rotations, recenter_score, reward = (
            self._bridge_recenter_candidate_score(bbox_idx, box_bounds, box_rotations)
        )
        self._bridge_sync_axis_state(
            recenter_bounds,
            recenter_rotations,
            recenter_score,
            1,
            touched_indices={bbox_idx},
        )
        self._trace_action("manifold_stateful_unscored_recenter", action, reward)
        return reward, self.done

    def _bridge_axis_refine_segment(self, max_steps):
        if not self._use_manifold_bridge_reward() or bool(self.args.tov):
            return None
        max_steps = int(max_steps)
        if max_steps <= 0:
            return [], [], self.done

        if self._use_manifold_stateful_reward():
            state = self._ensure_manifold_stateful_state()
            (
                next_bounds,
                next_rotations,
                rewards,
                actions,
                next_score,
            ) = state.greedy_axis_refine_segment(
                float(self.args.cover_penalty),
                float(self.pen_rate),
                max_steps,
            )
            if rewards:
                touched = {
                    int(action) // self._actions_per_bbox
                    for action in actions
                    if int(action) >= 0
                }
                self._bridge_sync_axis_state(
                    next_bounds,
                    next_rotations,
                    next_score,
                    len(rewards),
                    touched_indices=touched,
                    state_already_current=True,
                )
                for action, reward in zip(actions, rewards):
                    self._trace_action("manifold_stateful_segment", int(action), float(reward))
            return list(rewards), list(actions), self.done

        box_bounds = []
        box_rotations = []
        for bbox in self.bbox_list:
            box_bounds.append(list(bbox.box))
            if bool(self.args.tilted):
                box_rotations.append(bbox.rot.reshape(-1).tolist())
            else:
                box_rotations.append(np.eye(3).reshape(-1).tolist())

        (
            next_bounds,
            next_rotations,
            rewards,
            actions,
            next_score,
        ) = self._manifold_bridge_mesh.greedy_axis_refine_segment(
            box_bounds,
            box_rotations,
            int(self.num_action_scale),
            float(self.action_unit),
            float(self.volume_sum),
            float(self.last_bbox_score),
            float(self.args.cover_penalty),
            float(self.pen_rate),
            max_steps,
        )

        if rewards:
            touched = {
                int(action) // self._actions_per_bbox
                for action in actions
                if int(action) >= 0
            }
            self._bridge_sync_axis_state(
                next_bounds,
                next_rotations,
                next_score,
                len(rewards),
                touched_indices=touched,
            )
        return list(rewards), list(actions), self.done

    def _bridge_sync_axis_state(
        self,
        bounds,
        rotations,
        last_bbox_score,
        step_count,
        touched_indices=None,
        state_already_current=False,
    ):
        if len(bounds) != self.num_bbox or len(rotations) != self.num_bbox:
            raise RuntimeError("bridge state length does not match bbox count")

        self.step_cache = None
        self._cached_state_key = None
        self._rust_bbox_state = None
        self._bridge_apply_cache = {}
        # Keep exact memoized scores across state transitions. These caches are
        # keyed by bbox state and reward parameters, so retaining them lets MCTS
        # reuse repeated state/action evaluations without changing the metric.

        if touched_indices is None:
            touched_indices = range(self.num_bbox)

        # Exact bridge/stateful rewards use explicit bounds/rotations, so Python
        # bbox meshes can be rebuilt lazily only when rendering or legacy metrics
        # need them. This avoids repeated trimesh/pymanifold construction inside
        # refine/MCTS apply loops without changing the reward metric.
        defer_bbox_geometry = self._use_manifold_bridge_reward() and not bool(
            self.args.tov
        )
        for i in touched_indices:
            box = bounds[i]
            rotation = rotations[i]
            self.bbox_list[i].box = list(map(float, box))
            self.bbox_list[i].rot = np.asarray(rotation, dtype=float).reshape(3, 3)
            self.bbox_part_vol[i] = -1
            self.bbox_part_ov[i] = -1
            self.bbox_part_occ[i] = -1
            if self.bbox_list[i].valid_bbox():
                if defer_bbox_geometry:
                    self.bbox_mesh[i] = None
                    self.bbox_man[i] = None
                else:
                    self.bbox_mesh[i], self.bbox_man[i] = self.get_bbox_bmesh_bman(i)
            else:
                self.bbox_mesh[i] = None
                self.bbox_man[i] = None

        self.last_bbox_score = float(last_bbox_score)
        if self._manifold_stateful is not None:
            if state_already_current:
                self._manifold_stateful_key = self._state_cache_key()
                self._manifold_stateful_score = float(self.last_bbox_score)
            else:
                self._manifold_stateful.reset_to_state(
                    bounds,
                    rotations,
                    float(self.last_bbox_score),
                )
                self._manifold_stateful_key = self._state_cache_key()
                self._manifold_stateful_score = float(self.last_bbox_score)
        self.step_cnt += int(step_count)
        self._refresh_step_vec()
        if self.step_cnt >= self.max_step - 1:
            self.done = 1

    def _bridge_sync_axis_delta(
        self,
        bbox_idx,
        bounds,
        rotation,
        last_bbox_score,
        step_count,
        state_already_current=False,
    ):
        self.step_cache = None
        self._cached_state_key = None
        self._rust_bbox_state = None
        self._bridge_apply_cache = {}

        idx = int(bbox_idx)
        self.bbox_list[idx].box = list(map(float, bounds))
        self.bbox_list[idx].rot = np.asarray(rotation, dtype=float).reshape(3, 3)
        self.bbox_part_vol[idx] = -1
        self.bbox_part_ov[idx] = -1
        self.bbox_part_occ[idx] = -1
        if self.bbox_list[idx].valid_bbox():
            if self._use_manifold_bridge_reward() and not bool(self.args.tov):
                self.bbox_mesh[idx] = None
                self.bbox_man[idx] = None
            else:
                self.bbox_mesh[idx], self.bbox_man[idx] = self.get_bbox_bmesh_bman(idx)
        else:
            self.bbox_mesh[idx] = None
            self.bbox_man[idx] = None

        self.last_bbox_score = float(last_bbox_score)
        if self._manifold_stateful is not None:
            if not state_already_current:
                raise RuntimeError("delta sync requires the stateful backend to be current")
            self._manifold_stateful_key = self._state_cache_key()
            self._manifold_stateful_score = float(self.last_bbox_score)
        self.step_cnt += int(step_count)
        self._refresh_step_vec()
        if self.step_cnt >= self.max_step - 1:
            self.done = 1

    def random_sample(self) -> int:
        action = np.random.randint(
            self.num_bbox * self._actions_per_bbox, size=1
        )
        return action[0]

    def _decode_action(self, action):
        action = int(action)
        i = action // self._actions_per_bbox
        local = action % self._actions_per_bbox
        if local == self._actions_per_bbox - 1:
            return i, 6, 0
        return i, local // self.num_action_scale, local % self.num_action_scale

    def _trace_action(self, source, action, reward):
        trace_path = str(getattr(self.args, "trace_actions_path", "") or "")
        if not trace_path:
            return
        bbox_idx, coord_idx, scale_idx = self._decode_action(action)
        parent = os.path.dirname(trace_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        record = {
            "schema_version": 2,
            "category": str(getattr(self.args, "category", "")),
            "mesh": self.name,
            "source": str(source),
            "reward_backend": self.reward_backend,
            "manifold_volume_method": self.manifold_volume_method,
            "step": int(self.step_cnt),
            "action": int(action),
            "bbox_idx": int(bbox_idx),
            "coord_idx": int(coord_idx),
            "scale_idx": int(scale_idx),
            "num_bbox": int(self.num_bbox),
            "num_action_scale": int(self.num_action_scale),
            "actions_per_bbox": int(self._actions_per_bbox),
            "action_unit": float(self.action_unit),
            "reward": float(reward),
            "last_bbox_score": float(self.last_bbox_score),
            "bvs": float(
                sum(_box_volume(bbox.box) for bbox in self.bbox_list if bbox.valid_bbox())
                / self.volume_sum
            ),
            "volume_sum": float(self.volume_sum),
            "cover_penalty": float(self.args.cover_penalty),
            "pen_rate": float(self.pen_rate),
            "candidate_backend": str(self.candidate_backend),
        }
        with open(trace_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def refine_bbox(self, action, revert=False) -> None:
        i, j, k = self._decode_action(action)

        if not revert:
            self.step_cache = (
                self.bbox_mesh[i],
                self.bbox_man[i],
                self.bbox_part_vol[i],
                self.bbox_part_ov[i],
                self.bbox_part_occ[i],
                list(self.bbox_list[i].box),
                np.copy(self.bbox_list[i].rot),
                self._cached_state_key,
                self._rust_bbox_state,
            )
            self._cached_state_key = None
            if j != 6:
                self.bbox_list[i].box[j] += self.action_scale[k] * self.action_unit
                if self._rust_bbox_state is not None:
                    try:
                        self._rust_bbox_state = self._rust_bbox_state.after_axis_action(
                            int(action)
                        )
                    except Exception:
                        self._rust_bbox_state = None
            else:
                self._rust_bbox_state = None
                if self.args.tilted:

                    pts = np.copy(self.centroid)
                    rot_pts = np.matmul(pts, np.transpose(self.bbox_list[i].rot))

                    min_x, min_y, min_z = self.bbox_list[i].box[:3]
                    max_x, max_y, max_z = self.bbox_list[i].box[3:]

                    mask_x = (rot_pts[:, 0] >= min_x) & (rot_pts[:, 0] <= max_x)
                    mask_y = (rot_pts[:, 1] >= min_y) & (rot_pts[:, 1] <= max_y)
                    mask_z = (rot_pts[:, 2] >= min_z) & (rot_pts[:, 2] <= max_z)
                    mask = mask_x & mask_y & mask_z

                    nw_pts = []
                    masked_voxels = self.tetmsh.voxels[mask]
                    for j in range(len(masked_voxels)):
                        for k in range(4):
                            nw_pts.append(self.tetmsh.vertices[masked_voxels[j][k]])
                    try:
                        to_origin, _ = trimesh.bounds.oriented_bounds(
                            nw_pts, angle_digits=3
                        )
                        rot_mat = to_origin[:3, :3]

                        rot_pts = np.matmul(nw_pts, np.transpose(rot_mat))

                        mn = np.min(rot_pts, axis=0)
                        mx = np.max(rot_pts, axis=0)

                        cen = (mn + mx) / 2
                        cur_cen = (
                            np.array(self.bbox_list[i].box[:3])
                            + np.array(self.bbox_list[i].box[3:])
                        ) / 2

                        nw_bbox = np.array(self.bbox_list[i].box)

                        self.bbox_list[i].box = list(nw_bbox[:3] + cen - cur_cen) + list(
                            nw_bbox[3:] + cen - cur_cen
                        )
                    except:
                        rot_mat = self.bbox_list[i].rot
                    self.bbox_list[i].rot = rot_mat
                else:
                    self.bbox_list[i].rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if revert:
            assert (
                self.step_cache is not None
            ), "Revert option called twice or without any steps applied"
            (
                self.bbox_mesh[i],
                self.bbox_man[i],
                self.bbox_part_vol[i],
                self.bbox_part_ov[i],
                self.bbox_part_occ[i],
                self.bbox_list[i].box,
                self.bbox_list[i].rot,
                self._cached_state_key,
                self._rust_bbox_state,
            ) = self.step_cache
            self.step_cache = None
        else:
            if not self.bbox_list[i].valid_bbox():
                self.bbox_mesh[i] = None
                self.bbox_man[i] = None
                self.bbox_part_vol[i] = -1
                self.bbox_part_ov[i] = -1
                self.bbox_part_occ[i] = -1
                self._cached_state_key = None
                self._rust_bbox_state = None

                return

            self.bbox_mesh[i], self.bbox_man[i] = self.get_bbox_bmesh_bman(i)

            self.bbox_part_vol[i] = -1
            self.bbox_part_ov[i] = -1
            self.bbox_part_occ[i] = -1
            self._cached_state_key = None

    def step(
        self, action, apply=True, obs=False, prune_below=None
    ) -> Union[
        float, Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]
    ]:

        if self.done:
            assert 0, "Step called after done"

        reward_cache_key = None
        cached_trial_score = None
        if not apply and not obs and self.score_cache_size:
            reward_cache_key = self._step_reward_cache_key(action)
            cached_reward = self._cache_get(self.step_reward_cache, reward_cache_key)
            if cached_reward is not None:
                return self._cached_step_reward(cached_reward)
        elif apply and self.score_cache_size:
            reward_cache_key = self._step_reward_cache_key(action)
            cached_reward = self._cache_get(self.step_reward_cache, reward_cache_key)
            cached_trial_score = self._cached_step_score(cached_reward)

        self.refine_bbox(action)
        if prune_below is not None and not apply and not obs:
            upper_reward = self.evaluate_bbox_score_upper_bound() - self.last_bbox_score
            if upper_reward <= prune_below:
                self.refine_bbox(action, revert=True)
                return -sys.float_info.max

        if apply and cached_trial_score is not None:
            nw_bbox_score = cached_trial_score
        else:
            nw_bbox_score = self.evaluate_bbox_score()

        reward = nw_bbox_score - self.last_bbox_score

        if not apply:
            if obs:
                self.step_cnt += 1
                self._refresh_step_vec()
                done = 0
                if self.step_cnt == self.max_step - 1:
                    done = 1

                observation = self.current_observation()

                self.refine_bbox(action, revert=True)

                self.step_cnt -= 1
                self._refresh_step_vec()

                return reward, observation, done

            self.refine_bbox(action, revert=True)

            if reward_cache_key is not None:
                self._cache_set(
                    self.step_reward_cache,
                    reward_cache_key,
                    (reward, nw_bbox_score),
                )
            return reward

        self.step_cache = None
        self._bridge_apply_cache = {}
        self.last_bbox_score = nw_bbox_score
        if self._rust_bbox_state is not None:
            try:
                self._rust_bbox_state.set_last_bbox_score(nw_bbox_score)
            except Exception:
                self._rust_bbox_state = None
        if self._use_manifold_stateful_reward() and self._manifold_stateful is not None:
            bounds, rotations = self._bridge_current_bounds_rotations()
            self._manifold_stateful.reset_to_state(bounds, rotations, float(self.last_bbox_score))
            self._manifold_stateful_key = self._state_cache_key()
            self._manifold_stateful_score = float(self.last_bbox_score)
        self._trace_action("legacy_step", int(action), float(reward))

        self.step_cnt += 1
        self._refresh_step_vec()
        if self.step_cnt == self.max_step - 1:
            self.done = 1

        observation = self.current_observation()
        return reward, observation, self.done

    def postprocess_bbox(self) -> None:
        raise NotImplementedError

    def evaluate_bbox_score_upper_bound(self) -> float:
        bvs = self.BVS()
        return -abs(bvs - 1)

    def _action_upper_rewards(self, bbox_idx=None):
        rust_state = self._get_rust_bbox_state()
        if rust_state is not None:
            try:
                if bbox_idx is not None:
                    return rust_state.bbox_action_upper_rewards(int(bbox_idx))
                return rust_state.action_upper_rewards()
            except Exception:
                self._rust_bbox_state = None

        boxes = [bbox.box for bbox in self.bbox_list[: self.num_bbox]]
        if self._use_rust_action_helpers():
            if bbox_idx is not None:
                return smart_rust.bbox_action_upper_rewards(
                    boxes,
                    int(bbox_idx),
                    self.num_action_scale,
                    self.action_unit,
                    self.volume_sum,
                    self.last_bbox_score,
                )
            return smart_rust.action_upper_rewards(
                boxes,
                self.num_action_scale,
                self.action_unit,
                self.volume_sum,
                self.last_bbox_score,
            )

        old_volumes = []
        total_volume = 0.0
        for box in boxes:
            volume = _box_volume(box) if _box_valid(box) else 0.0
            old_volumes.append(volume)
            total_volume += volume

        out = []
        indexed_boxes = [(int(bbox_idx), boxes[int(bbox_idx)])] if bbox_idx is not None else enumerate(boxes)
        for bbox_idx, box in indexed_boxes:
            for coord_idx in range(6):
                for scale in self.action_scale:
                    candidate = list(box)
                    candidate[coord_idx] += scale * self.action_unit
                    new_volume = _box_volume(candidate) if _box_valid(candidate) else 0.0
                    bvs = (total_volume - old_volumes[bbox_idx] + new_volume) / self.volume_sum
                    out.append(-abs(bvs - 1) - self.last_bbox_score)
            bvs = total_volume / self.volume_sum
            out.append(-abs(bvs - 1) - self.last_bbox_score)
        return out

    def evaluate_bbox_score(self) -> float:
        assert self.num_bbox == len(self.bbox_list)

        cache_key = None
        if self.score_cache_size:
            cache_key = (
                self._state_cache_key(),
                float(self.pen_rate).hex(),
                float(self.args.cover_penalty).hex(),
                bool(self.args.tov),
            )
            cached_score = self._cache_get(self.score_cache, cache_key)
            if cached_score is not None:
                return cached_score

        score = 0.0

        bvs = self.BVS()

        shp_ratio = 1
        score -= shp_ratio * abs(bvs - 1)

        tov, covered = self.Covered(tov=self.args.tov)

        score -= (1 - covered) * self.pen_rate * self.args.cover_penalty

        # iou = self.IoU() * 3
        # score += iou

        # if self.args.mov:
        #     mov = self.MOV()
        #     score += self.args.mov_alpha / mov

        # if self.args.tov and abs(covered - 1.0) < 1e-10:
        #     score += self.args.tov_beta / tov

        if cache_key is not None:
            self._cache_set(self.score_cache, cache_key, score)
        return score

    def OCC(self) -> float:
        ret = 0.0
        for i in range(self.num_bbox):
            if self.bbox_list[i].valid_bbox():
                self._ensure_bbox_manifold(i)
                if self.bbox_man[i] is None:
                    assert 0, "Valid bounding box manifold is not initialized"

                if self.bbox_part_vol[i] == -1:
                    occ_man = self.bbox_man[i] ^ self.manmsh
                    self.bbox_part_vol[i] = _manifold_mesh_volume(occ_man)

                bbox_volume = _box_volume(self.bbox_list[i].box)
                ret += (self.bbox_part_vol[i] / self.volume_sum) * (
                    self.bbox_part_vol[i] / bbox_volume
                )

        return ret

    def MOV(self) -> float:
        ret = 0.0
        for i in range(self.num_bbox):
            if self.bbox_list[i].valid_bbox():
                self._ensure_bbox_manifold(i)
                if self.bbox_man[i] is None:
                    assert 0, "Bounding box manifold is not initialized"

                if self.bbox_part_ov[i] == -1:
                    part_man = self.bbox_man[i] - self.manmsh
                    part_volume = _manifold_mesh_volume(part_man)

                    if self.bbox_part_vol[i] == -1:
                        occ_man = self.bbox_man[i] ^ self.manmsh
                        self.bbox_part_vol[i] = _manifold_mesh_volume(occ_man)

                    if self.bbox_part_vol[i] < 1e-30:
                        self.bbox_part_ov[i] = 0
                    else:
                        self.bbox_part_ov[i] = part_volume / self.bbox_part_vol[i]

                ret = max(
                    ret,
                    self.bbox_part_ov[i],
                )

        return ret

    def Covered(self, tov=False) -> Tuple[float, float]:
        cache_key = None
        if self.score_cache_size:
            cache_key = (self._state_cache_key(), bool(tov))
            cached = self._cache_get(self.covered_cache, cache_key)
            if cached is not None:
                return cached

        if self._use_tet_clipping_reward() and not tov:
            metrics = self._tet_clipping_metrics()
            result = (0.0, float(metrics["Covered"]))
            if cache_key is not None:
                self._cache_set(self.covered_cache, cache_key, result)
            return result

        if self._use_manifold_bridge_reward() and not tov:
            result = (0.0, float(self._manifold_bridge_covered()))
            if cache_key is not None:
                self._cache_set(self.covered_cache, cache_key, result)
            return result

        bbxmans = []
        for i in range(self.num_bbox):
            if self.bbox_list[i].valid_bbox():
                self._ensure_bbox_manifold(i)
                if self.bbox_man[i] is None:
                    assert 0, "Bounding box manifold is not initialized"
                bbxmans.append(self.bbox_man[i])

        if not len(bbxmans):
            return 0, 0

        merged_bbox = bbxmans[0]
        for i in range(1, len(bbxmans)):
            merged_bbox = merged_bbox + bbxmans[i]

        cov_man = self.manmsh - merged_bbox
        cov = 1 - (_manifold_mesh_volume(cov_man) / self.volume_sum)

        ret = 0.0
        if tov:
            if cov >= 0.995:
                ret = (_manifold_mesh_volume(merged_bbox) - self.volume_sum) / self.volume_sum
            else:
                ret = _manifold_mesh_volume(merged_bbox - self.manmsh) / self.volume_sum

        result = (ret, cov)
        if cache_key is not None:
            self._cache_set(self.covered_cache, cache_key, result)
        return result

    def BVS(self) -> float:
        if self._use_manifold_stateful_reward():
            try:
                state = self._ensure_manifold_stateful_state()
                return float(state.bvs())
            except Exception:
                pass

        rust_state = self._get_rust_bbox_state()
        if rust_state is not None:
            try:
                return rust_state.bvs()
            except Exception:
                self._rust_bbox_state = None

        valid_boxes = []
        for i in range(self.num_bbox):
            if self.bbox_list[i].valid_bbox():
                valid_boxes.append(self.bbox_list[i].box)

        if self._use_rust_action_helpers():
            return smart_rust.total_bbox_volume(valid_boxes) / self.volume_sum

        ret = 0.0
        for i in range(self.num_bbox):
            if self.bbox_list[i].valid_bbox():
                ret += _box_volume(self.bbox_list[i].box)
        ret = ret / self.volume_sum
        return ret

    def IoU(self) -> float:
        bbxmans = []
        for i in range(self.num_bbox):
            if self.bbox_list[i].valid_bbox():
                self._ensure_bbox_manifold(i)
                if self.bbox_man[i] is None:
                    assert 0, "Bounding box manifold is not initialized"
                bbxmans.append(self.bbox_man[i])

        if not len(bbxmans):
            return 0.0

        merged_bbox = bbxmans[0]
        for i in range(1, len(bbxmans)):
            merged_bbox = merged_bbox + bbxmans[i]

        union_volume = _manifold_mesh_volume(merged_bbox + self.manmsh)
        if union_volume <= 0:
            return 0.0
        return _manifold_mesh_volume(merged_bbox ^ self.manmsh) / union_volume

    def num_valid_bboxs(self) -> None:
        rust_state = self._get_rust_bbox_state()
        if rust_state is not None:
            try:
                return rust_state.valid_count()
            except Exception:
                self._rust_bbox_state = None

        cnt = 0
        for i in range(self.num_bbox):
            if self.bbox_list[i].valid_bbox():
                cnt += 1

        return cnt

    def current_state_summary(self) -> tuple[float, float, float, float, int]:
        occ = 0.0
        if self.args.run_type != "mcts":
            occ = self.OCC()
        mov = self.MOV()
        bvs = self.BVS()
        iou = self.IoU()

        tov, covered = self.Covered(tov=True)

        return occ, mov, bvs, tov, covered, iou

    def get_bbox_bmesh_bman(self, idx, build_mesh=False):
        assert self.bbox_list[idx].valid_bbox(), "Get bbox called on invalid bbox"
        cache_key = None
        if self.score_cache_size:
            cache_key = (bool(build_mesh), self._bbox_geometry_cache_key(idx))
            cached = self._cache_get(self.bbox_geometry_cache, cache_key)
            if cached is not None:
                return cached

        bbox = self.bbox_list[idx]
        if self.args.tilted:
            vertices, faces = _bbox_vertices_faces(bbox.box, bbox.rot, oriented=True)
            bbx_mesh = _trimesh_from_bbox_vertices(vertices, faces) if build_mesh else None
            bbx_man = _manifold_from_vertices_faces(vertices, faces)
        else:
            lengths = np.asarray(bbox.box[3:], dtype=float) - np.asarray(
                bbox.box[:3], dtype=float
            )
            bbx_man = pymanifold.Manifold.cube(
                lengths[0], lengths[1], lengths[2]
            ).translate(bbox.box[0], bbox.box[1], bbox.box[2])
            if build_mesh:
                vertices, faces = _bbox_vertices_faces(
                    bbox.box, np.eye(3), oriented=False
                )
                bbx_mesh = _trimesh_from_bbox_vertices(vertices, faces)
            else:
                bbx_mesh = None

        if cache_key is not None:
            self._cache_set(self.bbox_geometry_cache, cache_key, (bbx_mesh, bbx_man))
        return bbx_mesh, bbx_man

    def _ensure_bbox_mesh(self, idx):
        if self.bbox_mesh[idx] is not None:
            return self.bbox_mesh[idx]
        if not self.bbox_list[idx].valid_bbox():
            return None
        self.bbox_mesh[idx], self.bbox_man[idx] = self.get_bbox_bmesh_bman(
            idx, build_mesh=True
        )
        return self.bbox_mesh[idx]

    def _ensure_bbox_manifold(self, idx):
        if self.bbox_man[idx] is not None:
            return self.bbox_man[idx]
        if not self.bbox_list[idx].valid_bbox():
            return None
        bbox_mesh, bbox_man = self.get_bbox_bmesh_bman(idx)
        if bbox_mesh is not None:
            self.bbox_mesh[idx] = bbox_mesh
        self.bbox_man[idx] = bbox_man
        return self.bbox_man[idx]

    def _bbox_geometry_cache_key(self, idx):
        bbox = self.bbox_list[idx]
        return (
            bool(self.args.tilted),
            tuple(float(value).hex() for value in bbox.box),
            tuple(float(value).hex() for value in bbox.rot.reshape(-1)),
        )

    def _state_cache_key(self):
        if self._cached_state_key is not None:
            return self._cached_state_key

        rust_state = self._get_rust_bbox_state()
        if rust_state is not None:
            try:
                self._cached_state_key = smart_rust.bbox_rot_state_key(
                    rust_state.bounds(),
                    [bbox.rot.reshape(-1).tolist() for bbox in self.bbox_list],
                )
                return self._cached_state_key
            except Exception:
                self._rust_bbox_state = None

        state = []
        for bbox in self.bbox_list:
            if not bbox.valid_bbox():
                state.append(("invalid",))
                continue
            state.append(
                (
                    tuple(float(value).hex() for value in bbox.box),
                    tuple(float(value).hex() for value in bbox.rot.reshape(-1)),
                )
            )
        self._cached_state_key = tuple(state)
        return self._cached_state_key

    def _get_rust_bbox_state(self):
        if smart_rust is None:
            return None
        if not self._use_rust_action_helpers():
            return None
        if self._rust_bbox_state is not None:
            return self._rust_bbox_state
        try:
            self._rust_bbox_state = smart_rust.BBoxState(
                [bbox.box for bbox in self.bbox_list[: self.num_bbox]],
                self.num_action_scale,
                self.action_unit,
                self.volume_sum,
                self.last_bbox_score,
            )
        except Exception:
            self._rust_bbox_state = None
        return self._rust_bbox_state

    def _use_rust_action_helpers(self):
        return (
            smart_rust is not None
            and self.num_bbox * self._actions_per_bbox >= _RUST_ACTION_HELPER_MIN
        )

    def _use_tet_clipping_reward(self):
        return (
            self.reward_backend == "tet_clipping"
            and self._tet_clipping_state is not None
            and smart_rust is not None
            and smart_rust.using_rust()
        )

    def _use_manifold_bridge_reward(self):
        return (
            self.reward_backend in {"manifold_bridge", "manifold_stateful"}
            and self._manifold_bridge_mesh is not None
            and smart_rust is not None
            and smart_rust.using_rust()
        )

    def _use_manifold_stateful_reward(self):
        return (
            self.reward_backend == "manifold_stateful"
            and smart_rust is not None
            and smart_rust.using_rust()
            and getattr(smart_rust, "ManifoldState", None) is not None
        )

    def _ensure_manifold_stateful_state(self):
        if not self._use_manifold_stateful_reward():
            return None
        bounds, rotations = self._bridge_current_bounds_rotations()
        state_key = self._state_cache_key()
        if self._manifold_stateful is None:
            self._manifold_stateful = smart_rust.ManifoldState(
                np.asarray(self.trimsh.vertices, dtype=float).tolist(),
                np.asarray(self.trimsh.faces, dtype=int).tolist(),
                bounds,
                rotations,
                int(self.num_action_scale),
                float(self.action_unit),
                float(self.volume_sum),
                float(self.last_bbox_score),
                bool(getattr(self.args, "stateful_union_cache", True)),
                int(getattr(self.args, "stateful_cache_capacity", 65536)),
                self.manifold_volume_method,
            )
            self._manifold_stateful_key = state_key
            self._manifold_stateful_score = float(self.last_bbox_score)
        elif (
            self._manifold_stateful_key != state_key
            or self._manifold_stateful_score != float(self.last_bbox_score)
        ):
            self._manifold_stateful.reset_to_state(
                bounds,
                rotations,
                float(self.last_bbox_score),
            )
            self._manifold_stateful_key = state_key
            self._manifold_stateful_score = float(self.last_bbox_score)
        return self._manifold_stateful

    def _manifold_stateful_cache_stats(self):
        state = self._ensure_manifold_stateful_state()
        if state is None or not hasattr(state, "cache_stats"):
            return {}
        return dict(state.cache_stats())

    def _candidate_prefilter_report(self):
        if self.candidate_backend != "bitset_topk":
            return {}
        return dict(self.candidate_prefilter_stats)

    def _tet_clipping_metrics(self):
        if self._tet_clipping_state is None:
            raise RuntimeError("tet clipping reward backend is not initialized")

        box_bounds = []
        box_rotations = []
        box_volumes = []
        for i in range(self.num_bbox):
            if not self.bbox_list[i].valid_bbox():
                continue
            bbox = self.bbox_list[i]
            box_bounds.append(list(bbox.box))
            if bool(self.args.tilted):
                box_rotations.append(bbox.rot.reshape(-1).tolist())
            else:
                box_rotations.append(np.eye(3).reshape(-1).tolist())
            box_volumes.append(_box_volume(bbox.box))

        if not box_bounds:
            return {
                "BVS": 0.0,
                "MOV": 0.0,
                "Covered": 0.0,
                "TOV": 0.0,
                "vIoU": 0.0,
            }

        try:
            return self._tet_clipping_state.metrics_for_boxes(
                box_bounds,
                box_rotations,
                max_boxes=self.tet_clipping_max_boxes,
            )
        except AttributeError:
            box_vertices = []
            for bounds, rotation in zip(box_bounds, box_rotations):
                vertices, _ = _bbox_vertices_faces(
                    bounds,
                    np.asarray(rotation, dtype=float).reshape(3, 3),
                    oriented=bool(self.args.tilted),
                )
                box_vertices.append(vertices.tolist())
            return self._tet_clipping_state.metrics(
                box_vertices,
                max_boxes=self.tet_clipping_max_boxes,
                box_volumes=box_volumes,
            )

    def _manifold_bridge_covered(self):
        if self._manifold_bridge_mesh is None:
            raise RuntimeError("Manifold bridge reward backend is not initialized")
        if self._use_manifold_stateful_reward():
            state = self._ensure_manifold_stateful_state()
            return float(state.covered())

        box_bounds = []
        box_rotations = []
        for i in range(self.num_bbox):
            if not self.bbox_list[i].valid_bbox():
                continue
            bbox = self.bbox_list[i]
            box_bounds.append(list(bbox.box))
            if bool(self.args.tilted):
                box_rotations.append(bbox.rot.reshape(-1).tolist())
            else:
                box_rotations.append(np.eye(3).reshape(-1).tolist())

        if not box_bounds:
            return 0.0

        try:
            if self.manifold_volume_method == "properties" and hasattr(
                self._manifold_bridge_mesh, "residual_volume_for_box_params_properties"
            ):
                residual_volume = self._manifold_bridge_mesh.residual_volume_for_box_params_properties(
                    box_bounds,
                    box_rotations,
                )
            else:
                residual_volume = self._manifold_bridge_mesh.residual_volume_for_box_params(
                    box_bounds,
                    box_rotations,
                )
            return 1.0 - (residual_volume / self.volume_sum)
        except AttributeError:
            pass

        box_vertices = []
        for bounds, rotation in zip(box_bounds, box_rotations):
            vertices, _ = _bbox_vertices_faces(
                bounds,
                np.asarray(rotation, dtype=float).reshape(3, 3),
                oriented=bool(self.args.tilted),
            )
            box_vertices.append(vertices.tolist())

        if self.manifold_volume_method == "properties" and hasattr(
            self._manifold_bridge_mesh, "residual_volume_for_boxes_properties"
        ):
            residual_volume = self._manifold_bridge_mesh.residual_volume_for_boxes_properties(
                box_vertices
            )
        else:
            residual_volume = self._manifold_bridge_mesh.residual_volume_for_boxes(box_vertices)
        return 1.0 - (residual_volume / self.volume_sum)

    def _refresh_step_vec(self):
        if self.args.run_type != "train":
            self.step_vec = None
            return
        self.step_vec = np.zeros((self.max_step), dtype=int)
        self.step_vec[self.step_cnt] = 1

    def _step_reward_cache_key(self, action):
        return (
            self._state_cache_key(),
            int(action),
            float(self.last_bbox_score).hex(),
            float(self.pen_rate).hex(),
            float(self.args.cover_penalty).hex(),
            bool(self.args.tov),
        )

    def _cache_get(self, cache, key):
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    def _cache_set(self, cache, key, value):
        if self.score_cache_size <= 0:
            return
        cache[key] = value
        while len(cache) > self.score_cache_size:
            cache.popitem(last=False)

    def _cached_step_reward(self, cached):
        if isinstance(cached, tuple):
            return cached[0]
        return cached

    def _cached_step_score(self, cached):
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached[1]
        return None


def _action_scales(num_action_scale):
    if smart_rust is not None:
        return smart_rust.action_scales(num_action_scale)
    half = num_action_scale // 2
    return [-(2**i) for i in range(half - 1, -1, -1)] + [2**i for i in range(half)]


def _as_numpy(array):
    array = array.squeeze()
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        return array.numpy()
    return np.asarray(array)


def _action_indices(max_bboxs, num_action_scale):
    if smart_rust is not None:
        return smart_rust.action_indices(max_bboxs, num_action_scale)

    action_indices = []
    for i in range(max_bboxs):
        for j in range(6):
            for k in range(num_action_scale):
                action_indices.append([i, j, k])
        action_indices.append([i, 6, 0])
    return action_indices


def _box_valid(box):
    return box[0] < box[3] and box[1] < box[4] and box[2] < box[5]


def _box_volume(box):
    return (
        max(0.0, box[3] - box[0])
        * max(0.0, box[4] - box[1])
        * max(0.0, box[5] - box[2])
    )


_BOX_FACES = np.asarray(
    [
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
    ],
    dtype=np.int64,
)


def _bbox_vertices_faces(box, rot, oriented):
    rot = np.asarray(rot, dtype=float)
    mn = np.asarray(box[:3], dtype=float)
    mx = np.asarray(box[3:], dtype=float)
    lengths = mx - mn
    base = np.matmul(mn, rot) if oriented else mn

    vertices = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                vertices.append(
                    base
                    + rot[0] * i * lengths[0]
                    + rot[1] * j * lengths[1]
                    + rot[2] * k * lengths[2]
                )
    vertices = np.asarray(vertices, dtype=float)
    faces = _BOX_FACES
    if oriented and np.linalg.det(rot) < 0:
        faces = faces[:, ::-1]
    return vertices, faces


def _trimesh_from_bbox_vertices(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _manifold_from_vertices_faces(vertices, faces):
    mesh = pymanifold.Mesh(vert_pos=np.asarray(vertices), tri_verts=np.asarray(faces))
    man = pymanifold.Manifold()
    return man.from_mesh(mesh)


def _manifold_mesh_volume(manifold_obj):
    mesh = manifold_obj.to_mesh()
    vertices = np.asarray(mesh.vert_pos, dtype=np.float64)
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return _triangle_mesh_volume(vertices, faces)


def _triangle_mesh_volume(vertices, faces):
    if len(vertices) == 0 or len(faces) == 0:
        return 0.0
    triangles = vertices[faces]
    vectors = triangles[:, 1:, :] - triangles[:, :2, :]
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    f1 = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]
    return float(np.sum(crosses[:, 0] * f1[:, 0]) / 6.0)
