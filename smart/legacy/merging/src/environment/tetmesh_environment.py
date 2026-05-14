import copy
import heapq
import os
import random
import time
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pymanifold
import pymesh
import trimesh
import trimesh.repair

from ..utils.bounding_box import axis_bbox, tilted_bbox
from ..utils.l2_preseg import distance_based_partition, farthest_point_sampling
from ..utils.preseg import presegmentation

try:
    import smart.rust as smart_rust
except ImportError:
    smart_rust = None


class TetMeshEnv:
    metadata = {"render.modes": ["file", "views", "video"]}

    def __init__(self, vertices, faces, voxels, args, name):
        super(TetMeshEnv, self).__init__()

        self.name = name
        self.exp_name = None
        self.args = args
        self.max_points = 0
        self.max_partition = 0
        self.num_meshes = 1

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

        tmp = self.tetmsh.get_attribute("voxel_centroid")
        self.centroid = np.asarray(tmp, dtype=float).reshape((-1, 3))

        data_path = os.path.join(args.path_to_msh_file, name)
        self.trimsh = trimesh.exchange.load.load(
            os.path.join(data_path, "tetra.msh__sf.obj"), file_type="obj", process=False
        )
        trimesh.repair.fix_normals(self.trimsh)

        mesh = pymanifold.Mesh(
            vert_pos=np.array(self.trimsh.vertices), tri_verts=np.array(self.trimsh.faces)
        )
        self.manmsh = pymanifold.Manifold()
        self.manmsh = self.manmsh.from_mesh(mesh)

        if self.args.run_type == "greedy":
            self.global_obs = None
            self.max_points = len(self.tetmsh.voxels)
        else:
            self.global_obs = []
            for i in range(len(self.tetmsh.voxels)):
                # tet_info = self.centroid[i]
                tet_info = []
                for j in range(4):
                    tet_info.append(list(self.tetmsh.vertices[self.tetmsh.voxels[i][j]]))
                tet_info.sort()
                tet_info = np.array(tet_info).reshape(-1)
                tet_info = np.concatenate((tet_info, np.array([self.volume[i]])))
                self.global_obs.append(tet_info)
            self.global_obs = np.array(self.global_obs)
            self.max_points = len(self.global_obs)

        self.greedy_map = dict()
        self.greedy_heap = []
        self.greedy_heap_order = 0
        self.greedy_past_reward = None
        self.left_part_set = set()
        self._bvs_cache = None

        self.reset(preseg_idx=0)
        if not self.args.print_off:
            print("Environment initialization done")

    def reset(self, preseg_idx=None):
        # Reset the state of the environment to an initial state
        self.done = 0
        self.greedy_map = dict()
        self.greedy_heap = []
        self.greedy_heap_order = 0
        self.greedy_cache_complete = False
        self.tetinf: List[TetInfo] = []
        self.partition: List[np.array] = []
        self.part_pts: List[np.array] = []

        # self.l2_preseg(self.args.K, preseg_idx)
        self.preseg()
        self.max_partition = len(self.partition)

        self.left_part = [i for i in range(len(self.partition))]
        self.left_part_set = set(self.left_part)
        self.part_pts = [[] for _ in range(len(self.partition))]
        self.part_merged = [True for _ in range(len(self.partition))]
        self.part_bmesh = [None for _ in range(len(self.partition))]
        self.part_bman = [None for _ in range(len(self.partition))]
        self.part_ov = [0 for _ in range(len(self.partition))]
        self.part_samples = [
            np.zeros((1 + self.args.sample_part * 3)) for _ in range(len(self.partition))
        ]
        self._bvs_cache = None

        part = None
        if self.args.only_nearby:
            part = [0 for i in range(len(self.volume))]
            for i in range(len(self.partition)):
                for j in range(len(self.partition[i])):
                    part[self.partition[i][j]] = i

        part_summaries = None
        if smart_rust is not None:
            try:
                part_summaries = smart_rust.partition_summaries(
                    self.tetmsh.vertices.tolist(),
                    self.tetmsh.voxels.tolist(),
                    self.volume.tolist(),
                    [list(map(int, partition)) for partition in self.partition],
                    unique_points=self.args.run_type == "greedy",
                )
            except Exception:
                part_summaries = None

        for i in range(len(self.partition)):
            if part_summaries is not None:
                volume = part_summaries[0][i]
                bounds = part_summaries[1][i]
                self.part_pts[i] = np.asarray(part_summaries[2][i], dtype=float).reshape((-1, 3))
            else:
                volume = 0
                first_vertex = self.tetmsh.vertices[
                    self.tetmsh.voxels[self.partition[i][0]][0]
                ]
                l_x, l_y, l_z = first_vertex
                r_x, r_y, r_z = first_vertex

                for j in range(len(self.partition[i])):
                    volume += self.volume[self.partition[i][j]]
                    for k in range(4):
                        vertex = self.tetmsh.vertices[
                            self.tetmsh.voxels[self.partition[i][j]][k]
                        ]
                        l_x = min(l_x, vertex[0])
                        l_y = min(l_y, vertex[1])
                        l_z = min(l_z, vertex[2])

                        r_x = max(r_x, vertex[0])
                        r_y = max(r_y, vertex[1])
                        r_z = max(r_z, vertex[2])

                        self.part_pts[i].append(vertex)
                self.part_pts[i] = np.array(self.part_pts[i])
                bounds = [l_x, l_y, l_z, r_x, r_y, r_z]

            if self.args.run_type == "greedy":
                self.part_pts[i] = _unique_points_exact(self.part_pts[i])

            nearby_part_id = set()
            if self.args.only_nearby:
                for j in range(len(self.partition[i])):
                    adj_voxel = self.tetmsh.get_voxel_adjacent_voxels(self.partition[i][j])
                    for k in range(len(adj_voxel)):
                        if part[adj_voxel[k]] != i:
                            nearby_part_id.add(part[adj_voxel[k]])

            self.tetinf.append(
                TetInfo(volume, bounds, nearby_part_id)
            )

        return self.current_observation()

    # def bbox_preseg(
    #     self,
    # ):
    #     """
    #     Presegment tetrahedral mesh with bounding box pre-segments
    #     """
    #     result_path = os.path.join(os.path.join(args.path_to_bbox_file, fn), "result")

    #     updates = [int(file[7:]) for file in os.listdir(result_path)]
    #     updates.sort()

    #     assert len(updates), "rl does not have any results"

    #     best_update = updates[-1]

    #     result_path = os.path.join(result_path, "updated%d" % (best_update))

    #     bbox_update_path = os.path.join(result_path, fn)
    #     steps = [int(file[11:]) for file in os.listdir(bbox_update_path)]
    #     steps.sort()
    #     best_step = steps[-1]

    #     bbox_list = []
    #     mesh_path = os.path.join(bbox_update_path, "bboxs_steps%d" % (best_step))
    #     bbox_dir = os.listdir(mesh_path)
    #     for i in range(len(bbox_dir)):
    #         if bbox_dir[i][:4] == "bbox":
    #             bbox_list.append(
    #                 trimesh.load(
    #                     os.path.join(mesh_path, bbox_dir[i]),
    #                     file_type="obj",
    #                     process=False,
    #                 )
    #             )

    #     bbox_list = []


    def preseg(
        self,
    ):
        """
        Presegment tetrahedral mesh with BSP-Net output pre-segment
        """
        self.partition = presegmentation(
            os.path.join(self.args.path_to_msh_file, self.name),
            self.tetmsh,
            self.args.init_type,
            self.args.path_to_bbox_file,
            self.name,
        )

    def l2_preseg(self, K, initial_idx=None):
        """
        Presegment tetrahedral mesh with l2 clustering
        """
        if initial_idx is None:
            initial_idx = int(np.random.randint(0, len(self.volume), 1))

        # FPS
        seed_idx, _ = farthest_point_sampling(self.centroid, K, initial_idx=initial_idx)
        seed_idx = seed_idx.squeeze()

        # Clustering
        self.partition = distance_based_partition(self.centroid, seed_idx)

    def render(self, num_update=None, mode="file", bboxs=False, index=None):
        # Render the tetmesh into .msh file
        if mode == "file":
            part = [0 for i in range(len(self.volume))]
            colors = [i for i in range(len(self.left_part))]
            random.shuffle(colors)
            for i in range(len(self.left_part)):
                nwi = self.left_part[i]
                for j in range(len(self.partition[nwi])):
                    part[self.partition[nwi][j]] = colors[i]
            part = np.array(part)
            self.tetmsh.set_attribute("voxel_partition", part)

            if self.exp_name is None:
                assert 0, "experiment name not set"
            f = os.path.join(
                os.path.join(
                    os.path.join(self.args.result_path, self.exp_name), "result"
                ),
                self.name,
            )
            os.makedirs(f, exist_ok=True)

            if num_update is None:
                filename = "%s_sz%d" % (
                    self.name[0:10],
                    len(self.left_part),
                )
            else:
                filename = "%s_sz%d_updated%d" % (
                    self.name[0:10],
                    len(self.left_part),
                    num_update,
                )

            pymesh.save_mesh(
                os.path.join(f, filename + ".msh"),
                self.tetmsh,
                "voxel_partition",
                ascii=True,
            )
            if index is None:
                index = ""

            if bboxs:
                os.makedirs(os.path.join(f, f"bboxs{index}"), exist_ok=True)

                trimesh.exchange.export.export_mesh(
                    self.trimsh,
                    os.path.join(os.path.join(f, f"bboxs{index}"), filename + ".obj"),
                    "obj",
                )

                bbox_cnt = 0
                for i in range(len(self.part_pts)):
                    if len(self.part_pts[i]):
                        if self.part_bman[i] is None:
                            (
                                self.part_bmesh[i],
                                self.part_bman[i],
                            ) = self.get_part_bmesh_bman(i, manifold=1)
                        trimesh.exchange.export.export_mesh(
                            self.part_bmesh[i],
                            os.path.join(
                                os.path.join(f, f"bboxs{index}"), "bbox%d.obj" % (bbox_cnt)
                            ),
                            "obj",
                        )
                        bbox_cnt += 1

        elif mode == "bbox":
            if self.exp_name is None:
                assert 0, "experiment name not set"
            f = os.path.join(
                    os.path.join(
                        os.path.join(self.args.result_path, self.exp_name)
                    , self.name)
                , num_update)
            os.makedirs(f, exist_ok=True)
            os.chmod(f, 0o777)
            if index is None:
                index = ""

            os.makedirs(os.path.join(f, f"bboxs{index}"), exist_ok=True)
            os.chmod(os.path.join(f, f"bboxs{index}"), 0o777)
            bbox_cnt = 0
            for i in range(len(self.part_pts)):
                if len(self.part_pts[i]):
                    if self.part_bman[i] is None:
                        (
                            self.part_bmesh[i],
                            self.part_bman[i],
                        ) = self.get_part_bmesh_bman(i, manifold=1)
                    trimesh.exchange.export.export_mesh(
                        self.part_bmesh[i],
                        os.path.join(
                            os.path.join(f, f"bboxs{index}"), "bbox%d.obj" % (bbox_cnt)
                        ),
                        "obj",
                    )
                    os.chmod(os.path.join(
                            os.path.join(f, f"bboxs{index}"), "bbox%d.obj" % (bbox_cnt)
                        ), 0o777)
                    bbox_cnt += 1

    def save_color_info(self, path_to_fd):
        box_list = []
        for i in range(len(self.part_pts)):
            if len(self.part_pts[i]):
                if self.part_bman[i] is None:
                    self.part_bmesh[i], self.part_bman[i] = self.get_part_bmesh_bman(
                        i, manifold=1
                    )
                box_list.append(self.part_bmesh[i])

        face_cen_list = []
        for i in range(len(self.trimsh.faces)):
            cen = list(
                (
                    self.trimsh.vertices[self.trimsh.faces[i][0]]
                    + self.trimsh.vertices[self.trimsh.faces[i][1]]
                    + self.trimsh.vertices[self.trimsh.faces[i][2]]
                )
                / 3
            )
            face_cen_list.append(cen)

        pt_sdf = [-1e30 for i in range(len(face_cen_list))]
        face_idx = [-1 for _ in range(len(face_cen_list))]

        for i in range(len(box_list)):
            sdf_query = trimesh.proximity.ProximityQuery(box_list[i])
            sdf = sdf_query.signed_distance(face_cen_list)

            for j in range(len(sdf)):
                if sdf[j] > pt_sdf[j]:
                    pt_sdf[j] = sdf[j]
                    face_idx[j] = i + 1

        np.save(
            os.path.join(
                path_to_fd,
                "face_color_seg%d_eps%.5g.npy" % (self.args.final_k, self.args.merge_eps),
            ),
            np.array(face_idx),
        )

    # Add random exiting -> sample i == j cases!!
    def random_sample(self) -> Tuple[int, int]:
        if self.args.only_nearby:
            # For random exit
            if np.random.rand() < 1 / len(self.left_part):
                return self.left_part[0], self.left_part[0]

            ac1 = np.random.randint(len(self.left_part), size=1)
            nearby = list(self.tetinf[self.left_part[ac1[0]]].nearby_part_id)
            ac2 = np.random.randint(len(nearby), size=1)

            return self.left_part[ac1[0]], nearby[ac2[0]]
        else:
            action = np.random.randint(len(self.left_part), size=2)
            # action = np.random.choice(len(self.left_part), size=2, replace=False)

            return self.left_part[action[0]], self.left_part[action[1]]

    def local_greedy_sample(self) -> Tuple[int, int]:
        action = None

        max_reward = -1e9
        n = len(self.left_part)
        nw = np.random.randint(len(self.left_part), size=1)[0]

        iter_arr = list(self.tetinf[self.left_part[nw]].nearby_part_id)
        for j in iter_arr:
            reward = self.step((self.left_part[nw], j), apply=0)
            if max_reward < reward:
                action = (self.left_part[nw], j)
                max_reward = reward

        if action is None:
            action = self.random_sample()
        return action

    def greedy_sample(self, until_k=0) -> Tuple[int, int]:
        """
        Warning:
        Modifying list, set with iteration (for) can result undefined behavior
        """
        action = None

        if until_k and until_k >= len(self.left_part):
            assert until_k == len(self.left_part), "Left part is smaller than until_k"
            return self.left_part[0], self.left_part[0]

        max_reward = -1e9 if until_k else -abs(self.args.merge_eps)

        if self.args.fast_merge and self.greedy_cache_complete:
            cached_action = self._best_cached_greedy_action(max_reward)
            if cached_action is not None:
                return cached_action

        for key in self._iter_greedy_candidate_keys():
            if key not in self.greedy_map:
                self._set_greedy_reward(key, self.step(key, apply=0))

            reward = self.greedy_map[key]
            if max_reward < reward:
                action = key
                max_reward = reward

        if self.args.fast_merge:
            self.greedy_cache_complete = True

        if action is None:
            action = (self.left_part[0], self.left_part[0])
        return action

    def step(
        self,
        action,
        apply=1,
        obs=False,
    ) -> Union[float, Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray], int]]:
        # Execute one time step within the environment
        first_idx, second_idx = action
        if first_idx > second_idx:
            first_idx, second_idx = second_idx, first_idx

        if self.done:
            assert 0, "Step called after done"

        if len(self.partition[first_idx]) == 0 or len(self.partition[second_idx]) == 0:
            assert 0, "Invalid partition selection"

        if first_idx == second_idx:
            if apply == 0:
                if obs == 0:
                    return 0
                else:
                    return 0, self.current_observation(), 1
            # Same index merge -> stop signal
            self.done = 1
            return 0, self.current_observation(), self.done

        if not apply and not obs and not self.args.mov and not self.args.tov:
            return self.fast_merge_reward(first_idx, second_idx)

        if not apply:
            backup_partition = copy.deepcopy(self.partition)
            backup_part = copy.deepcopy(self.part_pts)
            backup_tetinfo = copy.deepcopy(self.tetinf)
            backup_left = copy.deepcopy(self.left_part)
            backup_bvs_cache = self._bvs_cache

        # if self.args.only_nearby and (
        #     second_idx not in self.tetinf[first_idx].nearby_part_id
        # ):
        #     assert 0, "Trying to merging partition that are not nearby"

        prev_occ = self.compute_current_occupancy()
        self.merge_partition(first_idx, second_idx, apply)
        nw_occ = self.compute_current_occupancy()

        reward = nw_occ - prev_occ

        if not apply:
            if obs:
                done = 0
                if len(self.left_part) == 1:
                    done = 1

                observation = self.current_observation()

            self.partition = backup_partition
            self.part_pts = backup_part
            self.tetinf = backup_tetinfo
            self.left_part = backup_left
            self.part_bmesh[first_idx] = None
            self.part_bmesh[second_idx] = None
            self.part_bman[first_idx] = None
            self.part_bman[second_idx] = None
            self.part_ov[first_idx] = 0
            self.part_ov[second_idx] = 0
            self._bvs_cache = backup_bvs_cache
            if obs:
                return reward, observation, done
            else:
                return reward

        n = len(self.left_part)
        for i in range(n):
            self.greedy_map.pop(
                (self.left_part[i], first_idx),
                None,
            )
            self.greedy_map.pop(
                (self.left_part[i], second_idx),
                None,
            )
            self.greedy_map.pop(
                (first_idx, self.left_part[i]),
                None,
            )
            self.greedy_map.pop(
                (second_idx, self.left_part[i]),
                None,
            )

        if self.args.fast_merge and self.greedy_cache_complete:
            self._ensure_greedy_rewards_cached(changed_part=first_idx)

        if len(self.left_part) == 1:
            self.done = 1

        observation = self.current_observation()
        return reward, observation, self.done

    def _set_greedy_reward(self, action_key, reward):
        self.greedy_map[action_key] = reward
        heapq.heappush(self.greedy_heap, (-reward, self.greedy_heap_order, action_key))
        self.greedy_heap_order += 1

    def _best_cached_greedy_action(self, reward_threshold):
        while self.greedy_heap:
            neg_reward, _, action_key = self.greedy_heap[0]
            reward = self.greedy_map.get(action_key)
            if reward is None or reward != -neg_reward:
                heapq.heappop(self.greedy_heap)
                continue
            if not self._is_greedy_action_valid(action_key):
                heapq.heappop(self.greedy_heap)
                self.greedy_map.pop(action_key, None)
                continue
            if reward > reward_threshold:
                return action_key
            return None
        return None

    def _is_greedy_action_valid(self, action_key):
        first_idx, second_idx = action_key
        if first_idx == second_idx:
            return False
        if first_idx not in self.left_part_set or second_idx not in self.left_part_set:
            return False
        if (
            self.args.fast_merge
            and self.args.only_nearby
            and len(self.left_part) >= 25
        ):
            first_neighbors = self.tetinf[first_idx].nearby_part_id
            second_neighbors = self.tetinf[second_idx].nearby_part_id
            return second_idx in first_neighbors or first_idx in second_neighbors
        return True

    def _ensure_greedy_rewards_cached(self, changed_part=None):
        if changed_part is not None and not self.args.only_nearby:
            keys = self._iter_greedy_candidate_keys_for_part(changed_part)
        else:
            keys = self._iter_greedy_candidate_keys()
        for key in keys:
            if key not in self.greedy_map:
                self._set_greedy_reward(key, self.step(key, apply=0))

    def _iter_greedy_candidate_keys(self):
        n = len(self.left_part)
        for i in range(n):
            if self.args.fast_merge and self.args.only_nearby and n >= 25:
                for j in list(self.tetinf[self.left_part[i]].nearby_part_id):
                    yield self.left_part[i], j
            else:
                for j in range(i + 1, n):
                    yield self.left_part[i], self.left_part[j]

    def _iter_greedy_candidate_keys_for_part(self, part_idx):
        try:
            part_pos = self.left_part.index(part_idx)
        except ValueError:
            return
        for pos, other in enumerate(self.left_part):
            if pos == part_pos:
                continue
            if pos < part_pos:
                yield other, part_idx
            else:
                yield part_idx, other

    def fast_merge_reward(self, first_idx, second_idx):
        prev_bvs = self.BVS()
        left_volume = self.part_bbox_volume(first_idx)
        right_volume = self.part_bbox_volume(second_idx)
        merged_volume = self.merged_bbox_volume(first_idx, second_idx)
        if smart_rust is not None:
            return smart_rust.merge_bavf_reward(
                prev_bvs, left_volume, right_volume, merged_volume, self.volume_sum
            )

        new_bvs = (
            prev_bvs * self.volume_sum - left_volume - right_volume + merged_volume
        ) / self.volume_sum
        return -abs(new_bvs - 1) + abs(prev_bvs - 1)

    def current_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return observation which consist below information
        N = self.max_points
        N_P = self.max_partition
        Per tetrahedral info: N x [(x, y, z) x 4, volume] (4 points of tetrahedral mesh)
        Per partition info: N_P x [partition_sampled_points, vol_portion, min(x, y, z), max(x, y, z)]
        Available partition: N_P x 1 or N_P x N_P (only_nearby)
        """
        if self.args.run_type == "greedy":
            return None

        part = [0 for _ in range(len(self.volume))]
        for i in range(len(self.left_part)):
            nwi = self.left_part[i]
            for j in range(len(self.partition[nwi])):
                part[self.partition[nwi][j]] = nwi

        assert len(self.volume) == len(self.global_obs)

        tet_obs = []
        for i in range(len(self.global_obs)):
            tet_obs.append(self.global_obs[i])

        while len(tet_obs) < self.max_points:
            tet_obs.append(np.zeros_like(self.global_obs[0]))

        assert len(tet_obs) == self.max_points

        part_obs = []
        part_mask = []
        for i in range(len(self.partition)):
            if len(self.partition[i]) == 0:
                part_obs.append(np.zeros((1 + self.args.sample_part * 3)))

                if self.args.only_nearby:
                    part_mask.append(np.zeros(self.max_partition))
                else:
                    part_mask.append([0])
            else:
                tmp = [self.tetinf[i].volume / self.volume_sum]
                tmp = np.array(tmp)

                # adding sampled points in a partition using FPS
                if self.part_merged[i] is True:
                    self.part_merged[i] = False

                    part_pts = self.part_pts[i]

                    initial_idx = int(np.random.randint(0, len(part_pts), 1))

                    sampled_idx, _ = farthest_point_sampling(
                        part_pts, self.args.sample_part, initial_idx=initial_idx
                    )
                    sampled_idx = sampled_idx.squeeze()

                    sampled_pts = part_pts[sampled_idx, :]
                    sampled_pts = sampled_pts.reshape(-1)

                    self.part_samples[i] = np.copy(sampled_pts)
                else:
                    sampled_pts = np.copy(self.part_samples[i])

                part_obs.append(np.concatenate((sampled_pts, tmp)))

                if self.args.only_nearby:
                    nearby = np.zeros(self.max_partition)
                    nearby_part_id = list(self.tetinf[i].nearby_part_id)

                    for j in nearby_part_id:
                        nearby[j] = 1
                    nearby[i] = 1

                    part_mask.append(nearby)
                else:
                    part_mask.append([1])

        while len(part_obs) < self.max_partition:
            part_obs.append(np.zeros((1 + self.args.sample_part * 3)))
            if self.args.only_nearby:
                part_mask.append(np.zeros(self.max_partition))
            else:
                part_mask.append([0])

        assert len(part_obs) == self.max_partition
        assert len(part_mask) == self.max_partition

        return np.array(tet_obs), np.array(part_obs), np.array(part_mask)

    def environment_info(self):
        # N, d1, N_P, d2
        return *self.current_observation()[0].shape, *self.current_observation()[1].shape

    def merge_partition(self, left, right, apply):
        self.partition[left] = np.concatenate(
            (self.partition[left], self.partition[right])
        )
        self.part_pts[left] = np.concatenate((self.part_pts[left], self.part_pts[right]))
        if self.args.run_type == "greedy":
            self.part_pts[left] = _unique_points_exact(self.part_pts[left])
        self.part_bmesh[left] = None
        self.part_bman[left] = None
        self.part_ov[left] = 0
        self.tetinf[left].volume += self.tetinf[right].volume

        for i in range(6):
            if i >= 3:
                self.tetinf[left].box[i] = max(
                    self.tetinf[left].box[i], self.tetinf[right].box[i]
                )
            else:
                self.tetinf[left].box[i] = min(
                    self.tetinf[left].box[i], self.tetinf[right].box[i]
                )

        if apply:
            nearby_part_id = self.tetinf[right].nearby_part_id
            for id in nearby_part_id:
                assert id != right, "Invalid Neighbor index"

                self.tetinf[id].nearby_part_id.remove(right)
                if left != id:
                    self.tetinf[id].nearby_part_id.add(left)
                    self.tetinf[left].nearby_part_id.add(id)

            self.part_merged[left] = True
            self.part_merged[right] = True

        self.part_bmesh[right] = None
        self.part_bman[right] = None
        self.part_ov[right] = 0
        self.left_part.remove(right)
        self.left_part_set.discard(right)
        self.partition[right] = np.array([])
        self.part_pts[right] = np.array([])
        self.tetinf[right] = None
        self._bvs_cache = None

    def current_state_summary(self):
        occ = self.OCC()
        mov = self.MOV()
        tov = self.TOV()
        bvs = self.BVS()

        return occ, mov, tov, bvs

    def get_adjacency_matrix(self):
        part_ov = []
        part_id = []
        for i in range(len(self.part_pts)):
            if len(self.part_pts[i]):
                part_bmesh, part_bman = self.get_part_bmesh_bman(i, manifold=1)
                part_man = part_bman - self.manmsh
                part_ov.append(_manifold_mesh_volume(part_man) / self.tetinf[i].volume)
                part_id.append(i)

        arr = []
        for id in range(len(part_id)):
            nw = []
            for jd in range(len(part_id)):
                nw_pts = np.concatenate(
                    (
                        copy.deepcopy(self.part_pts[part_id[id]]),
                        copy.deepcopy(self.part_pts[part_id[jd]]),
                    )
                )

                if self.args.tilted:
                    bbx_mesh, bbx_man = tilted_bbox(nw_pts, manifold=1, pym=0)
                else:
                    bbx_mesh, bbx_man = axis_bbox(nw_pts, manifold=1, pym=0)

                part_man = bbx_man - self.manmsh
                part_volume = _manifold_mesh_volume(part_man)
                if id != jd:
                    nw.append(
                        part_volume
                        / (
                            self.tetinf[part_id[id]].volume
                            + self.tetinf[part_id[jd]].volume
                        )
                        - part_ov[id]
                        - part_ov[jd]
                    )
                else:
                    nw.append(
                        part_volume / (self.tetinf[part_id[id]].volume)
                        - part_ov[id]
                        - part_ov[jd]
                    )
            arr.append(nw)

        os.makedirs("./tmp/%s" % (self.name), exist_ok=True)
        np.save("./tmp/%s/adj.npy" % (self.name), arr)
        np.save("./tmp/%s/part_id.npy" % (self.name), part_id)
        return arr, part_id

    def compute_current_occupancy(self):
        metric = 0.0

        # metric += self.OCC()
        # print("occ", occ)

        bvs = self.BVS()

        metric -= abs(bvs - 1)

        # shp_ratio = 1
        # metric += 1 - shp_ratio * abs(bvs - 1)

        # tov = self.TOV()
        # metric += -tov

        if self.args.mov:
            mov = self.MOV()
            # print("mov", mov)
            metric += self.args.mov_alpha / mov
        if self.args.tov:
            tov = self.TOV()
            # print("tov", tov)
            metric += self.args.tov_beta / tov
        return metric

    def OCC(self):
        ret = 0.0
        for i in range(len(self.part_pts)):
            if len(self.part_pts[i]):
                if self.part_bmesh[i] is None:
                    self.part_bmesh[i], self.part_bman[i] = self.get_part_bmesh_bman(
                        i, manifold=(self.args.mov or self.args.tov)
                    )
                ret += (self.tetinf[i].volume / self.volume_sum) * (
                    self.tetinf[i].volume / self.part_bmesh[i].volume
                )

        return ret

    def BVS(self) -> float:
        if not self.args.mov and not self.args.tov and self._bvs_cache is not None:
            return self._bvs_cache

        ret = 0.0
        for i in range(len(self.part_pts)):
            if len(self.part_pts[i]):
                ret += self.part_bbox_volume(i)

        ret = ret / self.volume_sum
        if not self.args.mov and not self.args.tov:
            self._bvs_cache = ret
        return ret

    def part_bbox_volume(self, idx) -> float:
        if self.part_bmesh[idx] is None:
            self.part_bmesh[idx], self.part_bman[idx] = self.get_part_bmesh_bman(
                idx, manifold=(self.args.mov or self.args.tov)
            )
        return self.part_bmesh[idx].volume

    def merged_bbox_volume(self, left, right) -> float:
        if self.args.tilted:
            pts = np.concatenate((self.part_pts[left], self.part_pts[right]))
            return tilted_bbox(pts, manifold=False, pym=False)[0].volume

        left_box = self.tetinf[left].box
        right_box = self.tetinf[right].box
        if smart_rust is not None:
            return smart_rust.bbox_union_volume([left_box, right_box])
        merged = [
            min(left_box[0], right_box[0]),
            min(left_box[1], right_box[1]),
            min(left_box[2], right_box[2]),
            max(left_box[3], right_box[3]),
            max(left_box[4], right_box[4]),
            max(left_box[5], right_box[5]),
        ]
        return (
            max(0.0, merged[3] - merged[0])
            * max(0.0, merged[4] - merged[1])
            * max(0.0, merged[5] - merged[2])
        )

    def MOV(self) -> float:
        ret = 0.0
        for i in range(len(self.part_pts)):
            if len(self.part_pts[i]):
                if self.part_bman[i] is None:
                    self.part_bmesh[i], self.part_bman[i] = self.get_part_bmesh_bman(
                        i, manifold=1
                    )
                if self.part_ov[i] == 0:
                    part_man = self.part_bman[i] - self.manmsh
                    self.part_ov[i] = _manifold_mesh_volume(part_man) / self.tetinf[i].volume

                ret = max(
                    ret,
                    self.part_ov[i],
                )

        return ret

    def TOV(self):
        bbxmans = []
        for i in range(len(self.part_pts)):
            if len(self.part_pts[i]):
                if self.part_bman[i] is None:
                    self.part_bmesh[i], self.part_bman[i] = self.get_part_bmesh_bman(
                        i, manifold=1
                    )
                bbxmans.append(self.part_bman[i])

        merged_bman = bbxmans[0]
        for i in range(1, len(bbxmans)):
            merged_bman = merged_bman + bbxmans[i]
        # merged_bman = merged_bman - self.manmsh
        ret = (_manifold_mesh_volume(merged_bman) - self.volume_sum) / self.volume_sum

        return ret

    def get_part_bmesh_bman(self, idx, manifold=False, pym=False):
        pts = self.part_pts[idx]

        if self.args.tilted:
            bbx_mesh, bbx_man = tilted_bbox(pts, manifold=manifold, pym=pym)
        else:
            bbx_mesh, bbx_man = axis_bbox(pts, manifold=manifold, pym=pym)
        return bbx_mesh, bbx_man


def _manifold_mesh_volume(manifold_obj):
    mesh = manifold_obj.to_mesh()
    vertices = np.asarray(mesh.vert_pos, dtype=np.float64)
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return 0.0
    triangles = vertices[faces]
    vectors = triangles[:, 1:, :] - triangles[:, :2, :]
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    f1 = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]
    return float(np.sum(crosses[:, 0] * f1[:, 0]) / 6.0)


class TetInfo:
    def __init__(self, volume, box, nearby_part_id):
        self.volume: float = volume
        # l_x, l_y ,l_z, r_x, r_y, r_z
        self.box: List[float] = box
        self.nearby_part_id: Set[int] = nearby_part_id


def _as_numpy(array):
    array = array.squeeze()
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        return array.numpy()
    return np.asarray(array)


def _unique_points_exact(points):
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points
    return np.unique(points, axis=0)
