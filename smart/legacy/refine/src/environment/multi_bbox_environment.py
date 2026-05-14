import copy
import os
import time
from typing import List, Optional, Set, Tuple, Type, Union

import numpy as np
import pymanifold
import pymesh
import trimesh
import trimesh.repair

from .bbox_environment import BBox, MeshBBoxEnv

try:
    import smart.rust as smart_rust
except ImportError:
    smart_rust = None


class MultiMeshBBoxEnv(MeshBBoxEnv):
    metadata = {"render.modes": ["file", "views", "video"]}

    def __init__(self, args, dataset) -> None:
        self.exp_name = None
        self.args = args
        self.dataset = dataset
        self.iter = iter(self.dataset)
        self.max_points = 0
        self.max_bboxs = 0
        self.num_bbox = 0

        self.mesh_names = []
        self.mesh_tetmshs = []
        self.mesh_trimshs = []
        self.mesh_manmshs = []
        self.mesh_global_obs = []

        self.mesh_bbox_list: List[BBox] = []
        self.mesh_bbox_mesh: List[trimesh.Trimesh] = []
        self.mesh_bbox_man = []
        self.bbox_part_vol: List[float] = []
        self.bbox_part_ov: List[float] = []
        self.bbox_part_occ: List[float] = []

        self.mesh_idx = -1

        self.num_actions = 0
        self.action_unit = args.action_unit
        self.num_action_scale = args.num_action_scale * 2
        if smart_rust is not None:
            self.action_scale = smart_rust.action_scales(self.num_action_scale)
        else:
            self.action_scale = [
                -(2**i) for i in range(args.num_action_scale - 1, -1, -1)
            ] + [2**i for i in range(args.num_action_scale)]

        self.max_step = args.max_step

        self.step_vec = None
        self.action_mask = None

        self.pen_rate = 1.0

        self.load_meshes()

        self.num_actions = self.max_bboxs * (6 * self.num_action_scale + 1)
        self.action_mask = np.zeros(self.num_actions)
        self.action_mask[self.num_bbox * (6 * self.num_action_scale + 1) :] = 1

        self.action2idx = np.zeros(
            (self.max_bboxs * (6 * self.num_action_scale + 1), 3), dtype=int
        )
        self.idx2action = np.zeros(
            ((self.max_bboxs, 7, self.num_action_scale)), dtype=int
        )

        if smart_rust is not None:
            action_rows = smart_rust.action_indices(self.max_bboxs, self.num_action_scale)
        else:
            action_rows = []
            for i in range(self.max_bboxs):
                for j in range(6):
                    for k in range(self.num_action_scale):
                        action_rows.append([i, j, k])
                action_rows.append([i, 6, 0])

        for action_id, (i, j, k) in enumerate(action_rows):
            self.idx2action[i][j][k] = action_id
            self.action2idx[action_id] = np.array([i, j, k])

        self.reset(change_mesh=True)
        print("Environment initialization done")

    def load_meshes(
        self,
    ):
        self.num_meshes = 0
        while 1:
            try:
                # Samples the batch
                vertices, faces, voxels, name = next(self.iter)
                self.num_meshes += 1

                self.name = name[0]
                self.mesh_names.append(self.name)

                vertices = vertices.squeeze()
                faces = faces.squeeze()
                voxels = voxels.squeeze()

                self.tetmsh = pymesh.form_mesh(
                    vertices.numpy(), faces.numpy(), voxels.numpy()
                )
                self.tetmsh.enable_connectivity()
                self.tetmsh.add_attribute("voxel_volume")
                self.tetmsh.add_attribute("voxel_centroid")
                self.tetmsh.add_attribute("voxel_partition")

                self.mesh_tetmshs.append(self.tetmsh)

                self.volume = self.tetmsh.get_attribute("voxel_volume")

                tmp = self.tetmsh.get_attribute("voxel_centroid")
                self.centroid = np.array(
                    [tmp[3 * i : 3 * i + 3] for i in range(len(self.tetmsh.voxels))]
                )

                data_path = os.path.join(self.args.path_to_msh_file, name[0])
                self.trimsh = trimesh.exchange.load.load(
                    os.path.join(data_path, "tetra.msh__sf.obj"),
                    file_type="obj",
                    process=False,
                )
                trimesh.repair.fix_normals(self.trimsh)
                self.mesh_trimshs.append(self.trimsh)

                mesh = pymanifold.Mesh(
                    vert_pos=np.array(self.trimsh.vertices),
                    tri_verts=np.array(self.trimsh.faces),
                )
                self.manmsh = pymanifold.Manifold()
                self.manmsh = self.manmsh.from_mesh(mesh)
                self.mesh_manmshs.append(self.manmsh)

                self.global_obs = []
                for i in range(len(self.tetmsh.voxels)):
                    tet_info = []
                    for j in range(4):
                        tet_info.append(
                            list(self.tetmsh.vertices[self.tetmsh.voxels[i][j]])
                        )
                    tet_info.sort()

                    tet_info = np.array(tet_info).reshape(-1)
                    tet_info = np.concatenate((tet_info, np.array([self.volume[i]])))
                    self.global_obs.append(tet_info)

                self.global_obs = np.array(self.global_obs)
                self.mesh_global_obs.append(np.copy(self.global_obs))

                self.max_points = max(self.max_points, len(self.global_obs))

                self.bbox_list: List[BBox] = []
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

                self.mesh_bbox_list.append(self.bbox_list)

                self.max_bboxs = max(self.max_bboxs, len(self.bbox_list))
                self.num_bbox = len(self.bbox_list)

                self.bbox_mesh = [None for _ in range(self.num_bbox)]
                self.bbox_man = [None for _ in range(self.num_bbox)]
                self.bbox_part_vol: List[float] = [-1 for _ in range(self.num_bbox)]
                self.bbox_part_ov: List[float] = [-1 for _ in range(self.num_bbox)]
                self.bbox_part_occ: List[float] = [-1 for _ in range(self.num_bbox)]

                for i in range(self.num_bbox):
                    (
                        self.bbox_mesh[i],
                        self.bbox_man[i],
                    ) = self.get_bbox_bmesh_bman(i)

                self.mesh_bbox_mesh.append(copy.deepcopy(self.bbox_mesh))
                self.mesh_bbox_man.append(self.bbox_man)

            except StopIteration:
                print("Loaded %d meshes in the environment" % (self.num_meshes))
                return

    def reset(self, pen_rate=1.0, change_mesh=True):
        if change_mesh:
            self.mesh_idx = (self.mesh_idx + 1) % self.num_meshes

            self.name = self.mesh_names[self.mesh_idx]

            self.tetmsh = self.mesh_tetmshs[self.mesh_idx]
            self.trimsh = self.mesh_trimshs[self.mesh_idx]
            self.manmsh = self.mesh_manmshs[self.mesh_idx].copy_manifold()

            self.volume = self.tetmsh.get_attribute("voxel_volume")
            self.volume_sum = np.sum(self.volume)

            tmp = self.tetmsh.get_attribute("voxel_centroid")
            self.centroid = np.array(
                [tmp[3 * i : 3 * i + 3] for i in range(len(self.tetmsh.voxels))]
            )

            self.global_obs = np.copy(self.mesh_global_obs[self.mesh_idx])

        # Reset the state of the environment to an initial state

        self.done = 0
        self.step_cnt = 0
        self.last_bbox_score = 0

        self.step_cache = None

        self.bbox_list: List[BBox] = []
        self.bbox_man = []
        self.bbox_mesh: List[trimesh.Trimesh] = []
        self.bbox_part_vol: List[float] = []
        self.bbox_part_ov: List[float] = []
        self.bbox_part_occ: List[float] = []

        self.bbox_list = copy.deepcopy(self.mesh_bbox_list[self.mesh_idx])
        self.num_bbox = len(self.bbox_list)

        self.bbox_mesh = copy.deepcopy(self.mesh_bbox_mesh[self.mesh_idx])
        self.bbox_man = []
        for i in range(len(self.mesh_bbox_man[self.mesh_idx])):
            self.bbox_man.append(self.mesh_bbox_man[self.mesh_idx][i].copy_manifold())

        self.bbox_part_vol: List[float] = [-1 for _ in range(self.num_bbox)]
        self.bbox_part_ov: List[float] = [-1 for _ in range(self.num_bbox)]
        self.bbox_part_occ: List[float] = [-1 for _ in range(self.num_bbox)]

        self.step_vec = np.zeros((self.max_step), dtype=int)
        self.step_vec[self.step_cnt] = 1

        self.action_mask = np.zeros(self.num_actions)
        self.action_mask[self.num_bbox * (6 * self.num_action_scale + 1) :] = 1

        self.pen_rate = pen_rate

        self.last_bbox_score = self.evaluate_bbox_score()

        return self.current_observation()
