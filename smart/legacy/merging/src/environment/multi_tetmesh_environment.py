import copy
import os
import time
from typing import List, Optional, Set, Tuple, Type, Union

import numpy as np
import pymanifold
import smart.pymesh_compat as pymesh
import trimesh
import trimesh.repair
from matplotlib.style import available

from .tetmesh_environment import TetInfo, TetMeshEnv


class MultiTetMeshEnv(TetMeshEnv):
    metadata = {"render.modes": ["file", "views", "video"]}

    def __init__(self, args, dataset):
        self.exp_name = None
        self.args = args
        self.dataset = dataset
        self.iter = iter(self.dataset)
        self.max_points = 0
        self.max_partition = 0

        self.mesh_names = []
        self.mesh_tetmshs = []
        self.mesh_trimshs = []
        self.mesh_manmshs = []
        self.mesh_global_obs = []
        self.mesh_partitions = []
        self.mesh_tet_infos = []
        self.mesh_part_pts = []
        self.mesh_part_bmeshs = []
        self.mesh_part_bmans = []
        self.mesh_idx = -1

        self.greedy_map = dict()

        self.load_meshes()
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

                self.trimsh = trimesh.Trimesh(
                    vertices=self.tetmsh.vertices, faces=self.tetmsh.faces
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

                self.bsp_preseg()
                self.max_partition = max(self.max_partition, len(self.partition))
                self.mesh_partitions.append(copy.deepcopy(self.partition))

                part = [0 for i in range(len(self.volume))]
                for i in range(len(self.partition)):
                    for j in range(len(self.partition[i])):
                        part[self.partition[i][j]] = i

                self.tetinf: List[TetInfo] = []
                self.part_pts = [[] for _ in range(len(self.partition))]

                for i in range(len(self.partition)):
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

                    nearby_part_id = set()
                    for j in range(len(self.partition[i])):
                        adj_voxel = self.tetmsh.get_voxel_adjacent_voxels(
                            self.partition[i][j]
                        )
                        for k in range(len(adj_voxel)):
                            if part[adj_voxel[k]] != i:
                                nearby_part_id.add(part[adj_voxel[k]])

                    if self.args.only_nearby and len(nearby_part_id) == 0:
                        assert 0, "There isn't any nearby part"

                    self.tetinf.append(
                        TetInfo(volume, [l_x, l_y, l_z, r_x, r_y, r_z], nearby_part_id)
                    )

                self.mesh_tet_infos.append(copy.deepcopy(self.tetinf))
                self.mesh_part_pts.append(copy.deepcopy(self.part_pts))

                self.part_bmesh = [None for _ in range(len(self.partition))]
                self.part_bman = [None for _ in range(len(self.partition))]

                for i in range(len(self.part_pts)):
                    if len(self.part_pts[i]):
                        if self.part_bmesh[i] is None:
                            (
                                self.part_bmesh[i],
                                self.part_bman[i],
                            ) = self.get_part_bmesh_bman(
                                i, manifold=(self.args.tov or self.args.mov)
                            )

                self.mesh_part_bmeshs.append(copy.deepcopy(self.part_bmesh))
                self.mesh_part_bmans.append(self.part_bman)
            except StopIteration:
                print("Loaded %d meshes in the environment" % (self.num_meshes))
                return

    def reset(self, change_mesh=True):
        if change_mesh:
            self.mesh_idx = (self.mesh_idx + 1) % self.num_meshes

            self.name = self.mesh_names[self.mesh_idx]

            self.tetmsh = self.mesh_tetmshs[self.mesh_idx]
            self.trimsh = self.mesh_trimshs[self.mesh_idx]
            self.manmsh = self.mesh_manmshs[self.mesh_idx]

            self.volume = self.tetmsh.get_attribute("voxel_volume")
            self.volume_sum = np.sum(self.volume)

            tmp = self.tetmsh.get_attribute("voxel_centroid")
            self.centroid = np.array(
                [tmp[3 * i : 3 * i + 3] for i in range(len(self.tetmsh.voxels))]
            )

            self.global_obs = np.copy(self.mesh_global_obs[self.mesh_idx])

        # Reset the state of the environment to an initial state

        self.tetinf: List[TetInfo] = copy.deepcopy(self.mesh_tet_infos[self.mesh_idx])
        self.part_pts = copy.deepcopy(self.mesh_part_pts[self.mesh_idx])
        self.part_bmesh = copy.deepcopy(self.mesh_part_bmeshs[self.mesh_idx])
        self.partition: List[np.array] = copy.deepcopy(
            self.mesh_partitions[self.mesh_idx]
        )

        self.part_bman = []
        for i in range(len(self.mesh_part_bmans[self.mesh_idx])):
            if self.mesh_part_bmans[self.mesh_idx][i] is not None:
                self.part_bman.append(
                    self.mesh_part_bmans[self.mesh_idx][i].copy_manifold()
                )
            else:
                self.part_bman.append(None)

        self.done = 0
        self.left_part = [i for i in range(len(self.partition))]
        self.part_merged = [True for _ in range(len(self.partition))]
        self.part_ov = [0 for _ in range(len(self.partition))]
        self.part_samples = [
            np.zeros((7 + self.args.sample_part * 3)) for _ in range(len(self.partition))
        ]

        return self.current_observation()
