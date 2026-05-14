import os

import numpy as np
import pymesh
import trimesh
import trimesh.repair

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.multi_tetmesh_environment import MultiTetMeshEnv
from src.environment.tetmesh_environment import TetMeshEnv


def var3dseg(args):
    dataset = STM_DataLoader(False, args)

    for batch in dataset:
        vertices, faces, voxels, name = batch

        vertices = vertices.squeeze()
        faces = faces.squeeze()
        voxels = voxels.squeeze()

        tetmsh = pymesh.form_mesh(vertices.numpy(), faces.numpy(), voxels.numpy())
        tetmsh.enable_connectivity()
        tetmsh.add_attribute("voxel_volume")
        tetmsh.add_attribute("voxel_centroid")
        tetmsh.add_attribute("voxel_partition")

        volume = tetmsh.get_attribute("voxel_volume")
        volume_sum = np.sum(volume)

        tmp = tetmsh.get_attribute("voxel_centroid")
        centroid = np.array([tmp[3 * i : 3 * i + 3] for i in range(len(tetmsh.voxels))])

        trimsh = trimesh.Trimesh(vertices=tetmsh.vertices, faces=tetmsh.faces)
        trimesh.repair.fix_normals(trimsh)

        global_obs = []
        for i in range(len(tetmsh.voxels)):
            tet_info = np.array([])
            for j in range(4):
                tet_info = np.concatenate(
                    (tet_info, tetmsh.vertices[tetmsh.voxels[i][j]])
                )
            tet_info = np.concatenate((tet_info, np.array([volume[i]])))
            global_obs.append(tet_info)
        global_obs = np.array(global_obs)


def minimize_f(partition):

    pass


def run_var3dseg():

    pass
