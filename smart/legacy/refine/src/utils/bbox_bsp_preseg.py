import math
import os
import random
import shutil
import sys
from copy import deepcopy
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pymesh
import trimesh
import trimesh.exchange.export
import trimesh.proximity
import trimesh.repair

try:
    from .bounding_box import tilted_bbox
except ImportError:
    from bounding_box import tilted_bbox


def bbox_bsp_preseg(data_path, tetmsh, num_bbox, debug=False) -> List[list]:
    sys.setrecursionlimit(100000)

    def pts2box_mesh(x, y, z, lx, ly, lz, rot, pym=False):
        vertices = []
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
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    vertices.append(
                        np.array([x, y, z])
                        + rot[0] * i * lx
                        + rot[1] * j * ly
                        + rot[2] * k * lz
                    )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        trimesh.repair.fix_normals(mesh)

        if pym:
            mesh = pymesh.form_mesh(vertices=mesh.vertices, faces=mesh.faces)
        return mesh

    def render_bbox(box_idx) -> None:
        os.makedirs("./tmp/bboxs/", exist_ok=True)
        rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        x, y, z = bbox_list[box_idx][0], bbox_list[box_idx][1], bbox_list[box_idx][2]
        lx, ly, lz = (
            bbox_list[box_idx][3] - x,
            bbox_list[box_idx][4] - y,
            bbox_list[box_idx][5] - z,
        )

        trimesh.exchange.export.export_mesh(
            pts2box_mesh(x, y, z, lx, ly, lz, rot),
            "./tmp/bboxs/bbox%d.obj" % (box_idx),
            "obj",
        )

    tetmsh.add_attribute("voxel_centroid")
    tetmsh.add_attribute("voxel_partition")
    tetmsh.add_attribute("voxel_volume")
    tetmsh.enable_connectivity()

    tmp = tetmsh.get_attribute("voxel_centroid")
    centroid = np.array([tmp[3 * i : 3 * i + 3] for i in range(len(tetmsh.voxels))])

    trimsh = trimesh.Trimesh(vertices=tetmsh.vertices, faces=tetmsh.faces)
    trimesh.repair.fix_normals(trimsh)

    vertices = []
    faces = []
    mesh_list = []
    bsp_mesh_file = os.path.join(data_path, "bsp_seg.obj")

    axis = np.array([0, 1, 0])
    angle = math.radians(-90)
    rot = pymesh.Quaternion.fromAxisAngle(axis, angle)
    rot = rot.to_matrix()

    with open(bsp_mesh_file, "r") as f:
        sum = 1
        f.readline()
        for line in f:
            line_word = line.strip().split(" ")
            if line_word[0] == "usemtl":
                for i in range(len(faces)):
                    for j in range(3):
                        faces[i][j] -= sum
                if len(vertices) != 0:
                    vertices = np.transpose(np.dot(rot, np.transpose(vertices)))
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    trimesh.repair.fix_normals(mesh)
                    mesh_list.append(mesh)
                sum += len(vertices)
                vertices = []
                faces = []
            elif line_word[0] == "v":
                vertices.append(list(map(float, line_word[1:])))
            elif line_word[0] == "f":
                faces.append(list(map(int, line_word[1:])))
    for i in range(len(faces)):
        for j in range(3):
            faces[i][j] -= sum
    vertices = np.transpose(np.dot(rot, np.transpose(vertices)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh.repair.fix_normals(mesh)
    mesh_list.append(mesh)

    seg_part = [[] for i in range(len(mesh_list))]
    for i in range(len(mesh_list)):
        query = trimesh.proximity.ProximityQuery(mesh_list[i])
        sdf = query.signed_distance(centroid)
        for j in range(len(sdf)):
            if sdf[j] > 0:
                seg_part[i].append(j)

    seg_part.sort(key=lambda x: -len(x))

    assert (
        len(mesh_list) >= num_bbox
    ), "Selected number of bounding boxes are larger than number of bsp-net output"

    if num_bbox == 0:
        num_bbox = len(seg_part)

    bbox_list = []
    for i in range(num_bbox):
        pts = []
        for idx in seg_part[i]:
            for j in range(4):
                pts.append(tetmsh.vertices[tetmsh.voxels[idx][j]])

        mn = list(np.min(pts, axis=0))
        mx = list(np.max(pts, axis=0))

        bbox_list.append(mn + mx)

    assert len(bbox_list) == num_bbox

    if debug:
        if os.path.exists("./tmp/parts/"):
            shutil.rmtree("./tmp/parts/")

        trimesh.exchange.export.export_mesh(trimsh, "./tmp/test.obj", "obj")
        print("number of bounding boxes:", num_bbox)
        print(bbox_list)
        for i in range(num_bbox):
            render_bbox(i)

    return bbox_list


if __name__ == "__main__":
    dataset = "shapenet_table_e0.010000_l0.050000_nv-1"  # "tetmesh_table_0.01_0.05"
    data_path = "../Mesh2Tet/result/%s/1e4a2ed85bc9608d99138ce6d9b8fa3a/" % (dataset)
    msh_file = os.path.join(data_path, "tetra.msh")
    tetmsh = pymesh.load_mesh(msh_file)

    bbox_bsp_preseg(data_path, tetmsh, num_bbox=0, debug=True)
