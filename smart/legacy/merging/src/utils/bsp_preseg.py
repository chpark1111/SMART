import math
import os
import random
import shutil
import sys
from copy import deepcopy
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import smart.pymesh_compat as pymesh
import trimesh
import trimesh.proximity
import trimesh.repair

try:
    from .bounding_box import axis_bbox, tilted_bbox
except ImportError:
    from bounding_box import axis_bbox, tilted_bbox


def bsp_presegmentation(data_path, tetmsh, debug=False) -> List[list]:
    sys.setrecursionlimit(100000)

    def render_partition(part_idx):
        os.makedirs("./tmp/parts/", exist_ok=True)
        part_vertices = []
        part_faces = []
        part_voxels = []

        num_ver = 0
        for i in range(len(partition)):
            if idx2part[i] == part_idx:
                for j in range(4):
                    part_vertices.append(tetmsh.vertices[tetmsh.voxels[i][j]])
                part_faces.append([num_ver, num_ver + 1, num_ver + 2])
                part_faces.append([num_ver, num_ver + 1, num_ver + 3])
                part_faces.append([num_ver, num_ver + 2, num_ver + 3])
                part_faces.append([num_ver + 1, num_ver + 2, num_ver + 3])
                part_voxels.append([num_ver, num_ver + 1, num_ver + 2, num_ver + 3])
                num_ver += 4

        part_tetmsh = pymesh.form_mesh(
            np.array(part_vertices), np.array(part_faces), np.array(part_voxels)
        )
        pymesh.save_mesh(
            "./tmp/parts/out_seg%d.msh" % (part_idx),
            part_tetmsh,
        )

    def partition_bfs(part_idx, group_idx):
        vis[part_idx] = True
        partition[part_idx].append(group_idx)
        if group_idx not in part_vol.keys():
            part_vol[group_idx] = volume[part_idx]
        else:
            part_vol[group_idx] += volume[part_idx]

        adj_voxel = tetmsh.get_voxel_adjacent_voxels(part_idx)
        adj_voxel = list(map(int, adj_voxel))

        for k in range(len(adj_voxel)):
            if vis[adj_voxel[k]] is False:
                partition_bfs(adj_voxel[k], group_idx)
            elif partition[adj_voxel[k]][0] < len(mesh_list):
                st = " ".join(str(j) for j in partition[adj_voxel[k]])
                dict_idx = group_idx - len(mesh_list)
                if st not in nearby_vote[dict_idx].keys():
                    nearby_vote[dict_idx][st] = 1
                else:
                    nearby_vote[dict_idx][st] += 1

    def bsp_part_bfs(part_idx, group_idx):
        bsp_vis[part_idx] = True

        st = " ".join(str(j) for j in partition[i])
        bsp_bfs_idx[st][group_idx].append(part_idx)
        bsp_bfs_vol[st][group_idx] += volume[part_idx]

        adj_voxel = tetmsh.get_voxel_adjacent_voxels(part_idx)
        adj_voxel = list(map(int, adj_voxel))

        for k in range(len(adj_voxel)):
            nst = " ".join(str(j) for j in partition[adj_voxel[k]])
            if bsp_vis[adj_voxel[k]] is False and st == nst:
                bsp_part_bfs(adj_voxel[k], group_idx)

    tetmsh.add_attribute("voxel_centroid")
    tetmsh.add_attribute("voxel_partition")
    tetmsh.add_attribute("voxel_volume")
    tetmsh.enable_connectivity()

    volume = tetmsh.get_attribute("voxel_volume")
    volume_sum = float(np.sum(volume))
    tmp = tetmsh.get_attribute("voxel_centroid")
    centroid = np.array([tmp[3 * i : 3 * i + 3] for i in range(len(tetmsh.voxels))])
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
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
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
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)
    mesh_list.append(mesh)

    partition = [[] for i in range(len(centroid))]
    for i in range(len(mesh_list)):
        query = trimesh.proximity.ProximityQuery(mesh_list[i])
        sdf = query.signed_distance(centroid)
        for j in range(len(sdf)):
            if sdf[j] > 0:
                partition[j].append(i)

    # Selecting the main parts
    bsp_vis = [False if len(partition[i]) else True for i in range(len(centroid))]
    bsp_bfs_idx = dict()
    bsp_bfs_vol = dict()
    for i in range(len(partition)):
        if bsp_vis[i] is False:
            st = " ".join(str(j) for j in partition[i])
            if st not in bsp_bfs_idx.keys():
                bsp_bfs_idx[st] = [[]]
                bsp_bfs_vol[st] = [0]
                group_idx = 0
            else:
                group_idx = len(bsp_bfs_idx[st])
                bsp_bfs_idx[st].append([])
                bsp_bfs_vol[st].append(0)
            bsp_part_bfs(i, group_idx)

    for key in bsp_bfs_vol.keys():
        cnt = -1
        for i in range(len(bsp_bfs_vol[key])):
            if bsp_bfs_vol[key][i] / volume_sum < 0.002:
                for j in bsp_bfs_idx[key][i]:
                    partition[j] = []
            else:
                for j in bsp_bfs_idx[key][i]:
                    partition[j].append(cnt)
                cnt -= 1

    # If assigning a sub part tetrahedron to main part does not effect the bounding volume, assign such tetrahedron to it
    part_pts = dict()
    for i in range(len(partition)):
        if len(partition[i]) != 0:
            st = " ".join(str(j) for j in partition[i])
            if st not in part_pts.keys():
                part_pts[st] = []
            for j in range(4):
                part_pts[st].append(tetmsh.vertices[tetmsh.voxels[i][j]])

    for i in range(len(partition)):
        if len(partition[i]) == 0:
            cur_pts = []
            for j in range(4):
                cur_pts.append(tetmsh.vertices[tetmsh.voxels[i][j]])
            adj_voxel = tetmsh.get_voxel_adjacent_voxels(i)
            adj_voxel = list(map(int, adj_voxel))
            for j in adj_voxel:
                if len(partition[j]) == 0:
                    continue
                st = " ".join(str(k) for k in partition[j])
                prev_vol = tilted_bbox(np.array(part_pts[st]))[0].volume
                add_vol = tilted_bbox(np.array(part_pts[st] + cur_pts))[0].volume

                # prev_vol = axis_bbox(np.array(part_pts[st]))[0].volume
                # add_vol = axis_bbox(np.array(part_pts[st] + cur_pts))[
                #     0
                # ].volume

                if abs(prev_vol - add_vol) < 1e-7:
                    partition[i] = list(map(int, st.split(" ")))
                    break

    # Merging left sub parts and assigning them to the main parts (or making new main part) using bfs
    part_vol = dict()
    vis = [False if len(partition[i]) == 0 else True for i in range(len(centroid))]
    nearby_vote = []
    group_cnt = len(mesh_list)
    for i in range(len(centroid)):
        if vis[i] is False:
            nearby_vote.append(dict())
            partition_bfs(i, group_cnt)
            group_cnt += 1
    # Can further change the nearby voting to bbox volume minimal voting
    # for i in range(len(partition)):
    #     if (
    #         partition[i][0] >= len(mesh_list)
    #         and part_vol[partition[i][0]] / volume_sum < 0.005
    #     ):
    #         grp_idx = partition[i][0] - len(mesh_list)
    #         max_vote = 0
    #         max_str = ""
    #         for k in nearby_vote[grp_idx].keys():
    #             if nearby_vote[grp_idx][k] >= max_vote:
    #                 max_vote = nearby_vote[grp_idx][k]
    #                 max_str = k
    #         if max_str != "":
    #             partition[i] = list(map(int, max_str.split(" ")))

    mp = dict()
    hs = dict()
    cnt = 0
    for i in range(len(partition)):
        st = " ".join(str(j) for j in partition[i])
        if st not in mp.keys():
            mp[st] = 1
            if st == "":
                hs[st] = 0
            else:
                hs[st] = cnt
                cnt += 1
        else:
            mp[st] += 1

    part = [[] for i in range(len(hs))]
    for i in range(len(partition)):
        st = " ".join(str(j) for j in partition[i])
        part[hs[st]].append(i)

    # Force the each parts to have at least one nearby to merge
    # Almost does not effect the initial presegmentation quality
    idx2part = [0 for _ in range(len(volume))]
    for i in range(len(part)):
        for j in range(len(part[i])):
            idx2part[part[i][j]] = i

    cnt_vertex_nearby = 0
    # for i in range(len(part)):
    #     nearby_part_id = set()
    #     for j in range(len(part[i])):
    #         adj_voxel = tetmsh.get_voxel_adjacent_voxels(part[i][j])
    #         for k in range(len(adj_voxel)):
    #             if idx2part[adj_voxel[k]] != i:
    #                 nearby_part_id.add(idx2part[adj_voxel[k]])
    #     if len(nearby_part_id) == 0:
    #         for j in range(len(part[i])):
    #             adj_voxel = []
    #             for k in range(4):
    #                 adj_voxel = np.concatenate(
    #                     (
    #                         adj_voxel,
    #                         tetmsh.get_vertex_adjacent_voxels(
    #                             tetmsh.voxels[part[i][j]][k]
    #                         ),
    #                     )
    #                 )
    #             adj_voxel = list(map(int, adj_voxel))
    #             for k in range(len(adj_voxel)):
    #                 if idx2part[adj_voxel[k]] != i:
    #                     nearby_part_id.add(idx2part[adj_voxel[k]])

    #         assert len(nearby_part_id) != 0, "No vertex nearby partition exists"
    #         cnt_vertex_nearby += 1
    #         idx = list(nearby_part_id)[0]
    #         for j in range(len(part[i])):
    #             idx2part[part[i][j]] = idx
    #         part[idx] += deepcopy(part[i])
    #         part[i] = []

    part = list(filter(([]).__ne__, part))
    idx2part = [0 for _ in range(len(volume))]
    for i in range(len(part)):
        for j in range(len(part[i])):
            idx2part[part[i][j]] = i

    # New part sanity check
    check_exist = [0 for i in range(len(centroid))]
    for i in range(len(part)):
        for j in range(len(part[i])):
            check_exist[part[i][j]] += 1

    for i in range(len(check_exist)):
        if check_exist[i] != 1:
            assert 0, "Error in the partition"

    if debug:
        print(hs)
        print(mp)
        print("Vertex nearby merged partition: %d" % (cnt_vertex_nearby))
        bsp_mesh_file = os.path.join(data_path, "bsp_seg.obj")
        bsp_pymesh = pymesh.load_mesh(bsp_mesh_file)
        pymesh.save_mesh_raw(
            "./tmp/out.msh",
            np.transpose(np.dot(rot, np.transpose(bsp_pymesh.vertices))),
            bsp_pymesh.faces,
            bsp_pymesh.voxels,
        )
        print("number of partiton:", len(part))

        if os.path.exists("./tmp/parts/"):
            shutil.rmtree("./tmp/parts/")

        for i in range(len(part)):
            render_partition(i)

        color = [i for i in range(0, len(part))]
        # random.shuffle(color)
        part_colors = [0 for i in range(len(centroid))]
        for i in range(len(partition)):
            part_colors[i] = color[idx2part[i]]
        part_colors = np.array(part_colors)
        tetmsh.set_attribute("voxel_partition", part_colors)
        pymesh.save_mesh(
            "./tmp/out_seg.msh",
            tetmsh,
            "voxel_partition",
            ascii=True,
        )

    return part


if __name__ == "__main__":
    dataset = "shapenet_table_e0.004_l0.2"  # "tetmesh_table_0.01_0.05"
    data = "1b7dd5d16aa6fdc1f716cef24b13c00"  # "4b455e9b8dc1380dbd508cb7996d6164"
    data_path = "../Mesh2Tet/final_data/%s/%s/" % (dataset, data)

    msh_file = os.path.join(data_path, "tetra.msh")
    tetmsh = pymesh.load_mesh(msh_file)

    bsp_presegmentation(data_path, tetmsh, debug=True)
