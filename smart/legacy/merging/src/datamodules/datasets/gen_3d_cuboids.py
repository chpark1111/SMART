import argparse
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm

# Things to add!
# 1. Sample points nosiy (from gaussian)
# 2. Add other shapes


def parse_args():

    """parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Generate 3D point clouds sampled from cuboids and save it with ply format"
    )

    parser.add_argument(
        "--num_data", type=int, default=50, help="Number of data to generate"
    )
    parser.add_argument(
        "--num_pts",
        type=int,
        default=10000,
        help="Number of points in point clouds per data",
    )
    parser.add_argument(
        "--num_cbs", type=int, default=2, help="Number of cuboids per data"
    )
    parser.add_argument(
        "--max_length", type=int, default=10, help="Max lenght of each cuboid can have"
    )

    parser.add_argument(
        "--sv_path",
        type=str,
        default="/home/chpark1111/research/Unsup3DMeshSeg/data/3d_cuboids",
        help="Directory to save the point cloud data",
    )

    args = parser.parse_args()
    return args


def sdf_cuboid(pt, pm, ct):
    pt = pt - ct
    q = np.absolute(pt) - pm
    return np.linalg.norm(np.maximum(q, np.zeros_like(q))) + np.minimum(q.max(), [0.0])


def generate_cuboid(num_pts, num_cbs, max_length):
    # Sample parameters for cuboids
    cb_param = []
    for _ in range(num_cbs):
        pm = np.random.uniform(np.finfo(float).eps, max_length / 2, 3)
        ct = np.random.normal(0, 1, 3)

        cb_param.append([pm, ct])
    # Predefined variables
    pre_pts = []
    pts = []
    idx = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    pidx = [[0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0]]
    n_pt_cb = []
    total = 0
    # Set number of points proportional to surface area
    for i in range(num_cbs):
        n_pt_sf = []
        n_pt_sf.append(cb_param[i][0][1] * cb_param[i][0][2])
        n_pt_sf.append(cb_param[i][0][0] * cb_param[i][0][2])
        n_pt_sf.append(cb_param[i][0][0] * cb_param[i][0][1])
        n_pt_cb.append(n_pt_sf)
        total += sum(n_pt_sf)

    for i in range(num_cbs):
        for j in range(3):
            n_pt_cb[i][j] = int((n_pt_cb[i][j] / (total * 2)) * num_pts)

    # Sample random points at surface of cuboids
    for i in range(num_cbs):
        for j in range(6):
            base = cb_param[i][1] + np.multiply(cb_param[i][0], idx[j])
            for k in range(n_pt_cb[i][j // 2]):
                pt = base + np.multiply(
                    np.concatenate(
                        (
                            np.random.uniform(-cb_param[i][0][0], cb_param[i][0][0], 1),
                            np.random.uniform(-cb_param[i][0][1], cb_param[i][0][1], 1),
                            np.random.uniform(-cb_param[i][0][2], cb_param[i][0][2], 1),
                        )
                    ),
                    pidx[j],
                )
                pre_pts.append(pt)
    # Remove inside points of cuboids
    for pt in pre_pts:
        fg = 1
        for j in cb_param:
            if sdf_cuboid(pt, j[0], j[1]) < -1e-7:
                fg = 0
                break
        if fg:
            pts.append(pt)

    return pts


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.sv_path):
        os.makedirs(args.sv_path)

    # Point cloud saving
    for i in tqdm(range(args.num_data)):
        my_pts = np.array(generate_cuboid(args.num_pts, args.num_cbs, args.max_length))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(my_pts)
        o3d.io.write_point_cloud(
            os.path.join(
                args.sv_path,
                "num_pts%dnum_cbs%dmax_len%did%d.ply"
                % (args.num_pts, args.num_cbs, args.max_length, i + 1),
            ),
            pcd,
        )
