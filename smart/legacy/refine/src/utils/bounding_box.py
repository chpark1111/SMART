import math
import time

import numpy as np
import pymanifold
import smart.pymesh_compat as pymesh
import trimesh
import trimesh.repair
from pymanifold import Manifold

try:
    import smart.native as smart_native
except ImportError:
    smart_native = None


EPS = 1e-9


def _finite_points(pts):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return None
    return pts


def _safe_lengths(lengths):
    lengths = np.asarray(lengths, dtype=float)
    return np.maximum(lengths, EPS)


def pts2box_mesh(x, y, z, lx, ly, lz, rot, pym=False):
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
    vertices = None
    if smart_native is not None and hasattr(smart_native, "native_box_mesh"):
        try:
            vertices, faces = smart_native.native_box_mesh(
                float(x),
                float(y),
                float(z),
                float(lx),
                float(ly),
                float(lz),
                np.asarray(rot, dtype=float).reshape(-1).tolist(),
            )
        except Exception:
            vertices = None
    if vertices is None:
        vertices = []
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


def axis_bbox(pts, manifold=False, pym=False):
    pts = _finite_points(pts)
    if pts is None:
        pts = np.zeros((1, 3), dtype=float)
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    lengths = _safe_lengths(mx - mn)
    base = mn

    box_mesh = pts2box_mesh(
        base[0], base[1], base[2], lengths[0], lengths[1], lengths[2], np.eye(3), pym
    )

    # trimesh.exchange.export.export_mesh(box_mesh, "./tmp/AABB.obj", "obj")
    if manifold:
        box_man = Manifold.cube(lengths[0], lengths[1], lengths[2]).translate(
            base[0], base[1], base[2]
        )
        return box_mesh, box_man
    return box_mesh, None


def oriented_bbox(pts, rot, manifold=False, pym=False):
    pts = _finite_points(pts)
    rot = np.asarray(rot, dtype=float)
    if pts is None or not np.all(np.isfinite(rot)):
        return axis_bbox(np.zeros((1, 3), dtype=float) if pts is None else pts, manifold=manifold, pym=pym)
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    lengths = _safe_lengths(mx - mn)
    base = np.matmul(mn, rot)
    if not np.all(np.isfinite(base)):
        return axis_bbox(pts, manifold=manifold, pym=pym)

    box_mesh = pts2box_mesh(
        base[0], base[1], base[2], lengths[0], lengths[1], lengths[2], rot, pym
    )

    # trimesh.exchange.export.export_mesh(box_mesh, "./tmp/OBB.obj", "obj")
    if manifold:
        mesh = pymanifold.Mesh(
            vert_pos=np.array(box_mesh.vertices), tri_verts=np.array(box_mesh.faces)
        )
        box_man = pymanifold.Manifold()
        box_man = box_man.from_mesh(mesh)

        return box_mesh, box_man
    return box_mesh, None


def tilted_bbox(pts, angle_digits=3, manifold=False, pym=False):
    pts = _finite_points(pts)
    if pts is None or len(pts) < 4 or np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
        return axis_bbox(np.zeros((1, 3), dtype=float) if pts is None else pts, manifold=manifold, pym=pym)
    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(pts, angle_digits=angle_digits)
    except Exception:
        return axis_bbox(pts, manifold=manifold, pym=pym)
    rot_mat = to_origin[:3, :3]
    trans = to_origin[:3, 3:].squeeze()
    if not np.all(np.isfinite(rot_mat)) or not np.all(np.isfinite(trans)):
        return axis_bbox(pts, manifold=manifold, pym=pym)

    nw_pts = np.matmul(pts, np.transpose(rot_mat)) + trans
    if not np.all(np.isfinite(nw_pts)):
        return axis_bbox(pts, manifold=manifold, pym=pym)

    mn = np.min(nw_pts, axis=0)
    mx = np.max(nw_pts, axis=0)
    lengths = _safe_lengths(mx - mn)
    base = np.matmul(mn - trans, rot_mat)
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(lengths)):
        return axis_bbox(pts, manifold=manifold, pym=pym)

    box_mesh = pts2box_mesh(
        base[0],
        base[1],
        base[2],
        lengths[0],
        lengths[1],
        lengths[2],
        rot_mat,
        pym,
    )

    # trimesh.exchange.export.export_mesh(box_mesh, "./tmp/OBB.obj", "obj")
    if manifold:
        mesh = pymanifold.Mesh(
            vert_pos=np.array(box_mesh.vertices), tri_verts=np.array(box_mesh.faces)
        )
        box_man = pymanifold.Manifold()
        box_man = box_man.from_mesh(mesh)

        return box_mesh, box_man
    return box_mesh, None


if __name__ == "__main__":
    # vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    # faces = np.array([[3, 0, 1], [3, 2, 0], [3, 1, 2], [1, 2, 0]])
    # mesh1 = trimesh.Trimesh(vertices=vertices, faces=faces)
    # trimesh.repair.fix_normals(mesh1)
    # print(mesh1.volume)

    # vertices = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]])
    # faces = np.array([[3, 0, 1], [3, 1, 2], [3, 2, 0], [1, 0, 2]])
    # mesh2 = trimesh.Trimesh(vertices=vertices, faces=faces)
    # trimesh.repair.fix_normals(mesh2)
    # print(mesh2.volume)

    # mesh1 = pts2box_trimesh(0, 0, 0, 1, 1, 3, np.eye(3))
    # mesh2 = pts2box_trimesh(-1, 0, 1, 3, 1, 1, np.eye(3))
    # mesh4 = pts2box_trimesh(0, -1, 1, 1, 2, 2)

    # pymesh1 = pymesh.form_mesh(vertices=mesh1.vertices, faces=mesh1.faces)
    # pymesh2 = pymesh.form_mesh(vertices=mesh2.vertices, faces=mesh2.faces)
    # st = time.time()
    # pymesh3 = pymesh.boolean(pymesh1, pymesh2, operation="union", engine="cgal")
    # print(time.time()-st)
    # print(pymesh3.volume)

    # st = time.time()
    # mesh3 = trimesh.boolean.union([mesh1, mesh2], engine='scad')
    # print(time.time()-st)
    # mesh3 = trimesh.Trimesh(vertices=mesh3.vertices, faces=mesh3.faces, process=True, validate=True)
    # trimesh.exchange.export.export_mesh(mesh3, "./tmp/test.obj", "obj")
    # print(mesh3.volume)

    # Time comparision of libraries
    pymshs = []
    trimshs = []
    for i in range(50):
        pts = np.random.normal(0, 0.001, 3)
        length = np.abs(np.random.normal(0, 0.03, 3))
        py_bbx = pts2box_mesh(
            pts[0], pts[1], pts[2], length[0], length[1], length[2], np.eye(3), True
        )
        pymshs.append(py_bbx)

        tri_bbx = pts2box_mesh(
            pts[0], pts[1], pts[2], length[0], length[1], length[2], np.eye(3), False
        )
        trimshs.append(tri_bbx)

    bsp_pymesh = pymesh.load_mesh("./tmp/parts/out_seg9.msh")
    print(bsp_pymesh.volume)

    bsp_trimesh = trimesh.Trimesh(vertices=bsp_pymesh.vertices, faces=bsp_pymesh.faces)
    trimesh.repair.fix_normals(bsp_trimesh)

    pymsh = pymshs[0]
    st = time.time()
    for i in range(1, len(pymshs)):
        pymsh = pymesh.boolean(pymsh, pymshs[i], operation="union", engine="igl")
    pymsh = pymesh.boolean(pymsh, bsp_pymesh, operation="difference", engine="igl")
    print("igl", time.time() - st)
    print(pymsh.volume)

    pymsh = pymshs[0]
    st = time.time()
    for i in range(1, len(pymshs)):
        pymsh = pymesh.boolean(pymsh, pymshs[i], operation="union", engine="cgal")
    pymsh = pymesh.boolean(pymsh, bsp_pymesh, operation="difference", engine="cgal")
    print("cgal", time.time() - st)
    print(pymsh.volume)

    st = time.time()
    trimsh = trimesh.boolean.union(trimshs, engine="scad").difference(
        bsp_trimesh, engine="scad"
    )
    print("sacd", time.time() - st)
    trimesh.exchange.export.export_mesh(trimsh, "./tmp/test.obj", "obj")
    print(trimsh.volume)

    manmshs = []
    for i in range(len(trimshs)):
        mesh = pymanifold.Mesh(
            vert_pos=np.array(trimshs[i].vertices), tri_verts=np.array(trimshs[i].faces)
        )
        man = pymanifold.Manifold()
        man = man.from_mesh(mesh)
        manmshs.append(man)
    # for _ in range(1000):
    manmsh = manmshs[0]
    st = time.time()
    for i in range(1, len(manmshs)):
        manmsh = manmsh + manmshs[i]
    mesh = manmsh.to_mesh()
    meshOut = trimesh.Trimesh(vertices=mesh.vert_pos, faces=mesh.tri_verts)
    print("manifold", time.time() - st)
    print(meshOut.volume)

    # bsp_pymesh = pymesh.load_mesh("./tmp/parts/out_seg2.msh")
    # print(bsp_pymesh.volume)
    # trimsh = trimesh.Trimesh(vertices=bsp_pymesh.vertices, faces=bsp_pymesh.faces)
    # trimesh.exchange.export.export_mesh(trimsh, "./tmp/test.obj", "obj")

    # mesh_bbox, man_bbox = tilted_bbox(np.array(bsp_pymesh.vertices), manifold=True)
    # mesh = man_bbox.to_mesh()
    # meshout = trimesh.Trimesh(vertices=mesh.vert_pos, faces=mesh.tri_verts)
    # trimesh.exchange.export.export_mesh(meshout, "./tmp/man_tilt.obj", "obj")
    # print("titled:", mesh_bbox.volume)

    # mesh_bbox, man_bbox = axis_bbox(np.array(bsp_pymesh.vertices), manifold=True)
    # mesh = man_bbox.to_mesh()
    # meshout = trimesh.Trimesh(vertices=mesh.vert_pos, faces=mesh.tri_verts)
    # trimesh.exchange.export.export_mesh(meshout, "./tmp/man_axis.obj", "obj")
    # print("axis:", mesh_bbox.volume)
