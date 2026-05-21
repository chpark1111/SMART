from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import smart.native as smart_native
except ImportError:
    smart_native = None

__all__ = [
    "Mesh",
    "Quaternion",
    "boolean",
    "form_mesh",
    "load_mesh",
    "save_mesh",
    "save_mesh_raw",
]


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=int))
    voxels: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=int))
    _attributes: dict[str, np.ndarray] = field(default_factory=dict)
    _adjacency: list[list[int]] | None = None

    def is_closed(self) -> bool:
        if len(self.voxels):
            return True
        if len(self.faces) == 0:
            return False
        edges: dict[tuple[int, int], int] = {}
        for face in self.faces:
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = tuple(sorted((int(a), int(b))))
                edges[key] = edges.get(key, 0) + 1
        return all(count == 2 for count in edges.values())

    def enable_connectivity(self) -> None:
        if self._adjacency is not None:
            return
        if smart_native is not None:
            self._adjacency = smart_native.tetra_adjacency(self.voxels.tolist())
            return
        face_to_voxels: dict[tuple[int, int, int], list[int]] = {}
        for index, voxel in enumerate(self.voxels):
            for face in _tet_faces(voxel):
                key = tuple(sorted(int(v) for v in face))
                face_to_voxels.setdefault(key, []).append(index)
        adjacency = [set() for _ in range(len(self.voxels))]
        for owners in face_to_voxels.values():
            if len(owners) < 2:
                continue
            for owner in owners:
                adjacency[owner].update(other for other in owners if other != owner)
        self._adjacency = [sorted(values) for values in adjacency]

    def get_voxel_adjacent_voxels(self, index: int) -> np.ndarray:
        self.enable_connectivity()
        assert self._adjacency is not None
        return np.array(self._adjacency[int(index)], dtype=int)

    def add_attribute(self, name: str) -> None:
        if name in self._attributes:
            return
        if name == "voxel_volume":
            if smart_native is not None:
                self._attributes[name] = np.asarray(
                    smart_native.tetra_volumes(self.vertices.tolist(), self.voxels.tolist()),
                    dtype=float,
                )
            else:
                self._attributes[name] = _tetra_volumes(self.vertices, self.voxels)
        elif name == "voxel_centroid":
            if smart_native is not None:
                self._attributes[name] = np.asarray(
                    smart_native.tetra_centroids(self.vertices.tolist(), self.voxels.tolist()),
                    dtype=float,
                )
            else:
                centroids = np.mean(self.vertices[self.voxels], axis=1) if len(self.voxels) else np.zeros((0, 3))
                self._attributes[name] = centroids.reshape(-1)
        elif name == "voxel_partition":
            self._attributes[name] = np.zeros(len(self.voxels), dtype=int)
        else:
            self._attributes[name] = np.zeros(len(self.voxels))

    def get_attribute(self, name: str) -> np.ndarray:
        if name not in self._attributes:
            self.add_attribute(name)
        return self._attributes[name]

    def set_attribute(self, name: str, values: Iterable[float]) -> None:
        self._attributes[name] = np.array(values)

    @property
    def volume(self) -> float:
        return float(np.sum(self.get_attribute("voxel_volume")))


class Quaternion:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    @staticmethod
    def fromAxisAngle(axis: Iterable[float], angle: float) -> "Quaternion":
        axis_arr = np.array(axis, dtype=float)
        axis_arr = axis_arr / np.linalg.norm(axis_arr)
        x, y, z = axis_arr
        c = math.cos(angle)
        s = math.sin(angle)
        one_c = 1.0 - c
        matrix = np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=float,
        )
        return Quaternion(matrix)

    def to_matrix(self) -> np.ndarray:
        return self._matrix


def form_mesh(vertices, faces=None, voxels=None) -> Mesh:
    return Mesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces if faces is not None else [], dtype=int).reshape((-1, 3)),
        voxels=np.asarray(voxels if voxels is not None else [], dtype=int).reshape((-1, 4)),
    )


def load_mesh(filename: str | Path) -> Mesh:
    path = Path(filename)
    if path.suffix.lower() == ".obj":
        return _load_obj(path)
    return _load_gmsh(path)


def save_mesh(filename: str | Path, mesh: Mesh, *attributes: str, ascii: bool = True) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".obj":
        _save_obj(path, mesh)
    else:
        _save_gmsh(path, mesh)


def save_mesh_raw(filename: str | Path, vertices, faces, voxels=None, *args, **kwargs) -> None:
    save_mesh(filename, form_mesh(vertices, faces, voxels))


def boolean(*args, **kwargs):
    raise NotImplementedError("The SMART pymesh shim does not implement PyMesh boolean operations.")


def _load_obj(path: Path) -> Mesh:
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(part.split("/")[0]) - 1 for part in line.split()[1:4]])
    return form_mesh(vertices, faces)


def _save_obj(path: Path, mesh: Mesh) -> None:
    with path.open("w", encoding="utf-8") as file:
        for vertex in mesh.vertices:
            file.write("v %.9g %.9g %.9g\n" % tuple(vertex[:3]))
        for face in mesh.faces:
            file.write("f %d %d %d\n" % (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1))


def _load_gmsh(path: Path) -> Mesh:
    if smart_native is not None:
        vertices, faces, voxels = smart_native.load_gmsh(str(path))
        return form_mesh(vertices, faces, voxels)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if "$Nodes" not in lines or "$Elements" not in lines:
        raise ValueError(f"Unsupported or invalid Gmsh file: {path}")
    nodes_start = lines.index("$Nodes") + 1
    node_count = int(lines[nodes_start].split()[0])
    node_id_to_index: dict[int, int] = {}
    vertices = []
    for offset in range(node_count):
        parts = lines[nodes_start + 1 + offset].split()
        node_id = int(parts[0])
        node_id_to_index[node_id] = offset
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    elements_start = lines.index("$Elements") + 1
    element_count = int(lines[elements_start].split()[0])
    faces = []
    voxels = []
    for offset in range(element_count):
        parts = lines[elements_start + 1 + offset].split()
        element_type = int(parts[1])
        tag_count = int(parts[2])
        ids = [node_id_to_index[int(value)] for value in parts[3 + tag_count :]]
        if element_type == 2 and len(ids) >= 3:
            faces.append(ids[:3])
        elif element_type == 4 and len(ids) >= 4:
            voxels.append(ids[:4])
    if not faces and voxels:
        faces = _surface_faces(np.asarray(voxels, dtype=int))
    return form_mesh(vertices, faces, voxels)


def _save_gmsh(path: Path, mesh: Mesh) -> None:
    if smart_native is not None:
        smart_native.save_gmsh(
            str(path),
            mesh.vertices.tolist(),
            mesh.faces.tolist(),
            mesh.voxels.tolist(),
        )
        return

    elements = []
    for face in mesh.faces:
        elements.append((2, [int(v) + 1 for v in face[:3]]))
    for voxel in mesh.voxels:
        elements.append((4, [int(v) + 1 for v in voxel[:4]]))
    with path.open("w", encoding="utf-8") as file:
        file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        file.write("$Nodes\n%d\n" % len(mesh.vertices))
        for index, vertex in enumerate(mesh.vertices, start=1):
            file.write("%d %.17g %.17g %.17g\n" % (index, vertex[0], vertex[1], vertex[2]))
        file.write("$EndNodes\n")
        file.write("$Elements\n%d\n" % len(elements))
        for index, (element_type, ids) in enumerate(elements, start=1):
            file.write("%d %d 0 %s\n" % (index, element_type, " ".join(str(value) for value in ids)))
        file.write("$EndElements\n")


def _tetra_volumes(vertices: np.ndarray, voxels: np.ndarray) -> np.ndarray:
    if len(voxels) == 0:
        return np.zeros(0)
    pts = vertices[voxels]
    return np.abs(np.einsum("ij,ij->i", pts[:, 1] - pts[:, 0], np.cross(pts[:, 2] - pts[:, 0], pts[:, 3] - pts[:, 0]))) / 6.0


def _tet_faces(voxel: Iterable[int]) -> list[tuple[int, int, int]]:
    a, b, c, d = [int(v) for v in voxel]
    return [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]


def _surface_faces(voxels: np.ndarray) -> np.ndarray:
    if smart_native is not None:
        return np.array(smart_native.tetra_surface_faces(voxels.tolist()), dtype=int).reshape((-1, 3))
    counts: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for voxel in voxels:
        for face in _tet_faces(voxel):
            key = tuple(sorted(face))
            counts[key] = None if key in counts else face
    return np.array([face for face in counts.values() if face is not None], dtype=int).reshape((-1, 3))
