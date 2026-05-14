import copy
import math
import os
import shutil
import typing as t
from typing import List

import numpy as np
import pymesh
from typing_extensions import Literal


class STMDataset:
    def __init__(
        self,
        data_file_path: str,
        data_split: Literal["train", "test"],
        meshes: List[str],
        exclude: List[str],
        all: bool,
        mesh_txt: str,
    ) -> None:

        self.data_file_path = data_file_path
        self.data_split = data_split

        self.msh_paths = []
        self.msh_names = []
        self._mesh_cache = {}

        selected_prefixes = None
        if not all:
            if mesh_txt != "":
                selected_prefixes = []
                with open(mesh_txt, "r") as file:
                    nw = file.readline()
                    while nw:
                        selected_prefixes.append(nw.strip("\n")[:10])
                        nw = file.readline()
                selected_prefixes = set(selected_prefixes)
            else:
                selected_prefixes = set(meshes)
        cache_single_mesh = (
            self.data_split != "train"
            and selected_prefixes is not None
            and len(selected_prefixes) == 1
        )

        for f in os.listdir(data_file_path):
            if f[:10] in exclude:
                continue
            if selected_prefixes is not None and f[:10] not in selected_prefixes:
                continue

            msh_file = os.path.join(os.path.join(data_file_path, f), "tetra.msh")

            if not os.path.exists(msh_file):
                assert 0, "Broken mesh detected"
            tetmsh = pymesh.load_mesh(msh_file)
            if not tetmsh.is_closed():
                assert 0, "Broken mesh detected"

            index = len(self.msh_paths)
            self.msh_paths.append(msh_file)
            self.msh_names.append(f)
            if cache_single_mesh:
                self._mesh_cache[index] = tetmsh

        if len(self.msh_names) == 0:
            assert 0, "Mesh has not been selected"

    def __len__(self) -> int:
        return len(self.msh_names)

    def __getitem__(self, index: int):
        tetmsh = self._mesh_cache.get(index)
        if tetmsh is None:
            tetmsh = pymesh.load_mesh(self.msh_paths[index])

        vertices = np.asarray(np.copy(tetmsh.vertices), dtype=np.float32)
        faces = np.asarray(np.copy(tetmsh.faces), dtype=np.float32)
        voxels = np.asarray(np.copy(tetmsh.voxels), dtype=np.float32)
        if self.data_split == "train":
            import torch

            vertices = torch.Tensor(vertices).float()
            faces = torch.Tensor(faces).float()
            voxels = torch.Tensor(voxels).float()

        return vertices, faces, voxels, self.msh_names[index]


class _SimpleDataLoader:
    def __init__(self, dataset: STMDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        for index in range(len(self.dataset)):
            vertices, faces, voxels, name = self.dataset[index]
            yield (
                np.expand_dims(vertices, axis=0),
                np.expand_dims(faces, axis=0),
                np.expand_dims(voxels, axis=0),
                [name],
            )


def STM_DataLoader(training: bool, args):
    if training:
        data_split = "train"
    else:
        data_split = "test"

    dataset = STMDataset(
        args.path_to_msh_file,
        data_split,
        args.meshes,
        args.exclude,
        args.all,
        args.mesh_txt,
    )
    if not training and args.data_batch_size == 1 and args.worker == 0:
        return _SimpleDataLoader(dataset)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.data_batch_size,
        shuffle=training,
        drop_last=False,
        num_workers=args.worker,
    )
    return loader
