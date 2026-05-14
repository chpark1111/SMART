import os
import time

import numpy as np
import pymesh
import torch
import trimesh
from tqdm import tqdm

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.tetmesh_environment import TetMeshEnv
from src.utils.utils import calculate_reward


def gen_dataset(args):

    dataset = STM_DataLoader(False, args)

    print("Dataloader loaded %d meshes" % (len(dataset)))

    def identity(x):
        return x

    cover = tqdm if args.print_off else identity

    for batch in cover(dataset):
        vertices, faces, voxels, name = batch

        start = time.time()
        env = TetMeshEnv(vertices, faces, voxels, args, name[0])

        if args.final_k > env.max_partition:
            assert 0, "Target final partition is bigger that initial partition"

        env.exp_name = (
            "%s_data_eps%f"
            % (
                args.category,
                args.data_gen_eps,
            )
        )


        while 1:
            action = env.greedy_sample(until_k=1)
            if action[0] == action[1]:
                break
            if env.step(action, apply=0) < -env.args.data_gen_eps:
                break
            reward = env.step(action)[0]

        greedy_idx=0
        while 1:
            action = env.greedy_sample(until_k=1)
            if action[0] == action[1]:
                break
            
            env.render(mode="bbox", num_update=str(greedy_idx), index="_before")

            f = os.path.join(
                    os.path.join(
                        os.path.join(env.args.result_path, env.exp_name)
                    , env.name)
                , str(greedy_idx))
            trimesh.exchange.export.export_mesh(
                env.part_bmesh[action[0]],
                os.path.join(
                    f, "bbox_split1.obj"
                ),
                "obj",
            )
            os.chmod(os.path.join(
                    f, "bbox_split1.obj"),0o777)
            trimesh.exchange.export.export_mesh(
                env.part_bmesh[action[1]],
                os.path.join(
                    f, "bbox_split2.obj"
                ),
                "obj",
            )
            os.chmod(os.path.join(
                    f, "bbox_split2.obj"),0o777)
            reward = env.step(action)[0]
            
            env.render(mode="bbox", num_update=str(greedy_idx), index="_after")
            
            mn_idx = min(action[0], action[1])
            trimesh.exchange.export.export_mesh(
                env.part_bmesh[mn_idx],
                os.path.join(
                    f, "bbox_merge.obj"
                ),
                "obj",
            )
            os.chmod(os.path.join(
                    f, "bbox_merge.obj"),0o777)
            greedy_idx += 1