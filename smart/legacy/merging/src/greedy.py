import os
import time

import numpy as np
import pymesh
import trimesh
from tqdm import tqdm

from src.datamodules.tetmesh_datamodule import STM_DataLoader
from src.environment.tetmesh_environment import TetMeshEnv
from src.utils.utils import calculate_reward


def _dataset_name(path):
    return os.path.basename(os.path.normpath(path))


def greedy(args):

    dataset = STM_DataLoader(False, args)

    print("Dataloader loaded %d meshes" % (len(dataset)))

    def identity(x):
        return x

    cover = tqdm if args.print_off else identity

    for batch in cover(dataset):
        vertices, faces, voxels, name = batch
        if not args.print_off:
            print("Greedy search on %s mesh" % (name[0][:10]))
        start = time.time()
        env = TetMeshEnv(vertices, faces, voxels, args, name[0])

        if args.final_k > env.max_partition:
            assert 0, "Target final partition is bigger that initial partition"

        env.exp_name = (
            "greedy_%s%s_fastmerge%d_mgeps%.5g_final_k%dtilted%d_mov%.5gtov%.5g"
            % (
                _dataset_name(args.path_to_msh_file),
                ("_" + args.init_type if args.init_type != "bsp" else ""),
                1 if args.fast_merge else 0,
                args.merge_eps,
                args.final_k,
                (1 if args.tilted else 0),
                (args.mov_alpha if args.mov else 0),
                (args.tov_beta if args.tov else 0),
            )
        )
        env.render()

        rewards = []
        greedy_idx=0
        while 1:
            action = env.greedy_sample(until_k=args.final_k)
            if action[0] == action[1]:
                break

            reward = env.step(action)[0]
            rewards.append(reward)
            # env.render()
            env.render(bboxs=True, index=greedy_idx)
            greedy_idx += 1
            
            if not args.print_off:
                print("Action: (%d, %d)" % (action[0], action[1]))
                print("Reward: %f\n" % (reward))

        path_to_result = os.path.join(
            os.path.join(os.path.join(args.result_path, env.exp_name), "result"),
            name[0],
        )
        with open(
            os.path.join(
                path_to_result,
                "time.txt",
            ),
            "w",
        ) as f:
            f.write(str(env.max_partition))
            f.write("\n")
            f.write(str(len(env.left_part)))
            f.write("\n")
            f.write(str(time.time() - start))

        sum_reward = calculate_reward(rewards, args.gamma)
        final_occ = env.compute_current_occupancy()
        occ, mov, tov, bvs = env.current_state_summary()
        if not args.print_off:
            print(
                "Shape %s (%d -> %s %d), Final metric: %g, Improved metric: %g, Elapsed time: %g"
                % (
                    name[0][:10],
                    env.max_partition,
                    "Target partition" if args.final_k else "Find partition",
                    len(env.left_part),
                    final_occ,
                    sum_reward,
                    time.time() - start,
                )
            )
            print(
                "Metrics after merging, OCC: %g, MOV: %g, TOV: %g, BVS: %g"
                % (occ, mov, tov, bvs)
            )
        env.render(bboxs=True)

        # Save greedy merged result on the mesh file path
        mesh_name = name[0]
        path_to_fd = os.path.join(args.path_to_msh_file, mesh_name)

        save_preseg = []
        for i in range(len(env.left_part)):
            save_preseg.append([])
            nwi = env.left_part[i]
            for j in range(len(env.partition[nwi])):
                save_preseg[i].append(env.partition[nwi][j])

        with open(
            os.path.join(
                path_to_fd,
                "greedy_segment%d%s_mgeps%.5g%s.txt"
                % (
                    args.final_k,
                    ("_" + args.init_type) if args.init_type != "bsp" else "",
                    args.merge_eps,
                    "_fm" if args.fast_merge else "",
                ),
            ),
            "w",
        ) as f:
            f.writelines(str(len(env.left_part)))
            f.write("\n")

            for i in range(len(env.left_part)):
                for j in range(len(save_preseg[i])):
                    f.write(
                        str(save_preseg[i][j])
                        + (" " if j != len(save_preseg[i]) - 1 else "")
                    )

                if i != len(env.left_part) - 1:
                    f.write("\n")

        # env.save_color_info(path_to_fd)
