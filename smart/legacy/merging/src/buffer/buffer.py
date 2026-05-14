from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Type, Union

import config as p
import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        max_num_tet: int,
        tet_idim: int,
        num_init_part: int,
        part_idim: int,
        only_nearby: bool,
        n_step: int,
        gamma: float,
    ):

        super(ReplayBuffer, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.capacity = capacity
        self.max_num_tet = max_num_tet
        self.tet_idim = tet_idim
        self.num_init_part = num_init_part
        self.part_idim = part_idim
        self.only_nearby = only_nearby

        self.tet_obs_mem = np.zeros(
            (self.capacity, max_num_tet, tet_idim), dtype=np.float32
        )
        self.part_obs_mem = np.zeros(
            (self.capacity, num_init_part, part_idim), dtype=np.float32
        )
        if self.only_nearby:
            self.part_mask_mem = np.zeros(
                (self.capacity, num_init_part, num_init_part), dtype=np.float32
            )
        else:
            self.part_mask_mem = np.zeros(
                (self.capacity, num_init_part, 1), dtype=np.float32
            )

        self.reward_mem = np.zeros((self.capacity, 1))

        self.next_tet_obs_mem = np.zeros(
            (self.capacity, max_num_tet, tet_idim), dtype=np.float32
        )
        self.next_part_obs_mem = np.zeros(
            (self.capacity, num_init_part, part_idim), dtype=np.float32
        )
        if self.only_nearby:
            self.next_part_mask_mem = np.zeros(
                (self.capacity, num_init_part, num_init_part), dtype=np.float32
            )
        else:
            self.next_part_mask_mem = np.zeros(
                (self.capacity, num_init_part, 1), dtype=np.float32
            )

        self.action_mem = np.zeros((self.capacity, 2), dtype=np.int32)
        self.done_mem = np.zeros((self.capacity, 1), dtype=np.int32)

        self.memory_counter = 0

        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
        self.size = 0

    def store(
        self,
        last_obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        new_obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        done: np.ndarray,
    ):
        transition = (last_obs, action, reward, new_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        reward, new_obs, done = self.get_n_step_info(self.n_step_buffer, self.gamma)
        last_obs, action = self.n_step_buffer[0][:2]

        index = self.memory_counter % self.capacity
        obs_idx = len(last_obs[0])

        self.tet_obs_mem[index, :obs_idx] = last_obs[0]
        self.tet_obs_mem[index, obs_idx:] = np.zeros(
            (self.max_num_tet - obs_idx, self.tet_idim)
        )
        self.part_obs_mem[index, :] = last_obs[1]
        self.part_mask_mem[index, :] = last_obs[2]

        self.reward_mem[index, :] = reward

        self.next_tet_obs_mem[index, :obs_idx] = new_obs[0]
        self.next_tet_obs_mem[index, obs_idx:] = np.zeros(
            (self.max_num_tet - obs_idx, self.tet_idim)
        )
        self.next_part_obs_mem[index, :] = new_obs[1]
        self.next_part_mask_mem[index, :] = new_obs[2]

        self.action_mem[index, :] = action
        self.done_mem[index, :] = done

        self.memory_counter += 1
        self.size = min(self.capacity, self.size + 1)

        return self.n_step_buffer[0]

    def clear(self):
        self.memory_counter = 0

    def sample(self, batch: int):
        indices = np.random.choice(min(self.memory_counter, self.capacity), size=batch)

        last_obs = (
            self.tet_obs_mem[indices, :, :],
            self.part_obs_mem[indices, :],
            self.part_mask_mem[indices, :],
        )
        reward = self.reward_mem[indices, :]
        new_obs = (
            self.next_tet_obs_mem[indices, :, :],
            self.next_part_obs_mem[indices, :],
            self.next_part_mask_mem[indices, :],
        )

        action = self.action_mem[indices, :]
        done = self.done_mem[indices, :]

        data = [last_obs, action, reward, new_obs, done]
        return tuple(map(self.to_torch, data)), indices

    def sample_batch_from_idxs(self, indices: np.ndarray):
        last_obs = (
            self.tet_obs_mem[indices, :, :],
            self.part_obs_mem[indices, :],
            self.part_mask_mem[indices, :],
        )
        reward = self.reward_mem[indices, :]
        new_obs = (
            self.next_tet_obs_mem[indices, :, :],
            self.next_part_obs_mem[indices, :],
            self.next_part_mask_mem[indices, :],
        )

        action = self.action_mem[indices, :]
        done = self.done_mem[indices, :]

        data = [last_obs, action, reward, new_obs, done]
        return tuple(map(self.to_torch, data))

    def get_n_step_info(self, n_step_buffer: Deque, gamma: float):
        reward, new_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            reward = r + gamma * reward * (1 - d)
            new_obs, done = (n_o, d) if d else (new_obs, done)

        return reward, new_obs, done

    def to_torch(
        self,
        array: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        copy: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if type(array) == tuple:
            if copy:
                return (
                    torch.tensor(array[0]).to(self.device),
                    torch.tensor(array[1]).to(self.device),
                    torch.tensor(array[2]).to(self.device),
                )
            return (
                torch.as_tensor(array[0]).to(self.device),
                torch.as_tensor(array[1]).to(self.device),
                torch.as_tensor(array[2]).to(self.device),
            )
        else:
            if copy:
                return torch.tensor(array).to(self.device)
            return torch.as_tensor(array).to(self.device)
