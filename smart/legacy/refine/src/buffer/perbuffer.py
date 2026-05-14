# Refactored from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py

import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from .buffer import ReplayBuffer


class PERReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        max_step: int,
        max_num_action: int,
        max_num_tet: int,
        tet_idim: int,
        max_num_bbox: int,
        bbox_idim: int,
        alpha: float,
        n_step: int,
        gamma: float,
    ):
        super().__init__(
            capacity,
            max_step,
            max_num_action,
            max_num_tet,
            tet_idim,
            max_num_bbox,
            bbox_idim,
            n_step,
            gamma,
        )
        self.alpha = alpha

        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float("inf") for _ in range(2 * self.capacity)]

        self.max_priority = 1.0
        self.size = 0

    def _set_priority_min(self, idx: int, priority_alpha: float):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(
                self.priority_min[2 * idx], self.priority_min[2 * idx + 1]
            )

    def _set_priority_sum(self, idx: int, priority: float):
        idx += self.capacity
        self.priority_sum[idx] = priority

        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = (
                self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
            )

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum: float):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        return idx - self.capacity

    def store(
        self,
        last_obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        new_obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        done: np.ndarray,
    ):
        transition = super().store(last_obs, action, reward, new_obs, done)

        if transition:
            index = (self.memory_counter - 1) % self.capacity
            priority_alpha = self.max_priority**self.alpha

            self._set_priority_min(index, priority_alpha)
            self._set_priority_sum(index, priority_alpha)

        return transition

    def sample(self, batch_size: int, beta: float):
        assert beta > 0

        samples = {
            "weights": np.zeros(shape=batch_size, dtype=np.float32),
            "indexes": np.zeros(shape=batch_size, dtype=np.int32),
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples["indexes"][i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples["indexes"][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples["weights"][i] = weight / max_weight

        indices = samples["indexes"]

        last_obs = (
            self.tet_obs_mem[indices, :, :],
            self.bbox_obs_mem[indices, :],
            self.step_vec_mem[indices, :],
            self.action_mask_mem[indices, :],
        )
        reward = self.reward_mem[indices, :]
        new_obs = (
            self.next_tet_obs_mem[indices, :, :],
            self.next_bbox_obs_mem[indices, :],
            self.next_step_vec_mem[indices, :],
            self.next_action_mask_mem[indices, :],
        )

        action = self.action_mem[indices, :]
        done = self.done_mem[indices, :]

        data = [last_obs, action, reward, new_obs, done, samples["weights"]]

        return tuple(map(self.to_torch, data)), indices

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)

            priority_alpha = priority**self.alpha

            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
