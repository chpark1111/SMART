from __future__ import annotations

import random
from itertools import zip_longest
from typing import Iterable, List

import numpy as np

try:
    import smart.native as smart_native
except ImportError:
    smart_native = None


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def soft_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    """
    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Soft update", between 0 and 1)
    """
    import torch

    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer:
    :param learning_rate:
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def get_linear_schedular(start: float, end: float, end_fraction: float):
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return:
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def set_random_seed(seed: int, using_cuda: bool = False, seed_torch: bool = True) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    if not seed_torch:
        return

    import torch

    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_reward(rewards: List[float], gamma: float) -> float:
    if smart_native is not None and len(rewards) >= 128:
        try:
            return smart_native.discounted_reward(rewards, gamma)
        except Exception:
            pass

    ret = 0
    for i in range(len(rewards) - 1, -1, -1):
        ret = ret * gamma + rewards[i]

    return ret


def print_matrix(matrix: torch.Tensor) -> None:
    import torch

    np.set_printoptions(precision=5, linewidth=np.inf)
    matrix = matrix.cpu().numpy()
    print(matrix)
