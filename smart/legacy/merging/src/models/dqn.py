import json
import os
import shutil
import time
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..buffer.buffer import ReplayBuffer
from ..buffer.perbuffer import PERReplayBuffer
from ..environment.multi_tetmesh_environment import MultiTetMeshEnv
from ..environment.tetmesh_environment import TetMeshEnv
from ..utils.utils import (
    calculate_reward,
    get_linear_schedular,
    print_matrix,
    set_random_seed,
    soft_update,
    update_learning_rate,
)
from .policies import TetMeshPolicy


class DQN:
    """
    Deep Q-Network (DQN) for Unsupervised 3D Mesh Segmentation

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param max_num_tet: number of maximum tetrahedrals in data
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param eval_freq: Test the model every eval_freq updates
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param log_path: the log path location for tensorboard (if None, no logging)
    :param tag: string tag for the experiments
    :param seed: Seed for the pseudo random generators
    :param debug: Do not log or render when debugging
    """

    def __init__(
        self,
        policy: Callable[
            [int, int, int, float, str, bool, bool, bool, int], TetMeshPolicy
        ],
        env: Union[Type[TetMeshEnv], Type[MultiTetMeshEnv]],
        learning_rate: float = 1e-3,
        learning_rate_final: float = 1e-5,
        learning_rate_fraction: float = 0.3,
        buffer_size: int = 100000,
        learning_starts: int = 5000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        eval_freq: int = 1000,
        gradient_steps: int = 1,
        target_update_interval: int = 5000,
        exploration_fraction: float = 0.3,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        agent: str = "attn",
        ddqn: bool = True,
        duel: bool = True,
        noisy: bool = True,
        per: bool = True,
        n_step: int = 1,
        alpha: float = 0.6,
        beta: float = 1.0,
        n_head: int = 1,
        only_nearby: bool = False,
        log_path: Optional[str] = None,
        tag: Optional[str] = "",
        seed: Optional[int] = 7777,
        debug: bool = False,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            set_random_seed(
                seed, using_cuda=(self.device.type == torch.device("cuda:0").type)
            )
            os.environ["CHAINER_SEED"] = str(seed)

        self.env = env
        self.noisy = noisy
        self.duel = duel
        self.n_step = n_step
        max_num_tet, tet_idim, max_init_part, part_idim = self.env.environment_info()

        self.policy = policy(
            tet_idim,
            part_idim,
            max_init_part,
            learning_rate,
            agent,
            n_head,
            only_nearby,
            duel,
            noisy,
            self.env.args.sample_part,
        )
        if per:
            # We use a power of 2 for capacity in PER!
            capacity = 1
            while capacity < buffer_size:
                capacity *= 2
            buffer_size = capacity

            self.replay_buffer = PERReplayBuffer(
                buffer_size,
                max_num_tet,
                tet_idim,
                max_init_part,
                part_idim,
                only_nearby,
                alpha,
                1,
                gamma,
            )
            self.greedy_replay_buffer = PERReplayBuffer(
                buffer_size // 2,
                max_num_tet,
                tet_idim,
                max_init_part,
                part_idim,
                only_nearby,
                alpha,
                1,
                gamma,
            )
            if self.n_step != 1:
                self.nstep_replay_buffer = PERReplayBuffer(
                    buffer_size,
                    max_num_tet,
                    tet_idim,
                    max_init_part,
                    part_idim,
                    only_nearby,
                    alpha,
                    n_step,
                    gamma,
                )
            self.beta = beta
            self.init_beta = beta
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size,
                max_num_tet,
                tet_idim,
                max_init_part,
                part_idim,
                only_nearby,
                1,
                gamma,
            )
            self.greedy_replay_buffer = ReplayBuffer(
                buffer_size // 2,
                max_num_tet,
                tet_idim,
                max_init_part,
                part_idim,
                only_nearby,
                1,
                gamma,
            )
            if self.n_step != 1:
                self.nstep_replay_buffer = ReplayBuffer(
                    buffer_size,
                    max_num_tet,
                    tet_idim,
                    max_init_part,
                    part_idim,
                    only_nearby,
                    1,
                    gamma,
                )
        self.per = per

        self.max_num_tet = max_num_tet
        self.max_init_part = max_init_part
        print(
            "Max number of tetrahedral: %d, Max number of initial partitions %d"
            % (self.max_num_tet, self.max_init_part)
        )

        self.tau = tau
        self.gamma = gamma
        if n_step != 1:
            self.n_gamma = self.gamma**n_step
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps

        self.train_freq = train_freq
        self.last_obs = None
        self.learning_rate = learning_rate
        self.learning_rate_final = learning_rate_final
        self.learning_rate_fraction = learning_rate_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = 0.0
        self.current_progress_remaining = 1
        self.agent = agent
        self.ddqn = ddqn

        self.lr_schedule = get_linear_schedular(
            self.learning_rate, self.learning_rate_final, learning_rate_fraction
        )
        self.exploration_schedule = get_linear_schedular(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        self.target_update_interval = target_update_interval

        self.eval_freq = eval_freq
        self.learning_starts = learning_starts
        self.n_updates = 0  # number of total updates
        self.episode_num = 0  # number of total episodes called
        self.num_timesteps = 0  # number of timesteps
        self.n_calls = 0  # number of call of on_step
        self.max_avg_reward = 0  # max avg5 reward on test
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.q_net, self.q_net_target = self.policy.q_net, self.policy.q_net_target

        self.debug = debug

        self.exp_name = f"%s%s%s_%snstep%d_%s_tilted%dmov%.2ftov%.2fnearby%dlr{self.learning_rate:.1e}lre{self.learning_rate_final:.1e}lrf%.2fbfs%dbch%dtf%dls%def%.2fseed%d%s_" % (
            "ddqn" if self.ddqn else "dqn",
            "_duel" if self.duel else "",
            "_noisy" if self.noisy else "",
            "per_alpha%.3f_beta%.3f_" % (alpha, beta) if self.per else "",
            self.n_step,
            self.agent,
            (1 if self.env.args.tilted else 0),
            (self.env.args.mov_alpha if self.env.args.mov else 0),
            (self.env.args.tov_beta if self.env.args.tov else 0),
            only_nearby,
            self.learning_rate_fraction,
            buffer_size,
            self.batch_size,
            self.train_freq,
            self.learning_starts,
            self.exploration_fraction,
            seed,
            ("_" + tag),
        ) + (
            "mixed" if self.env.num_meshes > 1 else self.env.name[:10]
        )

        self.env.exp_name = self.exp_name
        self.log_path = log_path

        # Backup hyperparams
        if not debug:
            f = os.path.join(
                os.path.join(self.env.args.result_path, self.exp_name), "hyperparameter"
            )
            os.makedirs(f, exist_ok=True)
            with open(os.path.join(f, "args.json"), "w") as f:
                json.dump(self.env.args.__dict__, f, indent=4)
        # Backup source file
        if not debug:
            f = os.path.join(
                os.path.join(self.env.args.result_path, self.exp_name), "source"
            )
            os.makedirs(f, exist_ok=True)
            shutil.copytree(
                "./src/",
                f,
                ignore=shutil.ignore_patterns("*.pyc", "__pycache__"),
                dirs_exist_ok=True,
            )

        # Todo: add W&B support
        if log_path is None or debug:
            self.logger = None
        else:
            tensorboard_path = os.path.join(
                os.path.join(log_path, "tensorboard"), self.exp_name
            )
            os.makedirs(log_path, exist_ok=True)
            self.logger = SummaryWriter(tensorboard_path)

    def on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self.n_calls += 1
        if self.n_calls % self.target_update_interval == 0:
            soft_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self.current_progress_remaining)
        if self.logger is not None:
            self.logger.add_scalar(
                "rollout/exploration_rate", self.exploration_rate, self.n_calls
            )

    def calculate_loss(self, obs, action, reward, next_obs, done, n_step_loss=False):

        with torch.no_grad():
            gamma = self.n_gamma if n_step_loss else self.gamma
            next_q_values = self.q_net_target(next_obs).view(
                -1, self.max_init_part * self.max_init_part
            )

            if self.ddqn:
                best_action = (
                    self.q_net(next_obs)
                    .view(-1, self.max_init_part * self.max_init_part)
                    .max(dim=-1)[1]
                    .unsqueeze(-1)
                )

                target_q_values = reward + (1 - done) * gamma * torch.gather(
                    next_q_values, dim=-1, index=best_action.long()
                )
            else:
                next_q_values, _ = next_q_values.max(dim=-1)
                next_q_values = next_q_values.unsqueeze(-1)

                target_q_values = reward + (1 - done) * gamma * next_q_values

        # Get current Q-values estimates
        current_q_values = self.q_net(obs).view(
            -1, self.max_init_part * self.max_init_part
        )

        # Retrieve the q-values for the actions from the replay buffer
        action = (action[:, 0] * self.max_init_part + action[:, 1]).unsqueeze(-1)
        current_q_values = torch.gather(current_q_values, dim=-1, index=action.long())

        # Compute Huber loss (less sensitive to outliers) vs L2 loss
        if self.per:
            loss = F.smooth_l1_loss(
                current_q_values, target_q_values, reduction="none"
            ).squeeze()
        else:
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

        return loss

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            if self.per:
                (
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                    weights,
                ), indices = self.replay_buffer.sample(batch_size, self.beta)
                (
                    gd_obs,
                    gd_action,
                    gd_reward,
                    gd_next_obs,
                    gd_done,
                    gd_weights,
                ), gd_indices = self.greedy_replay_buffer.sample(batch_size, self.beta)

                element_loss = self.calculate_loss(
                    obs, action, reward, next_obs, done, False
                )
                gd_element_loss = self.calculate_loss(
                    gd_obs, gd_action, gd_reward, gd_next_obs, gd_done, False
                )

                if self.n_step != 1:
                    (
                        obs,
                        action,
                        reward,
                        next_obs,
                        done,
                    ) = self.nstep_replay_buffer.sample_batch_from_idxs(indices)
                    element_n_loss = self.calculate_loss(
                        obs, action, reward, next_obs, done, True
                    )
                    element_loss += element_n_loss

                loss = torch.mean(weights * element_loss)
                gd_loss = torch.mean(gd_weights * gd_element_loss)
                loss += gd_loss
                # loss based prioritization
                new_priorities = np.abs(element_loss.detach().cpu().numpy()) + 1e-7
                self.replay_buffer.update_priorities(indices, new_priorities)

                gd_new_priorities = np.abs(gd_element_loss.detach().cpu().numpy()) + 1e-7
                self.greedy_replay_buffer.update_priorities(gd_indices, gd_new_priorities)
            else:
                (
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                ), indices = self.replay_buffer.sample(batch_size)
                (
                    gd_obs,
                    gd_action,
                    gd_reward,
                    gd_next_obs,
                    gd_done,
                ), gd_indices = self.greedy_replay_buffer.sample(batch_size)

                loss = self.calculate_loss(obs, action, reward, next_obs, done, False)
                gd_loss = self.calculate_loss(
                    gd_obs, gd_action, gd_reward, gd_next_obs, gd_done, False
                )
                loss += gd_loss

                if self.n_step != 1:
                    (
                        last_obs,
                        action,
                        reward,
                        new_obs,
                        done,
                    ) = self.nstep_replay_buffer.sample_batch_from_idxs(indices)
                    n_loss = self.calculate_loss(
                        last_obs, action, reward, new_obs, done, True
                    )
                    loss += n_loss

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Update learning rate according to schedule
        self.update_learning_rate()
        # Update beta
        self.beta = self.init_beta + min(1.0, 1 - self.current_progress_remaining) * (
            1.0 - self.init_beta
        )
        if self.logger is not None:
            self.logger.add_scalar("train/beta", self.beta, self.n_updates)
        # Increase update counter
        self.n_updates += gradient_steps
        if self.logger is not None:
            self.logger.add_scalar("train/loss", np.mean(losses), self.n_updates)

    def test(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        avg_reward = []
        avg_left_part = []
        self.last_obs = self.env.reset(change_mesh=False)
        for idx in range(self.env.num_meshes):
            rewards = []
            obs = self.last_obs
            if not self.debug:
                self.env.render(self.n_updates)
            while 1:

                action = self.policy.predict(obs)

                reward, new_obs, done = self.env.step(action, apply=1)
                obs = new_obs

                rewards.append(reward)

                if done:
                    break

            if not self.debug:
                self.env.render(self.n_updates)

            avg_reward.append(calculate_reward(rewards, self.gamma))
            avg_left_part.append(len(self.env.left_part))

            self.last_obs = self.env.reset()

        avg_reward = np.mean(avg_reward)
        avg_left_part = np.mean(avg_left_part)

        if self.logger is not None:
            self.logger.add_scalar("test/avg_reward", avg_reward, self.n_updates)
            self.logger.add_scalar(
                "test/avg_left_partitions", avg_left_part, self.n_updates
            )

        if avg_reward > 0 and avg_reward >= self.max_avg_reward:
            self.max_avg_reward = avg_reward
            self.save_params()

    def predict(
        self,
        observation: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Tuple[int, int]:
        """
        Epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if np.random.rand() < self.exploration_rate:
            action = self.env.random_sample()
            # action = self.env.greedy_sample()
        else:
            action = self.policy.predict(observation, mask_exit=False)
        return action

    def learn(
        self,
        total_timesteps: int,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        reset_num_timesteps: bool = True,
    ):

        self._setup_learn(
            total_timesteps,
            self.eval_freq if eval_freq == -1 else eval_freq,
            reset_num_timesteps,
        )

        with tqdm(total=self.total_timesteps) as pbar:
            while self.num_timesteps < self.total_timesteps:
                self.collect_rollouts(
                    self.env,
                    train_freq=self.train_freq,
                    learning_starts=self.learning_starts,
                    pbar=pbar,
                )

                self.test()

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_freq: int = 500,
        reset_num_timesteps: bool = True,
    ):

        if reset_num_timesteps:
            self.num_timesteps = 0
            self.episode_num = 0

        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq

        if reset_num_timesteps or self.last_obs is None:
            self.last_obs = self.env.reset(change_mesh=False)

    def collect_rollouts(
        self,
        env: TetMeshEnv,
        train_freq: int,
        learning_starts: int = 0,
        pbar: Union[tqdm, None] = None,
    ) -> None:
        """
        Collect experiences and store them into a ReplayBuffer.

        :param env: The training environment
        :param train_freq: TrainFreq being an integer greater than 0 which is number of episodes to collect.
        :param replay_buffer:
        :param learning_starts: Number of steps before learning for the warm-up phase.
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_episodes = 0

        assert isinstance(env, TetMeshEnv) or isinstance(
            env, MultiTetMeshEnv
        ), "You must pass a TetMeshEnv"
        assert train_freq > 0, "Should collect at least one step or episode."

        self.last_obs = self.env.reset(change_mesh=False)
        while train_freq * self.env.num_meshes > num_collected_episodes:
            cnt = 0
            rewards = []
            greedy_rewards = []
            while 1:
                greedy_action = self.env.local_greedy_sample()

                reward, new_obs, done = env.step(greedy_action, apply=0, obs=True)

                self.store_transition(greedy_action, reward, new_obs, done, greedy=1)
                greedy_rewards.append(reward)

                # Select action randomly or according to policy
                action = self.sample_action(learning_starts)

                reward, new_obs, done = env.step(action, apply=1)
                rewards.append(reward)

                # Store data in replay buffer
                self.store_transition(action, reward, new_obs, done)
                self.last_obs = new_obs

                if pbar is not None:
                    pbar.update(1)

                self.num_timesteps += 1
                cnt += 1

                if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                    gradient_steps = (
                        self.gradient_steps
                        if self.gradient_steps >= 0
                        else self.train_freq
                    )

                    if gradient_steps > 0:
                        self.train(
                            batch_size=self.batch_size, gradient_steps=gradient_steps
                        )
                self.update_current_progress_remaining(
                    self.num_timesteps, self.total_timesteps
                )
                self.on_step()
                if self.logger is not None:
                    self.logger.add_scalar(
                        "rollout/reward",
                        rewards[-1],
                        self.num_timesteps,
                    )
                    self.logger.add_scalar(
                        "rollout/greedy_reward",
                        greedy_rewards[-1],
                        self.num_timesteps,
                    )

                if done:
                    break

            num_collected_episodes += 1
            self.episode_num += 1

            if self.logger is not None:
                self.logger.add_scalar(
                    "rollout/sum_reward",
                    calculate_reward(rewards, self.gamma),
                    self.episode_num,
                )
                self.logger.add_scalar("rollout/trajectory_length", cnt, self.episode_num)

            self.last_obs = env.reset()

    def store_transition(
        self,
        action: Tuple[int, int],
        reward: float,
        new_obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        done: int,
        greedy=False,
    ) -> None:

        action = np.array(list(action))
        reward = np.array([reward])
        done = np.array([done])

        if greedy:
            self.greedy_replay_buffer.store(self.last_obs, action, reward, new_obs, done)
        else:
            if self.n_step != 1:
                transition = self.nstep_replay_buffer.store(
                    self.last_obs, action, reward, new_obs, done
                )
            else:
                transition = (self.last_obs, action, reward, new_obs, done)

            if transition:
                self.replay_buffer.store(*transition)

    def sample_action(
        self,
        learning_starts: int,
    ) -> Tuple[int, int]:
        """
        Sample an action according to the exploration policy.
        This is done by sampling a random action (from a uniform distribution over the action space)

        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            action = self.env.random_sample()
        else:
            if self.env.args.noisy:
                self.policy.q_net.module.reset_noise()
                self.policy.q_net_target.module.reset_noise()

                action = self.policy.predict(self.last_obs, mask_exit=False)
            else:
                action = self.predict(self.last_obs)

        # Add scaling action in here
        return action

    def update_current_progress_remaining(
        self, num_timesteps: int, total_timesteps: int
    ) -> None:
        self.current_progress_remaining = 1.0 - float(num_timesteps) / float(
            total_timesteps
        )

    def update_learning_rate(self) -> None:
        # self.policy.lr_scheduler.step()
        update_learning_rate(
            self.policy.optimizer, self.lr_schedule(self.current_progress_remaining)
        )

        self.learng_rate = self.policy.optimizer.param_groups[0]["lr"]
        if self.logger is not None:
            self.logger.add_scalar(
                "train/learning_rate", self.learng_rate, self.n_updates
            )

    def save_params(self) -> None:
        save_path = os.path.join(
            os.path.join(self.env.args.result_path, self.exp_name), "checkpoint"
        )
        os.makedirs(save_path, exist_ok=True)

        torch.save(
            self.policy.q_net.state_dict(),
            os.path.join(save_path, "qnet_%d.pt" % (self.n_updates)),
        )
        torch.save(
            self.policy.q_net_target.state_dict(),
            os.path.join(save_path, "qnet_target_%d.pt" % (self.n_updates)),
        )
        torch.save(
            self.policy.optimizer.state_dict(),
            os.path.join(save_path, "optimizer_%d.pt" % (self.n_updates)),
        )
        # torch.save(self.policy.lr_scheduler.state_dict(), os.path.join(save_path, "lr_scheduler_%d.pt"%(self.n_updates)))

    def load_params(self, load_path: str, load_idx: int) -> None:
        pt = torch.load(os.path.join(load_path, "qnet_%d.pt" % (load_idx)))
        self.policy.q_net.load_state_dict(pt)

        pt = torch.load(os.path.join(load_path, "qnet_target_%d.pt" % (load_idx)))
        self.policy.q_net_target.load_state_dict(pt)

        pt = torch.load(os.path.join(load_path, "optimizer_%d.pt" % (load_idx)))
        self.policy.optimizer.load_state_dict(pt)

        # pt = torch.load(os.path.join(load_path, "lr_scheduler_%d.pt"%(load_idx)))
        # self.policy.lr_scheduler.load_state_dict(pt)

    def eval(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        if self.last_obs is None:
            self.last_obs = self.env.reset()

        for idx in range(self.env.num_meshes):
            self.env.exp_name = (
                "eval_bspseg_final_k%dtilted%d_mov%.3ftov%.3fonly_nearby%d_%s"
                % (
                    self.env.args.final_k,
                    (1 if self.env.args.tilted else 0),
                    (self.env.args.mov_alpha if self.env.args.mov else 0),
                    (self.env.args.tov_beta if self.env.args.tov else 0),
                    self.env.args.only_nearby,
                    self.env.name[0][0:10],
                )
            )

            init_part = len(self.env.left_part)
            assert (
                self.env.args.final_k <= init_part
            ), "Target final K is bigger than initial partition"

            rewards = []
            obs = self.last_obs
            self.env.render()

            st = time.time()
            while 1:
                action = self.policy.predict(obs, mask_exit=True)

                reward, new_obs, done = self.env.step(action, apply=1)
                obs = new_obs
                rewards.append(reward)

                if len(self.env.left_part) <= self.env.args.final_k:
                    break
            et = time.time()

            self.env.render()
            final_occ = self.env.compute_current_occupancy()
            sum_reward = calculate_reward(rewards, self.gamma)
            occ, mov, tov = self.env.current_state_summary()
            print(
                "Shape %s (%d -> %s %d), Final metric: %f, Improved metric: %f, Elapsed time: %f"
                % (
                    self.env.name[0][:10],
                    init_part,
                    "Target partition" if self.env.args.final_k else "Find partition",
                    len(self.env.left_part),
                    final_occ,
                    sum_reward,
                    et - st,
                )
            )
            print("Metrics after merging, OCC: %f, MOV: %f, TOV: %f" % (occ, mov, tov))
            self.last_obs = self.env.reset()

    def debug_test(self) -> None:
        self.policy.set_training_mode(False)

        reward = 0
        trj_len = 0

        rewards = []
        actions = []
        greedy_actions = []
        q_value_list = []
        if self.last_obs is None:
            self.last_obs = self.env.reset()

        for idx in range(self.env.num_meshes):
            self.env.exp_name = "debug_only_nearby%d_maxP%d_%s" % (
                self.env.args.only_nearby,
                len(self.env.left_part),
                self.env.name,
            )
            self.env.render()

            obs = self.last_obs
            while 1:
                obs = self.policy.obs2ten(obs)
                with torch.no_grad():
                    q_values = self.q_net(obs)
                    # Greedy action
                    q_values = q_values.squeeze()
                    action = int(q_values.argmax() / self.policy.num_init_part), int(
                        q_values.argmax() % self.policy.num_init_part
                    )

                    q_value_list.append(q_values)
                    actions.append(action)
                    greedy_actions.append(self.env.greedy_sample())

                reward, new_obs, done = self.env.step(action, apply=1)
                obs = new_obs

                rewards.append(reward)
                trj_len += 1

                if done:
                    break
            self.env.render()

            reward = calculate_reward(rewards, self.gamma)
            for i in range(len(rewards)):
                print(
                    "Greedy Action: ",
                    greedy_actions[i],
                    "Greedy Q-value: ",
                    q_value_list[i][greedy_actions[i][0], greedy_actions[i][1]],
                    "Action: ",
                    actions[i],
                    "Action Q-value: ",
                    q_value_list[i][actions[i][0], actions[i][1]],
                    "Reward: ",
                    rewards[i],
                )
                print("Q value matrix")
                print_matrix(q_value_list[i])
            print("Reward sum: ", reward)

            self.last_obs = self.env.reset()
