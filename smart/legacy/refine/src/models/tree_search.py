import copy
import json
import math
import os
import shutil
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from tqdm import tqdm

from ..environment.bbox_environment import MeshBBoxEnv
from ..utils.utils import calculate_reward, set_random_seed

try:
    import smart.rust as smart_rust
except ImportError:
    smart_rust = None

try:
    from smart.action_prior import load_action_prior
except ImportError:
    load_action_prior = None

_RUST_VECTOR_MIN = 128


class MCTSNode:
    num_bbox = 0
    num_action_scale = 0

    def __init__(
        self,
        node_id,
        parent_id=None,
        action_mask=None,
        num_bbox=0,
        num_action_scale=0,
        state_key=None,
    ) -> None:
        self.Q = -sys.float_info.max
        self.reward = -sys.float_info.max
        self.num_vis = 0
        self.node_id = node_id
        self.parent_id = parent_id
        self.state_key = state_key
        self.child_ids: List[int] = []
        self.child_actions: List[int] = []
        if node_id == 0:
            self.set_mcts_option(num_bbox, num_action_scale)

        if action_mask is None:
            self.action_mask = np.zeros(
                self.num_bbox * (6 * self.num_action_scale + 1), dtype=bool
            )
        else:
            self.action_mask = np.copy(action_mask)

        self.untried_actions: List[int] = self.get_untried()

    def addchild(self, action, child_id):
        self.untried_actions.remove(action)
        self.child_ids.append(child_id)
        self.child_actions.append(action)

    def get_untried(self):
        if _using_rust_backend() and len(self.action_mask) >= _RUST_VECTOR_MIN:
            try:
                return [int(action) for action in smart_rust.untried_actions(self.action_mask)]
            except Exception:
                pass
        return np.flatnonzero(~self.action_mask).astype(int).tolist()

    def set_mcts_option(self, num_bbox, num_action_scale):
        MCTSNode.num_bbox = num_bbox
        MCTSNode.num_action_scale = num_action_scale


class MCTSTreeSearch:
    def __init__(self, args, env) -> None:
        set_random_seed(args.seed, using_cuda=False, seed_torch=False)
        os.environ["CHAINER_SEED"] = str(args.seed)

        self.env: MeshBBoxEnv = env
        self.args = args
        self.exp_name = self.env.exp_name
        self.log_path = args.log_path

        # Backup hyperparams
        if not args.debug:
            f = os.path.join(
                os.path.join(self.env.args.result_path, self.exp_name), "hyperparameter"
            )
            os.makedirs(f, exist_ok=True)
            with open(os.path.join(f, "args.json"), "w") as f:
                json.dump(self.env.args.__dict__, f, indent=4)
        # Backup source file
        # if not args.debug:
        #     f = os.path.join(
        #         os.path.join(self.env.args.result_path, self.exp_name), "source"
        #     )
        #     os.makedirs(f, exist_ok=True)
        #     shutil.copytree(
        #         "./src/",
        #         f,
        #         ignore=shutil.ignore_patterns("*.pyc", "__pycache__"),
        #         dirs_exist_ok=True,
        #     )

        if not args.log_path or args.debug:
            self.logger = None
        else:
            tensorboard_path = os.path.join(
                os.path.join(args.log_path, "tensorboard"), self.exp_name
            )
            os.makedirs(args.log_path, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter

            self.logger = SummaryWriter(tensorboard_path)

        self.num_bbox = self.env.num_bbox
        self.num_action_scale = self.env.num_action_scale
        self.action_scale = copy.deepcopy(self.env.action_scale)

        self.action2idx = np.copy(self.env.action2idx)
        self.idx2action = np.copy(self.env.idx2action)

        self.exp_weight = args.exp_w

        self.id2Node: List[MCTSNode] = []
        self.use_transposition_table = bool(getattr(args, "transposition_table", False))
        self.transposition_table_size = max(
            0, int(getattr(args, "transposition_table_size", 8192))
        )
        self.transposition_table = OrderedDict()
        self.transposition_hits = 0

        self.root_id = 0
        self.id2Node.append(
            MCTSNode(
                self.root_id,
                None,
                None,
                self.num_bbox,
                self.num_action_scale,
                state_key=self._state_key(),
            )
        )
        self.node_cnt = 1

        self.not_updated = 0
        self.best_reward = 0.0
        self.best_actions = []
        self.exp_action_reward = np.zeros(
            (self.num_bbox * (6 * self.num_action_scale + 1)), dtype=float
        )
        self.exp_action_cnt = np.zeros(
            (self.num_bbox * (6 * self.num_action_scale + 1)), dtype=int
        )
        self.num_actions = self.num_bbox * (6 * self.num_action_scale + 1)
        self.action_prior_weight = float(getattr(args, "action_prior_weight", 0.0) or 0.0)
        self.puct_prior_weight = float(getattr(args, "puct_prior_weight", 0.0) or 0.0)
        self.action_prior = self._load_action_prior(str(getattr(args, "action_prior_path", "") or ""))
        self.action_prior_logits = self._action_prior_logits_for(range(self.num_actions))
        self._opposite_actions = self._build_opposite_actions()
        self.skip_summary_metrics = bool(getattr(args, "skip_summary_metrics", False))
        self.candidate_trace_path = str(getattr(args, "candidate_trace_path", "") or "")
        self.candidate_trace_top_k = max(0, int(getattr(args, "candidate_trace_top_k", 0) or 0))
        self._mcts_iter_index = 0
        self._rollout_step_index = 0
        if self.candidate_trace_path:
            os.makedirs(os.path.dirname(self.candidate_trace_path) or ".", exist_ok=True)

        if self.skip_summary_metrics:
            self.init_occ = 0.0
            self.init_mov = 0.0
            self.init_bvs = 0.0
            self.init_tov = 0.0
            self.init_covered = 0.0
            self.init_iou = 0.0
        else:
            (
                self.init_occ,
                self.init_mov,
                self.init_bvs,
                self.init_tov,
                self.init_covered,
                self.init_iou,
            ) = self.env.current_state_summary()
        self.init_num_bbox = self.env.num_valid_bboxs()

        if self.logger is not None and not self.skip_summary_metrics:
            self.logger.add_scalar("mcts/max_reward", 0, 0)
            self.logger.add_scalar("mcts/last_reward", 0, 0)
            self.logger.add_scalar("mcts/mov", self.init_mov, 0)
            self.logger.add_scalar("mcts/bvs", self.init_bvs, 0)
            self.logger.add_scalar("mcts/tov", self.init_tov, 0)
            self.logger.add_scalar("mcts/viou", self.init_iou, 0)
            self.logger.add_scalar("mcts/cov", self.init_covered, 0)
            self.logger.add_scalar("mcts/num_bbox", self.init_num_bbox, 0)
        print("MCTS tree initailized, Number of Bbox: %d" % (self.num_bbox))
        self._store_transposition(self.root_id)

    def _load_action_prior(self, path):
        if not path or (self.action_prior_weight == 0.0 and self.puct_prior_weight == 0.0):
            return None
        if not os.path.exists(path):
            warnings.warn("MCTS action_prior_path does not exist: %s" % path)
            return None
        if load_action_prior is not None:
            try:
                return load_action_prior(path)
            except Exception as exc:
                warnings.warn("failed to load MCTS action prior %s: %s" % (path, exc))
                return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            warnings.warn("failed to load MCTS action prior %s: %s" % (path, exc))
            return None
        return _LegacyActionPrior(payload)

    def _prior_context(self):
        if hasattr(self.env, "action_prior_context"):
            try:
                return self.env.action_prior_context()
            except Exception:
                pass
        return {
            "category": str(getattr(self.args, "category", "")),
            "step": int(getattr(self.env, "step_cnt", 0)),
            "max_step": int(getattr(self.args, "max_step", 0)),
            "num_bbox": int(self.num_bbox),
            "num_action_scale": int(self.num_action_scale),
            "action_unit": float(getattr(self.args, "action_unit", 0.0)),
            "cover_penalty": float(getattr(self.args, "cover_penalty", 100.0)),
            "pen_rate": float(getattr(self.env, "pen_rate", 1.0)),
        }

    def _action_prior_logits_for(self, actions):
        actions = [int(action) for action in actions]
        if (self.action_prior_weight == 0.0 and self.puct_prior_weight == 0.0) or self.action_prior is None:
            return np.zeros(len(actions), dtype=float)
        try:
            return np.array(
                self.action_prior.action_logits_for(
                    actions,
                    num_action_scale=self.num_action_scale,
                    context=self._prior_context(),
                ),
                dtype=float,
            )
        except Exception as exc:
            warnings.warn("failed to evaluate MCTS action prior: %s" % exc)
            return np.zeros(len(actions), dtype=float)

    def _state_key(self):
        if hasattr(self.env, "_state_cache_key"):
            return self.env._state_cache_key()
        return None

    def _seed_from_transposition(self, node):
        if not self.use_transposition_table or node.state_key is None:
            return
        try:
            cached = self.transposition_table.pop(node.state_key)
        except KeyError:
            return
        self.transposition_table[node.state_key] = cached
        node.Q = cached["Q"]
        node.reward = cached["reward"]
        node.num_vis = cached["num_vis"]
        self.transposition_hits += 1

    def _store_transposition(self, node_id):
        if not self.use_transposition_table or self.transposition_table_size <= 0:
            return
        node = self.id2Node[node_id]
        if node.state_key is None or node.num_vis <= 0:
            return
        self.transposition_table[node.state_key] = {
            "Q": node.Q,
            "reward": node.reward,
            "num_vis": node.num_vis,
        }
        while len(self.transposition_table) > self.transposition_table_size:
            self.transposition_table.popitem(last=False)

    def _build_opposite_actions(self):
        actions = np.arange(self.num_actions, dtype=int)
        if smart_rust is not None:
            return np.array(
                smart_rust.opposite_actions(self.num_bbox, self.num_action_scale),
                dtype=int,
            )

        per_bbox = 6 * self.num_action_scale + 1
        local = actions % per_bbox
        bbox_idx = actions // per_bbox
        coord_idx = local // self.num_action_scale
        scale_idx = local % self.num_action_scale
        opposite_scale = self.num_action_scale - 1 - scale_idx
        out = bbox_idx * per_bbox + coord_idx * self.num_action_scale + opposite_scale
        out[local == 6 * self.num_action_scale] = actions[local == 6 * self.num_action_scale]
        return out

    def get_opp_action_mask(self, action):
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[self._opposite_actions[int(action)]] = True
        return mask

    def _child_action_mask(self, action, parent_mask=None):
        if _using_rust_backend() and self.num_actions >= _RUST_VECTOR_MIN:
            try:
                parent_data = (
                    None
                    if parent_mask is None
                    else np.asarray(parent_mask, dtype=bool).tolist()
                )
                return np.array(
                    smart_rust.mcts_child_action_mask(
                        self.num_actions,
                        int(action),
                        self.num_action_scale,
                        parent_data,
                    ),
                    dtype=bool,
                )
            except Exception:
                pass
        mask = self.get_opp_action_mask(action)
        if parent_mask is not None:
            mask = mask | parent_mask
        return mask

    def prob_skip_exploration(self, node_id):
        eps = 1e-9
        mx_Q = 0.0
        for i, child_id in enumerate(self.id2Node[node_id].child_ids):
            if self.id2Node[child_id].reward > self.id2Node[node_id].reward:
                mx_Q = max(self.id2Node[child_id].Q, mx_Q)

        prob = mx_Q / (self.best_reward + eps)
        return np.clip(prob, 0, self.args.skip_rate)

    def get_exp_prob(self, actions, scale=100):
        if _using_rust_backend() and len(actions) >= _RUST_VECTOR_MIN:
            try:
                values = [
                    float(self.exp_action_reward[int(action)] * scale)
                    + self.action_prior_weight * float(prior_logit)
                    for action, prior_logit in zip(actions, self._action_prior_logits_for(actions))
                ]
                return np.array(smart_rust.softmax_scaled(values, 1.0), dtype=float)
            except Exception:
                pass
        x = self.exp_action_reward[actions] * scale
        if self.action_prior_weight != 0.0:
            x = x + self.action_prior_weight * self._action_prior_logits_for(actions)
        x = x - np.max(x)
        exp_x = np.exp(x)
        total = exp_x.sum()
        if total == 0:
            return np.ones_like(exp_x) / len(exp_x)
        return exp_x / total

    def _select(self, node_id):
        "Find an unexplored descendent of node"
        path = []
        rewards = []
        while True:
            path.append(node_id)
            assert len(path) - 1 == self.env.step_cnt

            if self.env.done:
                return path, rewards

            if len(self.id2Node[node_id].untried_actions) == 0:
                child_id, action = self._ucb_select(node_id)
                r, obs, done = self.env.step(action)
                rewards.append(r)

                node_id = child_id

            elif self.args.pns and np.random.rand() < self.prob_skip_exploration(node_id):
                child_id, action = self._ucb_select(node_id)
                r, obs, done = self.env.step(action)
                rewards.append(r)

                node_id = child_id
            else:
                if self.args.pns:
                    action = np.random.choice(
                        self.id2Node[node_id].untried_actions,
                        size=1,
                        p=self.get_exp_prob(self.id2Node[node_id].untried_actions),
                    )[0]
                else:
                    sample_id = np.random.randint(
                        len(self.id2Node[node_id].untried_actions), size=1
                    )[0]
                    action = self.id2Node[node_id].untried_actions[sample_id]

                r, obs, done = self.env.step(action)
                rewards.append(r)

                parent_mask = (
                    self.id2Node[node_id].action_mask if self.args.mask_prun else None
                )
                opp_mask = self._child_action_mask(action, parent_mask)

                child_node = MCTSNode(
                    self.node_cnt,
                    node_id,
                    opp_mask,
                    state_key=self._state_key(),
                )
                self._seed_from_transposition(child_node)
                self.id2Node.append(child_node)
                self.id2Node[node_id].addchild(action, self.node_cnt)
                path.append(self.node_cnt)

                self.node_cnt += 1
                self.exp_action = action
                return path, rewards

    def _ucb_select(self, node_id):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        if not self.args.pns:
            assert len(self.id2Node[node_id].untried_actions) == 0

        child_ids = self.id2Node[node_id].child_ids
        parent_visits = self.id2Node[node_id].num_vis
        child_qs = [self.id2Node[idx].Q for idx in child_ids]
        child_visits = [self.id2Node[idx].num_vis for idx in child_ids]
        if _using_rust_backend() and len(child_ids) >= _RUST_VECTOR_MIN and self.puct_prior_weight == 0.0:
            try:
                best_positions = smart_rust.ucb_best_indices(
                    parent_visits, child_qs, child_visits, self.exp_weight
                )
                next_pos = best_positions[np.random.randint(len(best_positions), size=1)[0]]
                return child_ids[next_pos], self.id2Node[node_id].child_actions[next_pos]
            except Exception:
                uct_scores = self._python_ucb_scores(parent_visits, child_qs, child_visits)
        else:
            uct_scores = self._python_ucb_scores(parent_visits, child_qs, child_visits)
        if self.puct_prior_weight != 0.0 and self.action_prior is not None:
            uct_scores = self._add_puct_prior(
                uct_scores,
                parent_visits,
                child_visits,
                self.id2Node[node_id].child_actions,
            )

        mx_uct = max(uct_scores)
        mx_id = []
        mx_action = []
        for i, child_id in enumerate(child_ids):
            if uct_scores[i] == mx_uct:
                mx_id.append(child_id)
                mx_action.append(self.id2Node[node_id].child_actions[i])

        next_idx = np.random.randint(len(mx_id), size=1)[0]

        return mx_id[next_idx], mx_action[next_idx]

    def _python_ucb_scores(self, parent_visits, child_qs, child_visits):
        child_qs = np.array(child_qs, dtype=float)
        child_visits = np.array(child_visits, dtype=float)
        if parent_visits <= 0:
            return np.full(len(child_qs), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            uct_scores = child_qs + self.exp_weight * np.sqrt(
                2 * math.log(parent_visits) / child_visits
            )
        uct_scores[child_visits <= 0] = np.inf
        return uct_scores

    def _add_puct_prior(self, uct_scores, parent_visits, child_visits, child_actions):
        if len(child_actions) == 0 or parent_visits <= 0:
            return uct_scores
        logits = self._action_prior_logits_for(child_actions)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        total = probs.sum()
        if total <= 0.0:
            return uct_scores
        probs = probs / total
        child_visits = np.array(child_visits, dtype=float)
        prior_bonus = self.puct_prior_weight * probs * math.sqrt(float(parent_visits)) / (1.0 + child_visits)
        return np.array(uct_scores, dtype=float) + prior_bonus

    def _action_trace_fields(self, action):
        action = int(action)
        per_bbox = 6 * int(self.num_action_scale) + 1
        bbox_idx = action // per_bbox
        local = action % per_bbox
        if local == per_bbox - 1:
            coord_idx = 6
            scale_idx = 0
        else:
            coord_idx = local // int(self.num_action_scale)
            scale_idx = local % int(self.num_action_scale)
        return bbox_idx, coord_idx, scale_idx

    def _trace_rollout_candidates(self, candidates, selected_action, node_id):
        if not self.candidate_trace_path or not candidates:
            return
        ordered = sorted(
            (
                (int(bbox_idx), int(action), float(reward))
                for bbox_idx, action, reward in candidates
                if action is not None and int(action) >= 0 and np.isfinite(float(reward))
            ),
            key=lambda item: item[2],
            reverse=True,
        )
        if self.candidate_trace_top_k > 0:
            ordered = ordered[: self.candidate_trace_top_k]
        if not ordered:
            return
        context = self._prior_context()
        rows = []
        for rank, (bbox_idx, action, reward) in enumerate(ordered):
            decoded_bbox_idx, coord_idx, scale_idx = self._action_trace_fields(action)
            rows.append(
                {
                    "schema_version": 3,
                    "record_type": "mcts_candidate",
                    "source": "mcts_rollout_bbox_best",
                    "category": str(context.get("category", "")),
                    "mesh": str(context.get("mesh", "")),
                    "reward_backend": str(context.get("reward_backend", "")),
                    "manifold_volume_method": str(context.get("manifold_volume_method", "")),
                    "mcts_iter": int(self._mcts_iter_index),
                    "rollout_step": int(self._rollout_step_index),
                    "node_id": int(node_id),
                    "rank": int(rank),
                    "action": int(action),
                    "bbox_idx": int(decoded_bbox_idx),
                    "candidate_bbox_idx": int(bbox_idx),
                    "coord_idx": int(coord_idx),
                    "scale_idx": int(scale_idx),
                    "num_bbox": int(self.num_bbox),
                    "num_action_scale": int(self.num_action_scale),
                    "actions_per_bbox": int(6 * self.num_action_scale + 1),
                    "action_unit": float(context.get("action_unit", 0.0) or 0.0),
                    "reward": float(reward),
                    "selected": bool(int(action) == int(selected_action)),
                    "bvs": float(context.get("bvs", 1.0) or 1.0),
                    "volume_sum": float(context.get("volume_sum", 0.0) or 0.0),
                    "cover_penalty": float(context.get("cover_penalty", 100.0) or 100.0),
                    "pen_rate": float(context.get("pen_rate", 1.0) or 1.0),
                    "max_step": int(context.get("max_step", 0) or 0),
                    "step": int(context.get("step", 0) or 0),
                }
            )
        with open(self.candidate_trace_path, "a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")

    def _simulate(self, path: List[int], rewards: List[float] = []):
        grd_cnt = 0
        mask_bbox = [1 for _ in range(self.num_bbox)]
        node_id = path[-1]
        fused_rollout_step = (
            getattr(self.args, "mcts_fused_rollout_step", False)
            and hasattr(self.env, "_bridge_mcts_greedy_rollout_step")
        )
        while not self.env.done:
            self._rollout_step_index += 1
            if fused_rollout_step:
                fused = self.env._bridge_mcts_greedy_rollout_step(mask_bbox)
                if fused is not None:
                    mx_ac, mx_reward, reward, done, next_mask = fused
                    mask_bbox = next_mask
                    if mx_ac is None or mx_reward <= 0:
                        break
                    if not np.isfinite(reward) or reward <= 0:
                        break
                    grd_cnt += 1
                    rewards.append(reward)

                    if self.args.grdexp:
                        node_id = self._append_greedy_expansion_node(
                            node_id, path, mx_ac
                        )
                    continue

            mx_ac = None
            mx_reward = -sys.float_info.max
            candidate_rows = []
            for idx in range(self.num_bbox):
                if mask_bbox[idx]:
                    ith_grd_action, ith_reward = self.env.ith_bbox_greedy_sample(idx)
                    candidate_rows.append((idx, ith_grd_action, ith_reward))
                    if mx_reward < ith_reward:
                        mx_reward = ith_reward
                        mx_ac = ith_grd_action
                    if ith_reward < 0:
                        mask_bbox[idx] = 0

            self._trace_rollout_candidates(candidate_rows, mx_ac if mx_ac is not None else -1, node_id)
            if mx_reward <= 0:
                break
            r, obs, done = self.env.step(mx_ac, apply=1)
            if not np.isfinite(r) or r <= 0:
                break
            if not math.isclose(r, mx_reward, rel_tol=1e-9, abs_tol=1e-12):
                mx_reward = r
            grd_cnt += 1
            rewards.append(r)

            if self.args.grdexp:
                node_id = self._append_greedy_expansion_node(node_id, path, mx_ac)

        return path, rewards, grd_cnt

    def _append_greedy_expansion_node(self, node_id, path, action):
        self.id2Node[node_id].untried_actions = [action]
        action_mask = self._single_untried_action_mask(action)
        self.id2Node[node_id].action_mask = np.copy(action_mask)

        opp_mask = self._child_action_mask(action)

        child_node = MCTSNode(
            self.node_cnt,
            node_id,
            opp_mask,
            state_key=self._state_key(),
        )
        self._seed_from_transposition(child_node)
        self.id2Node.append(child_node)
        self.id2Node[node_id].addchild(action, self.node_cnt)
        path.append(self.node_cnt)
        next_node_id = self.node_cnt

        self.node_cnt += 1
        return next_node_id

    def update_exp_action_reward(self, reward_sum):
        cnt = self.exp_action_cnt[self.exp_action]
        self.exp_action_reward[self.exp_action] = self.exp_action_reward[
            self.exp_action
        ] / (cnt + 1) * cnt + reward_sum / (cnt + 1)
        self.exp_action_cnt[self.exp_action] += 1

    def _backpropagate(self, path, rewards):
        reward_sum = 0
        if len(rewards) == 0:
            reward_sum = 0
        else:
            reward_sum = calculate_reward(rewards, self.args.gamma)

        if self.args.pns:
            self.update_exp_action_reward(reward_sum - self.best_reward)
        for node_id in reversed(path):
            if self.id2Node[node_id].num_vis == 0:
                self.id2Node[node_id].reward = reward_sum
            self.id2Node[node_id].num_vis += 1
            self.id2Node[node_id].Q = max(reward_sum, self.id2Node[node_id].Q)
            self._store_transposition(node_id)

    def _select_best(self, path, rewards, num_iter, grd_cnt):

        if len(rewards) == 0:
            reward_sum = 0
        else:
            reward_sum = calculate_reward(rewards, self.args.gamma)

        num_bbox = self.env.num_valid_bboxs()
        self.not_updated += 1
        if self.logger is not None:
            self.logger.add_scalar("mcts/last_reward", reward_sum, num_iter)
            self.logger.add_scalar("mcts/max_reward", self.best_reward, num_iter)
            self.logger.add_scalar("mcts/time", time.time() - self.start_time, num_iter)
            # self.logger.add_scalar(
            #     "mcts/g1_reward", calculate_reward(rewards, 1), num_iter
            # )
            # self.logger.add_scalar("mcts/num_bbox", num_bbox, num_iter)
            # self.logger.add_scalar("mcts/env_step", self.env.step_cnt, num_iter)
            # self.logger.add_scalar("mcts/grd_step", grd_cnt, num_iter)

        if reward_sum > self.best_reward:
            self.best_reward = reward_sum
            self.best_actions = copy.deepcopy(path)
            self.not_updated = 0
            if self.skip_summary_metrics:
                mov = bvs = tov = covered = iou = 0.0
            else:
                _, mov, bvs, tov, covered, iou = self.env.current_state_summary()

            self.env.render(num_iter)
            if self.logger is not None and not self.skip_summary_metrics:
                self.logger.add_scalar("mcts/mov", mov, num_iter)
                self.logger.add_scalar("mcts/bvs", bvs, num_iter)
                self.logger.add_scalar("mcts/tov", tov, num_iter)
                self.logger.add_scalar("mcts/viou", iou, num_iter)
                self.logger.add_scalar("mcts/cov", covered, num_iter)

    def run_mcts(self, num_iter):
        self.start_time = time.time()
        self.env.reset()
        for ith in tqdm(range(num_iter)):
            self._mcts_iter_index = ith + 1
            self._rollout_step_index = 0
            path, rewards = self._select(self.root_id)
            assert len(path) - 1 == len(rewards)

            path, rewards, grd_cnt = self._simulate(path, rewards)
            self._backpropagate(path, rewards)
            self._select_best(path, rewards, ith + 1, grd_cnt)

            self.env.reset()
            no_reward_stop_after = int(
                getattr(self.args, "mcts_no_reward_stop_after", 101)
            )
            if (
                no_reward_stop_after >= 0
                and ith + 1 > no_reward_stop_after
                and self.best_reward < 1e-2
            ):
                print("Ended of early stopping")
                break

            if self.not_updated > 400:
                print("Ended of early stopping")
                break

        if self.use_transposition_table:
            print("MCTS transposition hits: %d" % (self.transposition_hits))
        if hasattr(self.env, "_manifold_stateful_cache_stats"):
            stats = self.env._manifold_stateful_cache_stats()
            if stats:
                print("MCTS manifold_stateful cache: %s" % json.dumps(stats, sort_keys=True))
        if hasattr(self.env, "_candidate_prefilter_report"):
            stats = self.env._candidate_prefilter_report()
            if stats:
                print("MCTS candidate prefilter: %s" % json.dumps(stats, sort_keys=True))

    def _single_untried_action_mask(self, action):
        if _using_rust_backend() and self.num_actions >= _RUST_VECTOR_MIN:
            try:
                return np.array(
                    smart_rust.single_untried_action_mask(self.num_actions, int(action)),
                    dtype=bool,
                )
            except Exception:
                pass
        action_mask = np.ones(self.num_actions, dtype=bool)
        action_mask[int(action)] = 0
        return action_mask


class _LegacyActionPrior:
    def __init__(self, payload):
        self.coord_scale_logits = payload.get("coord_scale_logits", payload.get("priors", {}))
        self.action_logits = {
            int(key): float(value)
            for key, value in payload.get("action_logits", {}).items()
        }
        self.default_logit = float(payload.get("default_logit", 0.0))

    def action_logits_for(self, actions, *, num_action_scale, context=None):
        del context
        per_bbox = 6 * int(num_action_scale) + 1
        out = []
        for action in actions:
            action = int(action)
            if action in self.action_logits:
                out.append(self.action_logits[action])
                continue
            local = action % per_bbox
            if local == per_bbox - 1:
                key = "6:0"
            else:
                key = "%d:%d" % (local // int(num_action_scale), local % int(num_action_scale))
            out.append(float(self.coord_scale_logits.get(key, self.default_logit)))
        return out


def _using_rust_backend():
    return smart_rust is not None and smart_rust.using_rust()
