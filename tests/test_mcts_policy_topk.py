from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from argparse import Namespace

_fake_pymanifold = types.ModuleType("pymanifold")
_fake_pymanifold.__spec__ = ModuleSpec("pymanifold", loader=None)
sys.modules.setdefault("pymanifold", _fake_pymanifold)

from smart.legacy.refine.src.models.tree_search import MCTSNode, MCTSTreeSearch


class _FakePrior:
    def __init__(self, logits: dict[int, float], values: dict[int, float] | None = None) -> None:
        self.logits = logits
        self.values = values or {}

    def action_logits_for(self, actions, *, num_action_scale, context=None):
        del num_action_scale, context
        return [self.logits.get(int(action), 0.0) for action in actions]

    def action_values_for(self, actions, *, num_action_scale, context=None):
        del num_action_scale, context
        return [self.values.get(int(action), 0.0) for action in actions]


class _FakeEnv:
    def action_prior_context(self):
        return {"category": "table", "num_bbox": 1, "num_action_scale": 1}


def _search(prior: _FakePrior, *, top_k: int = 2) -> MCTSTreeSearch:
    search = MCTSTreeSearch.__new__(MCTSTreeSearch)
    search.env = _FakeEnv()
    search.args = Namespace()
    search.num_bbox = 1
    search.num_action_scale = 1
    search.num_actions = 7
    search.action_prior_weight = 1.0
    search.puct_prior_weight = 0.0
    search.action_value_weight = 0.0
    search.action_prior_top_k = top_k
    search.action_prior = prior
    search.prior_pruned_nodes = 0
    search.prior_pruned_actions = 0
    search.prior_kept_actions = 0
    return search


def test_mcts_policy_top_k_prunes_untried_actions_by_agent_score() -> None:
    node = MCTSNode(0, None, None, num_bbox=1, num_action_scale=1)
    search = _search(_FakePrior({0: 0.0, 1: 4.0, 2: 1.0, 3: 2.0, 4: -1.0, 5: 3.0, 6: 0.5}))

    search._prune_node_untried_actions(node)

    assert node.untried_actions == [1, 5]
    assert node.action_mask.tolist() == [True, False, True, True, True, False, True]
    assert search.prior_pruned_nodes == 1
    assert search.prior_pruned_actions == 5
    assert search.prior_kept_actions == 2


def test_mcts_policy_top_k_keeps_legacy_actions_when_disabled() -> None:
    node = MCTSNode(0, None, None, num_bbox=1, num_action_scale=1)
    search = _search(_FakePrior({1: 4.0}), top_k=0)

    search._prune_node_untried_actions(node)

    assert node.untried_actions == list(range(7))
    assert search.prior_pruned_nodes == 0
