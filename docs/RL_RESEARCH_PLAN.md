# SMART RL Research Plan

This plan keeps SMART's final geometric evaluation as the arbiter. Learned
models may change search order in research profiles, but they should guide
initialization, candidate ordering, or MCTS priors rather than silently replacing
the coverage/tightness objective.

## Main Direction

Use a category-general policy as an action-ordering model for MCTS and greedy
refinement.

- Input: bbox bounds, rotations, per-bbox volume, current BVS/Covered proxy,
  tet centroid/volume summaries, category embedding, and action-unit scale.
- Output: logits over SMART action coordinates/scales, plus optional value
  estimate for rollout ordering.
- Verification: every selected action is still scored by exact SMART reward
  (`manifold` or exact `manifold_stateful`), and final results are evaluated by
  `BVS`, `MOV`, `TOV`, `Covered`, `vIoU`, and `cub_CD`.

## Data Collection

1. Run the exact or stateful exact pipeline with `trace_actions_path` enabled.
2. Collect traces across airplane/chair/table and later more ShapeNet classes.
3. Store successful actions, reward deltas, bbox/action layout, backend, action
   unit, BVS, and final evaluation metrics. Trace schema v2 already records
   these accepted-action fields; rejected/fallback candidates are the next
   dataset expansion.
4. Split by mesh id, not action rows, so validation tests category
   generalization.

## Training Stages

1. Baseline prior: the current schema-v2 count-based trace prior from
   `smart build-prior` or `scripts/train_action_prior_from_traces.py`.
2. State-aware linear prior: `--model-type linear` uses category, BVS, step,
   action-unit, box-count, and penalty features as a lightweight policy baseline.
3. PyTorch MLP prior: `--model-type mlp` trains a compact shared network with
   `--device auto`, which probes Apple Silicon MPS before CUDA/CPU.
4. Category-specific priors: separate airplane/chair/table priors from traces.
5. Category-general MLP: larger shared network with category embedding.
6. Value head: predict rollout quality only for ordering, never as final reward.
7. Optional initializer: predict coarse bbox action schedule or improved
   initial boxes before exact SMART refinement.

## Promotion Rules

- Paper reproduction profile keeps the exact legacy `manifold` defaults.
- Research profiles may use learned priors or changed search order only when
  `allow_search_order_changes=true`.
- Report speed and quality separately.
- Reject a model if it improves MOV but worsens BVS/TOV/vIoU on aggregate.
- Keep learned priors behind `allow_search_order_changes=true` until a
  category-balanced validation sweep is positive.
- Compare against:
  - `accelerated_exact`
  - `stateful_union_cache_experimental`
  - `hybrid_local_search_experimental`

## Current Status

- Trace logging exists through `trace_actions_path`.
- `smart build-prior` and `smart.action_prior.build_action_prior_from_traces`
  build opt-in schema-v2 count priors with dynamic action-scale metadata.
- `smart build-prior --model-type linear` and
  `smart.build_linear_action_prior_from_traces` train the first state-aware
  category-general action prior without adding a deep-learning dependency.
- `smart build-prior --model-type mlp` and
  `smart.build_mlp_action_prior_from_traces` now train a PyTorch MLP action prior
  with automatic MPS/CUDA/CPU device selection. The exported prior is still a
  JSON weight file used only for action ordering.
- `scripts/train_action_prior_from_traces.py` is now the first explicit trainer
  entrypoint. It supports `counts`, `linear`, and `mlp`; the next upgrade is a
  larger category-general MLP/PUCT action scorer trained from schema-v2 traces.
- Generic smoke prior experiments produced small speedups but changed one table
  case, so they are not default.
- The first tiny linear-prior leave-one-out check kept reported metric diffs at
  `0` but was slower on MCTS2. A slightly larger three-airplane MCTS5 check kept
  metric diffs at `0` and measured `1.037x` at prior weight `0.1`, so the path is
  functional but not promoted.
- The first tiny PyTorch MLP prior check kept reported metric diffs at `0` but
  measured `0.952x` on CPU, so the MLP path is also wired but not promoted.
- `mcts.puct_prior_weight` is now available for PUCT-style prior-guided child
  selection. Its first tiny linear-prior check kept metric diffs at `0` and
  measured `1.055x`; it needs larger MCTS20/MCTS100 validation.
- The next research step is category-specific trace collection followed by a
  stronger category-general policy that only changes action ordering.
