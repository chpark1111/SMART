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
2. Category-specific priors: separate airplane/chair/table priors from traces.
3. Category-general MLP: shared network with category embedding.
4. Value head: predict rollout quality only for ordering, never as final reward.
5. Optional initializer: predict coarse bbox action schedule or improved
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
  build opt-in schema-v2 priors with dynamic action-scale metadata.
- `scripts/train_action_prior_from_traces.py` is now the first explicit trainer
  entrypoint. It currently trains the count-based baseline; the next upgrade is
  a category-general MLP action scorer trained from schema-v2 traces.
- Generic smoke prior experiments produced small speedups but changed one table
  case, so they are not default.
- The next research step is category-specific trace collection followed by a
  category-general policy that only changes action ordering.
