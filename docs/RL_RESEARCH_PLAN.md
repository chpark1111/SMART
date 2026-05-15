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
- Learned-search promotion does not require trajectory or metric identity.
  The acceptance rule is: same or lower search budget, no worse final SMART
  quality on category-balanced validation, and preferably improved quality.
- Treat `Covered` and `vIoU` as higher-is-better, and `BVS`, `MOV`, `TOV`,
  and `cub_CD` as lower-is-better. `num_box` is reported separately and should
  not override geometric quality unless explicitly used in a compactness study.
- Reject a model if it improves one secondary metric while worsening any
  critical final metric on aggregate.
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
- `smart build-prior --model-type rl-mlp` and
  `smart.build_rl_mlp_action_prior_from_traces` now train the first offline
  policy-gradient prior. It treats exact SMART trace rewards as a replay buffer,
  subtracts a category/mesh/global baseline to form advantages, increases
  probability for positive-advantage actions, and decreases probability for
  negative-advantage actions. The exported JSON is still only an MCTS
  action-ordering prior.
- `smart build-prior --model-type pg-agent` and
  `smart.build_policy_gradient_action_prior_from_traces` now train an
  action-level offline policy-gradient agent. This model scores concrete SMART
  action ids with bbox-index and local-action features, so it can guide which
  box MCTS explores rather than only choosing coordinate/scale classes. The
  training loss uses exact SMART trace rewards as advantages, but inference
  still only changes action ordering; exact SMART reward remains the evaluator.
- `scripts/train_action_prior_from_traces.py` is now the first explicit trainer
  entrypoint. It supports `counts`, `linear`, `mlp`, `rl-mlp`, and `pg-agent`;
  the next upgrade is collecting rejected candidates/final-return traces so the
  action-level policy has more informative negative examples.
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
- `scripts/benchmark_fixed_prior_by_category.py` now reports
  `quality_not_worse`, `quality_improved`, and `quality_worse` in addition to
  metric identity, so learned priors can be judged by final SMART quality rather
  than exact equality.
- The first all-trace offline RL prior was trained from `79` valid trace files
  under `runs/`, using all `10,502` action records from `119` meshes across
  airplane/chair/table. It is packaged at
  `smart/assets/priors/category_general_all_available_offline_rl_mlp_prior.json`
  and has a profile at `configs/rl_search_experimental.yaml`. A minimal
  category-balanced smoke (`1` mesh per category, `mcts_iter=10`,
  `max_step=10`, prior weight `0.1`) measured `1.094x` mean MCTS-stage speedup,
  all reported metrics identical, and no quality-worse cases. This is a smoke
  result only.
- A larger all-trace offline-RL sweep
  `runs/bench_exact/rl_prior_cat5_mcts20_weight_sweep.json` used `5` meshes per
  category, `mcts_iter=20`, `max_step=20`, and prior weights `0.05`, `0.1`, and
  `0.2`. Mean speedups were `1.032x`, `1.057x`, and `1.081x`. Weight `0.1`
  produced `14/15` quality-not-worse cases and `1/15` quality-improved case;
  weight `0.2` improved `2/15` but worsened `3/15`. The current recommendation
  is to keep `0.1` as an opt-in research setting and avoid promoting any RL
  prior without a quality guard or stronger policy validation.
- The next research step is a quality-guarded RL prior run, then a stronger
  category-aware policy that changes only action ordering.
- `scripts/run_quality_guarded_mcts.py` now provides the first quality guard.
  It runs baseline MCTS and learned-prior MCTS, evaluates both with SMART
  metrics, and copies the selected bbox directory into `mcts_guarded`. The prior
  result is accepted only when it is not worse than the baseline under the
  guarded metrics; otherwise the baseline result is kept. This changes the
  research target from “same trajectory” to “same-or-better final quality.”
- `runs/bench_exact/quality_guard_cat5_mcts20_rl01.json` is the first guarded
  15-mesh result. The raw global offline-RL prior was not-worse on `14/15`
  cases, worse on `1/15`, and improved `1/15`. The guard selected prior on
  `10/15` cases and baseline on `5/15`, producing `15/15` successful guarded
  outputs. Mean raw-prior stage speedup was `1.053x`.
- Category-specific offline-RL priors were trained for airplane/chair/table, but
  the first chair-specific check was not better than the global prior: on the
  five-chair subset, it had `2/5` worse prior cases and selected prior on only
  `2/5` guarded outputs. Keep the global prior as the current research default.
- The first action-level PG-agent model was trained from the same all-available
  trace list and packaged at
  `smart/assets/priors/category_general_policy_gradient_agent_prior.json`.
  Its first guarded check,
  `runs/bench_exact/quality_guard_cat3_mcts20_pg_agent_w005.json`, used `3`
  meshes per category, `mcts_iter=20`, `max_step=20`, and prior weight `0.05`.
  The raw prior was not-worse on `7/9` cases and worse on `2/9`; the guard
  selected prior on `4/9`, baseline on `5/9`, and mean raw-prior speedup was
  `1.006x`. This proves the new agent path is wired, but it is not better than
  the global coord/scale offline-RL prior yet.
- Candidate trace collection is now wired through
  `mcts.candidate_trace_path` / `--candidate_trace_path`. It records
  `record_type=mcts_candidate` rows from rollout candidates that SMART already
  exact-scored, including selected/non-selected labels and exact reward deltas.
  `--model-type pg-agent` consumes these rows as within-rollout comparisons,
  using the candidate-group mean as the advantage baseline. A smoke probe
  `runs/bench_exact/candidate_trace_probe.jsonl` generated `28` candidate rows
  from one 3-iteration MCTS run, and the trainer consumed all `28`.
- The first balanced candidate-trace PG-agent run
  `runs/bench_exact/candidate_pg_cat3_mcts10_collection.json` collected `541`
  candidate rows from `9` meshes. The retrained prior
  `runs/bench_exact/priors/category_general_candidate_pg_agent_cat3_prior.json`
  was benchmarked in
  `runs/bench_exact/candidate_pg_cat3_mcts10_benchmark.json`: at
  MCTS10/max-step10 and prior weight `0.05`, it was quality-not-worse on `9/9`
  cases and improved `1/9`, with `1.038x` mean raw-prior stage speedup. The
  follow-up sweep
  `runs/bench_exact/candidate_pg_cat3_mcts10_weight_sweep.json` showed that
  weight `0.2` is better on this subset: quality-not-worse `9/9`, improved
  `2/9`, no worse cases, and `1.067x` mean raw-prior stage speedup. The next RL
  step is a larger quality-guarded weight-`0.2` run, because the target is now
  equal-or-better final SMART metrics, not exact trajectory identity.
- The larger raw check
  `runs/bench_exact/candidate_pg_cat5_mcts10_w02_benchmark.json` showed the
  guard is required: weight `0.2` was quality-not-worse on `14/15`, improved
  `2/15`, worse on `1/15`, and measured `1.035x`. The bad table case
  `1040cd764facf6981190e285a2cbc9c` was then run through
  `runs/bench_exact/candidate_pg_guard_w02_table_badcase.json`, where the guard
  rejected the prior output and selected baseline. This is now the expected
  deployment shape for learned priors: try prior-guided search, evaluate with
  exact SMART metrics, and keep it only when it is not worse.
- The current packaged candidate-aware PG-agent prior is
  `smart/assets/priors/category_general_candidate_pg_agent_cat3_prior.json`.
  `runs/bench_exact/candidate_pg_guard_cat10_mcts10_w02.json` tested it with
  weight `0.2` on `10` meshes per category: guarded success was `30/30`, raw
  prior was not-worse on `25/30`, improved `2/30`, worse on `5/30`, and final
  guard selection was prior `15/30` and baseline `15/30`. This is the current
  strongest evidence that learned search should be deployed as a guarded
  quality-improvement candidate rather than as a raw replacement for baseline
  MCTS.
- Scaling candidate collection to `10` meshes per category produced
  `runs/bench_exact/candidate_pg_cat10_mcts10_collection.json` with `2206`
  candidate rows, but the retrained prior did not improve the policy:
  `runs/bench_exact/candidate_pg_cat10_prior_guard_cat10_mcts10_w02.json`
  selected prior on only `10/30`, improved `0/30`, had raw worse `5/30`, and
  measured `1.016x`. The conclusion is that candidate quantity alone is not the
  fix; the next model needs category-balanced/return-weighted training rather
  than simply adding more candidate rows.
- PG-agent loss weighting is now exposed through `--accepted-weight`,
  `--candidate-weight`, `--selected-candidate-weight`, and `--category-balance`.
  The first weighted cat10 prior did not improve quality:
  `runs/bench_exact/candidate_pg_cat10_weighted_guard_cat5_mcts10_w02.json`
  selected prior on `9/15`, improved `0/15`, and had raw worse `3/15`.
- Multi-candidate guard is now available through
  `scripts/run_quality_guarded_mcts.py --prior-weights 0.05,0.1,0.2`. The first
  cat3 check `runs/bench_exact/candidate_pg_multiweight_guard_cat3_mcts10.json`
  selected prior on `7/9`, improved `2/9`, and had no raw worse candidate. This
  is the cleanest quality-first learned-search direction so far, but it trades
  runtime for robustness by running several MCTS search orders.
- The larger cat5 multi-weight check
  `runs/bench_exact/candidate_pg_multiweight_guard_cat5_mcts10.json` selected
  prior on `13/15`, baseline on `2/15`, improved `2/15`, and caught one table
  mesh where all prior weights were worse. This reinforces the current research
  plan: learned policies propose multiple search orders, exact SMART evaluation
  selects the final bbox.
- `configs/rl_multiweight_guard_experimental.yaml` packages the current
  quality-first research profile. It should be launched through
  `scripts/run_quality_guarded_mcts.py --prior-weights ...`; the main SMART
  pipeline remains single-trajectory for paper reproduction.
- Adaptive multi-weight guard is now available. Conservative mode stops only
  after a quality-improved learned candidate; fast mode
  `--adaptive-stop-mode not_worse` can stop after a faster non-worse candidate.
  The first fast cat3 check
  `runs/bench_exact/candidate_pg_multiweight_adaptive_fast_cat3_mcts10.json`
  kept `9/9` guarded successes, selected prior on `7/9`, improved `1/9`, and
  skipped `12/27` candidate MCTS runs. This is a better runtime/quality tradeoff
  for exploratory sweeps, while full multi-weight guard remains the stronger
  quality-search setting.
- The follow-up cat5 fast-adaptive check
  `runs/bench_exact/candidate_pg_multiweight_adaptive_fast_cat5_mcts10.json`
  kept `14/14` guarded successes, selected prior on `12/14`, improved `1/14`,
  rejected `2` worse candidates, and reduced total MCTS launches from `56` to
  `32` on the refined subset. The next research target is making this cheaper
  policy improve more cases, not just select faster non-worse trajectories.
- On the full currently refined subset
  `runs/bench_exact/candidate_pg_multiweight_adaptive_fast_refined21_mcts10.json`
  (`10` airplane, `7` chair, `4` table), fast adaptive kept `21/21` guarded
  successes, selected prior on `13/21`, improved `1/21`, and reduced total MCTS
  launches from `84` to `61`. The method is now a practical guarded speed mode,
  but not yet a strong quality-improvement method.
- Hybrid local search is now the quality-improvement follow-up to guarded MCTS.
  `scripts/run_quality_guarded_local_refine.py` takes `mcts_guarded`, runs a
  small-action `local_refine`, evaluates both with exact SMART metrics, and
  copies the selected result into `local_refine_guarded`. On
  `runs/bench_exact/local_refine_guarded_refined21_covtol_improved_reuse_mcts_guarded.json`,
  it succeeded on the same `21` refined targets, selected local refine on
  `10/21`, and improved `10/21` cases. This is stronger for quality than the
  current learned prior itself, so the next RL target is to learn when and where
  this fine local search pays off, then use the policy to propose better
  post-MCTS actions.
- The larger manifest-selected run
  `runs/bench_exact/local_refine_guarded_expanded200_mcts_manifest52_covtol_improved.json`
  used `--from-input-manifest` and `input-stage=mcts` on all `52` successful
  processed MCTS outputs. It selected local refine on `29/52`, kept input on
  `23/52`, and improved `29/52` cases. Aggregate BVS/MOV/TOV/vIoU improved
  with nearly unchanged coverage. This is now the most useful supervised/RL
  target: predict the local-refine win probability from category, box count,
  and pre-refine SMART metrics, then run local search only when the gate expects
  a quality improvement.
- `scripts/export_local_refine_gate_dataset.py` exports that report to CSV or
  JSONL. The current manifest52 export has `52` rows and `29` positive
  improvement labels, which is enough for smoke-testing a gate but not enough
  for a final learned policy.
- `smart.local_refine_gate` and `scripts/train_local_refine_gate.py` now train a
  PyTorch gate for that decision. It uses only category and pre-local-refine
  SMART metrics, so it can run before the optional local search. The current
  packaged gate is
  `smart/assets/gates/local_refine_gate_manifest52.json`. On the `52`-row
  manifest dataset, leave-one-out validation measured accuracy `0.75`, F1
  `0.780`, and ROC-AUC `0.784` versus a majority baseline accuracy of `0.558`.
  This makes the next research step concrete: use the gate to skip local refine
  when it is unlikely to improve quality, then collect more rows from larger
  category-balanced runs.
- `scripts/evaluate_local_refine_gate.py` turns that into a time/quality sweep.
  On the current manifest52 dataset, threshold `0.5` catches every known
  local-refine improvement (`29/29`) while skipping `22/52` local-refine runs
  and saving `20.3%` of measured local-refine time. Thresholds above `0.5`
  save more time but miss improvements, so the gate is useful as a cost-control
  layer rather than a replacement for exact evaluation.
- Quality-first learned MCTS is now explicitly supported through
  `scripts/run_quality_guarded_mcts.py --selection-objective quality_score`.
  This keeps the exact per-metric guard but chooses a learned-prior output only
  when it gives positive scalar final SMART metric gain. The first cat5
  processed subset run
  `runs/bench_exact/candidate_pg_quality_score_guard_cat5_mcts10.json` selected
  the candidate-aware PG prior on `1/14` cases, kept baseline on `13/14`, and
  improved aggregate BVS/MOV/TOV/vIoU with no coverage drift. The effect is
  small; the next research target is collecting more candidate traces and
  training the action policy on final-return quality, not just local candidate
  rewards.
- `smart build-prior --model-type policy-value`,
  `scripts/train_action_prior_from_traces.py --model-type policy-value`, and
  `smart.build_policy_value_action_prior_from_traces` now train an action-level
  policy plus a scalar action-value head. The value head predicts normalized
  exact-reward advantage for a concrete SMART action; it is used only as an
  opt-in MCTS search bias through `mcts.action_value_weight`.
- `scripts/run_quality_guarded_mcts.py` now supports
  `--puct-prior-weight` and `--action-value-weight`, so the quality guard can
  test policy logits, PUCT child-selection bias, and action-value bias together.
  The packaged model
  `smart/assets/priors/category_general_policy_value_agent_prior.json` was
  trained from `13,439` accepted/candidate records over `119` meshes, including
  `2,937` candidate rows and `1,219,439` concrete action candidates.
- The first policy-value smoke
  `runs/bench_exact/policy_value_quality_guard_cat3_mcts10.json` used three
  meshes per category, MCTS10/max-step10, prior weights `0.05,0.1,0.2`,
  `puct_prior_weight=0.02`, and `action_value_weight=0.02`. It kept `9/9`
  guarded successes, selected the learned candidate on `1/9`, kept baseline on
  `8/9`, rejected `2` worse candidates, and improved aggregate BVS by
  `-0.001451`, MOV by `-0.020312`, TOV by `-0.000678`, and vIoU by
  `+0.000589` with no coverage drift. This is a real quality path, but still a
  weak one; it needs final-return labels or a stronger policy/value objective
  before promotion.
