# SMART Current Status

This page is the short operational status of the repository. It separates
paper-safe defaults from opt-in acceleration and search experiments.

## Official Defaults

- Pipeline order is `normalize -> tetra -> preseg -> merge -> refine -> mcts -> render`.
  An opt-in hybrid profile extends this to
  `normalize -> tetra -> preseg -> merge -> refine -> mcts -> local_refine -> render`.
- `configs/demo.yaml`, `configs/smoke_5.yaml`, and
  `configs/accelerated_exact.yaml` keep the paper-safe exact Manifold reward
  path.
- `configs/accelerated_exact.yaml` is the profile to use for exact reporting:
  `refine.reward_backend=manifold` and `mcts.reward_backend=manifold`.
- The fixed C++ Manifold source remains authoritative. It is not pulled,
  replaced, rewritten in Rust, or edited for optimization.
- Explicit Rust MCTS runners are blocked unless
  `mcts.allow_search_order_changes=true`, because they can change the search
  trajectory even when the reward evaluator is exact.

## Applied To The Main Pipeline

- Data normalization is a first-class stage and writes normalized meshes under
  `runs/<profile>/normalized/` without mutating `data/`.
- Mesh2Tet uses ManifoldPlus plus fTetWild with per-mesh retry/skip manifests.
  fTetWild failures are now classified as timeout, SIGSEGV crash, external kill,
  missing executable, or validation failure. The retry chain is:
  primary paper/category parameters, coarser retry, coarser bounded retry, and
  an optional general-winding-number fallback for open/degenerate ShapeNet
  meshes.
- CoACD is the default pre-segmentation path; BSP remains optional for closer
  paper reproduction.
- Merge uses the SMART BAVF merge logic with cached rewards and a lazy heap.
- Refine/MCTS skip log-only summary metrics and heavy partition snapshots by
  default while still writing final bbox OBJ outputs.
- MCTS now writes a deterministic initial bbox result when no improved update is
  found, so downstream render/evaluation cannot accidentally consume stale
  outputs.
- Renderer uses the legacy paper renderer path and defaults to transparent,
  box-only outputs, with mesh-overlay variants available by config.
- `smart-bbox` packaging is wired through maturin/PyO3, with `smart._rust`
  included in wheels and Python fallbacks available for debugging.
- `local_refine` is now available as an experimental post-MCTS local search
  stage. It reads the latest MCTS bbox output through `bbox_direct` and runs a
  smaller-action greedy refinement pass.
- OBB/recenter operations now guard NaN/Inf rotations and degenerate point sets.
  Invalid OBB fitting falls back to AABB construction instead of propagating
  non-finite states through refine/MCTS.
- MCTS can mark `success_timeout_output` when a timeout happens after a bbox
  output directory has already been written. This prevents completed-but-slow
  benchmark runs from being reported as hard failures.
- `scripts/analyze_pipeline_failures.py` summarizes manifest failures and
  slowest records by stage.

## Rust Exact Migration

Applied exact or parity-preserving Rust/C++ work:

- `pymesh.py` delegates tet volume, centroid, surface-face extraction,
  adjacency summaries, and Gmsh `.msh` IO to Rust when available.
- Bbox/action-state helpers, BVS summaries, action upper bounds, valid masks,
  reward accumulation, UCB/softmax helpers, and Chamfer fallback exist in Rust.
- Greedy refine can use Rust callback control while preserving the legacy reward
  path.
- Merge has exact Rust partition summaries and cached BAVF reward helpers.
- `reward_backend=manifold_bridge` and `reward_backend=manifold_stateful` call
  the fixed C++ Manifold implementation through SMART's Rust/C++ wrapper.
- `ManifoldState` has persistent exact reward cache entries, lazy
  `union_except_i` construction, initial bbox state caching, and delta sync for
  accepted axis actions.

Not promoted to default:

- `manifold_stateful` is still opt-in. The wrapper now uses the legacy
  sequential Manifold union order when `stateful_union_cache=false`, and the
  latest five-mesh MCTS20 smoke preserved all reported paper metrics and action
  traces exactly after matching legacy bbox mesh construction.
  Reward memoization is now independent from union caching, so exact repeated
  state/action scores are reused without changing boolean grouping order. Speed
  is a small win on the current smoke (`1.032x` on smoke5 mcts5 repeat-3,
  `1.189x` on smoke3 mcts20, `1.039x` on smoke5 mcts20 repeat-3), but this is
  not yet proven across the expanded ShapeNet set.
- `mcts.backend=rust_stateful` is a search-order-changing experiment. It may be
  faster on some cases, but it is not paper-compatible until the trajectory
  parity problem is solved.
- `reward_backend=tet_clipping` is experimental. It is promising for reducing
  Manifold calls, but it is not the official metric backend.
- `candidate_backend=bitset_topk` is an exact-verification experiment: bitsets
  rank candidates, but Manifold still verifies selected candidates.
  `configs/candidate_bitset_exact_experimental.yaml` packages the current
  smoke-checked setting (`candidate_top_k=8`) while keeping the legacy MCTS
  tree and `allow_search_order_changes=false`.

## Search And RL Experiments

- Trace logging is available through `trace_actions_path`.
- `smart build-prior` builds action-prior JSON files from traces.
- `configs/accelerated_search_experimental.yaml` uses the packaged smoke prior
  with `action_prior_weight=0.1`.
- `configs/rl_search_experimental.yaml` uses the packaged all-trace offline-RL
  MLP prior
  `smart/assets/priors/category_general_all_available_offline_rl_mlp_prior.json`
  with `action_prior_weight=0.1`.
- `smart build-prior --model-type pg-agent` trains the first action-level
  policy-gradient agent. It scores concrete SMART action ids with bbox-index
  features instead of only coord/scale classes.
- `mcts.candidate_trace_path` can now collect already-scored rollout candidates
  for richer policy-gradient data. This is opt-in and does not change search.
- `scripts/run_quality_guarded_mcts.py` can run baseline MCTS plus RL-prior
  MCTS and copy the selected non-worse result into `mcts_guarded`.
- Action priors and transposition tables are opt-in because they can change
  search order. They guide MCTS; they never replace the exact reward.
- Current smoke evidence supports trace priors as a useful research path, not a
  release default.

## Current Benchmark Facts

- `runs/expanded_full/manifests` timeout audit on 2026-05-14 found three
  separate causes. fTetWild `rc=-11` is SIGSEGV on bad/degenerate ShapeNet
  input, not a Python timeout. Refine `rc=124` at exactly 180s is a real stage
  budget timeout from experimental collection runs. One MCTS `rc=124` already
  had a final output path and was a timeout-wrapper false failure.
- Re-running table mesh `104ebf7f96c77fb46a0faccc2a4015d8` after the retry
  update succeeded with the primary fTetWild settings. Chair mesh
  `108b9cb292fd811cf51f77a6d7299806` was traced into fTetWild, not the fixed
  SMART Manifold boolean library. The first crash came from a debug
  `igl::is_vertex_manifold` diagnostic in `MeshImprovement.cpp::manifold_surface`
  after ManifoldPlus produced a watertight but highly degenerate repaired mesh.
  A second crash path came from `bfs_orient()` receiving an empty tracked
  surface during general-winding-number fallback. Both C++ crash paths now have
  guards in the local fTetWild source. The mesh still does not produce a useful
  tetrahedralization: primary/retry/coarse outputs contain `0` valid tet
  elements under validation, and the robust fallback reaches the 300s timeout.
  That case is now a clean skip/failure record instead of a process crash; it
  should be skipped or re-meshed upstream.

- `runs/bench_exact/exact_stateful_trace_smoke1_mcts1_auto.json`:
  `manifold_stateful` with legacy MCTS tree has no action divergence on one
  airplane smoke case and measured `1.16x`.
- `runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_auto_trace.json`:
  older five-mesh MCTS smoke before legacy-union matching; it exposed one chair
  near-tie metric drift.
- `runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_legacy_union_fresh_stats.json`:
  five-mesh MCTS smoke after legacy sequential union matching and fresh
  `rust_stats.json` writes; all reported metric diffs are `0`, mean speedup is
  `1.0001x`, and stateful stats correctly show `stateful_union_cache=0`.
- `runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_legacy_union_repeat3.json`:
  repeat-3 timing of the same smoke; all reported metric diffs remain `0`, mean
  speedup is `0.986x`, confirming current stateful exact MCTS is parity work,
  not a stable speed win.
- `runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_rewardcache_repeat3.json`:
  reward memoization enabled while keeping `stateful_union_cache=0`; all
  reported metric diffs remain `0`, mean speedup is `1.032x`, and cache stats
  show exact reward entries are reused without `union_except_i`.
- `runs/bench_exact/exact_stateful_sweep_smoke3_mcts20_rewardcache.json`:
  larger smoke with three targets at `mcts_iter=20`, `max_step=20`; all reported
  metric diffs remain `0`, mean speedup is `1.189x`.
- `runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_trace.json`:
  five targets at `mcts_iter=20`, `max_step=20` after changing the stateful
  bbox Manifold construction from `Cube().Transform()` to the same explicit
  eight-vertex/fixed-face mesh path used by legacy Python; all reported metric
  diffs are `0`, action traces have no divergence, max reward drift before
  divergence is only roundoff (`~1e-14`), and mean speedup is `1.024x`.
- `runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_repeat3.json`:
  same five-target MCTS20 smoke without trace logging, repeat 3; all reported
  metric diffs remain `0`, parity failures `{}`, and mean speedup is `1.039x`.
- `runs/bench_exact/airplane_mcts100_stateful_replacement.json`:
  `mcts.backend=rust_stateful` measured `1.33x` on one airplane case, but this
  is now classified as a search-order-changing experiment, not a paper-safe
  default.
- `runs/bench_exact/action_prior_generalization_smoke5_w01.json`:
  portable trace prior at weight `0.1` kept reported metric diffs at `0` on the
  checked smoke set and measured about `1.10x` mean speedup.
- `runs/bench_exact/mcts_accel_profiles_smoke3_iter10.json`:
  quick profile sweep over three targets, `mcts_iter=10`, `max_step=10`.
  `bitset_top8` kept all metric diffs at `0` and measured `1.101x`;
  `rust_stateful_tt` also kept metric diffs at `0` and measured `1.090x`;
  `rust_stateful_prior01_tt` was faster (`1.157x`) but changed chair metrics,
  so it remains a quality-changing search experiment.
- `runs/bench_exact/mcts_accel_profiles_smoke5_iter20_exact_candidates.json`:
  five-target MCTS20 profile sweep. `bitset_top8` kept all metric diffs at `0`
  and measured `1.015x`; `rust_stateful_tt` measured `1.023x` but changed
  metrics on airplane/chair near-tie cases, so it is not exact-compatible yet.
- `runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_bitset_all.json`:
  category-balanced processed sweep over three airplane, three chair, and three
  table meshes at `mcts_iter=20`, `max_step=20`. `bitset_top8` kept all metric
  diffs at `0` and measured `1.018x`.
- `runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_topk_tt_all.json`:
  same nine processed meshes comparing `bitset_top3`, `bitset_top8`, and
  `rust_stateful_tt`. `bitset_top3` and `bitset_top8` both kept all reported
  metric diffs at `0`; mean speedups were `1.025x` and `1.023x`. `rust_stateful_tt`
  measured `1.019x` but changed BVS/MOV/TOV/vIoU, so it remains an opt-in
  search-order experiment.
- `runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_instrumented_topk.json`:
  instrumented nine-mesh top-K run. `bitset_top8` kept metric diffs at `0` and
  measured `1.050x`, but exact reward misses only dropped from `12477` to
  `11584`. The fallback exact fraction was `0.439`, confirming that safe
  top-K still spends most time in exact Manifold verification.
- `runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_union_cache_probe.json`:
  exact leave-one-out union cache probe. `exact_union_cache` and
  `bitset_top8_union_cache` both kept all reported metric diffs at `0`.
  `exact_union_cache` measured `1.097x`; `bitset_top8_union_cache` measured
  `1.096x`. This is the strongest exact reward-evaluation speed path so far,
  because it does not change search order or candidate policy.
- `runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_union_cache.json`:
  current processed-set sweep over 16 meshes: six airplane, five chair, and
  five table. `exact_union_cache` kept all reported metric diffs at `0` and
  measured `1.047x`.
- `runs/bench_exact/mcts_accel_profiles_one_iter100_union_cache.json`:
  one 10-box airplane case at `mcts_iter=100`, `max_step=100`.
  `exact_union_cache` kept all reported metric diffs at `0` and measured
  `1.095x` (`81.80s -> 74.67s`). Cache stats show `1093`
  leave-one-out union builds and `12879` cache hits, so this is the best
  current exact MCTS speed path.
- `runs/bench_exact/mcts_accel_profiles_airplane3_iter20_fused_rollout.json`:
  three airplane cases at MCTS20/max-step100. The new opt-in
  `mcts.fused_rollout_step=true` kept reported metric diffs at `0` and
  measured `1.013x` over `exact_union_cache`. It helps some 10-box cases but is
  workload-dependent, so it remains an experiment.
- `runs/bench_exact/fast_stop20_probe_eval.json`:
  same 10-box airplane case as the MCTS100 run, but with
  `mcts.no_reward_stop_after=20`. It stopped after the no-improvement window,
  produced byte-identical bbox OBJ files versus the MCTS100 union-cache result,
  kept reported metric diffs at `0`, and reduced stage time from `73.83s` to
  `15.96s`. This is a search-budget shortcut, not a paper-safe default.
- `runs/bench_exact/manifold_volume_methods_processed16.json`:
  `GetMesh()` residual volume versus `GetProperties().volume` on deterministic
  AABB-derived probes across the 16 currently processed meshes. The maximum
  absolute difference is `4.064794009717154e-08`, and the latest properties path
  run is `1.105021890299984x` faster on average at the residual-volume
  micro-benchmark level.
- `runs/bench_exact/exact_stateful_properties_mcts20_target1.json`:
  first actual MCTS target using `mcts.manifold_volume_method=properties`.
  Reported evaluation metric diffs versus legacy `manifold` are all `0`, the
  accepted action sequence did not diverge, and stage speed was `1.069x` versus
  `manifold` (`18.74s -> 17.53s`). Reward traces differ by up to
  `2.2902652996314998e-04`, so this is promising but still opt-in until tested
  on a larger category-balanced sweep.
- `runs/bench_exact/hybrid_probe_local_refine_eval.json`:
  first hybrid MCTS + local-search check on the same 10-box airplane case.
  `local_refine` after MCTS improved `BVS` (`2.722 -> 2.609`), `MOV`
  (`3.034 -> 2.497`), `TOV` (`1.580 -> 1.508`), and `vIoU`
  (`0.3867 -> 0.3978`) while keeping `Covered > 0.998`.
- `scripts/run_quality_guarded_local_refine.py` and
  `runs/bench_exact/local_refine_guarded_refined21_covtol_improved_reuse_mcts_guarded.json`:
  guarded hybrid MCTS + local-search pass over every currently refined
  `mcts_guarded` output in `configs/expanded_200.yaml` (`10` airplane, `7`
  chair, `4` table). It ran `21/21` successfully, selected local refine on
  `10/21`, selected the input on `11/21`, and improved `10/21` cases. The stage
  evaluation
  `runs/bench_exact/local_refine_guarded_refined21_covtol_improved_stage_eval.json` shows the
  selected `local_refine_guarded` outputs improved aggregate BVS
  (`2.0373 -> 2.0010`), MOV (`1.5140 -> 1.3500`), TOV
  (`0.9780 -> 0.9480`), and vIoU (`0.6142 -> 0.6269`) versus `mcts_guarded`.
  This run used explicit `--covered-tolerance 0.001`; aggregate coverage stayed
  effectively unchanged (`0.999527 -> 0.999520`). A strict coverage guard
  selected local refine on only `2/21`, so this remains an explicit
  quality-first mode rather than a paper-safe default.
- `runs/bench_exact/local_refine_guarded_expanded200_mcts_manifest52_covtol_improved.json`:
  manifest-selected expansion of the same guarded hybrid local search, using
  `--from-input-manifest` and `input-stage=mcts` to cover all `52` successful
  processed MCTS outputs in `configs/expanded_200.yaml` (`18` airplane, `17`
  chair, `17` table). It ran `52/52` successfully, selected local refine on
  `29/52`, selected the input on `23/52`, and improved `29/52` cases. Stage
  evaluation
  `runs/bench_exact/local_refine_guarded_manifest52_covtol_improved_stage_eval.json`
  shows aggregate BVS (`1.7546 -> 1.7225`), MOV (`1.2327 -> 1.1410`), TOV
  (`0.7173 -> 0.6913`), and vIoU (`0.6835 -> 0.6970`) improved versus MCTS,
  while coverage stayed effectively unchanged (`0.999685 -> 0.999681`).
- `runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_prior_weight_sweep.json`:
  first RL/MCTS action-prior weight sweep on the same 16 meshes with exact
  reward and union cache. Prior weights `0.02`, `0.05`, and `0.1` measured
  `1.043x`, `1.053x`, and `1.055x` versus `exact_union_cache`, but all changed
  the same table case. MOV improved there, while BVS/TOV/vIoU worsened, so the
  current prior remains a search/quality experiment, not a default.
- `runs/bench_exact/rl_prior_cat5_mcts20_weight_sweep.json`:
  first all-trace offline-RL MLP prior sweep on `5` meshes per category at
  MCTS20/max-step20. Prior weights `0.05`, `0.1`, and `0.2` measured `1.032x`,
  `1.057x`, and `1.081x` mean speedup. Weight `0.1` had `14/15`
  quality-not-worse cases and `1/15` quality-improved case; weight `0.2` had
  more worse cases. This keeps the RL prior research-only until quality guard
  validation passes.
- `runs/bench_exact/quality_guard_cat5_mcts20_rl01.json`:
  guarded run for the global offline-RL prior at weight `0.1` on the same
  15-mesh subset. Raw prior was worse on `1/15`, but the guard selected baseline
  for that case and produced `15/15` successful `mcts_guarded` outputs. Prior
  was selected on `10/15`, baseline on `5/15`, with `1.053x` mean raw-prior
  stage speedup.
- `runs/bench_exact/quality_guard_chair5_mcts20_catrl01.json`:
  first chair-specific offline-RL prior check. It selected prior on `2/5` and
  rejected worse prior outputs on `2/5`, so category-specific priors are not
  promoted over the global prior yet.
- `runs/bench_exact/quality_guard_cat3_mcts20_pg_agent_w005.json`:
  first action-level policy-gradient agent check. The model was trained from
  the same all-available trace list and packaged at
  `smart/assets/priors/category_general_policy_gradient_agent_prior.json`. On
  `3` meshes per category with MCTS20/max-step20 and prior weight `0.05`, raw
  prior was not-worse on `7/9` cases and worse on `2/9`; the guard selected
  prior on `4/9`, baseline on `5/9`, and mean raw-prior speedup was `1.006x`.
  This path is wired, but the older global coord/scale offline-RL prior plus
  guard is still the stronger current research setting.
- `runs/bench_exact/candidate_trace_probe.jsonl`:
  first candidate-trace smoke. A 3-iteration airplane MCTS run produced `28`
  `record_type=mcts_candidate` rows. The PG-agent trainer consumed all `28`
  candidate records and uses rollout-candidate group means as advantages, which
  gives negative examples without replacing exact reward.
- `runs/bench_exact/candidate_pg_cat3_mcts10_collection.json` and
  `runs/bench_exact/candidate_pg_cat3_mcts10_benchmark.json`:
  first category-balanced candidate-trace PG-agent check. Trace collection ran
  `9/9` meshes successfully and collected `541` candidate rows. The retrained
  PG-agent
  `runs/bench_exact/priors/category_general_candidate_pg_agent_cat3_prior.json`
  used those candidate rows plus the existing trace list. On `3` meshes per
  category with MCTS10/max-step10 and prior weight `0.05`, raw prior outputs
  were quality-not-worse on `9/9` cases and improved `1/9`; no worse case was
  observed. Mean raw-prior stage speedup was `1.038x` overall, with category
  speedups airplane `1.148x`, chair `0.956x`, and table `1.009x`. This confirms
  candidate comparisons improve stability versus the first PG-agent, but the
  effect is still too small for a default.
- `runs/bench_exact/candidate_pg_cat3_mcts10_weight_sweep.json`:
  candidate-aware PG-agent weight sweep on the same `9` meshes. Weight `0.1`
  was quality-not-worse on `9/9`, improved `1/9`, and measured `1.059x`; weight
  `0.2` was quality-not-worse on `9/9`, improved `2/9`, and measured `1.067x`.
  No worse outputs were observed in this small sweep. The current best RL
  research setting for this candidate-aware PG-agent is therefore weight `0.2`
  plus quality guard, pending larger validation.
- `runs/bench_exact/candidate_pg_cat5_mcts10_w02_benchmark.json`:
  larger raw candidate-aware PG-agent check on `5` meshes per category. Weight
  `0.2` was quality-not-worse on `14/15`, improved `2/15`, worse on `1/15`, and
  measured `1.035x` mean raw-prior stage speedup. The worse case was table
  `1040cd764facf6981190e285a2cbc9c` with degraded BVS/MOV/TOV/Covered/vIoU.
  `runs/bench_exact/candidate_pg_guard_w02_table_badcase.json` confirms the
  quality guard rejects that prior output and selects baseline into
  `mcts_guarded`. So weight `0.2` is not a raw default, but is usable as a
  guarded research policy.
- `runs/bench_exact/candidate_pg_guard_cat10_mcts10_w02.json`:
  larger guarded check for the same candidate-aware PG-agent at weight `0.2`.
  It ran `10` meshes per category with `30/30` guarded successes. Raw prior was
  not-worse on `25/30`, improved `2/30`, worse on `5/30`, and measured `1.074x`
  mean raw-prior stage speedup. The guard rejected all `5` worse prior outputs;
  final selected outputs were prior `15/30` and baseline `15/30`. By category,
  prior was selected on airplane `8/10`, chair `4/10`, and table `3/10`. The
  research prior is packaged at
  `smart/assets/priors/category_general_candidate_pg_agent_cat3_prior.json`.
- `runs/bench_exact/candidate_pg_cat10_mcts10_collection.json` and
  `runs/bench_exact/candidate_pg_cat10_prior_guard_cat10_mcts10_w02.json`:
  expanded candidate-trace collection also succeeded on `30/30` meshes and
  collected `2206` candidate rows. A retrained cat10 candidate-aware PG-agent
  used those rows, but the guarded benchmark was worse as a policy: `30/30`
  guarded successes, raw prior not-worse `25/30`, improved `0/30`, worse
  `5/30`, selected prior `10/30`, baseline `20/30`, and mean raw-prior speedup
  `1.016x`. This prior is not packaged or promoted; more candidate data made
  this model conservative rather than better.
- `runs/bench_exact/priors/category_general_candidate_pg_agent_cat10_weighted_prior.json`
  and `runs/bench_exact/candidate_pg_cat10_weighted_guard_cat5_mcts10_w02.json`:
  first PG-agent loss-weight experiment using `--accepted-weight 2.0`,
  `--candidate-weight 0.5`, `--selected-candidate-weight 3.0`, and
  `--category-balance`. The 5/category guarded check succeeded on `15/15` and
  selected prior on `9/15`, but improved `0/15` and had raw worse `3/15`; this
  confirms weighting made the model faster/safer in some cases but not a better
  quality policy.
- `runs/bench_exact/candidate_pg_multiweight_guard_cat3_mcts10.json`:
  first multi-candidate guard using the packaged cat3 candidate prior at weights
  `0.05`, `0.1`, and `0.2`. It ran `3` meshes per category with `9/9` guarded
  successes, selected prior on `7/9`, selected baseline on `2/9`, improved
  `2/9`, and had no raw worse candidate on this subset. This is a
  quality-first research mode, not a speed default, because it runs multiple
  MCTS searches per mesh.
- `runs/bench_exact/candidate_pg_multiweight_guard_cat5_mcts10.json`:
  larger multi-candidate guard check on `5` meshes per category. It succeeded
  on `15/15`, selected prior on `13/15`, selected baseline on `2/15`, and
  improved `2/15`. One table mesh had all three prior weights worse than the
  baseline, and the guard selected baseline. This confirms the guard catches
  bad learned-search proposals while still letting improved/fast not-worse
  candidates through.
- `configs/rl_multiweight_guard_experimental.yaml` and
  `smart/configs/rl_multiweight_guard_experimental.yaml` package the current
  quality-first learned-search profile. Use it with
  `scripts/run_quality_guarded_mcts.py --prior-weights 0.05,0.1,0.2`; it is not
  a paper-default pipeline profile because it runs multiple MCTS trajectories.
- `runs/bench_exact/candidate_pg_multiweight_adaptive_cat3_mcts10.json` and
  `runs/bench_exact/candidate_pg_multiweight_adaptive_fast_cat3_mcts10.json`:
  first adaptive multi-weight guard checks on `3` meshes per category from
  `configs/expanded_200.yaml`. Conservative adaptive mode
  (`0.05,0.1,0.2`, stop only on quality improvement) kept `9/9` guarded
  successes but skipped no candidates because the improvement appeared at the
  last weight. Fast adaptive mode (`0.2,0.1,0.05`,
  `--adaptive-stop-mode not_worse`) kept `9/9` guarded successes, selected prior
  on `7/9`, selected baseline on `2/9`, improved `1/9`, rejected one worse
  candidate, and skipped `12/27` candidate MCTS runs. This is the first evidence
  that guarded learned search can reduce multi-weight overhead while preserving
  exact final quality checks.
- `runs/bench_exact/candidate_pg_multiweight_adaptive_fast_cat5_mcts10.json`:
  larger fast-adaptive check on all currently refined cat5 targets in
  `configs/expanded_200.yaml` (`5` airplane, `5` chair, `4` table). It kept
  `14/14` guarded successes, selected prior on `12/14`, selected baseline on
  `2/14`, improved `1/14`, rejected `2` worse candidates, and skipped `24/42`
  candidate MCTS runs. Total MCTS launches dropped from `56` possible launches
  for full baseline+3-candidate guard to `32`, while final selection still used
  exact SMART metric guard.
- `runs/bench_exact/candidate_pg_multiweight_adaptive_fast_refined21_mcts10.json`:
  fast-adaptive check on every currently refined target in `configs/expanded_200.yaml`
  (`10` airplane, `7` chair, `4` table). It kept `21/21` guarded successes,
  selected prior on `13/21`, selected baseline on `8/21`, improved `1/21`, and
  rejected `2` worse candidates. Candidate MCTS launches dropped from `63` to
  `40` (`36.5%` candidate-run reduction), and total baseline+candidate MCTS
  launches dropped from `84` to `61` (`27.4%` total reduction). This is the
  current best quality-safe learned-search runtime result, though the quality
  improvement rate is still low.
- `runs/bench_exact/local_refine_guarded_refined21_covtol_improved_reuse_mcts_guarded.json`:
  guarded local refine is currently the stronger quality-improvement step after
  learned MCTS selection. On the same refined21 subset it selected local search
  on `10/21` and improved `10/21` cases, with better aggregate
  BVS/MOV/TOV/vIoU and near-identical coverage under explicit
  `--covered-tolerance 0.001`. This remains an experimental quality-first
  post-process until larger category-balanced checks are complete.
- `runs/bench_exact/local_refine_guarded_expanded200_mcts_manifest52_covtol_improved.json`
  strengthens that result on `52` processed MCTS outputs: local search was
  selected on `29/52` and improved all selected cases. This makes local refine
  the strongest current quality-improvement stage after MCTS.
- `smart.local_refine_gate` and `scripts/train_local_refine_gate.py` now train a
  PyTorch local-refine gate from that report. The packaged research model is
  `smart/assets/gates/local_refine_gate_manifest52.json`. It uses only category
  and pre-local-refine metrics; leave-one-out validation on the `52` rows gives
  accuracy `0.75`, F1 `0.780`, and ROC-AUC `0.784`, compared with majority
  baseline accuracy `0.558`. This makes local-refine gating a better near-term
  RL target than raw MCTS action ordering: learn when the post-MCTS local search
  is likely to pay off, then still use exact SMART metrics as the final guard.
  `scripts/run_quality_guarded_local_refine.py --gate-path ... --gate-threshold`
  can now skip local refinement below the threshold; a forced skip smoke
  (`runs/bench_exact/local_refine_gate_skip_smoke.json`) scored one airplane
  case, skipped local refine, selected input, and wrote a successful guarded
  output.
- `scripts/evaluate_local_refine_gate.py` provides fast threshold analysis from
  the already-computed gate dataset. The current manifest52 sweep
  `runs/bench_exact/local_refine_gate_manifest52_threshold_sweep.json` shows
  threshold `0.5` runs local refine on `30/52`, skips `22/52`, catches all
  `29/29` known improvement cases, matches the full guarded local-refine
  aggregate metrics, and saves `20.3%` of measured local-refine stage time.
  Higher thresholds save more time but miss real improvement cases.
- `smart evaluate --from-manifest` now supports custom subset stages such as
  `local_refine_gate_guarded`. The current gated stage eval
  `runs/bench_exact/local_refine_gate_guarded_manifest52_t05_stage_eval.json`
  has `52/52` successes with BVS `1.7225`, MOV `1.1410`, TOV `0.6913`,
  Covered `0.999681`, and vIoU `0.6970`.
- `scripts/run_quality_guarded_mcts.py --selection-objective quality_score`
  now changes the learned-MCTS research target from speed/identity to final
  quality gain. The exact per-metric guard still rejects worse candidates, but
  among eligible candidates the selector keeps only positive scalar SMART metric
  gains. On
  `runs/bench_exact/candidate_pg_quality_score_guard_cat5_mcts10.json`, the
  candidate-aware PG prior was selected on `1/14` processed meshes, baseline was
  kept on `13/14`, and selected aggregate metrics improved over baseline by BVS
  `-0.00093`, MOV `-0.01306`, TOV `-0.00044`, and vIoU `+0.00038` with no
  coverage drift. This is a small but correctly targeted RL quality result.
- Action trace schema is now versioned. New traces include category, bbox/action
  layout, action unit, BVS, volume sum, backend, and Manifold volume method.
  `smart build-prior` and `scripts/train_action_prior_from_traces.py` now emit
  schema-v2 count priors with dynamic `num_action_scale` metadata.
- A state-aware linear prior is now wired through
  `smart build-prior --model-type linear`,
  `scripts/train_action_prior_from_traces.py --model-type linear`, and
  `smart.build_linear_action_prior_from_traces`. The tiny MCTS2 leave-one-out
  airplane smoke kept reported metric diffs at `0` but measured `0.852x`; the
  slightly larger three-airplane MCTS5 smoke
  `runs/bench_exact/action_prior_linear_smoke3_mcts5.json` also kept metric diffs
  at `0` and measured `1.037x` at prior weight `0.1`. This is an active
  RL/action-ordering path, not a default.
- A PyTorch MLP prior is now wired through `--model-type mlp` and
  `smart.build_mlp_action_prior_from_traces`. Training uses `--device auto`,
  which probes Apple Silicon MPS first, then CUDA, then CPU. The exported prior
  remains JSON weights and still only changes action order. The first tiny
  CPU smoke `runs/bench_exact/action_prior_mlp_airplane2_mcts2.json` kept metric
  diffs at `0` but measured `0.952x`, so it is not promoted.
- `mcts.puct_prior_weight` now adds a PUCT-style prior bonus during child
  selection. The tiny linear-prior PUCT smoke
  `runs/bench_exact/action_prior_puct_linear_airplane2_mcts2.json` kept metric
  diffs at `0` and measured `1.055x`, but it is not promoted.
- `runs/bench_exact/expanded_full_fixed_prior_cat20_mcts20_mlp120.json`:
  first larger expanded-full prior benchmark over 20 airplane, 20 chair, and
  20 table meshes. The prior was trained from
  `runs/bench_exact/trace_collection/expanded_full_cat20_20260514_*.jsonl`
  using PyTorch MLP training (`65` traced meshes, `3662` records seen,
  `2271` reward-positive records used). At `mcts_iter=20`, `max_step=20`,
  mean speedup was `0.967x` for prior weight `0.05` and `0.991x` for
  prior weight `0.1`, so the current learned ordering does not yet accelerate
  the overall benchmark. Category results were:
  airplane `0.941x/0.979x`, chair `0.952x/0.972x`, table `1.007x/1.022x`
  for weights `0.05/0.1`. Reported metrics were identical on `58/60`
  meshes for weight `0.05` and `56/60` for weight `0.1`; the largest drift was
  small (`max Avg_BVS 0.00613`, `max Avg_vIoU 0.00141`) but nonzero, so this
  remains a research-only search-order policy. The benchmark now also reports
  quality direction, not only identity: `Covered` and `vIoU` are
  higher-is-better, while `BVS`, `MOV`, `TOV`, and `cub_CD` are
  lower-is-better. Under that criterion, weight `0.05` was not worse on
  `59/60` meshes and improved `1/60`; weight `0.1` was not worse on `57/60`
  and improved `1/60`. This confirms the current MLP prior is mostly safe but
  not yet useful enough as a quality-improving policy.
- Action policy/value MCTS is now implemented as a real quality-first research
  path. `smart build-prior --model-type policy-value` and
  `smart.build_policy_value_action_prior_from_traces` train an action-level
  policy plus a scalar value head over concrete SMART action ids. The value
  head is consumed only when `mcts.action_value_weight > 0`, and PUCT policy
  bias is exposed through `mcts.puct_prior_weight`.
- The current packaged trained model
  `smart/assets/priors/category_general_policy_value_agent_prior.json` used
  `13,439` accepted/candidate records from `119` airplane/chair/table meshes.
  The first guarded table known-case check improved BVS `1.0960 -> 1.0829`,
  MOV `1.0834 -> 0.9006`, TOV `0.0759 -> 0.0698`, and vIoU
  `0.9294 -> 0.9347`.
- The category-balanced smoke
  `runs/bench_exact/policy_value_quality_guard_cat3_mcts10.json` kept `9/9`
  guarded successes, selected the learned candidate on `1/9`, kept baseline on
  `8/9`, and rejected `2` worse candidates. Aggregate selected deltas were BVS
  `-0.001451`, MOV `-0.020312`, TOV `-0.000678`, Covered `0`, and vIoU
  `+0.000589`. This is now wired and measurable, but still too weak to promote
  without larger final-return/value training.
- Final-return trace export is now implemented in
  `scripts/run_quality_guarded_mcts.py --final-return-trace-output`. It writes
  `record_type=mcts_final_return` rows where `reward` is the final exact SMART
  quality gain and `action_reward` preserves the immediate environment reward.
  Guard-failing candidates are forced negative. Current smoke files:
  `runs/bench_exact/policy_value_final_return_trace_smoke.jsonl` (`115` rows)
  and `runs/bench_exact/policy_value_final_return_table_known.jsonl` (`185`
  rows).
- A first final-return fine-tune
  `runs/bench_exact/priors/category_general_policy_value_final_return_smoke_prior.json`
  was trained and tested in
  `runs/bench_exact/policy_value_final_return_prior_table_known.json`. It did
  not beat the packaged policy-value prior: all three candidates were rejected
  by the guard and baseline was selected. Keep the packaged prior unchanged
  until final-return traces cover a much larger category-balanced set.

## Next Work

1. Expand exact stateful/bitset and prior sweeps from the current 60-mesh
   expanded-full benchmark to 50+ meshes per category, with timeout/skip
   manifests so slow Mesh2Tet/refine outliers do not block the run.
2. Repeat larger MCTS100/MCTS300 sweeps enough times to decide whether
   `stateful_union_cache=true` can become the recommended exact accelerator.
3. Rework Rust MCTS runner so it preserves the legacy tree trajectory before
   using it in exact profiles.
4. Run the multi-weight candidate-aware PG-agent guard on a larger 20-50
   mesh/category subset. The current cat5 result is robust (`15/15` guarded
   success, `2/15` improved), but promotion needs a larger category-balanced
   improvement rate and bounded runtime overhead.
5. Collect final-return traces on a larger 20-50 mesh/category sweep. The
   trace/export/training path is now wired, but the first tiny fine-tune was
   worse than the packaged policy-value prior.
6. Validate adaptive learned-search selection on a larger 20-50 mesh/category
   subset. The new `--adaptive-prior-weights` and
   `--adaptive-stop-mode not_worse` reduce candidate launches on cat3, but we
   need a larger sweep to quantify how often they miss full multi-weight quality
   improvements.
7. Collect larger category-specific traces, then compare count, linear, PyTorch
   MLP, and PUCT action priors as MCTS ordering policies. Research profiles may
   change search order, and metric identity is not required; promotion requires
   final SMART quality to be not worse and preferably improved under the same
   or lower search budget.
8. Keep a paper-safe reproduction profile with legacy `manifold` defaults, and
   keep faster/learned search profiles behind explicit research flags.
9. Continue removing PyMesh dependency by keeping `.msh` IO and tet summaries in
   Rust.
10. Keep RL/deep-learning work as action-prior or proposal ordering only, with
   exact Manifold reward verification.
