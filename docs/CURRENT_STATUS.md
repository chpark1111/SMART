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
- Action priors and transposition tables are opt-in because they can change
  search order. They guide MCTS; they never replace the exact reward.
- Current smoke evidence supports trace priors as a useful research path, not a
  release default.

## Current Benchmark Facts

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
- `runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_prior_weight_sweep.json`:
  first RL/MCTS action-prior weight sweep on the same 16 meshes with exact
  reward and union cache. Prior weights `0.02`, `0.05`, and `0.1` measured
  `1.043x`, `1.053x`, and `1.055x` versus `exact_union_cache`, but all changed
  the same table case. MOV improved there, while BVS/TOV/vIoU worsened, so the
  current prior remains a search/quality experiment, not a default.
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

## Next Work

1. Expand the exact stateful and bitset sweeps from 9 processed meshes to the
   prepared 150/expanded ShapeNet samples with `stateful_union_cache=false`.
2. Repeat larger MCTS100/MCTS300 sweeps enough times to decide whether
   `stateful_union_cache=true` can become the recommended exact accelerator.
3. Rework Rust MCTS runner so it preserves the legacy tree trajectory before
   using it in exact profiles.
4. Collect larger category-specific traces, then compare count, linear, PyTorch
   MLP, and PUCT action priors as MCTS ordering policies. Research profiles may change search
   order, but promotion requires aggregate quality to stay equal or improve under
   final SMART evaluation.
5. Keep a paper-safe reproduction profile with legacy `manifold` defaults, and
   keep faster/learned search profiles behind explicit research flags.
6. Continue removing PyMesh dependency by keeping `.msh` IO and tet summaries in
   Rust.
7. Keep RL/deep-learning work as action-prior or proposal ordering only, with
   exact Manifold reward verification.
