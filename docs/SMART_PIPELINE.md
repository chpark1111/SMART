# SMART Pipeline Notes

This document maps the 3DV 2024 SMART paper stages to the official package
layout and records the current optimization plan.

For the short current-status table, see
[`CURRENT_STATUS.md`](CURRENT_STATUS.md).

## Paper To Code

The paper pipeline is:

```text
mesh -> tetrahedral mesh -> pre-segment -> split/merge -> greedy refine -> MCTS -> render
```

The official runnable pipeline is:

```text
normalize -> tetra -> preseg -> merge -> refine -> mcts -> render
```

The opt-in hybrid quality profile adds a post-MCTS local search stage:

```text
normalize -> tetra -> preseg -> merge -> refine -> mcts -> local_refine -> render
```

`normalize`
: Centers the input OBJ and scales it before meshing. The default is bbox
  diagonal `1.0` because the current prepared ShapeNet v2 samples already use
  that scale well with the Mesh2Tet parameters. `unit_sphere` is available in
  config when radius-based normalization is required.

`tetra`
: Wraps the Mesh2Tet path used by the paper: ManifoldPlus watertight repair
  followed by fTetWild tetrahedral meshing. Failures are per mesh: timeout or
  tool errors are written to manifests, retried with coarser settings when
  configured, and skipped downstream. Failure records classify timeout,
  SIGSEGV crash, external kill, missing executable, and validation failures.
  The default retry chain tries primary category parameters, a coarser retry,
  a bounded coarse retry, and a general-winding-number fallback for difficult
  open meshes. ShapeNet meshes often produce watertight but disconnected
  fTetWild surfaces; these are accepted by default because the downstream SMART
  stages can handle them. Set `tetra.require_single_component: true` for strict
  debugging. This follows fTetWild's intended use on imperfect triangle soup
  inputs rather than treating disconnected-but-valid surface extraction as a
  fatal meshing failure.
  The official pipeline also validates that `tetra.msh` contains a minimum
  number of elements and that `tetra.msh__sf.obj` contains a minimum number of
  faces (`tetra.min_tetra_count`, `tetra.min_surface_faces`). This prevents
  fTetWild fallback attempts that exit successfully but leave an empty or tiny
  mesh from reaching CoACD/SMART merge. Local fTetWild source guards are applied
  only around crash-prone diagnostics and empty tracked-surface handling; the
  fixed SMART Manifold boolean library is not modified.

  fTetWild parameter notes: `epsilon` is an envelope/fidelity setting, where a
  smaller value preserves features better but costs more runtime. `edge_length`
  maps to fTetWild's relative ideal edge length, where smaller values produce
  denser tetrahedral meshes and slower runs. See Hu et al., "Fast Tetrahedral
  Meshing in the Wild", arXiv:1908.03581 / ACM TOG 2020.

`preseg`
: Uses CoACD as the default pre-segment for the official demo pipeline. This
  matches the paper's Objaverse/OmniObject3D path. BSP-Net remains optional for
  closer ShapeNet reproduction.

`merge`
: Runs the adapted `Unsup3DMerging` SMART split/flood-fill/merge code. The
  BAVF criterion corresponds to the paper's tightness-aware hierarchical
  merging. The current code uses `abs(merge_eps)`, so config value `0.02`
  corresponds to the paper's small negative stop threshold.

`refine`
: Runs the one-step greedy box refinement from `RLfor3DTightBBoxs` with the
  soft coverage penalty. Default paper-like values are `action_unit=0.01`,
  `max_step=2000`, and `cover_penalty=100`. The official pipeline keeps
  `summary_metrics=false`, `render_initial=false`, and
  `render_partition=false` by default so inference writes final bbox OBJ
  outputs without recomputing log-only metric summaries or partition snapshots.
  Use `smart evaluate` for official metrics.

`mcts`
: Runs the paper MCTS refinement over the discrete bbox action space. Demo uses
  small iteration counts for smoke tests. Paper-like profile uses larger search
  budgets such as `mcts_iter=10000`, `exp_w=0.001`, `grdexp=true`, `pns=true`,
  and `skip_rate=0.9`. Experimental search-order changes such as
  `transposition_table=true` are blocked unless
  `mcts.allow_search_order_changes=true`; keep them disabled for exact legacy
  metric compatibility. Like refine, MCTS defaults to
  `summary_metrics=false`, `render_initial=false`, and
  `render_partition=false`; search rewards and bbox outputs are unchanged.
  `mcts.no_reward_stop_after` is an opt-in development shortcut for stopping
  no-improvement searches earlier than the legacy threshold. It can save a lot
  of time on cases where MCTS never improves the refine initialization, but it
  changes the search budget and therefore is not a paper-safe default.

`local_refine`
: Experimental hybrid stage. Reads the latest MCTS bbox output through
  `bbox_direct`, then runs greedy local search with a smaller action unit
  (`0.005` in the packaged hybrid profile). Use it when evaluating whether
  MCTS should be followed by fine local search. It changes the post-MCTS search
  trajectory and should be reported separately from paper-safe MCTS results.

`render`
: Uses the legacy Box-Renderer paper teaser path,
  `smart/legacy/renderer/render_teaser.py`, with `boxes.blend`. Default output
  is transparent, box-only, and non-overlaid; `render.joint_mesh=true` or
  `render.variants=["boxes_only","with_mesh"]` enables mesh overlay outputs.

## Parameter Control

All stage parameters are controlled by YAML config files under `configs/`.
For quick runs, values can be overridden without editing YAML:

```bash
smart --config configs/smoke_5.yaml --set mcts.mcts_iter=100 run
smart --config configs/smoke_5.yaml --set refine.backend=python refine
smart --config configs/smoke_5.yaml --set mcts.backend=python mcts
smart --config configs/smoke_5.yaml --set merge.only_nearby=true merge
smart --config configs/smoke_5.yaml --set mcts.summary_metrics=true mcts
smart --config configs/smoke_5.yaml --set render.joint_mesh=true render
smart --config configs/smoke_5.yaml --set 'render.variants=["boxes_only","with_mesh"]' render
```

Use `smart --config <profile> doctor` before long runs to verify Mesh2Tet
binaries, Blender, Python modules, the fixed vendored Manifold binding path,
and optional Rust/PyO3 tooling. When the Rust extension is installed, `doctor`
reports how many compiled kernels are visible from the active Python
environment. The command also supports `--json` for CI.

Set `SMART_DISABLE_RUST=1` to force Python fallback kernels without uninstalling
the extension. This is used for exact parity and timing checks.

Use `smart --config <profile> evaluate --stage mcts` to run the paper-style
metrics adapted from `past_codes/Evaluation`: box count, `BVS`, `MOV`, `TOV`,
`Covered`, `vIoU`, and `cub_CD`. This is the regression gate for optimization
work: exact-speed changes should keep these metrics unchanged except for tiny
floating point noise.

## Rust Migration Status

The Rust crate is `rust/smart-core` and is exposed as optional `smart._rust`
through PyO3/maturin. Python imports it through `smart.rust`; if the extension
is unavailable, matching Python fallback kernels are used.

The fixed C++ Manifold binding is deliberately out of scope for Rust migration.
Keep `smart/vendor/manifold` as the authoritative source, and continue using it
for boolean operations. Do not pull upstream, replace it, rewrite it in Rust, or
make broad source changes there; only build-path integration is allowed unless
the SMART maintainer explicitly requests otherwise.

Migrated or prepared kernels:

- `BBoxState`, the first Rust-side bbox/action-space state object; the legacy
  greedy/MCTS env uses it for cached action upper-bound scoring when available
  and keeps its legacy score snapshot synchronized after accepted axis actions
- Rust-side `BBoxState` also provides exact BVS and valid-box counts for the
  legacy refine/MCTS environment
- Discounted reward accumulation in greedy/MCTS utility paths is available as
  an exact Rust kernel and falls back to the original Python loop
- Legacy merge/refine dataloaders filter selected mesh ids before loading
  `tetra.msh`, avoiding startup checks for unrelated meshes during single-mesh
  pipeline runs
- Official greedy merge, greedy refine, and MCTS inference skip construction of
  training-only tet observation tensors; this does not affect rewards, action
  masks, random sampling, or rendered outputs because those paths return
  `None` observations
- `pymesh.py` delegates exact tetra volume, centroid, surface-face extraction,
  and voxel adjacency summaries to Rust when the extension is available; Python
  fallback code keeps the same face order and sorted adjacency lists
- Gmsh `.msh` loading and writing in the `pymesh` shim can use Rust while
  preserving the legacy node-id remapping and generated surface-face order for
  loaded tet meshes; Python fallback code remains available
- Greedy merge can compute per-partition volume, bounds, and point-list
  summaries through Rust while keeping nearby-part insertion and candidate
  action order in the legacy Python loop
- Fast greedy merge now keeps cached merge rewards in a lazy max-heap with an
  insertion counter, so selecting the best cached positive reward no longer
  scans the whole cache while preserving legacy tie-breaking
- MCTS builds opposite-action masks on demand instead of allocating a dense
  `num_actions x num_actions` mask matrix; the opposite-action lookup order is
  unchanged
- MCTS has a Rust/Python parity helper for child action masks that combines the
  opposite action with an optional inherited parent mask while preserving the
  exact legacy masking semantics
- MCTS uses Rust softmax/UCB helpers only for sufficiently large action arrays,
  avoiding PyO3 overhead on tiny rollouts while keeping Python tie-breaking and
  RNG selection unchanged
- Tiny MCTS/refine action vectors also keep the original Python untried-action,
  child-mask, and discounted-reward loops so smoke tests do not get slower from
  crossing the Python/Rust boundary; larger action spaces still use Rust
- `refine.backend=auto` can run the greedy refine control loop through Rust
  helpers when `smart._rust` is installed. Exact C++ Manifold reward evaluation
  is preserved through the legacy environment.
- For exact Manifold-family reward backends, `mcts.backend=auto` keeps the
  legacy Python MCTS tree. Explicit `mcts.backend=rust` or
  `mcts.backend=rust_stateful` is available only as an opt-in search-order
  experiment guarded by `mcts.allow_search_order_changes=true`.
- `TetClippingState` stores tetrahedra in Rust and can evaluate exact
  tetrahedron-box clipping metrics from bbox bounds/rotations. It is wired as
  the opt-in `reward_backend=tet_clipping` experiment for refine/MCTS, while
  the official default remains `reward_backend=manifold` until larger parity
  and timing sweeps prove it should replace inner-loop Manifold calls.
- `ManifoldBridgeMesh` is an exact Rust/PyO3 handle around the fixed vendored
  C++ Manifold library. It preserves the legacy Manifold metric but avoids
  recreating the source mesh and repeated Python wrapper calls for non-TOV
  coverage rewards. Axis-action greedy reward selection for this backend runs
  inside Rust/C++; greedy refine can now keep consecutive accepted axis-action
  segments inside Rust and sync only touched bbox manifolds back to Python at
  the segment boundary. MCTS rollout also batches bbox-mask axis-action probes
  through Rust/C++. The recenter/rotation candidate geometry still uses the
  exact legacy `trimesh.bounds.oriented_bounds(angle_digits=3)` computation,
  while bridge reward scoring and selected-action apply avoid rerunning the full
  legacy `step()` for scored rollout moves. The C++ wrapper also reuses
  unchanged bbox Manifold objects when testing axis candidates. Use
  `refine.reward_backend=manifold_bridge` or
  `mcts.reward_backend=manifold_bridge` for opt-in timing/parity sweeps; the
  official default remains `reward_backend=manifold` until larger bridge sweeps
  show consistent wall-clock gains.
- `ManifoldState` is the stateful version of the exact bridge. Configure
  `reward_backend=manifold_stateful` with `refine.backend=rust_stateful` or
  `mcts.backend=rust_stateful` to keep bbox bounds, rotations, valid masks,
  current score, rollback history, and candidate action rewards inside
  `smart._rust`. For MCTS, keep `mcts.backend=auto` if you need legacy tree
  compatibility. It still calls the fixed vendored Manifold C++ library and is
  opt-in until parity and timing are verified on larger smoke/demo runs. The
  latest exact cache layer stores `union_except_i` residuals for candidate bbox
  replacement scoring and defers Python bbox manifold materialization after
  accepted stateful actions. Checked smoke timings are currently a small win for
  MCTS and about `1.27x` for table refine. The latest five-mesh MCTS smoke with
  `stateful_union_cache=false` and legacy sequential union order keeps all
  reported paper metric diffs at `0`; after decoupling exact reward memoization
  from union caching, repeat-3 timing is `1.032x`, and a three-target mcts20
  check is `1.189x`. The stateful wrapper now also uses the same explicit
  eight-vertex/fixed-face bbox mesh construction as legacy Python; the
  five-target MCTS20 trace check
  `runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_trace.json`
  has no action divergence, all metric diffs `0`, and `1.024x` mean speedup;
  the repeat-3 timing run
  `runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_repeat3.json`
  keeps metric diffs at `0` and measures `1.039x`.
  `configs/accelerated_exact.yaml` keeps legacy
  `reward_backend=manifold`; use `configs/stateful_exact_experimental.yaml` for
  stateful profiling with the legacy MCTS tree. Use
  `configs/candidate_bitset_exact_experimental.yaml` to test the exact
  Manifold-verified bitset top-K candidate profile (`candidate_top_k=8`) without
  changing the legacy MCTS tree. Explicit
  `mcts.backend=rust`/`rust_stateful` is guarded behind
  `mcts.allow_search_order_changes=true` because the Rust MCTS runner can change
  rollout trajectory even with exact rewards.
- For exact Manifold reward backends, `mcts.backend=auto` currently uses the
  legacy Python MCTS tree. The Rust callback runner is still available through
  explicit `mcts.backend=rust`/`rust_stateful`, but profiling showed that the
  callback path does not become a stable speed win until more rollout state is
  kept inside Rust.
- Refine/MCTS skip log-only `current_state_summary()` recomputation and
  internal partition snapshots in the official pipeline unless
  `summary_metrics=true` or `render_partition=true` is requested. This does not
  change action choice, reward, bbox OBJ outputs, or evaluation metrics.
- Refine/MCTS step rollback now decodes legacy action ids directly and copies
  bbox rollback state with typed value copies instead of generic `deepcopy`
- CLI and legacy inference startup are lazy where possible: `smart.__init__`
  does not import evaluation/trimesh during every `python -m smart` launch, and
  greedy/MCTS single-mesh inference avoids eager Torch/DataLoader imports while
  preserving legacy float32 input arrays
- Single-mesh inference reuses the validated `tetra.msh` load for the first
  batch, and default merge skips nearby adjacency setup unless `only_nearby` is
  explicitly enabled; fast merge only uses the nearby-restricted candidate path
  when that option is enabled, otherwise it keeps the full candidate set
- bbox volume and valid-bbox checks
- bbox union bounds/volume for cheap merge scoring
- centroid coverage masks
- BAVF-style score loops
- non-mutating BAVF merge reward arithmetic
- legacy action index generation
- opposite-action pruning masks
- single-action masks for greedy expansion
- stable softmax for MCTS progressive node search
- UCB score arrays
- incremental average updates
- symmetric Chamfer distance fallback for evaluation regression checks

Next high-value Rust targets:

- direct Rust MCTS rollout storage so Python only observes final/best states
  rather than every rollout state
- direct Rust environment storage for bbox lists/action masks and longer
  accepted action segments before calling back into Python
- cached tet centroid/volume coverage summaries
- direct Manifold callback integration from Rust, keeping the fixed C++
  Manifold implementation as the exact boolean backend

## Algorithm Improvement Ideas

The fastest near-term improvement is caching. Current refinement repeatedly
recomputes boolean volumes for candidate actions. For small axis moves, many
candidate states share unchanged boxes, so cache keys based on `(box_id, bounds,
rotation)` can avoid duplicate Manifold calls. The current pipeline exposes this
as `refine.score_cache_size` and `mcts.score_cache_size`; set either to `0` to
disable the cache for debugging.

Greedy refine and MCTS greedy rollout also use an exact upper-bound pruning
step before coverage booleans. Since the active reward is
`-abs(BVS - 1) - cover_penalty * uncovered`, `-abs(BVS - 1)` is the best score a
candidate can possibly achieve. If that upper bound cannot beat the current
best action, SMART skips the expensive Manifold coverage call without changing
the selected action. This now also applies to the legacy rotation/recentering
action: that action preserves bbox volume, so its BVS-only upper bound is the
current BVS reward bound. Full-action and single-bbox upper-bound arrays are
computed through the optional Rust backend when available and have Python
fallbacks. Accepted greedy actions also reuse the exact score computed during
candidate evaluation instead of recomputing the same state immediately on
apply. The MCTS action-mask bookkeeping uses the same Rust helpers for
deterministic mask expansion when available, but avoids dense opposite-mask
matrix allocation. Score cache keys reuse the Rust bbox-state bit key while
preserving rotation state in Python. The merge stage also skips unused
observation/FPS construction during official greedy runs; bbox outputs and
merge rewards are unchanged because greedy merge never consumes that
observation. The single-bbox variant avoids rescoring every box when MCTS
rollout asks for the best action of one bbox.

The current bitset candidate profile is measured by
`scripts/benchmark_mcts_acceleration_profiles.py`. In
`runs/bench_exact/mcts_accel_profiles_smoke5_iter20_exact_candidates.json`,
`bitset_top8` preserved all reported MCTS metrics on the five-target smoke set
and measured `1.015x`; `rust_stateful_tt` was slightly faster but changed
near-tie metrics, so it remains a search-order-changing experiment.
In
`runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_topk_tt_all.json`,
the same MCTS20 comparison over three airplane, three chair, and three table
processed meshes kept all reported metric diffs at `0` for both `bitset_top3`
and `bitset_top8`; mean speedups were `1.025x` and `1.023x`. The top-3 variant
is packaged separately as `configs/candidate_bitset_fast_experimental.yaml`.
The exact leave-one-out union cache profile is
`configs/stateful_union_cache_experimental.yaml`. It leaves candidate policy
unchanged and only caches exact Manifold unions inside the Rust/C++ bridge. In
`runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_union_cache_probe.json`,
`exact_union_cache` preserved all reported metrics on the same nine processed
meshes and measured `1.097x`.
`configs/expanded_processed_16.yaml` enumerates all currently processed expanded
meshes. In
`runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_union_cache.json`,
the same exact union-cache profile preserved all metrics across those 16 meshes
and measured `1.047x`.

Trace-derived action priors are available only as search-order-changing
experiments. In
`runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_prior_weight_sweep.json`,
prior weights `0.02`, `0.05`, and `0.1` were faster than exact union-cache but
changed one table case, so they are not defaults.

Manifold boolean outputs are still produced by the fixed C++ binding. For
volume-only paths, SMART now reads the returned triangle arrays and applies the
same trimesh mass-properties volume expression directly, avoiding temporary
`Trimesh` objects in search rewards and evaluation while preserving the metric
basis.

Candidate bbox updates also build only the Manifold object required for reward
evaluation. The matching `Trimesh` bbox is constructed lazily for final OBJ
export or optional per-tet partition rendering.

Use `scripts/benchmark_rust_parity.py` to run a Python-fallback vs Rust-enabled
comparison with the same config, seed, mesh, and stage parameters. The script
writes per-stage timing, evaluation summaries, metric diffs, and speedup ratios
under `runs/bench_exact/` by default. Use `--repeat N` to average noisy
subprocess-level timings while still evaluating the final output once. For
merge-only timing runs, pass `--skip-eval` because merge emits greedy segment
txt files, not bbox OBJ directories.

Use `scripts/sweep_tet_clipping_parity.py` before enabling any tet-clipping
reward experiment broadly. It reports missing bbox outputs separately from
actual computation failures, and compares `BVS`, `MOV`, `Covered`, `TOV`, and
`vIoU` against the Manifold evaluator.

The merge stage already uses the paper's BAVF form to avoid expensive state
copying for candidate scores when `mov=false` and `tov=false`: the candidate
reward is computed from current total bbox volume, the two old bbox volumes, and
the merged bbox volume. The actual accepted merge still follows the legacy path.

For a larger quality improvement, train a lightweight policy/value model over
tet centroid summaries and current bbox parameters to guide MCTS priors. This
keeps SMART non-supervised at inference if the policy is trained from SMART's
own successful search traces, and it can reduce random exploration while
preserving the exact coverage/tightness objective.

For generalization across categories, keep the geometry objective exact and use
learning only for action ordering or pre-pruning. Replacing the volumetric
objective with a learned loss would weaken the paper's main guarantee: tight
boxes with explicit full coverage recovery.
