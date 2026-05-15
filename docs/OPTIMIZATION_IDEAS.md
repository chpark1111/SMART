# SMART Optimization Ideas

This note records upgrade points to consider before moving more of SMART into
Rust. The fixed C++ Manifold binding remains unchanged.

For the short current applied-vs-experimental status, see
[`CURRENT_STATUS.md`](CURRENT_STATUS.md).

Current migration rule: port the legacy algorithm exactly first. Rust changes
must preserve the same action choices, bbox outputs, and evaluation metrics as
the Python/C++ path, aside from unavoidable floating-point representation noise.
Search-order changes, learned policies, approximate prefilters, and new memory
tables are deferred until after exact parity is locked.

## Highest Priority

0. Implemented exact greedy action pruning and Rust bbox state

   Greedy refine and the greedy rollout inside MCTS now use a safe upper bound
   before calling the expensive Manifold coverage boolean. The reward is
   `-abs(BVS - 1) - cover_penalty * uncovered`, so the best possible coverage
   term is zero. If this upper bound cannot beat the current best action, the
   exact coverage call is skipped. This preserves the selected action and the
   paper objective while reducing repeated Manifold calls. The upper-bound
   action array is now computed by the Rust backend when available, including a
   single-bbox variant used by MCTS rollout. The current Rust `BBoxState`
   object is an exact cache of the legacy bbox/action-space state, not a new
   search policy. It now also serves exact BVS and valid-box counts to avoid
   rebuilding those summaries in Python. Score cache keys now reuse the Rust
   bbox-state bit key for bbox coordinates while preserving rotation state in
   Python.

1. Implemented unused greedy-merge observation skip

   Official greedy merge does not consume the observation returned by
   `TetMeshEnv.step`; it only consumes the reward and then writes bbox outputs.
   For `run_type=greedy`, the environment now skips observation/FPS construction
   while preserving merge choices, rewards, and output files.
   Greedy refine and MCTS inference similarly skip construction of
   training-only tet observation tensors because their `current_observation`
   path returns `None` outside `run_type=train`.

2. Implemented exact discounted reward kernel

   The shared `calculate_reward` helper used by greedy/MCTS paths now delegates
   to a Rust `discounted_reward` kernel when available. The accumulation order is
   identical to the original Python loop.

3. Implemented selected-mesh dataloader filtering

   Legacy merge/refine dataloaders used to load and validate every `tetra.msh`
   in a category before filtering to `--meshes`. Official single-mesh pipeline
   stages now filter selected mesh prefixes first, so unrelated meshes are not
   loaded during startup. Selected mesh results and metrics are unchanged.

4. Implemented tetmesh summaries in Rust

   The `pymesh` shim now delegates exact tetra volume, centroid,
   surface-face extraction, and voxel adjacency summaries to Rust when the
   extension is available. The fallback Python code is still present and keeps
   the same face order and sorted adjacency lists. Gmsh `.msh` loading is also
   available through Rust with the same node-id remapping and generated surface
   order. The remaining packaging target is Rust `.msh` writing so pip users no
   longer depend on PyMesh-style IO behavior.

5. Implemented exact merge partition summaries

   Greedy merge can compute per-partition volume, bbox bounds, and point arrays
   through Rust while preserving the Python nearby-part set construction and
   candidate action order. This is an exact setup-time optimization; small
   smoke runs are noisy, so keep using metric parity as the regression gate.

6. Implemented cached merge reward heap

   Fast greedy merge now stores cached rewards in a lazy max-heap with an
   insertion counter. The cache is now treated as complete only after all
   currently valid candidates have been scored; after an applied merge, rewards
   for newly valid pairs touching the survivor are filled immediately. Heap
   selection uses the same merge threshold and then resolves ties in the legacy
   scan order, so it preserves greedy choices while avoiding broad rescans once
   the candidate cache is warm.

   Greedy merge also now asks the Rust partition-summary kernel to return
   exact-unique tetra vertices for greedy inference and avoids a defensive
   `deepcopy` before bbox construction. Duplicate vertices do not affect
   axis-aligned or tilted bounding boxes, so greedy partition outputs are
   unchanged. Smoke parity checks matched the previous `greedy_segment` outputs
   for airplane and chair; measured merge time improved from `0.878s` to about
   `0.855s` on the tested airplane and from `0.180s` to about `0.150s` on the
   tested chair.

7. Implemented exact MCTS bookkeeping reductions

   MCTS now builds opposite-action masks on demand instead of allocating a dense
   `num_actions x num_actions` matrix. It also routes softmax/UCB arrays through
   Rust only when the arrays are large enough to amortize PyO3 overhead; small
   rollouts keep the original Python math and random tie-breaking path. The
   same threshold is used for untried-action masks, child masks, and discounted
   rewards so tiny smoke runs do not pay Rust call overhead while larger action
   spaces still use the compiled kernels. A Rust/Python parity helper is also
   available for child action masks that combine the opposite-action block with
   an inherited parent mask without changing search order.

8. Implemented exact refine/MCTS step bookkeeping cleanup

   The bbox environment now decodes the legacy action id directly instead of
   indexing the NumPy `action2idx` table for every trial step, and uses direct
   value copies for bbox/rotation rollback state instead of generic
   `deepcopy`. This preserves the legacy action encoding and rollback values.

9. Implemented lazy inference imports and simple single-mesh loaders

   The package root no longer imports the API/evaluation stack during
   `python -m smart` startup, so CLI stage launches avoid importing `trimesh`
   until the selected command needs it. Legacy greedy/MCTS inference paths also
   avoid eager Torch/TensorBoard imports and use a NumPy single-mesh loader when
   `data_batch_size=1` and `worker=0`. The arrays keep the legacy `float32`
   tensor values, and training paths still import Torch and use Torch
   `DataLoader`.

10. Implemented default merge setup dead-work skips

   Single-mesh inference reuses the `tetra.msh` object loaded during dataset
   validation instead of parsing the same file again for the first batch. The
   merge environment also skips nearby-part adjacency construction when
   `only_nearby=false`, which is the official/default pipeline path. The
   adjacency logic is still preserved for explicit `--only_nearby` runs. Fast
   merge now uses the nearby-only candidate path only when `only_nearby=true`;
   the default path keeps the full candidate set so large CoACD segment counts
   do not silently stop with an empty nearby graph.

11. Implemented unused import cleanup from profiling

   MCTS profiling showed startup was dominated by geometry imports after
   TensorBoard/Torch were made lazy. The old bounding-box helpers imported
   `sklearn.PCA` even though the runtime code never used it, and several legacy
   files eagerly imported unused `trimesh` submodules. Removing those imports
   reduces startup work without touching bbox construction, Manifold booleans,
   candidate order, or evaluation metrics.

12. Implemented Rust callback runners for greedy refine and experimental MCTS

   `refine.backend=auto` can run the greedy refinement control loop in Rust
   when `smart._rust` is installed. The exact geometry state, rewards,
   rendering, and C++ Manifold boolean calls remain in the legacy Python
   environment. The Rust MCTS callback runner is implemented, but it is now
   treated as a search-order-changing experiment for exact Manifold-family
   reward backends and is guarded by `mcts.allow_search_order_changes=true`. The
   smoke benchmark `runs/bench_exact/smoke_5_rust_action_selectors_threshold.json`
   preserved all evaluation metrics (`BVS`, `Covered`, `MOV`, `TOV`, box count,
   and `vIoU`; `cub_CD` differs only by floating noise) and measured about
   `1.11x` MCTS, `1.14x` refine, and `1.10x` merge speedup on the small table
   smoke case. Per-candidate Rust action selection remains gated to larger
   action spaces because tiny smoke cases spend more time crossing PyO3 than
   they save in compiled loops.

13. Implemented Rust Gmsh writing in the PyMesh shim

   The `pymesh.save_mesh(... .msh)` path now delegates Gmsh 2.2 writing to Rust
   when `smart._rust` is installed, with the original Python writer as fallback.
   This does not change SMART metrics, but it removes another PyMesh-style IO
   dependency from merge/refine/MCTS render outputs and helps the future pip
   package path.

14. Implemented Rust tet-clipping parity kernel

   `scripts/compare_tet_clipping_manifold.py` computes exact convex
   tetrahedron-box intersection volumes and compares `BVS`, `MOV`, `Covered`,
   `TOV`, and `vIoU` against the current Manifold boolean evaluator. The script
   now supports `--backend rust`, which calls `smart.rust.tet_clipping_metrics`.
   On smoke MCTS outputs the Rust absolute differences are table `<=4.4e-7`,
   chair `<=1.9e-8`, and airplane `<=1.8e-6`. The same comparison took about
   `0.059s` for table, `0.093s` for chair, and `0.051s` for airplane with the
   Rust clipping backend, versus the earlier Python/SciPy experiment at about
   `2.0s`, `8.4s`, and `1.2s` respectively. A refine-stage airplane case with
   seven boxes initially exposed a volume parity bug; the Rust kernel now accepts
   explicit legacy bbox mesh volumes, matching Manifold/trimesh BVS while using
   tet clipping for intersections. Manifold remains the official verification
   path until this parity is measured on larger category sweeps.

15. Identified safe Manifold direct-call path

   The fixed vendored Manifold source already exposes a C API under
   `smart/vendor/manifold/bindings/c/include/manifoldc.h`. The minimum SMART
   subset is mesh construction, cube construction/transform, union,
   difference, intersection, `GetProperties().volume`, and mesh extraction for
   output. A future Rust wrapper should bind this C API or a tiny C++ shim
   around it, not rewrite Manifold. That would remove Python `pymanifold`
   round-trips while preserving the same C++ boolean implementation.

16. Implemented inference-only summary/render-output skips

   Refine and MCTS now separate required bbox OBJ output from log-only metric
   summaries and internal partition snapshots. The official pipeline passes
   `--skip_summary_metrics`, `--skip_initial_render`, and
   `--skip_render_partition` by default. Search rewards still use the same
   exact Manifold coverage objective, accepted actions are unchanged, and final
   evaluation still uses `smart evaluate`; this only removes duplicate
   Manifold metric work and heavyweight `.msh` snapshot exports from inference
   subprocesses.

17. Implemented exact accepted-action score reuse and rotation-action bound

   Greedy refine and MCTS greedy rollout evaluate candidate actions with
   `apply=0`, then immediately apply the winning action. The winning state score
   is now cached as `(reward, next_score)`, so the apply step can reuse the same
   exact score instead of calling Manifold coverage again for the identical
   state. The cache key includes the full bbox/rotation state, action,
   previous score, coverage penalty, and TOV flag.

   The legacy rotation/recentering action no longer has an infinite upper
   bound. It does not change bbox dimensions, so the best possible reward is
   the current BVS-only reward bound. SMART still evaluates that action whenever
   it could beat the current best candidate, but skips it when the exact reward
   cannot exceed the current best. The smoke benchmark
   `runs/bench_exact/smoke_airplane_rotation_upper_bound.json` preserved all
   reported metrics and reduced the checked airplane refine runtime to about
   `5.06s` on the Rust-enabled path.

18. Implemented direct triangle-volume extraction for Manifold outputs

   The legacy path still calls the fixed C++ Manifold boolean operations, but
   it no longer wraps every boolean result in a `trimesh.Trimesh` object just to
   read `.volume`. Instead, SMART reads `to_mesh()` arrays and applies the same
   triangle mass-properties volume expression used by trimesh. This keeps the
   metric basis aligned with the old path while removing object construction
   overhead in `Covered`, `MOV`, `TOV`, `IoU`, and evaluation. The benchmark
   `runs/bench_exact/smoke_airplane_fast_volume_formula.json` preserved all
   reported metrics and measured about `4.85s` refine and `6.73s` MCTS on the
   Rust-enabled checked airplane smoke case.

19. Implemented lazy bbox mesh construction in refine/MCTS

   Candidate reward evaluation only needs the Manifold representation of each
   bbox. The legacy path used to construct a `Trimesh` bbox object for every
   candidate as well, even though that mesh was only needed for final OBJ export
   or optional per-tet partition snapshots. The refine environment now builds
   the Manifold directly from the same bbox vertices/faces and creates the
   `Trimesh` object lazily when rendering/exporting asks for it. A parity check
   verified the direct vertices/faces path matches the old `oriented_bbox`
   output for positive- and negative-determinant rotations. The benchmark
   `runs/bench_exact/smoke_airplane_lazy_bbox_mesh.json` preserved all reported
   metrics on the checked airplane smoke case.

20. Implemented tet-clipping parity sweep gate

   `scripts/sweep_tet_clipping_parity.py` now checks existing merge/refine/MCTS
   bbox outputs across many meshes and reports Manifold-vs-Rust tet-clipping
   differences, missing outputs, and parity failures separately. This is the
   gate for any future Manifold inner-loop replacement. On the current smoke
   outputs, 10/10 records ran successfully; 9/10 were within `1e-5`, and the
   single outlier was an airplane MCTS MOV difference of `1.057e-5`. With a
   practical floating tolerance of `2e-5`, the current checked smoke and
   expanded refine records pass. This is not enough to change the official
   default yet; it is enough to continue opt-in experiments.

21. Implemented opt-in Rust tet-clipping reward backend

   The Rust extension now exposes `TetClippingState`, which stores tetra
   vertices/voxels once inside Rust and can evaluate bbox coverage metrics
   without resending the tet mesh from Python every reward call. The refine
   environment accepts `--reward_backend tet_clipping` and the pipeline can pass
   `refine.reward_backend` or `mcts.reward_backend`. The default remains
   `manifold`.

   A specialized `metrics_for_boxes` path builds the six exact bbox halfspace
   planes directly from bbox bounds and rotations, avoiding a per-action convex
   hull over bbox vertices. This preserves the same box geometry. Early timing
   on the small smoke table case still shows tet-clipping reward is slower than
   Manifold (`~0.40s` vs `~0.03s` for a one-step greedy smoke), so it must stay
   experimental until larger sweeps prove it helps.

22. Implemented narrower greedy-merge heap refresh

   Fast merge already uses a lazy heap and exact Rust BAVF reward helper. After
   an accepted merge, the default non-nearby path now refreshes only candidate
   rewards touching the survivor partition instead of scanning every remaining
   pair to find missing cache entries. Tie resolution still scans in the legacy
   candidate order before returning a heap winner, so merge choices are
   preserved. On the tiny smoke table case the measured speedup is mostly noise
   (`~1.02x`), but this reduces the remaining cache-refresh cost for larger
   CoACD partitions without changing reward math.

23. Implemented exact Manifold C++ bridge reward backend

   The Rust extension now optionally links against the fixed vendored
   `smart/vendor/manifold` build without modifying that source. It exposes
   `ManifoldBridgeMesh`, which stores the source mesh once and evaluates
   `source - union(boxes)` residual volume through the same C++ Manifold
   boolean implementation. The refine/MCTS environment accepts
   `reward_backend=manifold_bridge`; TOV summaries still fall back to the
   legacy path, so this accelerates the default non-TOV reward loop while
   preserving the metric.

   Reproducible checks:
   `runs/bench_exact/reward_backend_refine_table_axisrust_upperdirect_repeat3.json`
   reports zero metric differences and `1.06x` average speedup for a repeat-3
   10-step table refine smoke run.
   `runs/bench_exact/reward_backend_mcts_table_axisrust_upperdirect_repeat3.json`
   reports zero metric differences and `1.09x` average speedup for repeat-3
   10-iteration, 5-step table MCTS smoke runs. The current bridge path keeps
   axis-action greedy reward selection inside Rust/C++ and leaves only accepted
   steps and the recenter/rotation action in the Python env.
   `runs/bench_exact/smoke5_reward_backend_axisrust_bridge_summary.json`
   extends the parity check to the current five-mesh smoke set with zero metric
   differences; wall-clock timing is mixed at that size because process startup
   and Python environment work can dominate. Use
   `scripts/benchmark_reward_backends.py` to extend this to the full
   150/expanded set before changing defaults.

24. Implemented exact Rust refine axis-action segments

   Greedy refine now asks `ManifoldBridgeMesh` to continue applying consecutive
   axis-only actions inside Rust/C++ whenever the exact axis reward provably
   beats the legacy recenter upper bound. This preserves action order and the
   Manifold coverage metric, then syncs only touched bbox manifolds back to the
   Python environment at the segment boundary. The legacy recenter/rotation
   action remains in Python.

   Reproducible checks:
   `runs/bench_exact/reward_backend_refine_table_segment_private_repeat3.json`
   reports zero metric differences and `1.35x` average speedup for the checked
   repeat-3 10-step table refine run.
   `runs/bench_exact/smoke5_refine_segment_private_bridge_summary.json` reports
   zero metric differences and `1.18x` speedup on the current five-mesh smoke
   refine run.

25. Implemented exact MCTS bbox-mask axis batch scoring and scored apply

   MCTS rollout now calls `ManifoldBridgeMesh.best_axis_actions_for_mask` once
   per rollout step for the active bbox mask instead of probing each bbox
   through a separate Python/Rust bridge call. The batch path evaluates the same
   axis actions with the same C++ Manifold residual-volume metric. Per-bbox
   recenter/rotation candidate geometry still uses the exact legacy
   `trimesh.bounds.oriented_bounds(angle_digits=3)` computation. The Rust MCTS
   runner now applies the selected scored action through the bridge sync path
   instead of rerunning the full legacy `step()`. A C++ wrapper path also reuses
   unchanged bbox Manifold objects while testing candidate axis moves; this does
   not change the metric, but reduces part of the bridge conversion cost.

   Reproducible checks:
   `runs/bench_exact/reward_backend_mcts_table_cpp_batchboolean_repeat3.json` reports
   zero metric differences and `1.09x` speedup for a repeat-3 10-iteration,
   5-step table MCTS run.
   `runs/bench_exact/reward_backend_mcts_airplane10_cpp_batchboolean_iter300_repeat1.json`
   reports zero metric differences on a larger 10-box airplane case, but is
   still slower (`0.62x`) because the exact bridge continues to pay repeated
   Manifold union/residual costs for many boxes. Keep `reward_backend=manifold`
   as the official default while this path remains opt-in.

26. Cleaned up merge/evaluation volume-only Manifold paths

   Optional merge MOV/TOV/adjacency paths and the standalone evaluation module
   now reuse the same direct triangle-volume helper for Manifold `to_mesh()`
   outputs. Default BAVF merge is still governed by the exact same candidate
   scores and heap/cache ordering. For timing-only merge checks,
   `scripts/benchmark_rust_parity.py` now supports `--skip-eval`, because merge
   emits greedy segment txt files rather than bbox OBJ directories. The checked
   table smoke merge timing was about `0.98s` with Python fallback and `0.72s`
   with Rust enabled.

27. Stateful exact Manifold cache for refine/MCTS

   Added the opt-in `reward_backend=manifold_stateful` path. It keeps bbox
   bounds, rotations, valid mask, cached bbox Manifold objects, current exact
   score, rollback history, and candidate action rewards inside
   `smart._rust.ManifoldState`. Greedy axis refinement can now run through this
   stateful bridge with `greedy_backend=rust_stateful`. MCTS can use the same
   stateful reward backend while keeping the legacy tree with
   `mcts.backend=auto`; the explicit Rust MCTS runner is kept as a
   search-order experiment. This is still exact Manifold evaluation and does not
   modify the fixed vendored Manifold source. Optional `trace_actions_path`
   JSONL logging records accepted actions for later learned policy/RL priors
   without changing search behavior. The official default remains
   `reward_backend=manifold` until expanded parity and timing sweeps show a
   stable win.

   The current wrapper also caches exact `union_except_i` residuals for
   candidate replacement scoring, so unchanged bbox Manifold objects do not need
   to be rebuilt for every candidate. Latest checked timing with exact metric
   parity:

   - `runs/bench_exact/stateful_table_refine_step10_repeat3_deferredsync.json`:
     `1.13x` speedup, zero metric differences. This case converges after two
     accepted steps even when `max_step=10`.
   - `runs/bench_exact/stateful_table_refine_smoke_cache_keep.json`: `1.26x`
     speedup on the earlier short 2-step smoke, zero metric differences.
   - `runs/bench_exact/stateful_table_mcts_iter10_repeat3_exceptunion.json`:
     `1.04x` speedup, zero metric differences.
   - `runs/bench_exact/stateful_table_mcts_iter10_repeat3_deferredsync.json`:
     `0.98x`, zero metric differences. At this tiny scale the table MCTS timing
     is noise-level and not a stable win.
   - `runs/bench_exact/stateful_airplane_mcts_iter20_deferredsync.json`:
     `1.08x` speedup on a 7-box airplane smoke, zero metric differences.

   These results are directionally correct but small. The bottleneck is now the
   exact Manifold boolean/residual evaluation itself. The next exact-speed
   target is to keep more MCTS selection/simulation state entirely inside
   `ManifoldState` for longer segments, so Python only observes final/best states
   rather than every rollout state.

28. Re-measured exact MCTS backend selection

   Profiling the direct child process avoids the parent CLI subprocess wait
   noise. On a 7-box airplane 100-iteration MCTS smoke, the exact
   `manifold_stateful` path preserved all metrics but was slower than the
   baseline (`0.85x` in
   `runs/bench_exact/stateful_airplane_mcts_iter100_bottleneck.json`). The
   child profile showed the dominant stateful cost in
   `ManifoldState.score_action_batch` (`6.95s` cumulative self-time), with
   `trimesh.bounds.oriented_bounds` recenter work also visible (`2.59s`
   cumulative). The baseline exact Manifold path spent most of its time in
   `_manifold_mesh_volume`/`Covered` (`4.16s`) plus the same recenter family.

   Because the current Rust MCTS runner still calls Python env methods for
   exact reward/application, `mcts.backend=auto` now keeps the original Python
   tree for `reward_backend=manifold`, `manifold_bridge`, and
   `manifold_stateful`. Explicit `mcts.backend=rust` remains available for
   experiments, but it is not promoted as the exact default until rollout state
   and reward application live deeper inside Rust/C++.

29. Implemented persistent exact state/action memoization

   `ManifoldState` now keeps exact state/action reward cache entries across
   MCTS `env.reset()` calls. The previous cache was tied to a transient
   per-reset version, so repeated states reached in later MCTS iterations lost
   the C++ cache. The new key hashes the exact bbox bounds, rotations,
   `volume_sum`, action, action scale, action unit, coverage penalty, and
   penalty rate. This is pure memoization of the same fixed Manifold C++
   evaluator; it does not approximate or change the objective.

   The Python environment now preserves the stateful C++ object on reset and
   calls `reset_to_state()` instead of deleting the object. Cache stats are
   exposed through `ManifoldState.cache_stats()` and printed after Python MCTS
   when available. Checks:

   - small table MCTS, 10 iterations: `63` cache hits, `229` misses, zero metric
     differences, `1.017x` versus legacy `manifold` on repeat-3 timing.
   - airplane MCTS, 100 iterations: `5186` cache hits, `12118` misses, zero
     metric differences, still `0.90x` versus legacy `manifold`. This confirms
     the cache is active but the exact stateful wrapper still pays more per
     candidate than the legacy path on larger box counts.

30. Implemented opt-in trace action priors

   `smart build-prior` and the package API
   `smart.action_prior.build_action_prior_from_traces()` build JSON priors from
   MCTS trace JSONL records. MCTS accepts `--action_prior_path` and
   `--action_prior_weight`; the prior only changes PNS sampling logits and does
   not replace the exact reward. Relative config paths are resolved from the
   repo root before the legacy subprocess is launched, so traces and priors no
   longer land under `smart/legacy/refine` by accident.

   This is the first safe RL/deep-learning hook: learned models should emit
   action priors or top-K proposal order. SMART still verifies selected actions
   with exact Manifold or the configured exact backend.

31. Reduced exact stateful MCTS reset/apply overhead

   MCTS reset now caches deterministic initial bbox parameters in memory for
   `bsp_preseg`, `grd_merged`, and `bbox_direct` initializers. This avoids
   repeated bbox OBJ loading and oriented-bounds recomputation across rollouts
   without changing the exact reward. The option is controlled by
   `cache_initial_bbox_state` and is enabled by default.

   `smart._rust.ManifoldState` now exposes `apply_axis_action_delta()`, backed
   by a wrapper-level C++ `smart_manifold_state_copy_bbox()` API. Accepted
   stateful axis actions copy only the changed bbox bounds/rotation back to
   Python instead of materializing the full bbox state after every accepted
   action. The fixed vendored Manifold source remains unchanged.

   Airplane 20-iteration MCTS smoke:

   - cache off: `initial_bbox_cache_hits=0`, `initial_bbox_cache_misses=22`,
     `8.49s`.
   - cache on + delta sync + lazy bounds/rotation materialization:
     `initial_bbox_cache_hits=21`, `initial_bbox_cache_misses=1`, `8.25s`.
   - bbox OBJ outputs matched exactly by `diff -qr`.

   The time delta is noise-level on this small workload, so the remaining
   dominant cost is still exact Manifold boolean scoring. This change is still a
   useful structural migration because it removes repeated Python/file IO/state
   copying from the MCTS inner loop.

   A fused MCTS rollout-step experiment (`mcts.fused_rollout_step=true`) also
   merged batch scoring, best-action selection, and apply into one Python env
   call. It preserved identical bbox OBJ outputs and recorded
   `fused_rollout_steps=92` on the same 20-iteration airplane smoke, but timing
   was slower (`8.65s`) than the non-fused exact stateful path (`8.25s`).
   Therefore it stays opt-in and is not promoted to the default.

32. Implemented lazy exact `union_except_i` construction

   The stateful Manifold wrapper no longer eagerly builds every leave-one-out
   union for candidate replacement scoring. It now allocates the cache for the
   current state, but builds only the requested `union_except_i` entry on first
   use and reuses it until the bbox state changes. This keeps the reward metric
   identical because it still calls the same fixed Manifold boolean operations;
   it only changes cache construction timing.

   Checked airplane MCTS results:

   - 20-iteration smoke:
     `runs/smoke_5/mcts/airplane/.../1f5537f4747ec847622c69c3abc6f80_lazy_except_stats2`,
     `8.37s`, same bbox OBJ outputs and metrics as the pre-lazy stateful run,
     `except_union_builds=389`, `except_union_cache_hits=4174`,
     `reward_cache_hits=1296`, `reward_cache_misses=4563`.
   - 100-iteration smoke:
     `runs/smoke_5/mcts/airplane/.../1f5537f4747ec847622c69c3abc6f80_lazy_except_mcts100`,
     `18.48s` at `max_step=20`, exact metrics
     `BVS=2.119659339401995`, `Covered=0.999203597577755`,
     `TOV=1.0565878326097768`, `vIoU=0.4856670957154915`, `num_box=7`,
     `except_union_builds=880`, `except_union_cache_hits=9682`,
     `reward_cache_hits=8120`, `reward_cache_misses=10562`.

   This is useful but does not fully solve MCTS runtime. The remaining high
   cost is exact residual recomputation through Manifold for many candidate
   states. The next exact-only targets are larger repeated sweeps for
   `manifold_stateful`, then reducing candidate residual calls without changing
   search order or reward values.

33. Fixed MCTS no-output success cases

   While rechecking table MCTS, the command returned success even when the new
   exp tag produced only `time.txt`/`args.json` and no `bboxs_steps*` directory.
   The cause was the pipeline success check scanning the whole stage tree and
   accepting an older bbox output for the same mesh. `latest_bbox_dir()` now
   supports a `since=` filter and refine/MCTS use it after a forced run, so an
   old result cannot mask a current no-output run.

   MCTS also now writes a fallback result when search finds no improved update:
   it resets to the deterministic initial bbox state and renders that bbox as
   `updated0`. On the checked table 20-iteration smoke, baseline exact Manifold
   and `manifold_stateful` both write `bboxs_steps0`, and `diff -qr` confirms
   identical bbox OBJ files. This does not change the MCTS objective; it only
   guarantees the stage has a usable final bbox output.

34. Added exact stateful promotion sweep harness

   Added `scripts/benchmark_exact_stateful_sweep.py` to compare exact
   `reward_backend=manifold` against exact `reward_backend=manifold_stateful`
   over multiple configured meshes. The script runs the selected stage,
   evaluates paper metrics, records speedups, captures `rust_stats.json`, and
   fails if the SMART stage summary reports a failed run even when the Python
   process exits with return code `0`.

   The first MCTS smoke result with legacy MCTS tree exposed the remaining
   parity issue:

   - `runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_auto_trace.json`
   - five configured smoke meshes, `mcts_iter=5`, `max_step=5`
   - mean stateful speedup: `1.02x`
   - four meshes have zero metric drift
   - one chair mesh changes a near-tie action and produces small metric drift:
     `TOV=5.65e-6`, `vIoU=4.86e-6`, `BVS=8.46e-9`

   Trace comparison shows the drift starts at a near-tie bbox action:
   baseline chooses action `8`, while stateful scoring chooses action `3`.
   The bbox coordinate difference is one action unit (`~0.02`) in the final
   `bbox0.obj`. This means the exact stateful wrapper is still an opt-in
   accelerator; promotion requires either bit-for-bit legacy action ordering or
   an explicit tolerance policy accepted by the paper reproduction tests.

35. Split paper-safe exact and stateful experimental profiles

   `configs/accelerated_exact.yaml` now keeps `reward_backend=manifold` for
   refine and MCTS so it remains safe for paper-style exact reporting. The
   previous stateful settings moved to
   `configs/stateful_exact_experimental.yaml`, which explicitly opts into
   `reward_backend=manifold_stateful` for profiling while keeping
   `mcts.backend=auto` so MCTS still uses the legacy tree. Explicit
   `mcts.backend=rust`/`rust_stateful` is now guarded by
   `mcts.allow_search_order_changes=true`.
   The same split is mirrored in packaged `smart/configs/*.yaml`, and tests
   guard that `accelerated_exact` cannot silently drift back to the stateful
   backend before promotion criteria pass.

36. Added action-trace divergence diagnostics for stateful promotion

   `scripts/benchmark_exact_stateful_sweep.py --trace-actions` now writes one
   JSONL action trace per backend and records the first divergence versus
   `reward_backend=manifold`. The default stateful control backend is now
   `auto`, which keeps the legacy MCTS tree. A tiny one-mesh MCTS check
   (`runs/bench_exact/exact_stateful_trace_smoke1_mcts1_auto.json`) preserved
   all metrics, kept both traces at 11 actions, and reported no action
   divergence with max reward drift about `1.02e-7`. Running the same check with
   `--stateful-control-backend rust_stateful` confirmed the Rust MCTS runner is
   a search-order-changing experiment: it produced a shorter trace and different
   final metrics, so the pipeline now requires
   `mcts.allow_search_order_changes=true` for explicit Rust MCTS backend runs.

37. Matched legacy Manifold union order for stateful exact sweeps

   The stateful C++ wrapper now has a legacy-compatible sequential union path:
   bbox manifolds are unioned in bbox index order, matching the original Python
   `merged = merged + bbox_i` loop instead of using grouped `BatchBoolean`
   unions. The stateful exact experimental configs now set
   `stateful_union_cache=false`, because the faster `union_except_i` cache can
   change the boolean grouping order and therefore near-tie action ranking.

   The MCTS driver also now removes stale `rust_stats.json` at the start of a
   run and writes fresh stateful cache stats even when the legacy Python MCTS
   tree is used. This avoids benchmark reports mixing current metrics with cache
   stats from an older experimental run.

   Latest checked smoke result:

   - `runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_legacy_union_fresh_stats.json`
   - five configured smoke meshes, `mcts_iter=5`, `max_step=5`
   - all reported metric diffs vs `reward_backend=manifold` are `0`
   - mean stateful speedup: `1.0001x`
   - fresh stats show `stateful_union_cache=0` and
     `except_union_builds=0`
   - repeat-3 timing in
     `runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_legacy_union_repeat3.json`
     also keeps all reported metric diffs at `0`, with mean speedup `0.986x`

   This fixes the known metric drift in the small smoke sweep, but it does not
   prove default-readiness. Larger repeated sweeps are still required, and the
   current speed is effectively tied to slightly slower than the legacy backend.

38. Decoupled exact reward memoization from `union_except_i` caching

   `stateful_union_cache=false` must stay available because it preserves the
   original sequential Manifold union order. Previously that also disabled the
   exact reward cache. The stateful wrapper now keeps the reward cache active
   even when `union_except_i` caching is disabled, so repeated state/action
   scores are reused without changing boolean grouping or the reward metric.

   Checks:

   - `tests/test_rust_fallback.py` now asserts `ManifoldState` reuses reward
     cache entries when constructed with `stateful_union_cache=False`, while
     `except_union_builds` remains `0`.
   - `runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_rewardcache_repeat3.json`:
     five smoke meshes, repeat 3, all reported metric diffs `0`, mean speedup
     `1.032x`.
   - `runs/bench_exact/exact_stateful_sweep_smoke3_mcts20_rewardcache.json`:
     three smoke targets, `mcts_iter=20`, `max_step=20`, all reported metric
     diffs `0`, mean speedup `1.189x`; the 7-box airplane case recorded
     `2469` reward-cache hits and `5098` misses.

   This is the first exact stateful MCTS change that shows a repeatable speed
   signal without enabling a search-order-changing Rust MCTS runner. It is still
   not the official default until larger category sweeps pass.

39. Matched stateful bbox Manifold construction to legacy Python

   The legacy environment builds every bbox Manifold from eight explicit bbox
   vertices and the fixed `_BOX_FACES` triangle list. The stateful bridge had
   been using `Manifold::Cube(...).Transform(...)`, which is geometrically the
   same box but can present a different mesh/boolean input to Manifold. On a
   smoke5 MCTS20 chair case this produced tiny score drift that eventually
   changed a near-tie branch.

   The bridge now calls the same explicit vertex/fixed-face path used by
   legacy Python for all stateful bbox construction. The fixed vendored
   Manifold source remains untouched.

   Check:

   - `runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_trace.json`:
     five smoke targets, `mcts_iter=20`, `max_step=20`, all reported metric
     diffs `0`, no action trace divergence, max pre-divergence reward drift at
     roundoff level (`~1e-14`), mean speedup `1.024x`.
   - `runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_repeat3.json`:
     same smoke without trace logging, repeat 3, all reported metric diffs
     `0`, parity failures `{}`, mean speedup `1.039x`.

   This moves `manifold_stateful` from "metric parity on short smoke only" to
   "metric and action-trace parity on the current five-target MCTS20 smoke".
   It is still opt-in until repeated and expanded-data sweeps pass.

## Deferred Experiments

0. Bitset/voxel coverage prefilter

   Added `scripts/experiment_bitset_prefilter.py` to compare a
   voxel-centroid/bitset-style coverage proxy against exact Manifold action
   rewards. This does not change the pipeline; it measures whether a proxy could
   safely propose a small top-K candidate set for later exact verification.
   The proxy action scoring kernel now also exists in Rust as
   `smart.rust.centroid_proxy_axis_rewards`; it stores centroid coverage as
   packed bitsets and computes weighted coverage from tetra volumes.

   Initial smoke probes:

   - `runs/bench_exact/bitset_prefilter_airplane_mcts_rust_proxy_probe.json`: 84 axis
     actions, proxy top-1 equals exact top-1, Pearson `0.999`, proxy scoring
     about `72x` faster than exact action scoring.
   - `runs/bench_exact/bitset_prefilter_table_refine_rust_proxy_probe.json`: 24 axis
     actions, proxy top-1 equals exact top-1, proxy scoring about `35x` faster.
   - `runs/bench_exact/bitset_prefilter_chair_refine_rust_proxy_probe.json`: 48 axis
     actions, proxy top-1 equals exact top-1, proxy scoring about `120x` faster.
   - `runs/bench_exact/bitset_prefilter_chair_mcts_rust_proxy_probe.json`: 48 axis
     actions, proxy top-1 misses the exact best, but exact best is proxy rank 2;
     proxy top-3 recovers the exact best with zero reward gap, and proxy scoring
     is about `140x` faster.

   This is the strongest speed direction so far, but it is not exact by itself:
   centroid coverage is a proxy for the Manifold coverage objective. The safe
   version is top-K proposal plus exact Manifold verification. The experimental
   `candidate_backend=bitset_topk` path now exists for refine/MCTS, but it is
   not default because current checked timing is roughly neutral on the
   airplane 100-iteration smoke. The next promotion gate is category-sweep
   measurement: proxy top-K must contain the exact best action, or the stage
   must fall back to exact exhaustive scoring for that state.

   New profile sweep:

   - Added `scripts/benchmark_mcts_acceleration_profiles.py` to compare
     `exact_auto`, `bitset_top3`, `bitset_top8`, `rust_stateful_tt`, and
     `rust_stateful_prior01_tt` under the same target/iteration settings.
   - Added `configs/candidate_bitset_exact_experimental.yaml` for the current
     smoke-checked bitset profile: `reward_backend=manifold_stateful`,
     `candidate_backend=bitset_topk`, `candidate_top_k=8`,
     `stateful_union_cache=false`, and legacy MCTS tree.
   - `runs/bench_exact/mcts_accel_profiles_smoke3_iter10.json`: on three
     targets at `mcts_iter=10`, `max_step=10`, `bitset_top8` kept all metric
     diffs `0` and measured `1.101x`; `rust_stateful_tt` also kept metric diffs
     `0` and measured `1.090x`; `rust_stateful_prior01_tt` was faster
     (`1.157x`) but changed chair metrics.
   - `runs/bench_exact/mcts_accel_profiles_smoke5_iter20_exact_candidates.json`:
     on five targets at `mcts_iter=20`, `max_step=20`, `bitset_top8` kept all
     metric diffs `0` and measured `1.015x`; `rust_stateful_tt` measured
     `1.023x` but changed airplane/chair metrics. Current promotion candidate
     is therefore bitset top-K only, not TT/prior.
   - `runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_topk_tt_all.json`:
     on nine processed meshes balanced across airplane/chair/table, `bitset_top3`
     and `bitset_top8` both kept all reported metric diffs `0` at MCTS20.
     Mean speedups were `1.025x` and `1.023x`. `rust_stateful_tt` still changed
     BVS/MOV/TOV/vIoU, so TT remains opt-in and is not an exact-safe default.
   - Added `configs/candidate_bitset_fast_experimental.yaml` for the smaller
     `candidate_top_k=3` profile. The conservative bitset profile remains
     `configs/candidate_bitset_exact_experimental.yaml` with `candidate_top_k=8`.
   - Added runtime-stat collection to
     `scripts/benchmark_mcts_acceleration_profiles.py`. It now reads each
     run's `rust_stats.json` and aggregates candidate-prefilter and
     Manifold-state cache counters.
   - `runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_instrumented_topk.json`:
     `bitset_top8` kept all metric diffs `0` and measured `1.050x`, but exact
     reward misses only dropped from `12477` to `11584`; fallback exact
     fraction was `0.439`. This explains why bitset top-K is not yet a large
     speedup: the paper-safe fallback still performs most Manifold reward
     evaluations.

1. Exact leave-one-out union cache

   The current exact Manifold reward bottleneck is not Python arithmetic; it is
   rebuilding the union of unchanged boxes for each candidate replacement.
   `stateful_union_cache=true` caches leave-one-out unions inside the
   Rust/C++ bridge and evaluates candidate replacement as `union_except_i +
   candidate_i`. This preserves the exact Manifold reward path and does not
   change MCTS search order.

   - Added `exact_union_cache`, `bitset_top3_union_cache`, and
     `bitset_top8_union_cache` profiles to
     `scripts/benchmark_mcts_acceleration_profiles.py`.
   - Added `configs/stateful_union_cache_experimental.yaml` as the clean
     opt-in exact profile: `reward_backend=manifold_stateful`,
     `stateful_union_cache=true`, `candidate_backend=exact`, and
     `allow_search_order_changes=false`.
   - `runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_union_cache_probe.json`:
     over nine processed airplane/chair/table meshes at MCTS20,
     `exact_union_cache` kept all reported metric diffs `0` and measured
     `1.097x`. `bitset_top8_union_cache` also kept metric diffs `0` and measured
     `1.096x`.
   - Added `configs/expanded_processed_16.yaml`, which enumerates all currently
     processed expanded meshes: six airplane, five chair, and five table.
   - `runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_union_cache.json`:
     over those 16 processed meshes at MCTS20, `exact_union_cache` kept all
     reported metric diffs `0` and measured `1.047x`.
   - `runs/bench_exact/mcts_accel_profiles_one_iter100_union_cache.json`:
     on one 10-box airplane case at MCTS100/max-step100,
     `exact_union_cache` kept all reported metric diffs `0` and measured
     `1.095x` (`81.80s -> 74.67s`). The cache built `1093`
     leave-one-out unions and reused them `12879` times.

   Current conclusion: exact union caching is a better near-term default
   candidate than top-K. Top-K still helps, but its safe fallback limits the
   reduction in Manifold calls.

1. RL/MCTS action-prior sweep

   The current RL/deep-learning path is intentionally conservative: use traces
   to bias action ordering, keep the exact SMART reward, and evaluate the final
   boxes with the same metrics. It is not allowed to replace the objective.

   - Added union-cache action-prior profiles to
     `scripts/benchmark_mcts_acceleration_profiles.py`:
     `rust_stateful_prior002_union_cache`,
     `rust_stateful_prior005_union_cache`,
     `rust_stateful_prior01_union_cache`, and
     `rust_stateful_prior01_tt_union_cache`.
   - `runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_prior_union_cache.json`:
     prior `0.1` with and without TT measured about `1.03x` over
     `exact_union_cache`, but changed one table case.
   - `runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_prior_weight_sweep.json`:
     prior weights `0.02`, `0.05`, and `0.1` measured `1.043x`, `1.053x`,
     and `1.055x`, but all changed the same table case. The changed case had
     lower MOV but worse BVS/TOV/vIoU. This means the current generic smoke
     prior is not a quality upgrade yet.
   - Added a state-aware linear action-prior trainer:
     `smart build-prior --model-type linear`,
     `scripts/train_action_prior_from_traces.py --model-type linear`, and
     `smart.build_linear_action_prior_from_traces`. It uses schema-v2 trace
     fields such as category, BVS, step fraction, action unit, box count, and
     penalty settings. The first tiny MCTS2 leave-one-out airplane smoke kept
     reported metric diffs at `0` but measured `0.852x`. The slightly larger
     three-airplane MCTS5 smoke
     `runs/bench_exact/action_prior_linear_smoke3_mcts5.json` kept all reported
     metric diffs at `0` and measured `1.037x` at prior weight `0.1`. It is a
     functional RL/action-ordering baseline, not a promoted speed path.
   - Added a PyTorch MLP action-prior trainer:
     `smart build-prior --model-type mlp`,
     `scripts/train_action_prior_from_traces.py --model-type mlp`, and
     `smart.build_mlp_action_prior_from_traces`. `--device auto` probes Apple
     Silicon MPS first, then CUDA, then CPU. The saved model is a JSON weight
     file consumed by the existing MCTS prior loader, so PyTorch is used for
     training and exact SMART reward still evaluates the selected trajectory.
     The first tiny CPU smoke
     `runs/bench_exact/action_prior_mlp_airplane2_mcts2.json` kept metric diffs
     at `0` but measured `0.952x`; it verifies wiring, not a speed gain.
   - Added `mcts.puct_prior_weight`, an opt-in PUCT-style prior bonus for MCTS
     child selection. The tiny linear-prior PUCT smoke
     `runs/bench_exact/action_prior_puct_linear_airplane2_mcts2.json` kept metric
     diffs at `0` and measured `1.055x`; it is promising enough to test at
     MCTS20/MCTS100, but still too small to recommend.
   - Added an offline policy-gradient PyTorch MLP prior:
     `smart build-prior --model-type rl-mlp`,
     `scripts/train_action_prior_from_traces.py --model-type rl-mlp`, and
     `smart.build_rl_mlp_action_prior_from_traces`. It uses exact SMART trace
     rewards as replay data and exports a JSON policy, so runtime MCTS still
     uses the existing exact reward/evaluation. The packaged all-trace asset is
     `smart/assets/priors/category_general_all_available_offline_rl_mlp_prior.json`.
     On `runs/bench_exact/rl_prior_cat5_mcts20_weight_sweep.json`, prior weights
     `0.05`, `0.1`, and `0.2` measured `1.032x`, `1.057x`, and `1.081x`.
     Weight `0.1` is the current research setting; `0.2` is faster but worsened
     more cases.
   - Added `scripts/run_quality_guarded_mcts.py`, which runs baseline and
     learned-prior MCTS, evaluates both, and copies the non-worse selection into
     `mcts_guarded`. On
     `runs/bench_exact/quality_guard_cat5_mcts20_rl01.json`, the global
     offline-RL prior was selected on `10/15` cases, baseline was selected on
     `5/15`, and one raw-prior worse case was blocked by the guard. The guarded
     stage therefore produced `15/15` successful non-worse outputs.
   - Category-specific offline-RL priors were trained, but the first chair-only
     guarded check (`runs/bench_exact/quality_guard_chair5_mcts20_catrl01.json`)
     was not better than the global prior: `2/5` raw prior outputs were rejected
     for worse coverage. Keep category-specific priors experimental.

   Next RL/MCTS step: train a stronger category-aware policy and use the quality
   guard as the acceptance layer. A global prior is still too blunt for some
   chair cases. The prior/RL path is therefore active, but it is intentionally
   not a reproduction default until it improves or preserves final SMART quality
   metrics on larger sweeps.

1. Opt-in no-reward MCTS early stop

   The old legacy MCTS early stop only stopped after roughly 102 completed
   iterations when the best rollout reward stayed below `1e-2`. That is too
   late for difficult cases where MCTS never improves the refine initialization.
   The CLI now exposes `mcts.no_reward_stop_after`, wired to the legacy
   `--mcts_no_reward_stop_after` argument. The default preserves the legacy
   behavior; lower values are an opt-in search-budget shortcut.

   Check:

   - `runs/bench_exact/fast_stop20_probe_eval.json`: on the 10-box airplane
     MCTS100 case, `no_reward_stop_after=20` produced byte-identical bbox OBJ
     files and identical reported metrics versus the exact union-cache MCTS100
     output, while reducing stage time from `73.83s` to `15.96s`.

   This is not a proof for default promotion. If a shape only improves after a
   late rollout, an aggressive stop can reduce quality. Use it for development,
   smoke, or batch triage; keep paper-like runs at the legacy stop threshold.

1. Opt-in fused Python-tree rollout step

   The exact bridge already had `_bridge_mcts_greedy_rollout_step()`, but the
   legacy Python MCTS tree did not call it. `mcts.fused_rollout_step=true` now
   lets the Python tree fuse one greedy rollout step into one env call while
   preserving exact Manifold scoring and the legacy tree structure.

   Check:

   - `runs/bench_exact/mcts_accel_profiles_airplane3_iter20_fused_rollout.json`:
     three airplane cases at MCTS20/max-step100, all reported metric diffs `0`,
     mean speedup `1.013x`. It helped the 10-box case and was slightly slower
     on one smaller case, so it remains opt-in.

1. Implemented hybrid MCTS + local search stage

   Added a `local_refine` stage and
   `configs/hybrid_local_search_experimental.yaml`. The stage reads the latest
   MCTS bbox output through the existing `bbox_direct` initializer, then runs a
   smaller-action greedy local search. This is a quality experiment, not a
   paper-safe default, because it changes the post-MCTS trajectory.

   First check:

   - `runs/bench_exact/hybrid_probe_local_refine_eval.json`: on the checked
     10-box airplane case, local search after MCTS improved BVS
     (`2.722 -> 2.609`), MOV (`3.034 -> 2.497`), TOV (`1.580 -> 1.508`), and
     vIoU (`0.3867 -> 0.3978`) while keeping coverage above `0.998`.

   Guarded check:

   - `runs/bench_exact/local_refine_guarded_refined21_covtol_improved_reuse_mcts_guarded.json`:
     `scripts/run_quality_guarded_local_refine.py` ran the hybrid post-process
     on all `21` currently refined `mcts_guarded` targets. It selected local
     refine on `10/21`, selected input on `11/21`, and improved `10/21` cases.
     Aggregate selected-stage metrics improved BVS (`2.0373 -> 2.0010`), MOV
     (`1.5140 -> 1.3500`), TOV (`0.9780 -> 0.9480`), and vIoU
     (`0.6142 -> 0.6269`) versus `mcts_guarded`; coverage stayed effectively
     unchanged under the explicit `0.001` coverage tolerance. Strict coverage
     guard selected local refine on only `2/21`, which shows most quality gains
     trade a very small coverage change for much better tightness.

   - `runs/bench_exact/local_refine_guarded_expanded200_mcts_manifest52_covtol_improved.json`:
     after adding `--from-input-manifest`, the same guard ran on all `52`
     successful processed MCTS outputs (`18` airplane, `17` chair, `17` table).
     It selected local refine on `29/52`, selected input on `23/52`, and
     improved all `29` selected cases. Aggregate selected-stage metrics
     improved BVS (`1.7546 -> 1.7225`), MOV (`1.2327 -> 1.1410`), TOV
     (`0.7173 -> 0.6913`), and vIoU (`0.6835 -> 0.6970`) versus MCTS, with
     coverage effectively unchanged (`0.999685 -> 0.999681`).

   Next gate: run this over a larger category-balanced set and compare strict
   guard (`covered_tolerance=0`) versus quality-first guard
   (`covered_tolerance=0.001`). Promote only if coverage remains high and the
   aggregate quality tradeoff is consistently positive.

   Implemented local-refine gate baseline:
   `smart.local_refine_gate` and `scripts/train_local_refine_gate.py` train a
   PyTorch MLP classifier from the exported guard dataset. The model uses only
   category and pre-local-refine SMART metrics, avoiding local-output leakage.
   The current packaged research asset
   `smart/assets/gates/local_refine_gate_manifest52.json` was trained from the
   `52`-row manifest export. Leave-one-out validation measured accuracy
   `0.75`, F1 `0.780`, and ROC-AUC `0.784`, compared with a majority baseline
   accuracy of `0.558`. This does not replace exact SMART evaluation; it is a
   decision model for when to spend the extra local-refine time. The guarded
   runner now accepts `--gate-path` and `--gate-threshold`; a forced skip smoke
   confirmed that low-probability cases can bypass the local-refine subprocess
   and still produce a guarded bbox output. The threshold sweep script
   `scripts/evaluate_local_refine_gate.py` shows threshold `0.5` on manifest52
   skips `22/52` local-refine runs, catches all `29/29` known improvements, and
   saves `20.3%` of measured local-refine time while matching full guarded
   local-refine aggregate metrics. `smart evaluate --from-manifest` now
   evaluates custom subset stages cleanly; the current gated stage evaluation
   has `52/52` successes and the same aggregate quality numbers as the offline
   threshold analysis.

1. Optional MCTS transposition table

   Hash bbox states and reuse node values across equivalent states reached by
   different action orders. This can reduce duplicate exploration without
   changing the objective. The option is implemented as
   `mcts.transposition_table`; it is off by default and additionally guarded by
   `mcts.allow_search_order_changes` because reused statistics can change search
   order and therefore metrics.

2. Learned policy only for action ordering

   Train a small policy/value model from SMART search traces to order candidate
   actions or initialize MCTS priors. Keep the exact coverage/tightness reward
   as the evaluator. This preserves the paper's geometry guarantee while
   improving search efficiency and category generalization.

   Quality-first selection is now wired:
   `scripts/run_quality_guarded_mcts.py --selection-objective quality_score`
   ignores faster identical candidates and selects a learned-prior result only
   when exact metrics produce positive scalar quality gain under the per-metric
   guard. On
   `runs/bench_exact/candidate_pg_quality_score_guard_cat5_mcts10.json`, this
   selected prior on `1/14` processed meshes and improved aggregate BVS/MOV/TOV
   and vIoU over baseline with no coverage drift. The gain is small, so the
   next optimization is not more threshold tuning; it is richer final-return
   training data for the policy.

3. Opt-in tet-clipping reward backend

   Wire the verified Rust tet-clipping kernel into greedy/MCTS reward
   evaluation behind an explicit config flag. Pass the same bbox volumes used by
   the legacy mesh/Manifold path, and keep Manifold as the default and as the
   verification backend until larger category sweeps show the same metric parity
   across difficult meshes, reflected/tilted rotations, and higher box counts.

4. Candidate action prefilter before Manifold calls

   Most bbox actions can be rejected using cheap bounds checks, volume delta,
   centroid coverage masks, and duplicate-state hashes. This belongs after
   exact Rust parity unless every rejection can be proven identical to the
   legacy exhaustive scan.

5. Adaptive action units

   Start with coarse action units when boxes are loose, then shrink near
   convergence. This reduces long sequences of tiny moves and can be expressed
   as a schedule over bbox diagonal or reward plateau length.

6. Category-aware CoACD/Mesh2Tet profiles

   Airplanes, chairs, and tables fail in different ways during meshing and
   pre-segmentation. Keep paper-like defaults, but allow category profiles for
   CoACD resolution, max convex hulls, retry coarsening, and fTetWild edge
   length. The config system already supports this.

## What Not To Do First

- Do not rewrite the fixed Manifold C++ binding in Rust. It is a stable
  dependency and a high-risk rewrite.
- Do not replace the exact objective with a learned loss. Learning should guide
  search, not define coverage or tightness.
- Do not optimize renderer code before geometry kernels. Rendering is not the
  runtime bottleneck for SMART search.

## External Literature Notes

- AlphaGo and AlphaZero justify the safest learning direction for SMART:
  neural policy/value models should guide MCTS action ordering and priors, while
  the exact search/evaluator remains in charge of accepted results. This maps
  well to SMART trace learning because the accepted action logs can train a
  category-general proposal model without replacing the Manifold objective.
  Sources: https://www.nature.com/articles/nature16961 and
  https://arxiv.org/abs/1712.01815.
- MCTS transposition tables and progressive widening are standard tools for
  reducing duplicate search and high branching factors. For SMART, these should
  stay opt-in because they can change search order, but they are valid
  experiments once metric parity is measured. Source:
  https://link.springer.com/article/10.1007/s10462-022-10228-y and the
  progressive-strategy paper DOI `10.1142/S1793005708001094`.
- CoACD is already a good default pre-segmentation choice because it explicitly
  targets collision-aware concavity and uses tree search to reduce unnecessary
  cuts. This supports our current plan of sweeping CoACD profiles before merge
  rather than hardcoding one segmentation. Sources:
  https://colin97.github.io/CoACD/ and https://github.com/SarahWeiii/CoACD.
- Primitive/cuboid abstraction papers support a later learned initializer:
  predict coarse cuboid/box proposals across a category, then let SMART refine
  them with the exact objective. This can improve quality and initialization
  without making the learned model the final evaluator. Source:
  https://shubhtuls.github.io/volumetricPrimitives/.
- Recent learned or GPU convex-decomposition work is relevant for future
  pre-segmentation alternatives, not for replacing SMART reward now. Keep these
  behind optional profile sweeps because changing pre-segmentation can change
  final boxes even if refine/MCTS remains exact.
