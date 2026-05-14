# Manifold Performance Notes

This note records the current SMART-specific Manifold bottlenecks. The fixed
vendored Manifold source remains authoritative and should not be modified for
normal optimization work.

## Current Call Pattern

SMART's exact reward path evaluates candidate boxes through
`rust/smart-core/src/manifold_bridge.cpp`.

For each exact candidate score, the bridge generally:

1. Builds a candidate `manifold::Manifold` box from bbox parameters.
2. Unions the unchanged boxes plus the candidate box.
3. Computes `surface - merged_boxes`.
4. Calls `residual.GetMesh()`.
5. Computes residual volume by iterating over the returned triangles.

The expensive part is not just the boolean operation. `GetMesh()` forces the
lazy CSG tree to materialize an output mesh, and then SMART computes volume from
that mesh. This happens inside action scoring, so it is multiplied by MCTS and
greedy candidate count.

## Findings

- Vendored Manifold already contains useful union logic. `BatchBoolean` performs
  heap-ordered pairwise booleans on smaller meshes first, and `BatchUnion`
  composes disjoint children before boolean union.
- SMART's paper-safe exact path still uses ordered unions to preserve legacy
  search behavior. The faster leave-one-out union cache uses `BatchBoolean` and
  remains opt-in because different grouping can change near-tie search order.
- The strongest exact speed path so far is the leave-one-out union cache:
  `81.80s -> 74.67s` on the 10-box airplane MCTS100 case, with reported metric
  diffs at zero.
- Safe bitset/top-K prefiltering reduces some exact reward calls, but the safe
  fallback still verifies many candidates, so the observed speedup is modest.
- The bridge itself was previously compiled without explicit C++ optimization
  flags even when the vendored Manifold library was built in Release mode. The
  build script now compiles `manifold_bridge.cpp` with `-O3 -DNDEBUG`.

## Promising Exact Experiments

These should be tested before promotion:

1. Compare `residual.GetProperties().volume` against the current
   `signed_mesh_volume(residual.GetMesh())` metric on airplane/chair/table smoke
   and larger processed sets. If parity holds within existing float tolerance,
   use `GetProperties()` for reward residual volume to avoid materializing mesh
   output on every candidate.
2. Add an exact-order prefix cache for paper-safe candidate scoring. It should
   preserve bbox order more closely than `BatchBoolean`, but still reduce some
   repeated prefix union work.
3. Add cache instrumentation for candidate residual evaluation time split:
   candidate box construction, union construction, surface subtraction,
   `GetMesh()` or `GetProperties()`, and volume accumulation.
4. Keep `BatchBoolean` leave-one-out union cache as an opt-in speed profile until
   action trace parity is validated on larger runs.

## GetProperties Probe

Added `scripts/benchmark_manifold_volume_methods.py` and comparison methods on
`smart._rust.ManifoldBridgeMesh`:

- `volume_properties()`
- `residual_volume_for_boxes_properties(...)`
- `residual_volume_for_boxes_pair(...)`
- `residual_volume_for_box_params_properties(...)`
- `residual_volume_for_box_params_pair(...)`

The first processed-set probe used deterministic AABB-derived boxes on the 16
currently processed airplane/chair/table meshes:

```bash
python3 scripts/benchmark_manifold_volume_methods.py \
  --config configs/expanded_processed_16.yaml \
  --limit 16 \
  --repeat 5 \
  --output runs/bench_exact/manifold_volume_methods_processed16.json
```

Result:

- `num_targets`: `16`
- `num_probes`: `80`
- `max_abs_diff`: `4.064794009717154e-08`
- `max_rel_diff`: `4.064794009717154e-08`
- `mean_speedup_properties_vs_mesh`: `1.105021890299984x`

The bridge now exposes an opt-in runtime flag:

```bash
smart mcts --config configs/expanded_processed_16.yaml \
  --set mcts.reward_backend=manifold_stateful \
  --set mcts.manifold_volume_method=properties
```

The first actual MCTS20 target check is recorded at
`runs/bench_exact/exact_stateful_properties_mcts20_target1.json`. It kept all
reported metrics identical to legacy `manifold`, did not diverge in accepted
actions, and measured `1.069x` stage speedup. The reward trace still differed by
up to `2.2902652996314998e-04`, so `properties` remains opt-in until it passes a
larger airplane/chair/table sweep.

## Non-Goals

- Do not rewrite or replace the fixed vendored Manifold C++ implementation in
  this phase.
- Do not enable approximate tet clipping, learned pruning, or top-K candidate
  changes by default for paper-safe runs.
- Do not promote any path that changes reported metrics or action traces without
  a documented experiment.
