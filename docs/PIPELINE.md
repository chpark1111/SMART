# SMART Pipeline

The release pipeline follows the paper order:

1. Normalize each `model.obj` to the configured scale.
2. Run Mesh2Tet: ManifoldPlus repair followed by fTetWild tetrahedralization.
3. Run CoACD pre-segmentation on the tetrahedral surface mesh.
4. Run SMART merge with the native C++ backend.
5. Run greedy refine with the native C++ backend.
6. Run MCTS with the native C++ backend.
7. Render final boxes with the packaged software preview renderer by default,
   or with the adapted paper Blender renderer when explicitly requested.
8. Evaluate bbox outputs with the provided evaluation utilities.

Python owns CLI/config/manifests. The algorithm core is in `cpp/` and is
available as both `smart._cpp` and `smart-cpp-native`.

## Config Control

Use YAML profiles under `configs/` or override individual fields:

```bash
smart --config configs/example_3x3.yaml \
  --set tetra.ftetwild_threads=8 \
  --set refine.max_step=200 \
  --set mcts.mcts_iter=100 \
  run
```

Recommended public profiles:

- `configs/example_3x3.yaml`: local 3-per-category example.
- `configs/smoke_5.yaml`: quick smoke profile.
- `configs/demo.yaml`: 50-per-category demo profile.
- `configs/paper_like.yaml`: paper-style search settings.
- `configs/preprocess_fast.yaml`: opt-in first-run preprocessing speed preset.
- `configs/expanded_full.yaml`: larger local ShapeNet layout.

Research-only RL and acceleration configs live under ignored
`experiments/configs/`.

## Native Performance Profiling

`smart native-run` writes one `native_pipeline_stats.json` file per mesh under
`runs/<profile>/native_pipeline/<category>/<mesh>/`.  The file records
`normalize`, `manifoldplus`, `ftetwild`, `coacd`, partition, merge, and
`refine_mcts` wall time.

A June 25, 2026 local Apple Silicon smoke measurement with
`mcts.mcts_iter=3000`, `refine.max_step=200`, and rendering disabled showed:

| mesh | total | preprocessing | search |
| --- | ---: | ---: | ---: |
| airplane sample | 25.29s | 12.38s / 48.9% | 12.92s / 51.1% |
| chair sample | 18.34s | 17.51s / 95.4% | 0.84s / 4.6% |
| table sample | 5.71s | 5.62s / 98.4% | 0.09s / 1.6% |
| combined | 49.35s | 35.49s / 71.9% | 13.82s / 28.0% |

Combined stage totals for those three samples were:

- `ftetwild`: 26.69s / 54.1%;
- `refine_mcts`: 13.81s / 28.1%;
- `coacd`: 7.38s / 15.0%;
- `manifoldplus`: 1.50s / 3.1%.

This means learned routing mainly reduces the search portion.  End-to-end speed
still depends heavily on Mesh2Tet/CoACD preprocessing, especially fTetWild.
The current fTetWild build used by SMART does not expose `--max-threads`, so
`tetra.ftetwild_threads` is passed only when the configured binary supports it.

For repeated experiments on the same mesh, reuse preprocessing instead of
rerunning Mesh2Tet and CoACD:

```bash
smart --config configs/smoke_5.yaml \
  --set native_pipeline.reuse_preprocessing=true \
  --set stages.render=false \
  native-run --category table --mesh 1692563658149377630047043c6a0c50 --force
```

`native_pipeline.reuse_preprocessing=true` reuses normalized mesh,
ManifoldPlus output, fTetWild tetra mesh, and presegmentation metadata, then
reruns merge/refine/MCTS.  `native_pipeline.reuse_existing=true` is stronger:
it reuses every existing stage output and is useful only when checking or
summarizing a completed run.

To reuse preprocessing across different workspaces or profile runs, enable the
persistent hash cache:

```bash
smart --config configs/smoke_5.yaml \
  --set native_pipeline.preprocessing_cache=true \
  --set native_pipeline.preprocessing_cache_root=runs/.smart_preprocessing_cache \
  --set stages.render=false \
  native-run --category table --mesh 1692563658149377630047043c6a0c50 --force
```

The cache key includes the source mesh SHA-256, normalization settings,
tetrahedralization settings, CoACD settings, and external tool signatures.  It
intentionally excludes merge/refine/MCTS settings, so learned-router and MCTS
experiments can change search parameters while reusing the same normalized,
tetra, and CoACD artifacts.  On a cache hit SMART copies only preprocessing
artifacts into the run directory and passes `--reuse_preprocessing` to the
native executable.  Search outputs are never restored from this cache.

A June 27, 2026 local cache probe on the chair smoke mesh
`11b7c86fc42306ec7e7e25239e7b8f85` measured:

| mode | wall time | artifact check |
| --- | ---: | --- |
| cache miss | 17.81s | baseline preprocessing generated |
| cache hit in a fresh workspace | 0.54s | `tetra.msh`, `coacd_partitions.json`, and final bbox JSON hashes identical |

This is the strongest strict-quality speedup available for repeated runs: the
geometry artifact is reused rather than approximated or regenerated.

The next high-impact preprocessing optimizations are:

1. tune `tetra.manifold_depth` to keep ManifoldPlus repair enabled while
   reducing repair resolution and downstream fTetWild cost;
2. add a fast CoACD path that avoids Python CLI startup, ideally by linking
   CoACD directly or using a persistent worker;
3. add a quality-gated fast tetra preset that coarsens only when final SMART
   metrics remain stable;
4. evaluate alternative tetra backends or a newer fTetWild build with real
   threading support.

## Exact-Quality Speed Rules

If the requirement is **no quality regression at all**, preprocessing changes
must be treated differently from search changes.

Preprocessing changes are not exact-preserving.  Options such as
`tetra.skip_manifoldplus`, lower `tetra.manifold_depth`, altered CoACD
parameters, fTetWild threading, or concurrent batch execution can all change
the tetra mesh or CoACD partition graph.  Once those artifacts change, the
final SMART boxes can change even when the SMART seed is fixed.  A June 26,
2026 smoke probe showed this clearly:

| run path | wall time | exact-quality status |
| --- | ---: | --- |
| Python `native-run`, one mesh at a time | about 59s for 5 meshes | baseline |
| C++ `run-batch --jobs 1` | about 60s for 5 meshes | same execution mode class |
| C++ `run-batch --jobs 2` | about 31s for 5 meshes | faster throughput, but output metrics changed |

The two-worker batch run is useful for dataset throughput, but it is not a
strict identical-quality acceleration because fresh fTetWild/CoACD outputs can
differ under repeated or concurrent runs.  In the same probe, the per-mesh stage
budget was dominated by preprocessing:

| stage | share of one-worker C++ batch time |
| --- | ---: |
| fTetWild | 74.9% |
| CoACD | 16.1% |
| ManifoldPlus | 4.2% |
| SMART refine/MCTS | 4.8% |

Therefore the strict release rule is:

- keep default preprocessing unchanged for paper/release metrics;
- use preprocessing cache or `native_pipeline.reuse_preprocessing=true` for
  repeated runs on the same mesh and config;
- use learned routing only behind exact validation/fallback;
- allow `run-batch --jobs N` for throughput experiments, but do not claim
  bit-identical or metric-identical output unless the final metrics are checked;
- promote a fast preprocessing profile only if it passes an exact final metric
  gate on the target dataset.

`tetra.skip_manifoldplus=true` is intentionally opt-in.  It can save the
ManifoldPlus repair step on already clean/watertight meshes, but it is not a
same-role acceleration: the fTetWild input changes from repaired ManifoldPlus
output to the normalized source mesh.  In smoke probes it reduced wall time
substantially but also changed active box counts for chair/table, so it should
remain a validation-only shortcut.

`tetra.manifold_depth=<N>` is the safer first tuning knob when ManifoldPlus must
keep doing the repair job.  ManifoldPlus defaults to depth 8; lower depths can
reduce repaired mesh density and fTetWild runtime.  Because the repaired surface
still changes, publishable runs should report the depth and compare exact SMART
metrics against the default depth.

For a guarded first-run speed preset, use:

```bash
smart --config configs/preprocess_fast.yaml \
  --set stages.render=false \
  native-run --category chair --mesh 11b7c86fc42306ec7e7e25239e7b8f85 --force
```

The preset uses `tetra.manifold_depth_candidates_by_category`; SMART tries the
first candidate and falls back to depth 8 if the native run fails.  The current
profile keeps table on the default depth-0 path and sets
`merge.final_k_by_category.table=2`, because lower-depth table probes can
over-merge table parts into one box.

Smoke profiling showed why this knob must be quality-gated:

| mesh | setting | total | repaired faces | bbox count | BVS | vIoU | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| airplane | depth 8 | 25.29s | 74,742 | 7 | 2.868 | 0.380 | release baseline |
| airplane | depth 6 | 15.01s | 8,908 | 7 | 2.378 | 0.460 | faster, non-worse on smoke |
| airplane | depth 7 | 50.16s | 25,394 | 9 | 2.481 | 0.469 | fTetWild/search got slower |
| chair | depth 8 | 18.34s | 315,708 | 4 | 1.920 | 0.573 | release baseline |
| chair | depth 6 | 6.63s | 21,592 | 4 | 1.602 | 0.668 | faster, non-worse on smoke |
| table | depth 8 | 5.71s | 79,990 | 2 | 1.805 | 0.568 | release baseline |
| table | depth 7 | 3.69s | 21,266 | 2 | 1.842 | 0.570 | faster, mixed metric change |
| table | depth 6 | failed | 5,724 | - | - | - | CoACD crashed after tetra |

With the category fast preset, an early three-mesh probe completed in 24.86s
rather than 49.35s, but full smoke reruns showed why this is not the default
release path yet.  The default path succeeded on 4/5 meshes; the fast preset
succeeded on 5/5 by rescuing one CoACD failure, but common-case quality was
mixed:

| metric on common successful cases | fast result |
| --- | ---: |
| BVS non-worse than depth 8 | 2/4 |
| vIoU non-worse than depth 8 | 3/4 |
| Covered non-worse than depth 8 | 1/4 |
| box count delta sum | +3 |
| fast-only rescue cases | 1 |

For example, the first airplane changed from 7 to 9 boxes: vIoU improved
(`0.483 -> 0.540`) and Chamfer improved, but BVS worsened slightly
(`2.225 -> 2.263`) and coverage dropped slightly.  A table lower-depth probe
changed from 2 boxes to 1 box and degraded more clearly
(`BVS 1.592 -> 2.248`, `vIoU 0.644 -> 0.445`).  The packaged fast profile now
prevents that known over-merge by keeping table on depth 0 and requiring at
least two table partitions after merge, but it remains opt-in for speed/rescue
experiments rather than a default replacement.

Lower depth is therefore not a universal default.  It is useful as a
category/dataset-specific fast preset, or as a speculative fast attempt with an
exact fallback.  For strict reproducibility and no-quality-regression runs,
leave `tetra.manifold_depth=0` and rely on preprocessing cache/reuse or the
exact-validated learned search path instead.  Fresh fTetWild/CoACD runs can be
non-identical even with the same SMART seed, so changing preprocessing inputs
cannot be promoted as a guaranteed quality-preserving speedup without a final
exact metric gate.

## Rendering

Public configs use `render.backend: fallback` by default. This avoids launching
Blender on macOS during normal package runs, which also avoids OS crash dialogs
if a local Blender installation is unstable in background mode. The output is a
transparent boxes-only PNG by default and can also include a mesh overlay with
`render.joint_mesh=true`.

The paper-style Blender renderer remains available:

```bash
smart --config configs/example_3x3.yaml --set render.backend=blender render
```

If Blender crashes, SMART records the render log under `runs/.../logs/render/`
and falls back to the software renderer when `render.fallback=true`, but macOS
may still show a system crash dialog for the failed Blender process. Use
`render.backend=fallback` to avoid launching Blender entirely.

## Failure Handling

Mesh2Tet can fail on malformed meshes. The pipeline records per-mesh manifests,
retries with configured fallback settings where available, skips failed meshes,
and continues the batch. Inspect summaries with:

```bash
smart --config configs/example_3x3.yaml summary
```

Common tetra validation failures are:

- `tetra element count below minimum`: fTetWild produced an unusably tiny
  volume mesh. SMART now tries a finer retry before coarse retries because
  coarsening can make this failure worse.
- `surface is not watertight`: the exported `tetra.msh__sf.obj` is not a
  closed surface, usually because the source OBJ has holes, non-manifold edges,
  or severe self-intersections.
- `surface has multiple connected components`: this is allowed by default for
  ShapeNet-style assets, but can be enabled as a stricter validation check.
- fTetWild crashes or timeouts: usually external tetrahedralization robustness
  failures on degenerate or very thin meshes.

The tetra stage performs a conservative input cleanup before ManifoldPlus:
duplicate/degenerate faces are removed, unreferenced vertices are removed, and
normals are fixed when `trimesh` is installed. The source OBJ is not modified;
the repaired input is written under the run logs. SMART classifies failures in
the manifest and uses that class to queue targeted repair retries. For example,
`surface is not watertight` and low tetra-element validation failures queue
`fill_holes=true`, while crash/timeout failures queue a conservative
repaired-input retry. More aggressive repair can be enabled per config:

```yaml
tetra:
  input_repair:
    enabled: true
    fill_holes: true
    keep_largest_component: false
  min_tetra_count: 20
  # Optional guard for large dataset collection. Default 0 disables it.
  max_manifold_faces_for_ftetwild: 0
  retry:
    fine_retry:
      enabled: true
      epsilon_scale: 0.5
      edge_length_scale: 0.5
      coarsen: false
```

The default config also includes a non-destructive fallback input variant that
tries `fill_holes=true` only after the normal attempts fail. Stronger variants,
such as `keep_largest_component=true`, can be enabled for experiments but are
off by default because they may remove real shape parts. See
[`TETRA_FAILURE_PLAYBOOK.md`](TETRA_FAILURE_PLAYBOOK.md).

For long-tail datasets, ManifoldPlus can sometimes convert a small OBJ into a
very large repaired surface. Set `tetra.max_manifold_faces_for_ftetwild` to a
positive cap to classify those attempts as `repair_surface_too_large` and move
to the next repair/parameter route before fTetWild consumes the full timeout.
The release default is `0`, so this guard is opt-in.

Use the failure analyzer for per-stage categories:

```bash
python3 scripts/analyze_pipeline_failures.py --manifest-dir runs/example_3x3/manifests
```
