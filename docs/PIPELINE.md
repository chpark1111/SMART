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
- `configs/expanded_full.yaml`: larger local ShapeNet layout.

Research-only RL and acceleration configs live under ignored
`experiments/configs/`.

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
the repaired input is written under the run logs. More aggressive repair can be
enabled per config:

```yaml
tetra:
  input_repair:
    enabled: true
    fill_holes: true
    keep_largest_component: false
  min_tetra_count: 20
  retry:
    fine_retry:
      enabled: true
      epsilon_scale: 0.5
      edge_length_scale: 0.5
      coarsen: false
```

Use the failure analyzer for per-stage categories:

```bash
python3 scripts/analyze_pipeline_failures.py --manifest-dir runs/example_3x3/manifests
```
