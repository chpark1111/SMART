# Tetra Failure Playbook

Tetrahedralization is the most fragile part of the full paper pipeline because
it depends on raw mesh quality and external geometry tools. SMART should recover
when it can, but it should not silently destroy geometry just to force success.

## What Happens On Failure

For each mesh, SMART writes logs and manifest rows. A failed tetra mesh does not
abort the dataset:

- ManifoldPlus/fTetWild logs go to `runs/<profile>/logs/tetra/<category>/<id>/`;
- attempts and validation errors go to `runs/<profile>/manifests/tetra.jsonl`;
- downstream stages are skipped for that mesh;
- the remaining meshes continue.

## Built-In Recovery Order

Default recovery is conservative:

1. normalize mesh;
2. cleanup duplicate/degenerate faces and unreferenced vertices;
3. fix normals;
4. primary Mesh2Tet attempt;
5. finer retry;
6. coarser retry with `--coarsen`;
7. robust winding-number retry;
8. repaired input fallback with `fill_holes=true`.

The final fallback was added for common ShapeNet holes. It only changes the
temporary run input, never the original `data/` OBJ.

## Why Some Cases Still Fail

Common reasons:

- severe self-intersections;
- very thin parts below the target edge length;
- open surfaces with large missing regions;
- non-manifold edges that ManifoldPlus cannot repair;
- tiny disconnected components;
- fTetWild timeout or crash on degenerate geometry;
- output exists but fails validation: missing `tetra.msh__sf.obj`,
  non-watertight surface, too few tetrahedra, or too few surface faces.

## Config Knobs

Use less destructive knobs first:

```yaml
tetra:
  min_tetra_count: 20
  min_surface_faces: 20
  input_repair:
    enabled: true
    basic_cleanup: true
    fix_normals: true
    fill_holes: false
    keep_largest_component: false
    fallback_variants:
      - name: fill_holes
        fill_holes: true
        keep_largest_component: false
  retry:
    enabled: true
    fine_retry:
      enabled: true
      epsilon_scale: 0.5
      edge_length_scale: 0.5
      coarsen: false
    epsilon_scale: 2.0
    edge_length_scale: 2.0
    coarsen: true
```

For meshes that still fail, try stronger but riskier repair:

```yaml
tetra:
  input_repair:
    fallback_variants:
      - name: largest_component_fill_holes
        enabled: true
        fill_holes: true
        keep_largest_component: true
```

Only use `keep_largest_component=true` when losing disconnected small parts is
acceptable for that experiment.

## Practical Debug Commands

Run tetra only:

```bash
smart --config configs/example_3x3.yaml tetra
smart --config configs/example_3x3.yaml summary
```

Analyze failures:

```bash
python3 scripts/analyze_pipeline_failures.py \
  --manifest-dir runs/example_3x3/manifests
```

Inspect one mesh's logs:

```bash
ls runs/example_3x3/logs/tetra/<category>/<mesh_id>/
cat runs/example_3x3/manifests/tetra.jsonl
```

## Research Direction

If failure rate remains high on a larger dataset, the next robust path is a
separate mesh repair experiment:

- compare conservative repair, hole filling, largest-component filtering, and
  external repair tools;
- measure not only tetra success rate but also final SMART metrics;
- keep repaired meshes under `runs/` or `experiments/`, not `data/`;
- promote only repair steps that improve success without changing shape
  semantics in a harmful way.
