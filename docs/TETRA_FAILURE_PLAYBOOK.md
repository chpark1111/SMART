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

Default recovery is conservative and failure-aware:

1. normalize mesh;
2. cleanup duplicate/degenerate faces and unreferenced vertices;
3. fix normals;
4. primary Mesh2Tet attempt;
5. finer retry;
6. coarser retry with `--coarsen`;
7. robust winding-number retry;
8. targeted repaired-input retry when a matching failure class appears.

SMART records a `failure_class` in the tetra attempt metadata:

- `validation_open_surface`: output surface is not watertight; queues
  `fill_holes=true`.
- `command_crash` or `command_timeout`: fTetWild/ManifoldPlus crashed or timed
  out; queues the conservative repaired-input fallback.
- `validation_low_tetra_count`: output has too few tetrahedra; handled by the
  fine/coarse parameter retry schedule.
- `validation_disconnected`: output has multiple connected components; can
  queue `keep_largest_component=true` only if you opt in.

Repair only changes temporary inputs under `runs/.../logs/tetra/`; SMART never
mutates the original `data/` OBJ.

## Automatic Detection Map

| Failure class | Trigger text or condition | Default action | Why it is conservative |
| --- | --- | --- | --- |
| `validation_open_surface` | `surface is not watertight` from validation | queue `fill_holes` fallback | fills holes in a temporary copy only |
| `command_timeout` | external command timeout or rc `124` | queue repaired-input retry and continue parameter retries | does not change original mesh or force success |
| `command_crash` | negative return code such as `SIGSEGV` | queue repaired-input retry and continue parameter retries | isolates bad mesh cases without stopping the dataset |
| `validation_low_tetra_count` | `tetra element count below minimum` | rely on fine/coarse/robust retry schedule | avoids deleting geometry to inflate element count |
| `validation_disconnected` | `surface has multiple connected components` | no destructive default; opt-in largest-component fallback | avoids silently removing real disconnected parts |

If all retries fail, SMART records the failure and skips downstream stages for
that mesh. This is intentional: producing a silently corrupted tet mesh is worse
than skipping a bad input.

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
    auto_retry_by_failure: true
    immediate_retry_failures:
      - validation_open_surface
      - command_crash
      - command_timeout
    basic_cleanup: true
    fix_normals: true
    fill_holes: false
    keep_largest_component: false
    fallback_variants:
      - name: fill_holes
        triggers:
          - validation_open_surface
          - command_crash
          - command_timeout
          - command_failure
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
        triggers:
          - validation_disconnected
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
