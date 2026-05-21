# SMART Research Plan

This document tracks research directions that are intentionally separate from
the release path. Release defaults should remain reproducible and documented;
experimental learned systems should live under `experiments/` until they win on
held-out meshes.

## Baseline To Beat

Use the current native C++ pipeline as the reference baseline:

```bash
smart --config configs/paper_like.yaml run
smart --config configs/paper_like.yaml evaluate
```

Compare every experiment on both quality and runtime:

- runtime per mesh and per stage;
- bbox count;
- BVS, MOV, TOV, Covered, vIoU, cub_CD;
- tetra/preseg/merge/refine/MCTS success rates;
- category-level breakdown for airplane, chair, and table.

For quality-focused experiments, exact equality with the baseline is not
required. The rule is: equal or better quality on held-out meshes, or clearly
faster at no meaningful quality loss.

## Data Loop

1. Run the native baseline on all available ShapeNet data.
2. Save candidate/action traces from refine and MCTS.
3. Label actions by exact SMART reward improvement and final return.
4. Keep train/validation/test splits by mesh id, not by action row.
5. Report results per category and pooled across categories.

Useful splits:

- category-general: train on airplane/chair/table together;
- per-category: train one policy per category;
- cross-category: train on two categories, test on the third.

## Policy/Value Agent

The first learned system should guide search, not replace the geometry metric.

Inputs:

- normalized bbox bounds and rotations;
- action descriptor: bbox index, axis, direction, scale/recenter flag;
- current bbox volumes and aspect ratios;
- tet/mesh summary features such as centroid bounds, covered volume, uncovered
  volume, and bbox overlap summaries;
- optional category embedding.

Outputs:

- policy prior over candidate actions;
- value estimate for expected final improvement;
- optional escape score when the current local search has stalled.

Integration:

- pass learned priors through `mcts.action_prior_path`;
- use PUCT/action ordering to spend exact reward calls on better candidates;
- keep exact SMART reward for accepted states and final evaluation.

## Local-Minimum Escape Agent

MCTS exists because greedy refine can get trapped. A learned escape agent should
trigger only after no-improvement windows:

```yaml
mcts:
  escape_policy: true
  escape_after_no_update: 20
  escape_action_top_k: 8
  escape_probability: 0.5
```

Training target:

- states where greedy did not improve for `N` steps;
- actions from successful later MCTS branches;
- positive label if the branch improves final BVS/coverage/tightness.

## Memory And Table-Based Search

Use memory as a deterministic accelerator before relying on learned inference:

- transposition table keyed by quantized bbox state;
- reward cache keyed by `(state_hash, action)`;
- category-level action statistics from successful traces;
- shape-neighbor retrieval from global tet/bbox summary features.

These systems are safe only if they do not change final exact reward semantics.
They may change search order; therefore evaluate as quality/time experiments,
not parity tests.

## Deep Learning Backend

Use PyTorch for research training. On Apple Silicon, use MPS when available:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

Keep model artifacts out of the release package until they are validated:

```text
experiments/assets/priors/
experiments/assets/value/
experiments/assets/escape/
```

## Promotion Rule

Move an experiment into the supported package only after:

1. it improves held-out category metrics or runtime;
2. it has a documented config profile;
3. it has deterministic smoke tests;
4. it has a fallback path when no model file is installed;
5. the release default remains stable and reproducible.
