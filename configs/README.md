# SMART Configs

YAML configs define the pipeline, parameters, data roots, and backend choices.

## Public Profiles

- `smoke_5.yaml`: fastest local smoke profile.
- `example_3x3.yaml`: local 3-per-category example profile.
- `demo.yaml`: small demo data profile.
- `paper_like.yaml`: paper-style parameters.
- `expanded_full.yaml`: larger local ShapeNet layout.

## Research Profiles

Research profiles for acceleration, learned priors, pruning, tet clipping, and
hybrid search experiments live under `experiments/configs/`. That directory is
ignored by git and excluded from release packages.

## Wheel Copy

The source configs in this directory are mirrored under `smart/configs/` so an
installed wheel can run without a full repository checkout.
