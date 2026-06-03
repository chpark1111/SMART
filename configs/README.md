# SMART Configs

YAML configs define the pipeline, parameters, data roots, and backend choices.

## Public Profiles

- `smoke_5.yaml`: fastest local smoke profile.
- `example_3x3.yaml`: local 3-per-category example profile.
- `learned_auto_safe.yaml`: opt-in production-candidate guarded learned MCTS profile.
- `learned_frontier.yaml`: opt-in guarded DeepSets MCTS-prior production-candidate profile.
- `demo.yaml`: small demo data profile.
- `paper_like.yaml`: paper-style parameters.
- `expanded_full.yaml`: larger local ShapeNet layout.

## Research Profiles

Research profiles for acceleration, learned priors, pruning, tet clipping, and
hybrid search experiments live under `experiments/configs/`. That directory is
ignored by git and excluded from release packages.

## Learned Prior Promotion Status

The MCTS learned-prior safe profile is exposed as `mode=guarded` and
`mode=auto_safe`.  It remains opt-in for release builds, but it is the current
default-candidate profile: all accepted states still use exact SMART/Manifold
reward, risky multibox states fall back to exact shallow MCTS, and the current
local state-level validation has zero observed quality losses.

Do not make learned macro-skill polishing the global default yet.  It remains
an opt-in post-refinement quality controller until the 500-case fresh matched
benchmark has zero-loss replacement evidence.

## Wheel Copy

The source configs in this directory are mirrored under `smart/configs/` so an
installed wheel can run without a full repository checkout.
