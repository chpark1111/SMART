# SMART Configs

YAML configs define the pipeline, parameters, data roots, and backend choices.

## Public Profiles

- `smoke_5.yaml`: fastest local smoke profile.
- `example_3x3.yaml`: local 3-per-category example profile.
- `learned_auto_safe.yaml`: opt-in production-candidate guarded learned MCTS profile.
- `learned_macro_safe.yaml`: opt-in release-candidate learned MCTS plus exact-validated macro-skill polishing profile.
- `learned_macro_program_gate_top3.yaml`: opt-in stage-source validated macro-skill profile using pure program-gate top-3 exact validation after MCTS.
- `learned_macro_refine_only.yaml`: opt-in research profile that skips MCTS and runs the exact-validated substructure planner directly after refine.
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

Do not make learned macro-skill polishing the global default yet.  Use
`learned_macro_safe.yaml` when you explicitly want the post-MCTS macro-skill
planner.  Use `learned_macro_program_gate_top3.yaml` when you want the tighter
stage-source validated setting: pure program-gate order, top-3 exact
validation, and no macro-memory re-rank.  That setting passed 456-case refine
and MCTS stage-source gates with zero accepted-state regressions and 81.25%
fewer exact skill attempts than the 16-skill portfolio, but it remains opt-in
until full mesh-level pipeline validation clears.

Use `learned_macro_refine_only.yaml` only for MCTS-replacement research. It
feeds the refine output into the same exact-validated planner and keeps the
same rollback contract, but it is not default-ready until matched refine-only
versus MCTS+macro evaluation clears the 500+ fresh-state zero-regression gate.

## Wheel Copy

The source configs in this directory are mirrored under `smart/configs/` so an
installed wheel can run without a full repository checkout.
