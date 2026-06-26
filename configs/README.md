# SMART Configs

YAML configs define the pipeline, parameters, data roots, and backend choices.

## Public Profiles

- `smoke_5.yaml`: fastest local smoke profile.
- `example_3x3.yaml`: local 3-per-category example profile.
- `learned_default.yaml`: learned default-agent profile for non-paper-reproduction runs; skips MCTS and uses the guarded macrohash MCTS-replacement controller with exact fallback.
- `learned_auto_safe.yaml`: opt-in production-candidate guarded learned MCTS profile.
- `learned_macro_safe.yaml`: opt-in release-candidate learned MCTS plus exact-validated macro-skill polishing profile.
- `learned_macro_program_gate_top3.yaml`: opt-in stage-source validated macro-skill profile using pure program-gate top-3 exact validation after MCTS.
- `learned_macro_refine_only.yaml`: opt-in research profile that skips MCTS and runs the exact-validated substructure planner directly after refine.
- `learned_macro_mcts_replacement_guarded.yaml`: underlying guarded MCTS-replacement profile used by the learned default agent. It skips MCTS, ranks the 16-skill exact portfolio with the packaged macrohash selector, tries learned top-3 first, then falls back to the exact portfolio when confidence is low.
- `learned_macro_mcts_replacement_learned_only.yaml`: research/production-candidate profile that skips MCTS and does not use the 16-skill exact portfolio fallback. It uses the packaged cross-model macrohash selector and exact-validates only learned top-4 programs.
- `learned_frontier.yaml`: opt-in guarded DeepSets MCTS-prior production-candidate profile.
- `demo.yaml`: small demo data profile.
- `paper_like.yaml`: paper-style parameters.
- `expanded_full.yaml`: larger local ShapeNet layout.

## Research Profiles

Research profiles for acceleration, learned priors, pruning, tet clipping, and
hybrid search experiments live under `experiments/configs/`. That directory is
ignored by git and excluded from release packages.

## Learned Prior Promotion Status

For the current accuracy/speed table and claim boundary, see
[`docs/LEARNED_ROUTER_RELEASE.md`](../docs/LEARNED_ROUTER_RELEASE.md).

The default CLI/API run path now uses `learned_default.yaml`: exact native
refine followed by the guarded macrohash MCTS-replacement controller, with exact
SMART/Manifold validation and rollback/fallback.  Use explicit paper configs
such as `paper_like.yaml` when reproducing the original paper baseline.  Use
`learned_macro_safe.yaml` when you explicitly want the post-MCTS macro-skill
planner.  Use `learned_macro_program_gate_top3.yaml` when you want the tighter
stage-source validated setting: pure program-gate order, top-3 exact
validation, and no macro-memory re-rank.  That setting passed 456-case refine
and MCTS stage-source gates with zero accepted-state regressions and 81.25%
fewer exact skill attempts than the 16-skill portfolio.  It remains separate
from the learned MCTS-replacement default profile below because it is a
post-MCTS polishing configuration.

Use `learned_default.yaml` when you want the current learned MCTS-replacement
default agent for non-paper-reproduction runs.  It is an alias profile around
the guarded macrohash selector path: refine output is routed into learned top-3
macro skills, and uncertain cases fall back to the 16-skill exact portfolio.
The guarded path passed the current 510-state default-candidate gate with zero
losses versus the exact fallback portfolio and 26.3% fewer exact skill attempts.
Use `learned_macro_mcts_replacement_learned_only.yaml` when you explicitly want
the stricter research setting: no exact-portfolio fallback, learned top-4
programs only, and exact validation/rollback for those top-4 attempts.  On the
401-case cross-model replay gate this learned-only top-4 setting matched the
8-skill tried oracle with zero losses and uses 75% fewer exact skill attempts
than a 16-skill exact portfolio.  It still needs fresh end-to-end mesh
wall-time validation before becoming the conservative default.  Use
`learned_macro_refine_only.yaml` only for planner-oriented MCTS-replacement
research, and use `learned_macro_mcts_replacement_guarded.yaml` when you want
the underlying guarded profile name explicitly.

CLI shortcut:

```bash
smart run
smart agent-run
smart run --agent
smart --config <your_config.yaml> run --agent
```

`smart run` and `agent-run` load `configs/learned_default.yaml` when no
`--config` is supplied.  This is the user-facing learned default-agent
entrypoint: MCTS is disabled, the macrohash MCTS-replacement controller runs
after refine, and every accepted update is still exact-validated with
rollback/fallback.  If an explicit config is supplied, `smart run` respects that
config normally; `run --agent` overlays the guarded learned agent on the
selected config, so custom datasets can reuse the MCTS-replacement controller
without copying the bundled config.

Python users can call the same profile with `smart.run_agent(...)` or overlay it
on an existing config with `smart.run("configs/smoke_5.yaml", agent=True)`.
Calling `smart.run()` with no config uses the learned default-agent profile.

## Native Pipeline Reuse Controls

For repeated profiling or learned-search experiments on the same mesh, use:

```bash
smart --config configs/smoke_5.yaml \
  --set native_pipeline.reuse_preprocessing=true \
  native-run --category table --mesh <mesh_id> --force
```

`native_pipeline.reuse_preprocessing=true` keeps normalized, ManifoldPlus,
fTetWild, and presegmentation outputs, then reruns merge/refine/MCTS.  This is
the preferred knob when comparing learned router/MCTS settings because
preprocessing can dominate end-to-end time.  `native_pipeline.reuse_existing`
reuses all stage outputs, including search outputs.

For cross-workspace reuse, enable the persistent preprocessing cache:

```bash
smart --config configs/smoke_5.yaml \
  --set native_pipeline.preprocessing_cache=true \
  native-run --category table --mesh <mesh_id> --force
```

The cache is keyed by source mesh content, normalization, tetra, CoACD, and
tool signatures.  Search settings are excluded, so changing learned router or
MCTS parameters reuses the same preprocessing artifacts while rerunning the
actual optimization stage.

For known-clean meshes, `--set tetra.skip_manifoldplus=true` bypasses the
ManifoldPlus repair step and lets fTetWild consume the normalized OBJ directly.
Leave this off for arbitrary web meshes unless validation shows the input is
stable.

`tetra.manifold_depth` keeps ManifoldPlus enabled but changes its repair
resolution.  The upstream default is `8`; lower values such as `7` can reduce
the repaired mesh face count and downstream tetra time, but they can also alter
the fitted box decomposition.  Treat it as a quality-gated speed knob, not as a
bit-identical optimization.

`configs/preprocess_fast.yaml` enables a guarded version of this idea with
category-specific depth candidates.  SMART tries the first candidate and falls
back to depth 8 if the native run fails, so this accelerates first-run
preprocessing rather than reusing previous outputs.

## Wheel Copy

The source configs in this directory are mirrored under `smart/configs/` so an
installed wheel can run without a full repository checkout.
