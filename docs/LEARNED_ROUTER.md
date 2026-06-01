# Learned Geometry Router

This document summarizes the opt-in learned routing work for SMART native
refinement.  The release default remains exact native SMART; the learned router
is a research acceleration path.

## Idea

SMART refinement evaluates candidate bounding-box edits with an exact
Manifold-backed reward.  That exact reward is reliable, but scoring the full
candidate pool dominates runtime.  The learned router reduces this cost by
ranking candidate actions before exact scoring.

The model does not replace SMART's reward.  The flow is:

1. Build cheap geometry features for candidate axis edits.
2. Rank candidates with the packaged DeepSets policy.
3. Exact-score only the selected candidate subset with native SMART/Manifold.
4. Apply the exact-best action among the checked candidates.

This preserves the reward semantics for accepted actions while reducing the
number of exact geometry calls.

## Packaged Policy

The bundled policy checkpoint is:

```text
smart/assets/policies/deepset_setaware_v2_h128_v1.smartmlp
```

It is loaded through:

```python
import smart.cpp as sc

policy = sc.load_builtin_deepset_policy()
defaults = sc.native_deepset_refine_defaults("auto")
```

## Pipeline Usage

Enable the router explicitly:

```bash
smart --config configs/smoke_5.yaml refine \
  --set refine.learned_router.enabled=true \
  --set refine.learned_router.profile=auto
```

For quality reinvestment experiments, increase the refine turn budget while the
router is enabled:

```bash
smart --config configs/smoke_5.yaml refine \
  --set refine.max_step=6 \
  --set refine.learned_router.enabled=true \
  --set refine.learned_router.profile=auto
```

The corresponding YAML form is:

```yaml
refine:
  backend: cpp_native
  max_step: 6
  learned_router:
    enabled: true
    policy: default
    profile: auto
    overrides: {}
```

For the current best-known opt-in routing policy, use the portfolio helper:

```python
import smart.cpp as sc

engine = sc.NativeSmartEngine(...)
result = sc.run_builtin_deepset_portfolio_refine(engine, mode="speed")
```

The portfolio keeps one-box states on native exact C++ refine and uses the
learned router only for multibox states, where candidate filtering is large
enough to pay for model inference.

## Current Validation Snapshot

Current local validation is intentionally reported as research evidence, not a
paper-level benchmark.  The router was compared against exact candidate-pool
scoring on native refine states.

### Same-Turn Acceleration

| split | profile | quality | exact-call change | wall-time change |
| --- | --- | --- | --- | --- |
| unseen probe50, 120 states | `auto` | zero regret, no oracle-loss cases | 67.1% fewer exact checks | 1.34x vs oracle pool |
| mixed case41, 39 states | `auto` | zero regret, no oracle-loss cases | 55.5% fewer exact checks | 1.59x vs oracle pool |
| hard airplane multibox, 28 states | `hard` | zero regret, no oracle-loss cases | 79.9% fewer exact checks | 2.02x vs oracle pool |

### Budget Reinvestment

The saved exact-call budget can also be spent on additional refinement turns.
This can improve quality while staying within or near the original exact
runtime budget.

| split | baseline | router | score change | exact-call change | wall-time change | quality losses |
| --- | --- | --- | --- | --- | --- | --- |
| hard airplane, 28 states | exact 4-turn | hard router 5-turn | +0.2087 | 69.3% fewer | 31.0% faster | 0 |
| hard airplane, 28 states | exact 4-turn | hard router 6-turn | +0.4010 | 58.2% fewer | 18.4% faster | 0 |
| mixed case41, 39 states | exact 4-turn | auto router 5-turn | +0.1817 | 40.6% fewer | 20.8% faster | 0 |
| mixed case41, 39 states | exact 4-turn | hard router 6-turn | +0.3449 | 55.2% fewer | 9.6% faster | 0 |
| mixed case41, 39 states | exact 4-turn | auto router 6-turn | +0.3464 | 25.1% fewer | approximately equal | 0 |

The 8-turn router improves quality more, but can become slower on mixed states.
The current best tradeoff is profile-dependent: hard airplane benefits from
hard router 6-turn when quality is prioritized, while mixed case41 has a
cleaner quality/time tradeoff with hard or mixed router 6-turn.  The conservative
`auto` profile remains the safest public opt-in default because it has extra
small-pool exact fallbacks.

### Portfolio Scheduler

The current research portfolio separates exact-native and learned-router cases:

| split | mode | route | score change | exact-check change | wall-time change | quality losses |
| --- | --- | --- | --- | --- | --- | --- |
| hard airplane, 28 multibox states | `speed` | learned `auto` 5-turn | +0.8821 | 25.3% fewer | 3.3% faster | 0 |
| hard airplane, 28 multibox states | `balanced` | learned `hard` 6-turn | +1.0744 | 45.1% fewer | 4.1% slower | 0 |
| mixed live, 86 one-box states | `speed` | native exact 5-turn | +0.3016 | not comparable | 44.3% faster | 0 |
| mixed live, 86 one-box states | `balanced` | native exact 6-turn | +0.4980 | not comparable | 37.6% faster | 0 |

The one-box rows are not learned-router wins; they are exact C++ shortcut wins.
A learned-only check on those one-box states improved score but was about 5x
slower.  This is why the portfolio intentionally routes one-box states to exact
native refine.

### Native MCTS Priors

The same DeepSets candidate scorer can also create `action_prior_logits` for
native C++ MCTS:

```python
result = sc.run_builtin_deepset_prior_mcts(
    engine,
    num_iter=50,
    max_step=2,
)
```

There is also a more experimental state-refreshed variant:

```python
result = sc.run_builtin_deepset_dynamic_prior_mcts(
    engine,
    num_iter=50,
    max_step=2,
)
```

The dynamic variant refreshes DeepSets scores when each C++ MCTS node is
created.  It is useful for research, but it is not the current recommended
setting because the extra neural scoring overhead outweighed the benefit on the
114-state validation set.

Current research result:

| split | setting | score change | node change | wall-time change | quality losses |
| --- | --- | --- | --- | --- | --- |
| hard airplane, 28 multibox states | MCTS 50, top8, weight 0.05 | +0.0567 | 51.0 -> 17.25 | 63.4% faster | 0 |
| mixed live, 86 one-box states | MCTS 50, top15, weight 0.05 | +0.0000 | 37.94 -> 25.85 | 11.4% faster | 0 |
| combined, 114 states | MCTS 50, box-count top-k | +0.0139 | 41.15 -> 23.74 | 61.8% faster | 0 |
| combined, 114 states | MCTS 50, box-count top-k + TT | +0.0139 | 41.15 -> 22.75 | 70.6% faster | 0 |
| combined, 114 states | exact 50/depth2 vs prior 100/scheduled depth | +0.1075 | 41.15 -> 25.82 | 13.7% faster | 0 |
| combined, 114 states | exact 50/depth2 vs tuned prior 100/scheduled depth | +0.1075 | 41.15 -> 24.34 | 34.4% faster | 0 |
| combined, 114 states | tuned prior 100/scheduled depth + TT | +0.1075 | 41.15 -> 24.34 | 35.0% faster | 0 |
| combined, 114 states | dynamic node prior, MCTS 50, top6/top15 | +0.0139 | 41.15 -> 22.75 | 64.3% faster | 0 |
| combined, 114 states | dynamic node prior, MCTS 100/scheduled depth | +0.1075 | 41.15 -> 24.30 | 24.4% faster | 0 |
| combined, 114 states | dynamic node prior, top4/top15 | +0.0127 | 41.15 -> 21.77 | 72.5% faster | 2 |
| combined, 114 states | static prior, depth3, top5/top15 + TT | +0.3933 | 41.15 -> 32.45 | 43.2% faster | 0 |
| combined, 114 states | static prior, depth4, top4/top15 + TT | +0.7161 | 41.15 -> 41.22 | 29.6% faster | 0 |
| combined, 114 states | static prior, depth4, top1/top15 + TT, 25 iters | +0.7121 | 41.15 -> 20.52 | 81.3% faster | 0 |

Aggressive one-box top8 was faster but caused quality-loss cases, so it is not
a safe setting.  Earlier safe rules used top6 for multibox states and
top15/top20 for one-box states.  The newer depth4 frontier is much narrower for
multibox states, but it is still validation-only because top1 pruning needs
larger held-out checks.  Dynamic top4/top15 is rejected because it introduced
two quality-loss cases; dynamic top6/top15 is safe in this check but slower than
the static-root tuned prior.  The transposition table is safe in this check and
gives a small additional runtime gain, but the main gain comes from using the
learned prior to make deeper exact MCTS affordable.

The newest result is different from pure acceleration: it uses the learned
prior to make a deeper but narrower MCTS practical.  The baseline is exact MCTS
with 50 iterations and depth 2.  The aggressive `mode="frontier"` setting uses
multibox top1, one-box top15, depth4, 25 iterations, and TT.  It is very fast
and improved the 114-state validation pool, but a larger unseen probe exposed
low-confidence multibox states where top1 pruning could miss the exact
baseline.  The recommended research setting is therefore `mode="guarded"`.

`guarded` keeps the same frontier route for normal states, but if a multibox
state has an initial exact score above `-0.5`, it disables learned pruning and
runs exact shallow MCTS for that state.  The guard is intentionally simple: the
failure analysis showed that the frontier losses clustered in low-confidence
multibox states with near-zero or weakly negative initial score, while stable
hard multibox states had clearly lower initial scores.  This preserves most of
the frontier runtime gain while removing the observed quality-loss cases.

Additional local validation:

| split | cases | mean score change | wall-time ratio | quality losses |
| --- | ---: | ---: | ---: | ---: |
| hard airplane multibox | 28 | +0.7073 | 0.172x | 0 |
| live one-box/mixed | 86 | +0.7137 | 0.920x | 0 |
| unseen probe | 150 | +0.8939 | 0.817x | 0 |
| case41 mixed category | 39 | +0.5771 | 0.191x | 0 |
| category expand | 6 | +0.2225 | 0.245x | 0 |
| combined checked pool | 309 | +0.7738 | 0.695x | 0 |

The same three-split full check was repeated with seeds 1111 and 2222.  All six
additional rows also had zero quality-loss cases; the hard-airplane multibox
score gain ranged from +0.7062 to +0.7403, while live and unseen rows were
unchanged for these deterministic token pools.

The guard threshold was swept on the unseen probe.  Thresholds from `-0.8` to
`-0.5` removed all observed losses; `-0.4` and above reintroduced a multibox
loss.  The packaged default is therefore `guard_multibox_score_gt=-0.5`, the
least conservative value that was safe in the sweep.

The exact fallback budget was also swept.  25 and 30 fallback iterations caused
seed-dependent losses, while 35 iterations removed the observed losses on the
three-seed multibox/unseen check.  The packaged fallback therefore uses exact
MCTS depth2/35 iterations rather than the full depth2/50 baseline.
For very near-zero risky states (`initial_score > -0.05`), a second tier uses
30 fallback iterations; this preserved the three-seed checks and slightly
reduced the unseen-probe runtime.

The validation summary is stored at:

```text
experiments/macro_search/runs/frontier_validation/guarded_summary.json
```

The MCTS-prior validation harness accepts the same preset names as the packaged
API:

```bash
python experiments/macro_search/evaluate_deepset_mcts_prior.py \
  --preset guarded \
  --token-glob 'experiments/macro_search/runs/<token_set>/*/*/turn_*.json' \
  --limit 0 \
  --transposition-table \
  --output experiments/macro_search/runs/frontier_validation.json
```

For multi-split checks and guard sweeps:

```bash
python experiments/macro_search/run_deepset_mcts_frontier_validation.py \
  --preset guarded \
  --limit 0 \
  --seeds 7777,1111,2222 \
  --split multibox_airplane28 \
  --split unseen_probe50 \
  --transposition-table \
  --guard-multibox-score-gt -0.5 \
  --guard-num-iter 35
```

The report records both configured values and the effective scheduled values
for box-count rules, e.g. `mean_scheduled_action_prior_top_k` and
`mean_scheduled_mcts_max_step`.

Available helper presets:

| mode | intent | multibox top-k | one-box top-k | depth | recommended iters |
| --- | --- | --- | --- | --- | --- |
| `speed` | shallow MCTS acceleration | 6 | 15 | 2 | 50 |
| `balanced` | conservative budget reinvestment | 6 | 15 | multibox 3, one-box 2 | 100 |
| `quality` | safer deeper search | 4 | 15 | 4 | 50 |
| `frontier` | aggressive speed+quality frontier | 1 | 15 | 4 | 25 |
| `guarded` | frontier plus exact fallback for multibox score > -0.5 | 1 or exact | 15 | 4 or exact depth2 | 25 or exact 35 |

Example:

```python
import smart.cpp as sc

result = sc.run_builtin_deepset_prior_mcts(
    engine,
    mode="guarded",
    transposition_table=True,
)
```

Pipeline config example:

```yaml
mcts:
  backend: cpp_native
  direct_file_runner: true
  learned_prior:
    enabled: true
    policy: default
    mode: guarded
    transposition_table: true
```

For a complete sample profile, use:

```bash
smart --config configs/learned_frontier.yaml run
```

That profile keeps the exact C++ SMART pipeline, enables in-process
`smart._cpp` DeepSets MCTS priors only for the MCTS stage, and leaves final
state scoring on exact native SMART/Manifold evaluation.

This uses the packaged policy through `smart._cpp`, not the standalone
`smart-cpp-native` executable, because DeepSets inference is currently exposed
as an in-process native extension API.

A separate budget-scheduler probe compared the fast MCTS-50 route against the
quality MCTS-100/depth route.  A ridge gate trained on bbox/proxy token
features matched the quality score on the held-out split with zero losses, but
only saved a small amount beyond always using the quality route.  The current
practical recommendation is still the tuned static prior, not a learned budget
gate.

## Profiles

| profile | intent |
| --- | --- |
| `auto` | conservative default: exact route for one-box states and small-pool fallback |
| `mixed` | balanced research setting for mixed-category multibox validation |
| `hard` | faster hard-case profile with fewer small-pool exact fallbacks |
| `fast` | aggressive probe profile; can introduce quality losses |

## Status

Implemented:

- packaged policy checkpoint,
- C++ native inference path through `smart._cpp`,
- Python API wrapper,
- pipeline opt-in config,
- route diagnostics and exact-check stats,
- tests for the direct API and pipeline opt-in path.
- portfolio helper for choosing exact native vs learned-router refine.

Not promoted:

- the router is not the default reproduction backend,
- MCTS integration is not yet the default,
- larger held-out ShapeNet validation is still needed before claiming a
  generalizable control agent.
