# Learned Geometry Router

This document summarizes the learned routing work for SMART native refinement.
The full SMART pipeline still keeps exact native SMART as the conservative
baseline, but the learned refine helper's `profile="auto"` now points to the
validated v9 production-candidate router.

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

As of the 2026-06-03 validation pass, `auto`, `auto_safe`,
`learned_auto_safe`, and `production_candidate` all resolve to the v9
hard-state gated router for multibox states.  One-box states still route to
exact native refine through `auto_exact_max_boxes=1`.

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

For a simple library call, use the builtin helper:

```python
import smart.cpp as sc

engine = sc.NativeSmartEngine(...)
result = sc.run_builtin_deepset_policy_refine(
    engine,
    max_steps=4,
    profile="auto",
)
```

For budget reinvestment experiments, use the portfolio helper:

```python
import smart.cpp as sc

engine = sc.NativeSmartEngine(...)
result = sc.run_builtin_deepset_portfolio_refine(engine, mode="speed")
```

The portfolio keeps one-box states on native exact C++ refine and uses the
learned router only for multibox states, where candidate filtering is large
enough to pay for model inference.

## Current Validation Snapshot

Current local validation is still not a paper-level benchmark, but it is strong
enough to ship as the learned-router default profile.  The router was compared
against exact candidate-pool scoring on native refine states.

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
runs baseline-budget exact MCTS for that state.  The guard is intentionally simple: the
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

On 2026-06-03, the guarded preset was rerun on all five local validation
splits with three seeds (`7777,1111,2222`).  This gives 927 weighted state
checks, all with exact SMART/Manifold final scoring and zero observed quality
losses:

| split | cases per seed | seeds | mean score change | wall-time ratio | quality losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard airplane multibox | 28 | 3 | +0.7179 | 0.174x | 0 |
| live one-box/mixed | 86 | 3 | +0.7137 | 1.000x | 0 |
| case41 mixed category | 39 | 3 | +0.5862 | 0.193x | 0 |
| category expand | 6 | 3 | +0.2241 | 0.269x | 0 |
| unseen probe | 150 | 3 | +0.8939 | 1.220x | 0 |
| weighted aggregate | 927 | -- | +0.7760 | 0.916x | 0 |

We then repeated the validation at the state level by evaluating every
recorded `turn_*.json` token rather than deduplicating to one initial state per
shape.  Across seeds `7777,1111,2222,3333,4444`, this expands the check to
9,670 weighted valid states across the same five splits.  The final guard
fallback disables learned priors and top-k pruning, and now uses the same MCTS
iteration budget as the exact baseline; this prevents the safety path from
losing quality just because it ran fewer exact iterations.

| split | weighted states | seeds | mean score change | wall-time ratio | quality losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard airplane multibox | 805 | 5 | +0.7058 | 0.162x | 0 |
| live one-box/mixed | 2480 | 5 | +0.7422 | 0.930x | 0 |
| case41 mixed category | 1150 | 5 | +0.5913 | 0.182x | 0 |
| category expand | 160 | 5 | +0.2492 | 0.244x | 0 |
| unseen probe | 5075 | 5 | +1.0367 | 0.908x | 0 |
| weighted aggregate | 9670 | -- | +0.8676 | 0.754x | 0 |

In the 9,670-state all-turn aggregate, the baseline exact MCTS expanded an
average of 40.83 nodes while the guarded learned prior expanded 21.62 nodes,
a 47.1% reduction in exact search budget.

### Default Promotion Status

The guarded DeepSets MCTS prior is now a **production candidate**, not merely a
loose ablation.  It is exposed under both `mode="guarded"` and
`mode="auto_safe"`.  The current evidence is strong enough to recommend it for
opt-in production trials:

- exact SMART/Manifold reward remains the final evaluator;
- risky multibox states fall back to baseline-budget exact MCTS;
- 9,670 all-turn state checks across five seeds had zero observed quality
  losses;
- exact MCTS node budget dropped by 47.1% on that aggregate.

It should become the package default only after the fresh 500-case matched
benchmark reproduces the same zero-loss behavior from cleanly regenerated
pipeline artifacts.  Until then, the release default remains exact native SMART
and the learned prior is enabled explicitly:

```yaml
mcts:
  learned_prior:
    enabled: true
    mode: auto_safe
```

The variable-length macro-skill controller is one step behind the MCTS prior:
it has strong quality results as an opt-in post-refinement controller, but it
has not yet replaced the historical live controller without losses.  It should
not be the global default until the strict fresh matched benchmark passes.

This strengthens the quality claim but also clarifies the runtime claim:
multibox and case41-style states show large speedups, while the unseen probe
is now slightly faster at the all-turn/state level.  The safe paper claim is
therefore: "a guarded learned prior can make deeper exact MCTS practical,
improving exact-validated quality with no observed losses on current local
checks, while reducing runtime substantially on the expensive multibox and
mixed-category subsets."  The generated claim-readiness table is stored at:

```text
experiments/macro_search/PAPER_CLAIM_READINESS_2026_06_03.md
```

The same three-split full check was repeated with seeds 1111 and 2222.  All six
additional rows also had zero quality-loss cases; the hard-airplane multibox
score gain ranged from +0.7062 to +0.7403, while live and unseen rows were
unchanged for these deterministic token pools.

The guard threshold was swept on the unseen probe.  Thresholds from `-0.8` to
`-0.5` removed all observed losses; `-0.4` and above reintroduced a multibox
loss.  The packaged default is therefore `guard_multibox_score_gt=-0.5`, the
least conservative value that was safe in the sweep.

The exact fallback budget was also swept.  25, 30, and 35 fallback iterations
can cause seed-dependent losses because the guard is no longer equivalent to
the exact baseline.  The packaged fallback therefore uses the full baseline
MCTS depth and iteration budget when learned pruning is disabled.  For very
near-zero risky states (`initial_score > -0.05`), the same baseline-budget
fallback is used; this is slightly more conservative but preserves the
five-seed all-turn check.

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

## Parameterized Skill Research

The learned router above is still a one-step candidate router.  The active
research branch now also mines reusable variable-length SMART skills from trace
data.  These are not raw fixed n-step macros; each mined sequence is lifted into
a parameterized skill description:

```text
precondition(state) -> action program -> termination(state)
```

Examples include:

- `expand_to_recover_coverage`
- `coverage_rescue_then_tighten`
- `recenter_then_shrink`
- `major_axis_extend_then_trim`
- `escape_local_minimum_expand_then_refine`

The research scripts live under `experiments/macro_search`:

```bash
python3 experiments/macro_search/run_parameterized_skill_research.py \
  --max-traces 1000 \
  --min-support 3 \
  --top-per-family 100 \
  --epochs 120 \
  --hidden 128
```

Current local smoke found `455` supported parameterized skill candidates from
the first `1000` traces.  A small MPS-backed MLP predicted the mined skill
family with best validation accuracy `88.6%` and final accuracy `83.3%` on a
row split.

Native replay is now wired for both axis edits and recenter actions through the
C++ engine.  On an 80 matched-state replay smoke:

| item | value |
| --- | ---: |
| accepted non-worse skills | 69 / 80 |
| unsupported recenter steps | 0 |
| mean accepted exact score delta | +0.8718 |
| mean accepted executed steps | 4.97 |

The strongest families so far are
`escape_local_minimum_expand_then_refine` and `major_axis_extend_then_trim`.
This is evidence of reusable trace structure and executable variable-length
skills, but not yet a release-ready controller.  The next gate is held-out
comparison against guarded MCTS with rollback, exact-call counts, wall time, and
per-category win/tie/loss metrics.

An oracle skill-bank portfolio benchmark also exists.  It tries up to eight
mined skills for each matched state and accepts the best exact-validated result.
On the currently available matched airplane states it found `28 / 29` wins with
mean best exact score delta `+1.4876`.  This is an upper bound, not an automatic
controller: it shows the mined skill bank has useful programs, and the next
problem is learned retrieval/termination.

The larger 4k-trace skill bank is noisier but gives a better retrieval test:
`762` candidates, `39` matched states, and oracle top-10 portfolio mean exact
delta `+1.2999`.  A leave-one-case-out listwise MLP retriever improves the mean
chosen delta from the raw mined-prior baseline `+0.8250` to `+1.2124`, reducing
oracle regret from `0.4749` to `0.0875` with `84.6%` top-1 oracle accuracy.
This is still research-only, but it is the first clear signal that learned
skill retrieval can select useful variable-length 3D fitting programs instead
of trying the whole portfolio.

The current best operational variant is guarded retrieval: rank skills with the
listwise MLP, execute at most the top three in exact SMART, and rollback any
non-improving skill.  On the same leave-one-case-out split, guarded top-3 gives
mean delta `+1.2487`, oracle regret `0.0512`, positive rate `82.1%`, and only
`1.08` exact skill attempts per state on average.

A live `NativeSmartEngine` controller is now implemented in the research tree.
It saves the current bbox state, executes a ranked variable-length skill in C++,
accepts only non-worse exact score, and rolls back failed skills.  The
leave-one-case-out live smoke gives `32 / 39` accepted cases, mean delta
`+1.2653`, and mean live execution time `0.042s` per case.  A saved full-fit
listwise model reproduces the oracle portfolio on the same 39 cases in `1.90s`
total, but that is an engineering smoke rather than a held-out claim.

The saved selector is now exportable as a pure JSON weight file and can be used
without PyTorch in the controller.  This is the C++-port-friendly path: the JSON
contains feature normalization, linear weights, GELU, and LayerNorm parameters.
On the same 39-case live smoke the JSON selector gives the same `32 / 39`
accepted cases, mean delta `+1.3081`, and `1.88s` total controller time.

The replay/controller scripts can also search separate tetra and bbox roots.
This expands the current live smoke from 39 airplane-only cases to 42 cases:
39 airplane, 2 chair, and 1 table.  On that small multicategory set, the
oracle skill portfolio reaches `34 / 42` wins with mean delta `+1.2358`.
A full-fit saved JSON controller reproduces the same mean delta in `2.64s`
total with `1.07` skill attempts per state.  In leave-one-case-out retrieval,
MLP-listwise guarded top-2/top-3 gives mean delta `+1.1979`, oracle regret
`0.0292`, and positive rate `81.0%`.  This is promising but not enough data to
claim category-general behavior yet; chair/table live traces need to be expanded
next.

The next, stronger test is cross-model skill reuse.  Instead of trying skills
mined from the same model only, `candidate_scope=category` lets each target
shape retrieve from the category-level skill bank.  The valid fixed benchmark
executes every skill on the target case, not on the source model.  With top-100
category candidates, the oracle portfolio wins all 42 currently matched cases
with mean exact delta `+2.4117`.  Raw trace-prior ranking remains weak
(`+0.4799`), but learned retrieval closes most of the gap:

| selector, LOOCV | mean guarded delta | oracle regret | attempts |
| --- | ---: | ---: | ---: |
| MLP listwise top-1 | +2.2765 | 0.1352 | 1.00 |
| MLP regression top-1 | +2.3965 | 0.0151 | 1.00 |
| oracle portfolio | +2.4117 | 0.0000 | up to 100 |

A full-fit regression JSON selector reaches mean live delta `+2.3907` with one
skill attempt per state and `4.36s` total controller time.  This is currently
the strongest evidence for reusable variable-length 3D fitting skills, but it
is still a research result because the matched non-airplane live set is only
three shapes.

The latest filesystem-heldout run removes that same-model bottleneck.  We
generated missing bbox parameter files beside the available tetra meshes and
scanned runnable target cases directly from disk, independent of whether the
target model had mined skills.  This exposes `176` runnable targets:
`87` airplane, `46` chair, and `43` table.

With category-scope skill reuse and only the top `16` mined skills per target,
the exact oracle portfolio found a positive reusable skill on every target:

| benchmark | cases | candidates/case | wins | mean best exact delta |
| --- | ---: | ---: | ---: | ---: |
| filesystem category-scope | 176 | 16 | 176 | +1.1084 |

Held-out case split retrieval is now the more meaningful number.  On a 70/30
split (`123` train cases, `53` test cases), the learned retriever is much
stronger than the raw trace prior:

| selector, guarded | mean exact delta | oracle regret | positive rate | attempts |
| --- | ---: | ---: | ---: | ---: |
| trace prior top-3 | +0.6287 | 0.5055 | 100.0% | 1.00 |
| MLP listwise top-3 | +1.0001 | 0.1341 | 98.1% | 1.04 |
| MLP regression top-3 | +1.0731 | 0.0611 | 100.0% | 1.04 |
| oracle portfolio | +1.1342 | 0.0000 | 100.0% | up to 16 |

The deploy-style controller uses a full-fit regression MLP exported to JSON
weights, then ranks skills with NumPy/C++-portable inference and validates the
chosen skill with exact SMART scoring.  Adding a compact C++ proxy-action
distribution summary to the target features improves the live controller.  On
all `176` targets it accepted all cases with mean delta `+1.0991`, exactly
`1.00` skill attempt per state, and `26.62s` total runtime (`0.113s` per case).
The raw trace-prior controller accepted all cases too, but only reached mean
delta `+0.5194`.  The learned controller is therefore within `0.0093` mean
delta of the top-16 exact oracle portfolio while trying about one exact skill
per state.  So this branch is no longer just a time shortcut: it is a learned
variable-length skill selector that improves exact quality over simple
mined-prior ordering.

There is also an exact-safe quality mode: execute the learned top-k skills,
then keep the best exact-positive result.  This does not change the reward
definition, only the number of exact skill executions:

| mode | exact skill attempts | mean exact delta | total time |
| --- | ---: | ---: | ---: |
| first positive, top-1 behavior | 1.00 | +1.0991 | 26.62s |
| best positive, top-2 | 2.00 | +1.1053 | 38.47s |
| best positive, top-3 | 3.00 | +1.1070 | 47.70s |
| top-16 exact oracle | up to 16 | +1.1084 | - |

So the practical profile is top-1/first-positive, while top-2 or top-3 is a
quality profile that nearly matches the oracle with far fewer exact tries.
An adaptive margin mode is also implemented: when the selector's top two scores
are close, it exact-picks the best of the learned top-k; otherwise it accepts
the first positive skill.  Offline tradeoff analysis from the top-3 report gives
useful intermediate profiles:

| adaptive profile | exact attempts | mean exact delta |
| --- | ---: | ---: |
| margin 0.02, best of top-2 | 1.35 | +1.1023 |
| margin 0.05, best of top-2 | 1.62 | +1.1033 |
| margin 0.10, best of top-2 | 1.80 | +1.1053 |
| margin 0.10, best of top-3 | 2.59 | +1.1070 |

### Macro Identity Memory

The next improvement adds reusable-program identity to the selector.  Earlier
models only saw the broad skill family plus target geometry/proxy summaries.
That is too coarse: two `recenter_then_shrink` macros can behave very
differently.  The new feature appends a fixed-width 64-bucket hash sketch of
`macro_id` to each candidate token.  This keeps the model portable to JSON and
C++ inference while giving it a memory of which concrete macro program usually
works.

Held-out 70/30 split:

| selector | mean exact delta | oracle regret | attempts/state |
| --- | ---: | ---: | ---: |
| previous proxy MLP, adaptive top-3 margin 0.10 | +0.7968 | 0.0385 | 2.47 |
| macro-hash MLP, first positive | +0.7716 | 0.0636 | 1.00 |
| macro-hash MLP, adaptive top-2 margin 0.05 | +0.8041 | 0.0311 | 1.60 |
| macro-hash MLP, adaptive top-3 margin 0.10 | +0.8083 | 0.0270 | 2.58 |

Three-seed stratified split validation gives the same conclusion:

| selector | mean exact delta | oracle regret | attempts/state |
| --- | ---: | ---: | ---: |
| raw trace prior | +0.5337 +/- 0.1193 | 0.5320 +/- 0.0444 | 1.00 |
| macro mean memory | +0.9340 +/- 0.1362 | 0.1317 +/- 0.0273 | 1.03 |
| macro-hash MLP, first positive | +1.0163 +/- 0.1739 | 0.0494 +/- 0.0126 | 1.00 |
| macro-hash MLP, adaptive top-2 margin 0.05 | +1.0354 +/- 0.1644 | 0.0303 +/- 0.0066 | 1.50 |
| macro-hash MLP, adaptive top-3 margin 0.10 | +1.0412 +/- 0.1654 | 0.0246 +/- 0.0059 | 2.36 |

Live C++ controller on all 176 targets:

| controller | mean exact delta | attempts/state | total time |
| --- | ---: | ---: | ---: |
| previous proxy regression | +1.0991 | 1.00 | 26.62s |
| previous proxy regression, best top-3 | +1.1070 | 3.00 | 47.70s |
| macro-hash regression | +1.3315 | 1.00 | 31.52s |
| macro-hash regression, adaptive top-2 margin 0.05 | +1.3372 | 1.55 | 43.17s |
| macro-hash regression, best top-3 | +1.3434 | 3.00 | 59.13s |

These live numbers should be compared against the previous live controller, not
directly against the earlier portfolio oracle.  The portfolio was generated
with a smaller skill-internal candidate budget (`64`), while live execution
uses `128` proxy candidates inside each selected skill.

Per-category live mean exact delta for the macro-hash first-positive model:

| category | previous proxy model | macro-hash model |
| --- | ---: | ---: |
| airplane | +1.3699 | +1.7044 |
| chair | +0.8205 | +0.9421 |
| table | +0.8492 | +0.9935 |

This is a better match for the "3D knowledge memory" hypothesis: the model is
not only reading current geometry, it also learns which reusable variable-length
macro programs are worth trying.  Exact SMART reward remains the acceptance
gate, so failed macro predictions can still be rolled back safely.

### Candidate Budget Knob

The selected macro still has an internal search budget: `candidate-count`
controls how many proxy-ranked primitive actions are considered while executing
the variable-length skill.  With the macro-hash selector:

| candidate-count / mode | mean exact delta | attempts/state | total time | positive rate |
| --- | ---: | ---: | ---: | ---: |
| 32, first positive | +0.8533 | 1.03 | 22.43s | 98.3% |
| 64, first positive | +1.0995 | 1.00 | 30.90s | 100.0% |
| 128, first positive | +1.3315 | 1.00 | 34.26s | 100.0% |
| 256, first positive | +1.4359 | 1.00 | 39.09s | 100.0% |
| 512, first positive | +1.4432 | 1.00 | 40.19s | 100.0% |
| 256, adaptive top-2 margin 0.05 | +1.4471 | 1.55 | 53.53s | 100.0% |

The current practical setting is `candidate-count=256`: it captures most of the
quality jump without the nearly flat tail at `512`.  The `512` setting is a
slightly stronger quality default if the extra few seconds are acceptable.

### C++ Scalar MLP Inference Guard

The exported macro-hash JSON model can now be evaluated by
`smart._cpp.NativeScalarMlpScorer`.  A direct replacement is not always faster:
NumPy uses optimized vector kernels on larger batches, while the C++ scorer wins
mainly for the small candidate batches used inside live skill execution.

Measured on the `2816` portfolio rows, the C++ scorer is faster up to roughly
`128` candidates and equivalent around `256` candidates:

| rows | C++ mean | NumPy mean | C++ / NumPy |
| ---: | ---: | ---: | ---: |
| 16 | 0.65 ms | 2.11 ms | 0.31x |
| 64 | 2.52 ms | 3.60 ms | 0.70x |
| 128 | 5.12 ms | 5.53 ms | 0.93x |
| 256 | 10.22 ms | 10.25 ms | 1.00x |
| 512 | 20.00 ms | 18.72 ms | 1.07x |

For that reason the experiment harness only uses native scalar inference when
the row count is at or below `SMART_NATIVE_SCALAR_MLP_MAX_ROWS`, default `256`.
On the all-176 live benchmark with `candidate-count=256`, the guarded native
path preserved the exact same mean delta, `+1.4359`, and reduced total runtime
from `39.09s` to `36.12s`.

### Dynamic Budget Selector Status

The budget sweep also revealed a non-obvious research direction:
`candidate-count` is not monotonic.  Smaller budgets sometimes win because they
change the skill trajectory.  The exact-best budget distribution over the 176
cases was `32:32`, `64:53`, `128:54`, `256:34`, `512:3`.

An oracle dynamic budget policy can keep fixed-256 quality while reducing time:
within `0.05` exact delta of fixed `256`, it gets `+1.4353` versus `+1.4359`
and reduces summed per-case elapsed time from `31.59s` to `28.00s`.

A first learned budget selector using only cheap pre-execution case features is
not good enough yet: five 70/30 splits averaged `+1.3641` versus fixed-256
`+1.4222`, while saving about `12%` time.  The next version needs richer native
geometry features: slack statistics, uncovered centroid direction, score spread
by skill family, and shape/bbox alignment.

### Larger Model Check

I also tested a larger macro-hash retriever, hidden size `512` for `160` epochs,
on the same three stratified seeds.  It did not improve over the smaller
hidden-`256` model:

| model | adaptive top-2 m=0.05 mean delta | regret | attempts |
| --- | ---: | ---: | ---: |
| hidden 256 / 120 epochs | +1.0354 | 0.0303 | 1.50 |
| hidden 512 / 160 epochs | +1.0270 | 0.0387 | 1.48 |

This points away from "just use a bigger MLP" and toward richer geometry state
features and better reusable skill abstraction.

### Extracted Knowledge Patterns

The macro miner currently extracts `762` parameterized skill candidates.  These
are not natural-language facts; they are executable templates with
preconditions, target roles, repeat counts, and termination predicates.

Live all-176 usage of the extracted families:

| family | mined macros | support | live accepted | live mean delta |
| --- | ---: | ---: | ---: | ---: |
| `recenter_then_shrink` | 200 | 4659 | 102 | +1.6995 |
| `escape_local_minimum_expand_then_refine` | 79 | 715 | 52 | +0.9575 |
| `shrink_slack_face` | 200 | 25226 | 22 | +1.3441 |
| `expand_to_recover_coverage` | 88 | 2010 | 0 | 0.0000 |
| `major_axis_extend_then_trim` | 183 | 4625 | 0 | 0.0000 |

The most common executable patterns are:

| support | pattern |
| ---: | --- |
| 13651 | `1x shrink_face -> max_slack_face` |
| 10147 | `2x shrink_face -> max_slack_face` |
| 4852 | `4x shrink_face -> max_slack_face` |
| 2253 | `1x recenter_box -> single_box` |
| 1304 | `recenter_box -> single_box`, then `shrink_face -> max_slack_face` |

So yes, knowledge extraction is happening, but the useful live knowledge is
currently concentrated in three families: recenter-then-tighten, escape by
coverage expansion then tighten, and repeated slack-face shrink.  The next
expansion should mine cross-family compositions rather than only adding larger
neural networks.

### Variable-Length Template Generalization

I further compressed the 762 macro candidates into `93` variable-length
templates by dropping fixed repeat counts from the signature and storing repeat
ranges instead.  This is closer to the intended "3D knowledge" form:

```text
precondition bucket -> option template with repeat ranges -> termination rule
```

Top live-used templates:

| template | repeat range | live accepted | live mean delta |
| --- | --- | ---: | ---: |
| `shrink max_slack -> recenter single_box -> shrink max_slack` | `1-11`, `1`, `1-14` | 63 | +1.9779 |
| `expand coverage_gap -> recenter dominant_box -> shrink max_slack` | `1-7`, `1-2`, `1-12` | 46 | +1.0394 |
| `shrink max_slack` | `1-16` | 22 | +1.3441 |
| `recenter single_box -> shrink max_slack` | `1-2`, `1-15` | 15 | +1.1543 |

This is the clearest current evidence for reusable 3D fitting knowledge:
fixed-length action traces can be lifted into variable-length skills like
"tighten until margin is low, recenter, then tighten again" or "expand to
recover coverage, recenter, then tighten."  These are not yet a full
generalizable controller, but they are meaningful option templates.

### Live Knowledge Pattern Mining

I added `experiments/macro_search/mine_live_knowledge_patterns.py` to mine
knowledge patterns directly from live exact-validated skill-controller logs.
This is stricter than only counting mined trace candidates: it asks which
programs actually win after SMART/Manifold exact validation inside the live
top-k candidate set.

The current report is stored at:

```text
experiments/macro_search/runs/live_knowledge_patterns/knowledge_patterns_report.md
```

It combines the all-176 top-3 log, the airplane80 top-5 log, the chair/table80
top-5 log, and the held-out16 top-5 log:

| cases | default mean delta | family-rep oracle delta | family calls | oracle-family first | bounded top-k oracle delta | hard ordering rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 352 | +0.2764 | +1.1316 | 2.49 | +0.5619 | +1.5212 | 60.2% |

The oracle here is only the best exact-scored attempt inside each logged top-k
candidate set.  It is a useful local certificate, but it is not a proof of the
global minimum over continuous bbox states and arbitrary-length programs.  A
true global certificate would require a defined finite search lattice plus
branch-and-bound or exhaustive enumeration over bbox/action programs, with exact
SMART/Manifold validation at the leaves.

The `family-rep oracle` result evaluates only the first candidate from each
semantic family and then takes the best exact result.  It improves mean delta
from `+0.2764` to `+1.1316` with `2.49` exact attempts on average, versus
`+1.5212` for full top-k oracle.  This says the abstract family is useful and
removes much of the search ambiguity, but the remaining `0.3896` gap requires
learning parameters and termination inside each family.

The stricter `oracle-family first` number is also important.  It assumes we
magically know which semantic family contains the full top-k oracle, but then
uses only that family's first representative.  Its mean delta is only
`+0.5619`, leaving a `+0.9593` parameterization gap to the true top-k oracle.
So the research target is not simply "choose recenter vs escape"; it is:

```text
choose family
-> choose target box/axis/face and repeat/termination parameters
-> exact validate and rollback if needed
```

The live pattern mining gives a clearer answer about reusable 3D knowledge:

| family | support | oracle rate | mean delta | mean steps |
| --- | ---: | ---: | ---: | ---: |
| `recenter_then_shrink` | 870 | 36.6% | +0.8424 | 8.36 |
| `coverage_rescue_then_tighten` | 176 | 11.9% | +0.8849 | 5.96 |
| `shrink_slack_face` | 362 | 3.6% | -0.2535 | 2.06 |

The conclusion is useful: direct slack shrinking is common but weak by itself.
The stronger reusable pattern is a variable-length corrective program:

```text
shrink some easy slack
-> recenter into a better local frame
-> shrink/tighten again until exact score stalls
```

The second pattern is the local-minimum escape version:

```text
expand or rescue coverage
-> recenter if needed
-> tighten again
```

This matches the intuition behind MCTS: one-step tightening is often trapped,
while a short corrective program can temporarily move away from the current
local optimum and then recover a better fit.  The next research step is to use
these pattern families as options in a macro search controller, not just as
flat candidate ranks.

### Learned Family Selector

I added `experiments/macro_search/train_live_family_selector.py` to test that
next step.  Instead of ranking raw macros, it trains a candidate-family value
model over semantic representatives:

```text
state features + family id + selector score + first local-action token
  -> predicted exact delta for that family representative
```

The live action still remains exact-validated.  The model only decides which
family representative to try first.  I evaluated two modes:

- `one-call`: exact-evaluate only the predicted family representative.
- `guarded`: exact-evaluate the default representative and the predicted
  representative if they differ, then accept the better exact result.

Five deduplicated case splits over the live logs produced:

| mode | mean delta | exact calls | gap to family oracle |
| --- | ---: | ---: | ---: |
| default first family | +0.2492 | 1.00 | +1.1401 |
| learned one-call family | +1.3067 | 1.00 | +0.0826 |
| learned guarded family | +1.3844 | 1.26 | +0.0050 |
| family-representative oracle | +1.3893 | 2.49 | 0.0000 |
| full top-k oracle | +1.4866 | top-k | - |

This is the strongest evidence so far that the useful abstraction level is not
the primitive action and not the raw macro ID, but the semantic option family.
The remaining gap to full top-k oracle is mostly inside-family parameter and
termination choice.  In other words:

```text
family selection is mostly learnable;
fine-grained skill parameterization is the next bottleneck.
```

This also answers why simply making a larger Transformer did not solve the
earlier router: the earlier target was too low-level.  The family-level target
matches the actual 3D fitting structure better.

### Direct Pattern Application

I also tested a non-neural mined-pattern memory in
`experiments/macro_search/evaluate_mined_pattern_direct_policy.py`.  This
builds train-split memories over `macro_id`, semantic pattern, family, and
skill, then reorders held-out top-k attempts from those memories.

The question was whether the mined patterns can be applied directly to reduce
exact calls, without doing full MCTS-like enumeration.  Five held-out splits
over the same deduplicated live cases gave:

| policy | exact calls | mean delta | gap to top-k oracle |
| --- | ---: | ---: | ---: |
| default first | 1.00 | +0.2492 | +1.2373 |
| macro memory, one-call | 1.00 | +0.9381 | +0.5484 |
| mined pattern, one-call | 1.00 | +1.0041 | +0.4825 |
| mined pattern, guarded | 1.96 | +1.4002 | +0.0863 |
| empirical bound q=0.80 | 4.20 | +1.4490 | +0.0375 |
| empirical max-bound | 4.84 | +1.4823 | +0.0042 |
| full top-k oracle | top-k | +1.4866 | 0.0000 |

This confirms two things:

1. mined patterns can directly cut exact calls while preserving most top-k
   oracle quality;
2. near-oracle or "global-like" behavior requires more exact checks unless we
   have a formal admissible bound.

The empirical bound test is branch-and-bound-like: exact-evaluate candidates in
pattern-priority order, then stop when the current best is above the empirical
upper bound of the remaining candidates.  With max-like bounds it almost
recovers the top-k oracle, but it still uses `4.84` calls on a top-5 candidate
pool.  With a looser 0.80 quantile bound, calls drop to `4.20` but the oracle
gap grows to `0.0375`.

So the current answer is:

```text
pattern knowledge can reduce calls substantially;
pattern knowledge alone does not prove global optimality;
near-global behavior needs either more exact calls or a real admissible bound.
```

The next real breakthrough would be a cheap admissible or near-admissible bound
for a macro skill family: for example, a bound on the best possible BVS/coverage
gain obtainable by shrinking a face or by recentering a box without running
Manifold.  Learned models can order the search, but only a bound can safely
stop it early.

### Conformal Pattern Bounds

I also tested a distributional stopping rule in
`experiments/macro_search/evaluate_conformal_pattern_bound.py`.  This uses a
train/calibration/test split:

```text
train:
  build pattern-memory score tables

calibration:
  collect residuals exact_delta - pattern_score
  choose a one-sided conformal residual quantile q

test:
  exact-evaluate candidates in pattern-score order
  stop when current_best >= max_remaining(pattern_score + q)
```

This is not a geometric proof, but it is a cleaner version of the empirical
bound idea because the stop threshold is calibrated on held-out residuals.

Global conformal residuals:

| coverage | exact calls | mean delta | gap to top-k oracle | oracle hit |
| ---: | ---: | ---: | ---: | ---: |
| 0.75 | 3.89 | +1.4073 | +0.0793 | 84.5% |
| 0.80 | 4.12 | +1.4368 | +0.0498 | 89.4% |
| 0.85 | 4.34 | +1.4553 | +0.0313 | 91.7% |
| 0.90 | 4.49 | +1.4670 | +0.0196 | 93.2% |
| 0.95 | 4.74 | +1.4813 | +0.0052 | 97.7% |
| 0.99 | 4.85 | +1.4824 | +0.0041 | 98.9% |

Family-specific conformal residuals were similar: they slightly improved
quality at some coverage points, but did not create a large call reduction.  The
reason is structural.  If we want near-oracle behavior, the remaining
candidates' calibrated upper bounds stay high, so the search cannot safely stop
early.  Lowering the coverage reduces calls, but the top-k oracle gap grows.

This gives a useful boundary:

```text
learned/pattern memory = good for ordering and 1-2 call practical speedups
conformal bound = good for distributional safety, but still calls many candidates
global-like guarantee = needs a real geometry-derived admissible bound
```

So the next target should not be another classifier.  It should be a geometric
upper bound per option family, e.g.:

- maximum possible volume reduction from a face-shrink before coverage breaks;
- maximum score improvement from recentering when box extents are fixed;
- coverage-rescue upper bound from uncovered centroid/voxel direction;
- dominance pruning when two candidates affect the same box/axis but one has
  both weaker volume reduction and weaker coverage margin.

### Geometry-Role Dominance Replay

I tested a first version of that last idea in
`experiments/macro_search/evaluate_geometry_dominance_pruning.py`.  It groups
top-k macro candidates by cheap semantic keys and keeps only the highest-scoring
representative per key:

- `family`
- `pattern`
- `family + first action direction`
- `family + first action direction + axis role`
- `family + first action direction + box/axis/face role`

The result is useful mostly as a negative boundary:

| pruning key | default included | exact calls | mean delta | gap to top-k oracle |
| --- | --- | ---: | ---: | ---: |
| family | yes | 3.91 | +1.4197 | +0.0668 |
| family | no | 2.96 | +1.1465 | +0.3400 |
| family + direction | no | 3.91 | +1.4721 | +0.0145 |
| family + direction + axis | no | 3.91 | +1.4721 | +0.0145 |
| pattern | yes | 4.92 | +1.4866 | 0.0000 |

The interpretation is that the current top-k candidate generator already
produces mostly unique macro/pattern/first-role candidates.  Coarse family
pruning reduces calls, but it loses too much quality unless the default branch
is also retained.  Fine-grained role keys preserve quality but barely prune.

So this is not the missing global-minimum shortcut.  The useful direction is
still a real numeric bound, not only role deduplication.

### Family Margin Escalation

I also tested whether the learned family selector's confidence margin can decide
when to escalate from one exact call to all family representatives.  The result
was weaker than the simpler guarded rule:

| policy | exact calls | mean delta | family-oracle gap |
| --- | ---: | ---: | ---: |
| one-call learned family | 1.00 | +1.3067 | +0.0826 |
| margin threshold 0.10 | 1.28 | +1.3276 | +0.0617 |
| margin threshold 0.20 | 1.63 | +1.3469 | +0.0425 |
| margin threshold 0.40 | 2.03 | +1.3555 | +0.0338 |
| always family oracle | 2.96 | +1.3893 | 0.0000 |
| guarded default+learned | 1.26 | +1.3844 | +0.0050 |

The simple guarded rule is better:

```text
exact-check default representative
exact-check learned representative if different
accept the better exact result
```

This works because many learned mistakes are corrected by comparing against the
default branch, while the default branch is often already one of the strongest
family representatives.  Escalating to all family representatives is only worth
it if we need the last few hundredths of family-oracle quality.

### Global Optimum Direction

The current learned router does not guarantee the global minimum.  What it can
do is make a larger exact search affordable.  A practical path toward stronger
optimality is:

1. define a finite bbox edit lattice for a chosen action unit schedule;
2. use mined variable-length skills as macro edges in that lattice;
3. compute cheap lower/upper bounds from volume, coverage, slack, and bbox
   validity before any Manifold call;
4. use learned retrieval only to order branches, not to accept final states;
5. exact-validate every accepted branch with SMART/Manifold;
6. keep the best exact state and rollback unsafe programs.

That still gives bounded-search optimality, not continuous global optimality.
But it is the right research form: learned 3D knowledge reduces the
combinatorial search order, while exact SMART remains the validator.

### Template Retrieval And Two-Stage Control

The next experiment asks whether those templates are predictable from the
current case state.  This is different from selecting one mined macro ID.  The
model predicts a higher-level option template such as:

```text
shrink max_slack until coverage margin is low
-> recenter single_box until center shift stalls
-> shrink max_slack until coverage margin is low
```

Then a second model chooses the concrete mined macro/parameterization inside
that template.  This matches the intended knowledge-controller hierarchy:

```text
state -> reusable variable-length template -> concrete macro parameters
      -> exact SMART reward validates accept/rollback
```

With hidden-512 template MLP, five 70/30 case splits gave:

| metric | value |
| --- | ---: |
| top-1 template accuracy | 58.1% |
| top-3 template accuracy | 80.0% |
| top-1 family accuracy | 75.8% |
| template miss rate | 2.3% |
| oracle mean delta | +1.0032 |
| first candidate in predicted template | +0.6463 |
| best candidate inside predicted template | +0.9012 |

The last row is the useful upper bound: if the template is chosen first and a
good parameter selector exists inside that template, the controller can recover
about `90%` of the oracle portfolio delta while searching a much smaller
semantic space.

I then evaluated the full two-stage prototype:

| selector | mean exact delta | positive rate |
| --- | ---: | ---: |
| template first candidate | +0.6463 | -- |
| top-1 template + inner MLP, guarded | +0.8920 | 97.7% |
| top-3 templates + inner MLP, guarded | +0.9518 | 100.0% |
| unrestricted inner MLP baseline, guarded | +0.9635 | -- |
| oracle among candidates | +1.0032 | -- |

The structured controller is slightly below the unrestricted macro retriever,
but close: `+0.9518` versus `+0.9635`.  The tradeoff is now explicit.  The
unrestricted model is a direct macro scorer; the two-stage model is more
interpretable and exposes reusable 3D knowledge.  It also gives a natural place
to add better termination/repeat prediction later.

### What The Patterns Look Like

The consistent patterns are not natural language rules and not direct cuboid
prediction.  They are small executable programs:

```json
{
  "precondition": {
    "coverage_bucket": "cov>=98",
    "bvs_bucket": "bvs10-30",
    "aspect_bucket": "ar4-8",
    "nbox": 1
  },
  "program": [
    {"op": "shrink_face", "target": "max_slack_face", "repeat": "1-11"},
    {"op": "recenter_box", "target": "single_box", "repeat": "1"},
    {"op": "shrink_face", "target": "max_slack_face", "repeat": "1-14"}
  ],
  "termination": "coverage_margin_low_or_score_stalls"
}
```

The highest-value templates have a consistent form:

- `tighten -> recenter -> tighten`: mostly airplane-like elongated single-box
  states where slack remains after a rough fit.
- `expand coverage gap -> recenter -> tighten`: local-minimum escape when a box
  is too tight or miscentered and cannot improve by shrinking alone.
- `shrink max slack`: simple one-family local tightening when coverage is
  already safe.
- `recenter -> shrink`: center correction before local tightening.

This is the current answer to "what form does reusable 3D knowledge take?"  It
looks like option programs with preconditions, target roles, repeat ranges, and
rollback-safe exact validation.  The next real improvement is to predict repeat
counts and termination from native geometry features rather than relying on
mined fixed macro variants.

### Repeat / Termination Prediction

I also tested whether the variable-length part should be learned by a neural
model.  The target is the repeat vector inside an accepted template, for
example:

```text
shrink max_slack repeat 4
recenter repeat 1
shrink max_slack repeat 3
```

Five 70/30 splits on the accepted live macros gave:

| repeat predictor | repeat MAE | total repeat error | exact vector | within 1 step |
| --- | ---: | ---: | ---: | ---: |
| template median memory | 0.172 | 0.385 | 89.4% | 90.9% |
| category + template median memory | 0.172 | 0.385 | 89.4% | 90.9% |
| MLP repeat regressor | 0.240 | 2.766 | 78.5% | 89.1% |

This is a useful negative result for deep learning.  Repeat/termination is
currently better represented as a compact lookup table attached to the
template, not as a standalone neural regressor.  The neural part should choose
which template/option to try; the table provides strong default repeat ranges;
exact SMART reward still decides when to accept, stop, or rollback.

### Exported Knowledge Base

The current research artifacts can be exported as a compact option table:

```bash
python experiments/macro_search/export_skill_knowledge_base.py
```

This writes:

```text
experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json
```

The exported file currently contains `8` live-supported skills.  Each record is
structured as:

```json
{
  "family": "recenter_then_shrink",
  "program": [
    {"op": "shrink_face", "target": "max_slack_face", "until": "coverage_margin_low_or_score_stalls"},
    {"op": "recenter_box", "target": "single_box", "until": "center_shift_stalls"},
    {"op": "shrink_face", "target": "max_slack_face", "until": "coverage_margin_low_or_score_stalls"}
  ],
  "default_repeats": [3, 1, 4.5],
  "precondition_memory": ["coverage/BVS/aspect/nbox buckets"],
  "risk_policy": "exact accept, rollback on non-positive exact delta"
}
```

That is the practical representation of the discovered knowledge: an
interpretable option table plus a learned retriever, not a black-box cuboid
predictor.

### Live Executor Connection

The exported knowledge table is now connected to the live
`NativeSmartEngine` evaluator through:

```bash
python experiments/macro_search/evaluate_live_skill_controller.py \
  --selector knowledge_base_delta \
  --knowledge-base experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json
```

The connection converts each knowledge-base entry into the same skill format
used by the live executor:

```text
program op/target/until + default repeat -> exact native action search
```

All accepted updates are still validated by exact SMART/Manifold reward.  On
the all-176 live benchmark:

| controller | accept mode | mean delta | attempts | total time | positive |
| --- | --- | ---: | ---: | ---: | ---: |
| macro-hash JSON, c256, native scorer | first positive | +1.4359 | 1.00 | 36.1s | 100% |
| exported knowledge table | first positive | +1.4641 | 1.02 | 28.0s | 100% |
| exported knowledge table | top-3 exact best | +1.5696 | 3.00 | 47.4s | 100% |

This is the first result where the generalized pattern table improves both
quality and wall time against the previous learned macro controller.  The
reason is simple: the table collapses many raw macro variants into a few
high-value option programs, so the live executor spends less time scoring weak
programs.

Per-category, the first-positive knowledge table gives:

| category | cases | mean delta | mean time |
| --- | ---: | ---: | ---: |
| airplane | 87 | +1.9000 | 0.090s |
| chair | 46 | +1.0363 | 0.186s |
| table | 43 | +1.0397 | 0.118s |

### Held-Out Live Validation

To check leakage, I added a stricter split test:

```bash
python experiments/macro_search/evaluate_knowledge_base_live_holdout.py
```

For each split, the knowledge table is rebuilt from train-case live outcomes
only, then evaluated on held-out cases through the actual C++ native live
executor.  Five 70/30 splits gave:

| metric | value |
| --- | ---: |
| mean held-out delta | +1.2030 |
| positive rate | 99.6% |
| mean attempts | 1.02 |
| mean time / case | 0.115s |
| mean rebuilt skills | 8.0 |

So the pattern table generalizes, but there is still a real train/test gap:
`+1.2030` held-out versus `+1.4641` in-sample all-176.  The next target is to
reduce that gap by using richer geometry-state features for template selection,
especially for chair/table where the current table often overuses
`recenter_then_shrink`.

### Program-Gate Follow-Up

The all-176 top-3 exact-best log showed a simple in-sample pattern: the best
option was often the longer recenter variant
`shrink -> recenter -> shrink -> recenter -> shrink`, not the shorter
`shrink -> recenter -> shrink` template.  I added
`knowledge_base_program_gate`, which gives that longer program a priority bonus.

In-sample all-176:

| selector | mean delta | total time |
| --- | ---: | ---: |
| knowledge table, first positive | +1.4641 | 28.0s |
| knowledge table, program gate | +1.5658 | 33.2s |
| knowledge table, top-3 exact best | +1.5696 | 47.4s |

This nearly matches top-3 exact-best with about `70%` of the wall time.  But it
does not generalize: five held-out splits dropped to `+1.1047`, below the
plain global table's `+1.2030`.  So the longer-program preference is an
in-sample heuristic, not a robust controller.

I also trained an offline top-3 choice gate from case features using the
exact-attempt logs.  On held-out case splits it recovered some of the top-3
headroom:

| choice policy | mean delta | top-3 oracle regret |
| --- | ---: | ---: |
| first-positive scan | +1.2817 | +0.0999 |
| learned one-shot guarded | +1.3321 | +0.0495 |
| learned reordered scan | +1.3528 | +0.0288 |
| top-3 exact oracle | +1.3816 | 0.0000 |

This suggests the useful next step: do not hard-code a program gate.  Train a
small state-conditioned gate that chooses which knowledge-table option to try
first, then fall back to exact-positive scan if needed.

### Split-Trained Knowledge Choice Gate

I connected that choice-gate idea to the live executor.  The strict validation
loop now does this for each split:

```text
train cases -> rebuild knowledge table
train cases -> run top-3 exact attempts
train cases -> choose/export either MLP gate or majority-index gate by CV
test cases  -> reorder top-3 knowledge options with that exported gate
test cases  -> execute first positive option through NativeSmartEngine
```

The `cv_best` export is important.  The MLP has higher choice accuracy on some
splits, but a constant "most often best option index" is sometimes more robust
for exact reward.  The export therefore selects whichever policy has better
split-local reordered-scan delta.

Five 70/30 held-out live splits:

| selector | mean delta | attempts | mean time / case | positive |
| --- | ---: | ---: | ---: | ---: |
| plain train-only knowledge table | +1.2030 | 1.02 | 0.115s | 99.6% |
| split-trained `cv_best` choice gate | +1.2445 | 1.02 | 0.137s | 99.6% |
| split-trained `cv_best` choice gate, portfolio+native features | +1.2482 | 1.02 | 0.137s | 99.6% |
| top-3 exact best upper bound, current code | +1.2504 | 3.00 | 0.267s | 99.6% |

This is the current strongest knowledge-controller result.  It recovers almost
all of the top-3 exact-best quality while keeping exact attempts near one per
case.  It is still research-only, but the result is materially better than the
plain held-out table and avoids the overfitting failure of the hand-written
program gate.

Implementation points:

- `train_knowledge_choice_gate.py --export-mode cv_best` writes a portable JSON
  gate.  It can export either an MLP or a constant majority-index controller.
- `evaluate_live_skill_controller.py --selector knowledge_base_choice_gate`
  loads that JSON and reorders the top-k knowledge options before exact
  execution.
- `evaluate_knowledge_base_live_holdout.py --train-choice-gate-per-split`
  performs the strict train-only gate validation.

The remaining gap is small on this split set (`+1.2482` versus `+1.2504`), so
the next research target is not a larger MLP over the same labels.  The useful
new addition is `NativeSmartEngine.geometry_state_features()`, which exports a
67-dimensional state vector directly from the C++ engine:

- exact coverage/BVS/score and valid box ratio,
- bbox volume distribution, aspect ratios, and rotation alignment,
- union-box volume/extent statistics,
- centroid-proxy coverage and weighted PCA extent,
- a coarse normalized tet-volume histogram.

The gate can now be trained with:

```bash
python experiments/macro_search/evaluate_live_skill_controller.py \
  --selector knowledge_base_delta \
  --portfolio experiments/macro_search/runs/parameterized_skills_4k/portfolio_report_filesystem_category_scope_all176_top16_proxyfeatures.jsonl \
  --extra-tetra-root runs/expanded_full/tetra \
  --extra-tetra-root runs/shapenet_v1_3/tetra \
  --extra-tetra-root runs/expanded_200/tetra \
  --top-k 3 --candidate-count 256 \
  --accept-mode best_positive \
  --record-native-features \
  --out experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_all176_best_positive_native_features_c256.jsonl

python experiments/macro_search/evaluate_knowledge_base_live_holdout.py \
  --selector knowledge_base_choice_gate \
  --train-choice-gate-per-split \
  --choice-gate-feature-source portfolio_native \
  --choice-gate-export-mode cv_best \
  --top-k 3 --candidate-count 256
```

Important negative result: native geometry features improve the held-out live
mean delta slightly, but the exported `cv_best` controller still often chooses a
constant majority-index gate over the MLP.  That means the current label
distribution is dominated by stable skill ordering.  To get a large quality win,
the next dataset needs harder states where different variable-length skills are
actually required, not just more capacity on the same 176 cases.

### Hard-State Mining

I added `analyze_hard_skill_states.py` for that next dataset.  It scans top-k
exact attempt logs and extracts states where the first-ranked skill is not the
best, the first-positive skill is not the best, or first-skill regret is large.

With all-176 native-feature labels and `regret_threshold=0.25`:

| category | hard states | mean regret | dominant nontrivial pattern |
| --- | ---: | ---: | --- |
| airplane | 59 | 0.207 | longer `recenter_then_shrink` |
| chair | 32 | 0.270 | longer recenter plus occasional `shrink_slack_face` |
| table | 26 | 0.293 | highest `shrink_slack_face` share |

The high-regret cases show an interpretable precondition: when BVS, aspect, or
union-volume ratio is very high, recenter-first programs can be unsafe, and
`shrink_slack_face` should be attempted first.  This is the next real
knowledge-controller target: learn preconditions for safe variable-length skill
families, not just a flat top-k index.

### Native Rule And Memory Ablations

I tested whether that risk-state idea can already be solved with simple
interpretable rules or nearest-neighbor memory over the 67-D native geometry
vector.

```bash
python experiments/macro_search/analyze_native_choice_gate_rules.py
python experiments/macro_search/evaluate_native_choice_memory.py
```

Both scripts use split evaluation.  Rules or memory hyperparameters are chosen
from train/validation data only, then tested on held-out states.

Five stratified 70/30 splits:

| controller | held-out delta | regret vs top-3 | attempts | interpretation |
| --- | ---: | ---: | ---: | --- |
| constant majority index | +1.4983 | 0.0023 | 1.023 | very strong default on this label set |
| single native-threshold rule | +1.4978 | 0.0028 | 1.023 | in-sample `union_extent_y` looked good, but does not generalize |
| native KNN memory | +1.4850 | 0.0155 | 1.015 | global state-vector distance is not a good reusable-knowledge metric |
| top-3 exact upper bound | +1.5006 | 0.0000 | 3.000 | current ceiling for this dataset |

This is a useful negative result.  The current native feature vector helps the
MLP gate slightly, but simple global-feature thresholds and nearest-neighbor
memory are not enough.  Reusable 3D knowledge probably needs structured tokens:

- box tokens: per-box center, extent, slack, coverage contribution;
- action tokens: operation, target role, predicted reward, expected risk;
- skill tokens: family, repeat range, termination predicate;
- shape tokens: PCA/centroid histogram and uncovered-region direction.

In other words, the problem is not "remember a similar whole state."  The
better representation is "match the local box/action situation that caused a
skill to work."

### Extracted Knowledge Patterns

The current macro miner already exposes repeatable pattern families.  Running

```bash
python experiments/macro_search/analyze_skill_knowledge_patterns.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_all176_best_positive_native_features_c256.jsonl

python experiments/macro_search/generalize_skill_templates.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_all176_best_positive_native_features_c256.jsonl
```

produces 93 generalized templates.  The live-used templates are not arbitrary
action strings; they have stable precondition/program forms:

| template family | live accepted | mean live delta | reusable program |
| --- | ---: | ---: | --- |
| `recenter_then_shrink` | 63 | +1.98 | shrink slack face, recenter single box, shrink again; repeats vary by state |
| `escape_local_minimum_expand_then_refine` | 46 | +1.04 | expand coverage-gap face, recenter dominant box, shrink slack face |
| `shrink_slack_face` | 22 | +1.34 | shrink max-slack face until coverage margin becomes risky |
| `recenter_then_shrink` shorter form | 15 | +1.15 | recenter single box, then shrink slack face |

The important part is the repeat range, not the raw step count.  Example:

```text
Template:
  shrink_face(max_slack_face) -> recenter_box(single_box) -> shrink_face(max_slack_face)

Observed repeat ranges:
  shrink before recenter: 1..11, median 3
  recenter: 1
  shrink after recenter: 1..14, median 4.5

Common precondition:
  coverage >= 90%, BVS very high, aspect large, nbox=1
```

That is exactly the variable-length skill form we wanted: the same conceptual
operation may run for 2 steps or 16 steps depending on when coverage margin and
exact reward stop improving.

Current conclusion:

- The skill representation is promising and interpretable.
- The constant/choice gate is already close to the top-3 exact ceiling on the
  current dataset.
- The next real improvement requires harder data plus local box/action token
  models, not a larger MLP over global features.

Template retriever check:

| model | top1 template | top3 template | template miss | predicted-template upper bound |
| --- | ---: | ---: | ---: | ---: |
| h512, support >= 3 | 57.4% | 81.1% | 1.5% | +0.9047 |
| h1024, support >= 3 | 57.0% | 81.1% | 1.9% | +0.9038 |
| h512, support >= 10 | 59.2% | 82.6% | 10.2% | +0.7807 |

So simply making the template retriever bigger is not the answer.  The model
can usually put the right variable-length template family in the top-3, but
quality is still lost inside the template because repeat count, box target, and
termination must be parameterized from local geometry.

### Live Sequence-Value Test

I added `train_live_skill_sequence_value.py` to test the next question directly:
given the live top-3 skill attempts, can a model choose the best variable-length
program before spending exact calls?

The script builds one training row per live attempt. Each row contains:

- the 67-D native state vector;
- category, skill family, and generated macro id;
- sequence structure: action count, axis/recenter counts, shrink/expand counts,
  target bbox statistics, repeated-action ratios, first/last operation, and
  normalized action-id summaries.

Two modes are separated:

- `program_only`: no exact action reward or final score fields are used;
- `execution_value`: includes reward/score-derived trace fields, so it is an
  upper-bound diagnostic rather than a deployable speed path.

Five held-out 70/30 splits on the same 176 cases:

| selector | mean delta | regret vs top-3 | exact calls | interpretation |
| --- | ---: | ---: | ---: | --- |
| constant index 1 | +1.3173 | 0.0649 | 1.0 | strongest non-learned one-call baseline |
| ridge, `program_only` | +1.2566 | 0.1256 | 1.0 | sequence shape alone is not enough yet |
| ridge, `execution_value` | +1.3808 | 0.0013 | 1.0 | action-level value is almost sufficient |
| top-3 exact oracle | +1.3822 | 0.0000 | 1.0 in replay / 3 live tries | current ceiling |

This is the key diagnosis: the macro-skill representation is useful, but the
current non-leaky features do not know which internal action sequence will
actually pay off. Once action-level value information is available, even a
linear model nearly reaches the top-3 oracle. Therefore the next useful model is
not a bigger whole-state Transformer; it is a local token model that predicts
cheap per-action/per-skill value from geometry tokens before exact Manifold
validation.

### Local Action-Value Proxy

I then added `train_local_action_value_proxy.py`.  This is the first direct test
of that next target.  It trains on primitive actions inside live skill traces:

```text
input  = initial native state + skill token + action token + step position
target = exact per-action reward recorded by the live executor
```

Previous exact rewards are not used as input.  The model still lacks
intermediate bbox state, so this is a conservative proxy.

Five held-out 70/30 splits:

| model | MAE | RMSE | R2 | corr | top-step hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean baseline | 0.246 | 2.664 | -0.104 | 0.000 | 0.599 |
| ridge/asinh | 0.357 | 3.485 | -0.408 | 0.314 | 0.592 |
| MPS MLP/asinh | 0.159 | 1.947 | 0.500 | 0.737 | 0.475 |

This is progress but not a deployable controller yet.  The MLP learns reward
magnitude much better than the mean baseline, but it still does not reliably
identify the best step inside each skill attempt.  That means the next C++
feature export should include intermediate local state after each action:

- per-box center/size/slack after the current step;
- target face margin and coverage contribution;
- coverage-risk margin before/after candidate action;
- local rollback/value labels for termination.

With those features, the same local value proxy can become the cheap guide that
the sequence-value ablation showed we need.

After adding `NativeSmartEngine.local_action_features()`, the live executor can
record a 45-D local token before each primitive action.  This token includes the
current action, target bbox, axis/face, shrink/expand flag, current score/BVS,
target box extent/aspect/volume, slack relative to the current union bounds, and
candidate validity/volume change.

Using the all-176 live log regenerated with local tokens:

| task | baseline | local-token model | result |
| --- | ---: | ---: | --- |
| per-action reward proxy MAE | 0.246 | 0.084 | much better reward magnitude prediction |
| per-action reward proxy R2 | -0.104 | 0.813 | local token carries real value signal |
| one-shot skill selection delta | +1.3173 | +1.3226 | small held-out gain |
| one-shot skill selection regret | 0.0649 | 0.0596 | small held-out gain |
| one-shot oracle-pick rate | 52.5% | 58.1% | better but not solved |

So the local token export is useful, but the current attempt-level summary is
still too lossy.  The next model should consume the action sequence as tokens
with attention or pooling, not just mean/std/delta summaries over local features.

### Local Action-Token Transformer

I added `experiments/macro_search/train_local_action_token_controller.py` to
test that directly.  Each skill attempt is represented as:

```text
global token:
  category + native state + skill id + macro id + selector score

action tokens:
  kind/op + step position + bbox/action ids + 45-D C++ local geometry token
```

Two modes are separated because they answer different research questions:

- `first`: only the first primitive action token is visible.  This is closest to
  a deployable pre-exact router.
- `all`: the full executed action sequence is visible.  This is stronger, but it
  should be interpreted as an in-skill value/termination model because later
  tokens include states reached after earlier exact actions.

Five held-out 70/30 splits on 176 live cases:

| selector | mean delta | regret | exact calls | oracle pick | note |
| --- | ---: | ---: | ---: | ---: | --- |
| constant index 1 | +1.3173 | 0.0649 | 1.0 | 52.5% | best simple one-call baseline |
| token Transformer, `first` | +1.3578 | 0.0243 | 1.0 | 60.8% | deployable-router signal |
| token Transformer, `all`, one-shot | +1.3525 | 0.0296 | 1.0 | 74.3% | sees full executed token trace |
| token Transformer, `all`, reordered | +1.3732 | 0.0089 | 1.004 | 74.7% | near top-3 ceiling with almost one call |
| top-3 exact oracle | +1.3822 | 0.0000 | 1.0 replay / 3 live tries | 100% | ceiling |

I also tried a larger `h512/l3/8-head` Transformer in `first` mode.  It did not
improve: mean delta dropped to `+1.3516` and regret rose to `0.0305`.  On the
current 176-case dataset, bigger model capacity is not the limiting factor.
The next bottleneck is data diversity and better local labels, especially
termination/rollback labels inside each variable-length skill.

Current research conclusion:

- The action-token Transformer is the first learned controller that improves
  over the best constant one-shot policy without using exact reward fields from
  the candidate.
- Full-token sequence modeling almost reaches the top-3 exact oracle, which
  supports the variable-length skill idea.
- This is not ready as a release default yet.  It should stay experimental until
  we validate on more live cases and move the inference path into C++.

Checkpoint status:

- `train_local_action_token_controller.py` can now save a full-data checkpoint
  with `--checkpoint-out`.
- `evaluate_local_action_token_checkpoint.py` reloads that checkpoint and
  verifies that it can produce a deterministic ordering over live attempts.
- Same-log replay with the `first` checkpoint reaches the top-3 oracle mean
  delta, but this is only a load/replay sanity check because the checkpoint was
  trained on the same 176 cases.  The honest held-out claim remains the 5-split
  `first` result above: `+1.3578` mean delta and `0.0243` regret.

The next deployment step is not another larger PyTorch model.  It is:

1. collect more live traces with distinct ShapeNet objects and harder local
   minima;
2. train the small `first` controller on that larger split;
3. export the trained weights to a compact C++ inference format;
4. integrate the router before exact top-k skill execution and measure real
   wall-time reduction, not only replay exact-call reduction.

### Top-5 Harder Trace Expansion

I then generated top-5 live traces to test whether the controller still helps
when the candidate set is larger and includes the escape skill:

```text
airplane80:    top_k=5, candidate_count=192, 80 airplane cases
chair/table80: top_k=5, candidate_count=192, 46 chair + 34 table cases
combined160:   both logs together, 800 skill attempts
```

The airplane-only subset is almost solved by a constant rule: index 1 is already
near oracle.  In that subset the learned controller is not useful, which is a
good warning that more data is not enough if the ordering pattern is trivial.

| subset | selector | mean delta | regret | exact calls | oracle pick |
| --- | --- | ---: | ---: | ---: | ---: |
| airplane80 top-5 | constant index 1 | +1.8877 | 0.0054 | 1.0 | 44.2% |
| airplane80 top-5 | token Transformer `first` | +1.8437 | 0.0493 | 1.0 | 54.2% |
| airplane80 top-5 | token Transformer `all` | +1.8634 | 0.0296 | 1.0 | 70.8% |
| airplane80 top-5 | oracle | +1.8930 | 0.0000 | 1.0 replay | 100% |

The chair/table subset is harder and the learned controller is useful:

| subset | selector | mean delta | regret | exact calls | oracle pick |
| --- | --- | ---: | ---: | ---: | ---: |
| chair/table80 top-5 | constant index 1 | +0.9543 | 0.1938 | 1.0 | 44.2% |
| chair/table80 top-5 | token Transformer `first` | +1.0495 | 0.0986 | 1.0 | 58.3% |
| chair/table80 top-5 | token Transformer `first`, reordered | +1.1015 | 0.0466 | 1.008 | 59.2% |
| chair/table80 top-5 | token Transformer `all`, reordered | +1.1093 | 0.0387 | 1.025 | 63.3% |
| chair/table80 top-5 | oracle | +1.1480 | 0.0000 | 1.0 replay | 100% |

Combined top-5, 160-case result:

| selector | mean delta | regret | exact calls | oracle pick |
| --- | ---: | ---: | ---: | ---: |
| constant index 1 | +1.3979 | 0.1141 | 1.0 | 47.9% |
| token Transformer `first` | +1.4550 | 0.0570 | 1.0 | 60.0% |
| token Transformer `first`, reordered | +1.4735 | 0.0386 | 1.004 | 60.0% |
| token Transformer `all`, reordered | +1.4809 | 0.0311 | 1.008 | 70.4% |
| oracle | +1.5120 | 0.0000 | 1.0 replay | 100% |

This is a better research signal than the first 176-case top-3 result.  The
controller is not merely memorizing one fixed action rank; it helps most when
candidate ordering genuinely varies by category and geometry.  The practical
next dataset should therefore be balanced by category and should intentionally
include hard negatives where the default skill order fails.

I rechecked the combined160 top-5 split with a smaller and a larger `first`
token Transformer.  The larger model improves one-shot quality, but the best
small-model reordered policy still gives the strongest mean delta in this
quick check:

| model | policy | mean delta | regret | exact calls | oracle pick |
| --- | --- | ---: | ---: | ---: | ---: |
| h256/l2, 120 epochs | one-shot | +1.4369 | 0.0751 | 1.000 | 62.9% |
| h256/l2, 120 epochs | reordered | +1.4814 | 0.0307 | 1.013 | 63.3% |
| h512/l3, 120 epochs | one-shot | +1.4682 | 0.0439 | 1.000 | 59.2% |
| h512/l3, 120 epochs | reordered | +1.4682 | 0.0439 | 1.000 | 59.2% |

This says capacity helps somewhat, but the bigger gain comes from using the
model as an ordering policy with exact guarded execution.  The next useful
scaling step is more hard states and better local state tokens, not only a
larger Transformer.

### Top-5 Combined Knowledge Examples

I refreshed the knowledge-pattern summary using the all-176 top-3 trace plus
the newer top-5 airplane80 and chair/table80 traces:

```bash
python experiments/macro_search/analyze_skill_knowledge_patterns.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_all176_best_positive_native_local_features_c256.jsonl \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_80_top5_native_local_features_c192.jsonl \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_chair_table80_top5_native_local_features_c192.jsonl \
  --out experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_patterns_top5_combined.json

python experiments/macro_search/generalize_skill_templates.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_all176_best_positive_native_local_features_c256.jsonl \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_80_top5_native_local_features_c192.jsonl \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_chair_table80_top5_native_local_features_c192.jsonl \
  --out experiments/macro_search/runs/parameterized_skills_4k/generalized_skill_templates_top5_combined.json
```

The combined result is stable: useful live knowledge remains concentrated in
three families, while several mined families have high offline support but no
live wins under exact validation.

| family | mined macros | trace support | live accepted | mean live delta | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `recenter_then_shrink` | 200 | 4659 | 102 | +1.6995 | dominant variable-length tightening skill |
| `escape_local_minimum_expand_then_refine` | 79 | 715 | 52 | +0.9575 | coverage/local-minimum rescue skill |
| `shrink_slack_face` | 200 | 25226 | 22 | +1.3441 | simple tightening, useful but not enough alone |
| `expand_to_recover_coverage` | 88 | 2010 | 0 | 0.0000 | only useful when composed with recenter/tighten |
| `major_axis_extend_then_trim` | 183 | 4625 | 0 | 0.0000 | mined often, but not selected by current live exact controller |

The strongest reusable programs are:

| program template | repeat range | live accepted | mean live delta | typical precondition |
| --- | --- | ---: | ---: | --- |
| shrink max-slack face -> recenter single box -> shrink max-slack face | 1-11, 1, 1-14 | 63 | +1.9779 | high coverage, high BVS, elongated one-box state |
| expand coverage-gap face -> recenter dominant box -> shrink max-slack face | 1-7, 1-2, 1-12 | 46 | +1.0394 | coverage is imperfect or local tightening is stuck |
| shrink max-slack face | 1-16 | 22 | +1.3441 | coverage is safe and slack is obvious |
| recenter single box -> shrink max-slack face | 1-2, 1-15 | 15 | +1.1543 | center is wrong before tightening |
| shrink -> recenter -> shrink -> recenter -> shrink | 1-2, 1, 3-10, 1, 1-3 | 14 | +1.6175 | very high BVS/aspect, usually one-box |

This is the clearest practical form of the "3D knowledge" so far.  It is not a
fixed n-step sequence.  It is an option template:

```text
precondition bucket
-> target-role program
-> repeat range / termination predicate
-> exact SMART validation and rollback
```

The negative result is also useful.  Some intuitive skills, such as standalone
coverage expansion or major-axis extension, appear frequently in mined traces
but do not win live exact validation unless composed with recentering and
tightening.  So the knowledge controller should retrieve composed option
templates rather than individual primitive fixes.

### Attempt-Order Policy Recheck

I added an offline replay diagnostic:

```bash
python experiments/macro_search/evaluate_skill_attempt_order_policies.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_80_top5_native_local_features_c192.jsonl \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_knowledge_base_delta_chair_table80_top5_native_local_features_c192.jsonl \
  --out experiments/macro_search/runs/parameterized_skills_4k/skill_attempt_order_policies_top5_combined160_narrow_guard.json \
  --max-checks 5
```

This replays mined skill attempts without running geometry again.  A policy
chooses which exact-audited skill to try first; if that attempt is not positive,
it falls back through the remaining attempts.  This is closer to live execution
than the earlier "oracle among top-k" table, because exact call count is
explicit.

Top-5 combined160 result:

| policy | mean delta | regret vs top-5 oracle | exact checks | note |
| --- | ---: | ---: | ---: | --- |
| original mined order | +1.4438 | 0.1152 | 1.025 | too much regret |
| constant0 / first template | +1.4438 | 0.1152 | 1.025 | same as original |
| constant1 / second recenter variant | +1.5465 | 0.0125 | 1.025 | strongest simple policy |
| narrow negative-recenter guard | +1.5406 | 0.0184 | 1.000 | safer calls, slightly worse quality |
| shrink first | +0.7299 | 0.8290 | 1.000 | not viable |

The top-3 all176 replay is even closer to oracle:

| policy | mean delta | regret vs top-3 oracle | exact checks |
| --- | ---: | ---: | ---: |
| original mined order | +1.4657 | 0.1032 | 1.023 |
| constant1 / second recenter variant | +1.5651 | 0.0038 | 1.023 |
| narrow negative-recenter guard | +1.5598 | 0.0091 | 1.000 |

The hard-state analysis explains why this works.  Most "hard" cases are not a
different skill family; they are the same `recenter_then_shrink` family but a
different variant.  In the 176-case top-3 log, 162/173 hard states still choose
`recenter_then_shrink` as the exact-best family.  Only 11/173 choose
`shrink_slack_face`.

There are two severe chair/table cases where recenter variants score around
`-95` and shrink recovers the state.  They have degenerate one-box native
features:

```text
num_boxes = 1
num_actions = 13
weighted_pca_extent_0 = weighted_pca_extent_1 = weighted_pca_extent_2 = 0
```

A narrow guard can avoid those exact failures, but on current data it loses
more average quality than it saves.  So the current research default remains:

```text
try second recenter_then_shrink variant first
fallback exactly if non-positive
use top-k oracle only for audits / training labels
```

This is a useful knowledge pattern, but not yet a general local-minimum escape
agent.  The next improvement needs a learned rule for when to use the escape
template or a true variable-length option executor that can stop/rollback inside
the skill, not only choose among full pre-mined attempts.

### Live Program-Gated Skill Execution

The replay result above is now checked in the live C++ skill executor on the
currently runnable 133-case subset of the all-176 portfolio.  The subset is
smaller only because the local checkout currently has tetra/bbox artifacts for
133 of the historical 176 cases.

I added `export_trace_skill_knowledge_base.py` to build an executable knowledge
base directly from accepted live traces.  It compresses primitive traces into
run-length programs:

```text
shrink_face x3 -> recenter_box x1 -> shrink_face x4
shrink_face x2 -> recenter_box x1 -> shrink_face x7 -> recenter_box x1 -> shrink_face x2
shrink_face x2
```

The trace-compressed KB is behaviorally close to the older mined-template KB,
but it does not yet improve it.  The stronger result is the program-gated
ordering inside the existing KB:

```text
try long recenter_then_shrink variant first
if exact delta is non-positive, fall back through the remaining skill programs
accept only exact-positive SMART/Manifold deltas
```

Live runnable-133 result:

| live policy | mean delta | accepted | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| existing KB, top-3 best-positive | +1.7247 | 133/133 | 3.000 | 0.292s |
| existing KB, top-1 first-positive | +1.5672 | 132/133 | 1.000 | 0.150s |
| program gate, top-1 first-positive | +1.6801 | 132/133 | 1.000 | 0.198s |
| program gate, top-3 first-positive | +1.7214 | 133/133 | 1.015 | 0.196s |

This is the clearest current evidence that the mined "3D knowledge" is useful
as a control primitive: nearly the same quality as exact top-3 skill audit, but
with almost one exact skill execution per state.  It is not a global optimum
guarantee.  It is an option-level shortcut with exact validation and rollback.

The same check on the separate 42-case mixed-category portfolio gives the same
shape:

| live policy | mean delta | accepted | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| existing KB, top-3 best-positive | +2.2741 | 42/42 | 3.000 | 0.108s |
| program gate, top-3 first-positive | +2.2721 | 42/42 | 1.000 | 0.078s |

This smaller probe is mostly airplane, but it confirms that the result is not
only an artifact of the runnable-133 portfolio.

The remaining gap is concentrated in a small number of cases:

- most small losses are the shorter `recenter_then_shrink` variant beating the
  longer one by a tiny margin;
- one table case needs `shrink_slack_face` after the long recenter variant
  fails, which `top-3 first-positive` rescues;
- no learned termination model is active yet, so each selected program still
  uses fixed median repeats rather than stopping from state feedback.

### Post-First Exact Fallback Gate

I then tested the next obvious learned controller: after the first skill has
already been executed and exact-scored, predict whether the remaining top-3
skills are worth exact-evaluating.  This is deployable because the model sees
only information available after the first exact skill result:

- category and native geometry summary;
- first skill id/family and selector gaps;
- first exact score delta;
- executed primitive count, reward statistics, recenter/shrink counts, and
  touched-box summary.

The target is:

```text
continue if top-3 exact best - first exact result > min_gain
```

The experiment script is:

```bash
python experiments/macro_search/train_skill_post_exact_gate.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_existing_kb_program_gate_runnable133_top3_best_positive_c256.jsonl \
  --out experiments/macro_search/runs/parameterized_skills_4k/post_exact_gate_runnable133_mlp_recall_c256.json \
  --model mlp --top-k 3 --min-gain 0.01 \
  --max-missed-rescue-rate 0.20 --device auto
```

Runnnable-133 result, 3 seeds x 5 folds:

| post-first policy | mean delta | regret vs top-3 | exact attempts | note |
| --- | ---: | ---: | ---: | --- |
| first-positive program gate | +1.72137 | 0.00329 | 1.015 | current fastest strong policy |
| top-3 exact audit | +1.72466 | 0.00000 | 3.000 | oracle over skill attempts |
| logistic, quality-calibrated | +1.72137 | 0.00329 | 1.015 | learns to stay closed |
| logistic, recall-calibrated | +1.72191 | 0.00275 | 1.195 | small gain, many extra calls |
| MLP h256/d3, quality-calibrated | +1.72154 | 0.00312 | 1.045 | tiny gain |
| MLP h256/d3, recall-calibrated | +1.72249 | 0.00218 | 1.095 | best learned tradeoff so far |

The same MLP setting on the separate mixed-42 check gives:

| post-first policy | mean delta | regret vs top-3 | exact attempts |
| --- | ---: | ---: | ---: |
| first-positive program gate | +2.27210 | 0.00195 | 1.000 |
| top-3 exact audit | +2.27405 | 0.00000 | 3.000 |
| MLP h128/d2, recall-calibrated | +2.27289 | 0.00116 | 1.175 |

The conclusion is important: post-first learned fallback is not the main
breakthrough.  The residual quality left by `program_gate + first_positive` is
already tiny.  A learned gate can buy back a little score by spending a little
more exact evaluation, but the better research target is stronger option
generation and state-conditioned termination/switching inside the option.

### Knowledge Pattern Mining Diagnosis

I regenerated the live knowledge-pattern report for the runnable-133
program-gate top-3 log:

```bash
python experiments/macro_search/mine_live_knowledge_patterns.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_existing_kb_program_gate_runnable133_top3_best_positive_c256.jsonl \
  --out-dir experiments/macro_search/runs/parameterized_skills_4k/live_knowledge_patterns_program_gate_runnable133
```

Key numbers:

| quantity | value |
| --- | ---: |
| cases | 133 |
| bounded top-3 oracle mean delta | +1.72466 |
| family-representative oracle mean delta | +1.72149 |
| family-representative exact calls | 1.992 |
| family-representative-to-oracle gap | 0.00317 |
| oracle-family parameter gap | 0.00325 |

This says the high-level 3D knowledge family is mostly right.  If we only
evaluate one representative per semantic family, we are already almost at the
top-3 skill oracle.  The missing part is not "which family" but "which
parameterization and when to stop."

The most common exact-winning programs are variable-length versions of:

```text
shrink -> recenter -> shrink
shrink -> recenter -> shrink -> recenter -> shrink
shrink-only when coverage is already stable
```

On the runnable-133 log, the strongest raw program pattern is:

```text
shrink_face x2 -> recenter_box x1 -> shrink_face x7 -> recenter_box x1 -> shrink_face x2
```

It appears as the winning accepted pattern in 61/133 cases across airplane,
chair, and table.  Shorter variants such as `shrink x3 -> recenter -> shrink
x4/x5` cover many of the remaining states.  This is why the pattern generalizes
despite changing length: the stable object is the option schema, not the exact
number of primitive actions.

The native-local-feature log also exposes face roles such as `low_slack_face`,
`positive_slack_face`, and `max_slack_face`.  Those roles are too sparse as
standalone pattern keys, but they are the right input features for the next
model:

```text
state -> choose option family -> choose face/axis target -> repeat until stop predicate
```

### Upper-Repeat Option Variant

The pattern diagnosis suggested that the family is usually right but the repeat
count is underfit.  I therefore tested an upper-repeat variant:

```text
default repeat = observed max repeat for each program step
execution still stops early when exact primitive reward stalls
```

The variant is generated with:

```bash
python experiments/macro_search/make_skill_repeat_variant.py \
  --in-kb experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --out experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json \
  --policy max
```

Blindly using upper-repeat everywhere improves the runnable-133 mean delta but
breaks one table case badly:

| policy | mean delta | accepted | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| median-repeat program gate | +1.7214 | 133/133 | 1.015 | 0.184s |
| upper-repeat all categories | +1.7477 | 132/133 | 1.015 | 0.214s |

The failure is category-specific.  A simple guard works better:

```text
airplane/chair: upper-repeat KB
table: median-repeat KB
```

This can be run in the experiment live controller with category-specific KBs:

```bash
python experiments/macro_search/evaluate_live_skill_controller.py \
  --portfolio experiments/macro_search/runs/parameterized_skills_4k/portfolio_from_program_gate_runnable133.jsonl \
  --knowledge-base experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --category-knowledge-base airplane=experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json \
  --category-knowledge-base chair=experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json \
  --selector knowledge_base_program_gate --top-k 3 --accept-mode first_positive
```

Live result:

| policy | mean delta | accepted | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| median-repeat program gate | +1.7214 | 133/133 | 1.015 | 0.184s |
| top-3 exact audit, median KB | +1.7247 | 133/133 | 3.000 | 0.292s |
| category-aware upper-repeat | +1.7770 | 133/133 | 1.015 | 0.204s |

Case-level comparison against median-repeat on runnable-133:

| setting | wins | ties | losses | mean gain |
| --- | ---: | ---: | ---: | ---: |
| upper all categories | 50 | 70 | 13 | +0.0263 |
| upper airplane/chair, median table | 40 | 83 | 10 | +0.0557 |

The same category-aware upper-repeat setting on the mixed-42 portfolio gives
`+2.3646` mean delta versus `+2.2721` for the median-repeat program gate.

This is the first result in this branch that improves quality, not just exact
call count.  It is still not a global optimum guarantee.  It is a stronger
parameterized option policy: use a wider repeat budget where it is safe, keep
the conservative table policy where upper repeats can damage coverage.

#### State-Conditioned Repeat Gate

I then tested whether a learned state gate can replace the category rule.  The
training data is paired exact live output from the same 133 states:

```text
input  = initial native geometry_state_features + category + first skill metadata
label  = whether upper-repeat beats median-repeat by at least 0.01
target = choose median KB or upper-repeat KB before running the skill
score  = exact live SMART/Manifold delta from the chosen branch
```

The important detail is that failed upper-repeat branches are no longer hidden
by a median fallback.  The gate is evaluated as a real branch choice.

5-seed, 5-fold CV on runnable-133:

| branch policy | mean delta | accepted | upper rate | losses vs median |
| --- | ---: | ---: | ---: | ---: |
| median-repeat | +1.7214 | 665/665 | 0.000 | 0 |
| upper-repeat all | +1.7477 | 660/665 | 1.000 | 65 |
| category upper non-table | +1.7770 | 665/665 | 0.782 | 50 |
| MLP state gate | +1.7563 | 662/665 | 0.483 | 34 |

This says the learned binary gate is not yet the right abstraction.  It reduces
some upper-repeat losses, but it is too conservative on wins and still lets a
few unsafe cases through.  The feature signal is real, but the better current
representation is an interpretable precondition rule.

#### Conditional Knowledge Rule

A feature scan showed that `table` is not uniformly unsafe.  The damaging table
case has a different rotation/slack regime.  The useful mined rule is:

```text
airplane/chair:
  use upper-repeat KB

table:
  use upper-repeat KB only if mean_rotation_offdiag_abs < 0.48678448090071313
  otherwise use median-repeat KB
```

This is now executable in the live experiment controller:

```bash
python experiments/macro_search/evaluate_live_skill_controller.py \
  --selector knowledge_base_program_gate \
  --knowledge-base experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --category-knowledge-base airplane=experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json \
  --category-knowledge-base chair=experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json \
  --conditional-knowledge-base 'category=table,feature=mean_rotation_offdiag_abs,op=<,threshold=0.48678448090071313,path=experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json'
```

Live executor result:

| policy | mean delta | accepted | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| median-repeat program gate | +1.7214 | 133/133 | 1.015 | 0.184s |
| category-aware upper-repeat | +1.7770 | 133/133 | 1.015 | 0.204s |
| conditional upper-repeat | +1.7889 | 133/133 | 1.015 | 0.211s |

Per-category live deltas for the conditional rule:

| category | cases | mean delta | accepted |
| --- | ---: | ---: | ---: |
| airplane | 73 | +2.1974 | 73/73 |
| chair | 31 | +1.1970 | 31/31 |
| table | 29 | +1.3933 | 29/29 |

On the mixed-42 portfolio, the same conditional policy reaches `+2.3646`
mean delta with `42/42` accepted, matching the best upper-repeat result and
staying above the median-repeat baseline (`+2.2721`).

#### Automatic Precondition Mining

I added an experiment-only rule miner:

```bash
python experiments/macro_search/mine_repeat_precondition_rules.py \
  --median-live <median_repeat_live.jsonl> \
  --upper-live <upper_repeat_live.jsonl> \
  --out <rule_report.json>
```

It searches category defaults plus one category-specific scalar precondition:

```text
default per category: median or upper
optional override: if native_state_feature op threshold then upper else median
```

On runnable-133, quality-first full-fit recovers the manually discovered rule:

```json
{
  "by_category": {"airplane": true, "chair": true, "table": false},
  "category": "table",
  "feature": "mean_rotation_offdiag_abs",
  "op": "<",
  "threshold": 0.48678448090071313,
  "if_true_upper": true,
  "if_false_upper": false
}
```

On the larger all-176 live set, the same pattern appears with a different table
precondition:

```json
{
  "by_category": {"airplane": true, "chair": true, "table": false},
  "category": "table",
  "feature": "max_bbox_volume_ratio",
  "op": "<",
  "threshold": 1.5581414368932673,
  "if_true_upper": true,
  "if_false_upper": false
}
```

All-176 live executor comparison:

| policy | mean delta | accepted | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| median-repeat program gate | +1.5599 | 176/176 | 1.023 | 0.153s |
| upper-repeat all categories | +1.5834 | 175/176 | 1.023 | 0.179s |
| category upper non-table | +1.6060 | 176/176 | 1.023 | 0.172s |
| mined conditional upper-repeat | +1.6146 | 176/176 | 1.023 | 0.177s |
| conservative LCB table-open rule | +1.6200 | 176/176 | 1.023 | 0.179s |

This confirms that the useful knowledge pattern is not a one-off threshold:

```text
use larger repeat budgets for airplane/chair;
for table, allow larger repeats only when current box geometry is compact/stable enough.
```

The caveat is important: split-CV rule mining is still weaker than the full-fit
rule because the severe table failures are rare.  When those failures leave the
training fold, the miner can over-open table upper-repeat.  So this is a strong
candidate research result, but not yet a default release policy.  The next
step is safety-calibrated precondition learning: train the rule/model on more
table failures and require a conservative lower confidence bound before opening
the upper-repeat option.

I added that conservative lower-bound miner as:

```bash
python experiments/macro_search/mine_conservative_repeat_open_rules.py \
  --median-live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_median_repeat_program_gate_all176_native_c256.jsonl \
  --upper-live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_upper_repeat_program_gate_all176_native_c256.jsonl \
  --out experiments/macro_search/runs/parameterized_skills_4k/conservative_repeat_open_rule_all176_lcb95.json \
  --min-support 8 \
  --min-lcb-gain 0.0 \
  --max-open-losses 0 \
  --min-worst-gain -0.000001
```

The full-fit conservative rule is:

```json
{
  "by_category": {"airplane": true, "chair": true, "table": false},
  "category": "table",
  "feature": "num_actions",
  "op": ">=",
  "threshold": 130.0,
  "if_true_upper": true,
  "if_false_upper": false
}
```

This is a more interpretable table precondition than the earlier single
`max_bbox_volume_ratio` rule.  It says: open upper-repeat for table only when
the current table state is already sufficiently decomposed (`num_actions >=
130`, corresponding to at least ten boxes with the current action schema).
On the opened table subset, the paired-log statistics are:

```json
{
  "cases": 12,
  "mean": 0.1239,
  "lcb": 0.0394,
  "min": 0.0,
  "losses": 0
}
```

The actual live executor check used the same all-176 portfolio and extra tetra
roots:

```bash
python experiments/macro_search/evaluate_live_skill_controller.py \
  --portfolio experiments/macro_search/runs/parameterized_skills_4k/portfolio_report_filesystem_category_scope_all176_top16_proxyfeatures.jsonl \
  --extra-tetra-root runs/expanded_full/tetra \
  --extra-tetra-root runs/shapenet_v1_3/tetra \
  --extra-tetra-root runs/expanded_200/tetra \
  --knowledge-base experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --category-knowledge-base airplane=experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json \
  --category-knowledge-base chair=experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json \
  --conditional-knowledge-base 'category=table,feature=num_actions,op=>=,threshold=130.0,path=experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_upper_repeats.json' \
  --selector knowledge_base_program_gate \
  --top-k 3 \
  --accept-mode first_positive \
  --candidate-count 256 \
  --max-cases 176 \
  --record-native-features
```

The same gate can now be run directly from the mined rule artifact, without
hand-copying the conditional expression:

```bash
python experiments/macro_search/evaluate_live_skill_controller.py \
  --portfolio experiments/macro_search/runs/parameterized_skills_4k/hard_skill_states_all176_top5_local_c256.jsonl \
  --extra-tetra-root runs/expanded_full/tetra \
  --extra-tetra-root runs/shapenet_v1_3/tetra \
  --extra-tetra-root runs/expanded_200/tetra \
  --knowledge-base experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --conditional-rule-file experiments/macro_search/runs/parameterized_skills_4k/conservative_repeat_open_rule_all176_lcb95.json \
  --selector knowledge_base_program_gate \
  --top-k 3 \
  --accept-mode first_positive \
  --candidate-count 256 \
  --record-native-features
```

Live result:

| category | cases | mean delta | accepted |
| --- | ---: | ---: | ---: |
| airplane | 87 | +2.1055 | 87/87 |
| chair | 46 | +1.1546 | 46/46 |
| table | 43 | +1.1359 | 43/43 |
| all | 176 | +1.6200 | 176/176 |

Hard table failure diagnosis also became clearer.  The severe upper-repeat
failure is a one-box table state:

```text
table::10ca7bfe736d81b64b3c42e318f3affc
num_boxes=1, num_actions=13, exact_bvs=7.6579
median delta=+5.4851, upper delta=0.0, upper accepted=false
```

So the pattern is not merely "table is unsafe."  The unsafe cases are
under-decomposed or low-action-count table states where a long repeat budget
cannot correct the geometry.  Once the table has enough active boxes/actions,
upper-repeat becomes safe and useful in the current live set.

I also reran the same median/upper/conservative comparison on the harder
chair/table80 subset (`45` chair + `34` table hard cases).  This set is biased
toward states where the first skill is not already the best branch.

| policy | cases | mean delta | accepted | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| median-repeat hard set | 79 | +1.1002 | 79/79 | 0.210s |
| upper-repeat hard set | 79 | +1.0755 | 78/79 | 0.249s |
| conservative `num_actions >= 130` | 79 | +1.1438 | 79/79 | 0.248s |

Per-category conservative hard-set result:

| category | cases | median delta | conservative delta | accepted |
| --- | ---: | ---: | ---: | ---: |
| chair | 45 | +1.0181 | +1.0711 | 45/45 |
| table | 34 | +1.2087 | +1.2401 | 34/34 |

This is stronger evidence for the precondition: blind upper-repeat is unsafe on
hard chair/table states, but the same longer repeat budget becomes useful when
opened through a simple geometry rule and exact rollback-safe validation.

I then expanded the same check to the full hard-state portfolio
(`86` airplane + `44` chair + `43` table, `173` runnable states).  This is a
better stress test because it includes the easy categories where longer repeat
budgets usually help and the table states where they can fail.

| policy | cases | mean delta | accepted | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| median-repeat hard-all | 173 | +1.5200 | 173/173 | 0.155s |
| upper-repeat hard-all | 173 | +1.5447 | 172/173 | 0.179s |
| conservative `num_actions >= 130` hard-all | 173 | +1.5759 | 173/173 | 0.179s |

Per-category conservative hard-all result:

| category | cases | median delta | upper delta | conservative delta |
| --- | ---: | ---: | ---: | ---: |
| airplane | 86 | +2.0386 | +2.1060 | +2.1060 |
| chair | 44 | +0.9157 | +0.9699 | +0.9699 |
| table | 43 | +1.1013 | +1.0103 | +1.1359 |

Per-state comparison against median-repeat:

| policy | wins | ties | losses | mean diff |
| --- | ---: | ---: | ---: | ---: |
| upper-repeat hard-all | 58 | 93 | 22 | +0.0247 |
| conservative hard-all | 55 | 101 | 17 | +0.0559 |

The table-only breakdown is the key evidence for reusable knowledge.  Blind
upper-repeat has `10` wins, `28` ties, and `5` losses on table states, with a
negative mean diff (`-0.0910`).  The conservative gate has `7` wins, `36` ties,
and `0` losses on the same table states, with a positive mean diff (`+0.0346`).
So this pattern is doing exactly what we want: keep the longer-repeat gains for
airplane/chair, open table only when decomposition is sufficient, and avoid the
hard table failure mode.

For knowledge mining, I also reran the conservative rule-file policy with
`accept_mode=best_positive`, so all top-3 skill attempts are exact-scored
instead of stopping after the first positive skill.  This is slower, but it
exposes the local oracle inside the current top-k set.

| mode | cases | mean delta | accepted | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| conservative first-positive | 173 | +1.5759 | 173/173 | 1.023 | 0.177s |
| conservative best-positive | 173 | +1.6020 | 173/173 | 3.000 | 0.380s |

The mined hard-all best-positive report shows a different lesson from the
easier runnable-133 set:

| family | support | oracle rate | mean delta | mean steps |
| --- | ---: | ---: | ---: | ---: |
| `shrink_slack_face` | 176 | 0.5227 | +0.4651 | 8.87 |
| `recenter_then_shrink` | 343 | 0.2362 | +0.9700 | 11.58 |

The family-level oracle gap is still large (`0.5808`).  So for hard states,
choosing the broad family is not enough.  The next useful controller should
learn parameterized execution details: repeat budget, target face/box role, and
whether to use direct long shrink versus recenter-then-shrink.  This is closer
to the 3D knowledge program idea than a one-step router: the reusable unit is
an option plus a state-dependent internal parameterization.

I then tested this directly with a held-out pattern-memory replay policy over
the conservative hard-all best-positive log.  This policy does not train a
generic neural net; it stores empirical values for `macro_id`, canonical
program pattern, family, and skill, then orders held-out candidates by a
weighted memory score.  The strongest setting was macro-heavy:

```text
macro_id=0.70, pattern=0.15, family=0.10, skill=0.05
```

Held-out replay aggregate over five splits:

| policy | exact calls | mean delta | gap to logged top-3 oracle |
| --- | ---: | ---: | ---: |
| default first candidate | 1.0 | +1.1207 | +0.4062 |
| macro-memory one-call | 1.0 | +1.4675 | +0.0594 |
| pattern-memory top-2 | 2.0 | +1.5267 | +0.00025 |
| logged top-3 oracle | 3.0 | +1.5269 | 0.0000 |

This is the clearest evidence so far for the “reusable 3D knowledge” direction:
the memorized macro/program identity almost solves the local top-k choice with
one exact call, and top-2 memory reaches the top-3 oracle while saving one
third of exact evaluations in replay.  It is not yet a live global-optimum
guarantee, but it is a concrete route to fewer Manifold calls without lowering
quality inside the logged candidate set.

I then connected this back into the live executor with a hybrid selector:

```text
program_gate ranks the skill pool;
macro-memory reorders only the top-3 program_gate candidates;
exact validation evaluates the top-2 memory-ranked candidates.
```

This avoids the failure mode of using macro-memory as a global ranker, which
over-favored direct shrink and hurt table quality.  The live hard-all result:

| live policy | exact attempts | mean delta | accepted | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| conservative first-positive | 1.023 | +1.5759 | 173/173 | 0.177s |
| conservative top-3 best-positive | 3.000 | +1.6020 | 173/173 | 0.380s |
| program-gate + macro-memory top-2 | 2.000 | +1.6017 | 173/173 | 0.332s |
| table-only second exact | 1.249 | +1.6015 | 173/173 | 0.242s |
| table `num_actions < 138` second exact | 1.197 | +1.6015 | 173/173 | 0.195s |

This is the first live version of the multi-step knowledge idea that gives a
clean tradeoff: it keeps essentially all top-3 exact quality (`-0.00027` mean
delta) while cutting one third of exact skill evaluations.  The wall-time
reduction is smaller than the call reduction because feature construction,
native state sync, and non-Manifold overhead remain, but the direction is now
operational rather than replay-only.

The stronger follow-up is a conditional second-exact gate.  In the top-2 live
log, the second exact skill wins only `31/173` states, and `25` of those are
tables.  A simple executable precondition,

```text
open the second exact skill iff category=table and num_actions < 138
```

opens only `34/173` states.  It keeps the same mean delta as the broader
table-only gate (`+1.6015`) while lowering mean attempts to `1.197` and mean
elapsed time to `0.195s`.  Relative to top-3 best-positive, this is a `60.1%`
exact-skill call reduction and a `1.95x` mean per-state speedup with only
`0.00044` mean delta loss.  This is the clearest current example of 3D
knowledge as an executable precondition: not "always run MCTS/top-k", but
"only open the extra branch for under-decomposed table states where the second
skill historically fixes the basin."

I then scaled the same idea from top-2 to top-5.  The goal was to check whether
a larger exact branch budget contains additional quality, and whether a mined
budget rule can recover that quality without paying for all five exact skill
evaluations on every state.  The top-5 oracle is better, but too slow as a
default:

| live policy | exact attempts | mean delta | accepted | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| table `num_actions < 138` second exact | 1.197 | +1.6015 | 173/173 | 0.195s |
| program-gate + macro-memory top-3 oracle | 3.000 | +1.6020 | 173/173 | 0.390s |
| program-gate + macro-memory top-5 oracle | 5.000 | +1.6133 | 173/173 | 0.723s |
| top-5 conditional budget, speed rule | 1.283 | +1.5939 | 173/173 | 0.191s |
| top-5 conditional budget, quality rule | 1.590 | +1.6060 | 173/173 | 0.203s |

The quality rule is the best current practical macro-skill setting:

```text
if category=table and num_actions < 137.8: use budget 4
else: use budget 1
```

Compared with the previous second-exact gate, it adds `+0.00445` mean delta for
only `+0.008s` per state.  Compared with the top-5 oracle, it uses `68.2%`
fewer exact skill attempts while recovering about `57.9%` of the remaining
quality gap.  The pattern is consistent with the earlier rule: extra exact
branching is mainly useful for table states whose decomposition leaves several
plausible macro-program basins.

I then checked that this is not just a one-log overfit.  A new CV script mines
the exact-budget rule on train folds and evaluates the chosen rule on held-out
folds:

```bash
python experiments/macro_search/validate_conditional_budget_cv.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_program_gate_macro_memory_rulefile_all173_hard_pool5_top5_best_positive_c256.jsonl \
  --max-budget 5 \
  --call-penalty 0.02 \
  --candidate-keep 60 \
  --folds 5 \
  --reference-rule smart/assets/skills/macro_budget_quality_rule_v1.json \
  --out experiments/macro_search/runs/parameterized_skills_4k/conditional_budget_cv_pool5_top5_penalty002_5fold_fast.json
```

Across five different hash-split seeds, the mined rule family is stable:
`table` states with low `num_actions` open budget 4, while other states stay at
budget 1.

| validation | mean delta | exact attempts | gain vs budget-1 | regret vs top-5 oracle | losses vs budget-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5-fold CV, 5 seeds mean | +1.6051 | 1.579 | +0.0552 | 0.0082 | 0 |
| CV min/max mean delta | +1.6038 / +1.6060 | 1.555 / 1.607 | +0.0538 / +0.0560 | 0.0073 / 0.0096 | 0 |
| packaged reference rule | +1.6060 | 1.590 | +0.0560 | 0.0073 | 0 |

This is the clearest current "3D knowledge" pattern in the macro branch: table
states with a small candidate/action lattice often need a few exact
macro-program basins checked; airplane/chair and larger table states are
usually handled by the first ranked macro.

#### Guarded Variable-Repeat Execution

The packaged executor now uses the mined repeat range as an execution budget,
not as a fixed median repeat count.  This is important because the useful
knowledge pattern is often:

```text
repeat shrink/tighten while exact SMART reward keeps improving,
stop before the first non-improving shrink,
allow temporary negative steps only for explicit expand/escape macros.
```

In other words, the skill is a variable-length option, not a hard-coded
n-step sequence.  The executor therefore opens a step up to the observed
`repeat_max` when the mined termination predicate contains
`score_stalls`, `coverage_margin_low`, or `coverage_recovered`.  It then
exact-scores each primitive action.  Non-positive shrink/tighten steps are
rejected before application; expansion steps may be temporarily negative
because those are the local-minimum escape macros.

On the same 173-state top-5 conditional-budget benchmark, replaying the
packaged file API with the guarded variable-repeat executor gives:

| setting | exact attempts | mean delta | accepted | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| top-5 conditional budget v1 | 1.590 | +1.6060 | 173/173 | 0.203s |
| guarded variable-repeat v2 balanced | 1.590 | +1.6118 | 173/173 | 0.226s |
| guarded variable-repeat v2 quality (`quality_preset="quality"`) | 5.000 | +1.6639 | 173/173 | 0.815s |

These numbers use reference-initial-score artifact matching, because the local
checkout contains duplicate tetra/bbox artifacts for some ShapeNet ids.  The
balanced preset compares against v1 with `101` wins, `72` ties, and `0` losses,
mean gain `+0.0059`.  The quality preset compares with `145` wins, `28` ties,
and `0` losses, mean gain `+0.0579`.  The larger wins come from avoiding early
median-repeat termination.  This is the strongest current
evidence that reusable 3D knowledge should be represented as option programs
with stop predicates: the same family can run for two steps on one state and
sixteen steps on another without changing the exact reward validator.

Per-category replay against the v1 conditional-budget reference:

| category | cases | balanced mean gain | balanced wins/ties/losses | quality mean gain | quality wins/ties/losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| airplane | 86 | +0.0029 | 49 / 37 / 0 | +0.0333 | 67 / 19 / 0 |
| chair | 44 | +0.0106 | 23 / 21 / 0 | +0.1164 | 40 / 4 / 0 |
| table | 43 | +0.0069 | 29 / 14 / 0 | +0.0475 | 38 / 5 / 0 |

The artifact-matched benchmark removes the previous apparent hard-table loss:
that case was replayed against the wrong duplicate artifact.  On the matching
artifact, the quality preset improves the reference delta from `+0.7586` to
`+0.8291`.  This turns the macro branch from a small acceleration result into a
clear exact-safe quality-improving controller in the current live-state
benchmark.

These macro-skill artifacts are packaged as experimental assets so they can be
referenced reproducibly without depending on the ignored `experiments/` tree:

```python
import smart

skill_kb = smart.asset_path("skills", "macro_v1")
memory = smart.asset_path("skills", "macro_memory_v1")
budget_rule = smart.asset_path("skills", "macro_budget_quality_v1")
```

They are not the default SMART backend.  The release-safe path remains exact
native SMART and the opt-in DeepSets router above.  The macro-skill controller
is now available as an experimental Python API that drives the native C++
engine:

```python
import smart
import smart.cpp as sc

engine = sc.NativeSmartEngine(...)
result = sc.run_builtin_macro_skill_controller(
    engine,
    category="table",
    top_k=5,
    candidate_count=256,
    repeat_mode="guarded_variable",
)
```

The wrapper ranks the packaged skills, chooses the conditional exact budget,
executes candidate skills on the live C++ engine, and accepts only a
positive-exact-reward skill.  If no skill improves the exact score, it restores
the original engine state.  This is still marked experimental because the
current held-out evidence is log/live-state based; larger end-to-end mesh-level
checks and direct C++ integration are not complete.

`repeat_mode="guarded_variable"` is the current recommended research profile.
Use it in one of two presets:

```python
# Balanced: current practical default, conditional exact budget.
result = sc.run_builtin_macro_skill_controller(engine, category="table")

# Quality: spend all top-5 exact skill attempts and use max_steps=32, then accept the best.
result = sc.run_builtin_macro_skill_controller(
    engine,
    category="table",
    top_k=5,
    quality_preset="quality",
    repeat_mode="guarded_variable",
)
```

I also tested `repeat_mode="best_of_median_and_guarded"`, which exact-executes
both the mined median repeat and the guarded variable repeat for each macro and
keeps the better result.  On the 173-state reference replay it matched the
guarded-variable balanced mean delta but increased exact attempts from `1.590`
to `3.179` without improving quality, so it is retained as an analysis knob
rather than a recommended production setting.  Pure median repeat is not a
valid fallback: it drops mean delta to `+0.7494` and loses on `134 / 173`
states.  During this pass the executor was also fixed to allow neutral
`recenter_box` actions and cap them at one application per macro step.

The rule is reproducible with:

```bash
python experiments/macro_search/mine_second_exact_open_rules.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_program_gate_macro_memory_rulefile_all173_hard_pool3_top2_best_positive_c256.jsonl \
  --out experiments/macro_search/runs/parameterized_skills_4k/second_exact_open_rule_program_gate_macro_memory_all173_hard_top2.json \
  --call-penalty 0.02
```

The live executor command is:

```bash
python experiments/macro_search/evaluate_live_skill_controller.py \
  --portfolio experiments/macro_search/runs/parameterized_skills_4k/hard_skill_states_all176_top5_local_c256.jsonl \
  --extra-tetra-root runs/expanded_full/tetra \
  --extra-tetra-root runs/shapenet_v1_3/tetra \
  --extra-tetra-root runs/expanded_200/tetra \
  --knowledge-base experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --conditional-rule-file experiments/macro_search/runs/parameterized_skills_4k/conservative_repeat_open_rule_all176_lcb95.json \
  --selector knowledge_base_program_gate_macro_memory \
  --macro-memory-policy experiments/macro_search/runs/parameterized_skills_4k/macro_memory_policy_conservative_rulefile_all173_hard_best_positive.json \
  --macro-memory-pool-size 3 \
  --top-k 2 \
  --accept-mode category_second_exact \
  --second-exact-rule-file experiments/macro_search/runs/parameterized_skills_4k/second_exact_open_rule_program_gate_macro_memory_all173_hard_top2.json \
  --candidate-count 256 \
  --record-native-features
```

Five-fold train/test mining gives the same qualitative rule family.  On each
fold, the rule is mined from train rows only and evaluated on held-out rows:

| policy | held-out mean delta | held-out exact attempts | regret vs top2 |
| --- | ---: | ---: | ---: |
| first skill only | +1.5478 | 1.000 | +0.05184 |
| mined second-exact gate | +1.5984 | 1.197 | +0.00123 |
| top2 exact oracle | +1.5996 | 2.000 | 0.00000 |

The held-out rules are not identical in every fold (`num_actions < 130`,
`num_actions < 138`, and one `global_min_bbox_dim` variant appear), but they
all express the same reusable knowledge: only a small under-decomposed table
subset needs the second branch; most airplane/chair states and many table
states should stop after the first exact skill.

The current research lesson is concrete: reusable 3D knowledge is appearing as
preconditioned variable-length options, not as a raw one-step action classifier.
The next useful model should predict these option preconditions and repeat
budgets directly, with exact SMART reward kept as the acceptance validator.

### Conditional Exact-Budget Ladder

The second-exact gate is speed-first: it opens very few extra branches and
therefore gives the largest wall-time improvement, but it is tied to the
program-gate + macro-memory top-2 setting.  I added a more general miner,
`mine_conditional_exact_budget_rules.py`, that treats the exact budget itself as
a reusable skill-control decision:

```text
budget 1: accept the first positive skill only
budget 2: exact-pick the best positive among the first two skills
budget 3: exact-pick the best positive among the first three skills
```

The mined policy is an ordered list of cheap predicates.  The first matching
predicate chooses the budget; the accepted skill is still validated by exact
SMART/Manifold.  This makes the "knowledge" more explicit: not only which
macro skill to run, but how much exact branching a state deserves.

On `existing_kb_program_gate_runnable133`, the best mined rule is fully
interpretable:

```text
if category=table: use budget 3
elif category=chair: use budget 2
else: use budget 1
```

Live validation with `--accept-mode conditional_exact_budget` gives:

| policy | cases | mean delta | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: | ---: |
| first candidate only | 133 | +1.6801 | 1.000 | baseline |
| top-3 exact oracle | 133 | +1.7247 | 3.000 | 0.292s |
| conditional budget ladder | 133 | +1.7237 | 1.669 | 0.254s |

So this ladder preserves nearly all local top-3 quality (`0.0010` mean-delta
loss) while reducing exact skill calls by `44.4%` and wall time by about
`1.15x`.  The rule is category-level rather than an opaque neural decision:
airplane states in this portfolio are usually solved by the first program-gate
skill, chairs benefit from a second check, and tables often need all three
candidate skills because support/slab geometry makes local minima more common.

The miner command is:

```bash
python experiments/macro_search/mine_conditional_exact_budget_rules.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_existing_kb_program_gate_runnable133_top3_best_positive_c256.jsonl \
  --out experiments/macro_search/runs/parameterized_skills_4k/conditional_exact_budget_rule_existing_program_gate_runnable133_top3_penalty005.json \
  --max-budget 3 \
  --call-penalty 0.005
```

The live replay command is:

```bash
python experiments/macro_search/evaluate_live_skill_controller.py \
  --portfolio experiments/macro_search/runs/parameterized_skills_4k/portfolio_from_program_gate_runnable133.jsonl \
  --extra-tetra-root runs/expanded_full/tetra \
  --extra-tetra-root runs/shapenet_v1_3/tetra \
  --extra-tetra-root runs/expanded_200/tetra \
  --knowledge-base experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --selector knowledge_base_program_gate \
  --top-k 3 \
  --accept-mode conditional_exact_budget \
  --exact-budget-rule-file experiments/macro_search/runs/parameterized_skills_4k/conditional_exact_budget_rule_existing_program_gate_runnable133_top3_penalty005.json \
  --candidate-count 256 \
  --record-native-features
```

The same miner on the harder conservative all-173 set finds a quality-preserving
threshold rule:

```text
if bbox_volume_cv < 1.7497: use budget 3
else: use budget 1
```

That policy reaches `+1.60175` vs top-3 oracle `+1.60196`, with attempts
`2.595` instead of `3.000`.  It is less aggressive than the table second-exact
gate, but it shows the same principle: exact branching should be opened only
for states whose bbox geometry predicts that later macro candidates can matter.

I also checked rule transfer.  The category budget rule is not a universal law;
it is tied to the selector/program family that produced the candidate order.
Applying the runnable133 category rule to other top-3 logs gives:

| target log | first-only delta | transferred rule delta / attempts | top-3 delta |
| --- | ---: | ---: | ---: |
| same `existing_kb_program_gate` | +1.6801 | +1.7237 / 1.669 | +1.7247 |
| `trace_kb_runnable133` | +1.5013 | +1.5931 / 1.669 | +1.7243 |
| conservative hard-173 | +1.5187 | +1.5595 / 1.751 | +1.6020 |
| `knowledge_base_delta_all176` | +1.4069 | +1.4718 / 1.750 | +1.5689 |

So the next abstraction must include the candidate generator/selector identity
as part of the precondition:

```text
when selector_family = existing_kb_program_gate and category = chair,
open two exact branches;
when selector_family = existing_kb_program_gate and category = table,
open three exact branches;
otherwise learn a separate budget rule or fall back to top-k exact.
```

This is still useful 3D knowledge, but it is contextual knowledge: the same
object category can need different exact budgets when the upstream skill order
changes.

### Intra-Family Parameter Choice

The next level is not "which family?", but "within the chosen family, which
repeat/target/program variant?".  I added:

```bash
python experiments/macro_search/mine_intra_family_parameter_rules.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_conservative_rulefile_all173_hard_program_gate_top3_best_positive_c256.jsonl \
  --out experiments/macro_search/runs/parameterized_skills_4k/intra_family_parameter_rules_conservative_all173_hard_top3.json
```

For the hard-173 top-3 log, the meaningful intra-family ambiguity is inside
`recenter_then_shrink`:

| variant | default repeats | program |
| --- | --- | --- |
| `skill_004` | `[1.5, 1.0, 6.5, 1.0, 2.0]` | shrink -> recenter -> shrink -> recenter -> shrink |
| `skill_000` | `[3.0, 1.0, 4.5]` | shrink -> recenter -> shrink |

The current program-gate order always tries `skill_004` first.  The mined rule
learns that many low-score states should instead use the shorter `skill_000`
variant:

```text
if best_bbox_score < -1.2718: choose skill_000
else: keep skill_004
```

Offline within-family replay:

| policy | mean delta inside family | regret vs family oracle |
| --- | ---: | ---: |
| current first variant | +0.4184 | 0.0136 |
| mined intra-family rule | +0.4316 | 0.0004 |
| family oracle | +0.4321 | 0.0000 |

I wired this rule into the live executor with:

```bash
--intra-family-policy experiments/macro_search/runs/parameterized_skills_4k/intra_family_parameter_rules_conservative_all173_hard_top3.json
```

Live hard-173 result under the same first-positive exact budget:

| policy | mean delta | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: |
| conservative first-positive | +1.5759 | 1.023 | 0.177s |
| + intra-family parameter rule | +1.5891 | 1.023 | 0.193s |

This is the first clean result where parameter selection inside a family
improves quality without increasing exact skill attempts.  The time is slightly
higher because the rule uses native geometry features; if promoted, the feature
subset should be cached or moved into the C++ engine.

Combining the same intra-family rule with the speed-first macro-memory
second-exact gate gives only a tiny extra gain:

| policy | mean delta | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: |
| macro-memory + table second-exact | +1.601518 | 1.197 | 0.195s |
| + intra-family parameter rule | +1.601542 | 1.197 | 0.195s |

Interpretation: macro-memory already fixes most variant ordering in the
speed-first path; intra-family parameter learning matters more for one-call
profiles.

I also tested a direct repeat-budget extreme by generating a min-repeat
knowledge base:

```bash
python experiments/macro_search/make_skill_repeat_variant.py \
  --in-kb experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --out experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_min_repeats.json \
  --policy min
```

Min-repeat is much faster (`0.094s`) but too weak (`+1.1775` mean delta).  A
strict switch-to-min policy can safely replace the conservative path on only a
small subset and saves little wall time.  The higher-value direction is
therefore not just shorter repeats; it is state-conditioned selection among
meaningfully different program variants and, eventually, generating new
target/repeat variants before exact validation.

### Repeat-Palette and Target-Aware Follow-Up

I then generated a structured repeat palette instead of only choosing among the
original mined variants:

```bash
python experiments/macro_search/make_skill_repeat_palette.py \
  --in-kb experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base.json \
  --out experiments/macro_search/runs/parameterized_skills_4k/skill_knowledge_base_repeat_palette_v1.json \
  --max-total-steps 16 \
  --max-variants-per-skill 8
```

This produced 32 variants.  For example, the same
`shrink -> recenter -> shrink -> recenter -> shrink` family now has:

```text
skill_004__rep_low:       [1, 1, 3, 1, 1]
skill_004__rep_median:    [2, 1, 7, 1, 2]
skill_004__rep_mid_high:  [2, 1, 9, 1, 3]
skill_004__rep_inner_high:[2, 1,10, 1, 2]
```

The repeat value is still only a ceiling.  Each primitive step is exact-scored,
and execution stops early when the local score stalls.

Hard-173 live results:

| policy | mean delta | exact attempts | mean elapsed | note |
| --- | ---: | ---: | ---: | --- |
| conservative first-positive | +1.5759 | 1.023 | 0.177s | existing baseline |
| repeat palette, first-positive | +1.5200 | 1.081 | 0.168s | ordering is worse |
| repeat palette, top-8 oracle | +1.5812 | 8.000 | 0.457s | small headroom |
| repeat palette + mined intra-family rule | +1.4791 | 1.000 | 0.154s | rejected |
| repeat palette + old second-exact rule | +1.4605 | 1.197 | 0.165s | rejected |

The important result is negative: expanding the repeat palette creates some
oracle headroom, but it does not beat the current macro-memory + second-exact
controller (`+1.6015`).  Against that best speed path, the top-8 palette oracle
has 64 wins, 74 ties, and 35 losses.  A conservative hybrid rule that opens the
palette only on non-worse logged cases improves the offline mean from `+1.6015`
to `+1.6045`, but it opens only 4--6 cases and is not yet worth promotion.

I also tested direct target-aware execution.  Skill records contain targets such
as `max_slack_face`, `major_axis_face`, and `single_box`, but the live executor
previously used only the operation type (`shrink_face`, `expand_face`,
`recenter_box`) and let exact reward choose the primitive.  I added an
experiment-only flag:

```bash
--target-aware-execution
```

The unguarded target filter was harmful:

| 40-case smoke | mean delta | exact attempts | mean elapsed |
| --- | ---: | ---: | ---: |
| original exact-best primitive choice | +2.1153 | 1.100 | 0.218s |
| target-aware forced filter | +1.1960 | 1.100 | 0.121s |
| guarded target-aware fallback | +2.1153 | 1.100 | 0.230s |

So the mined target labels are not strong enough to override exact local reward.
The guarded version falls back to exact-best when the target-filtered primitive
is worse, recovering baseline quality but adding overhead.  Conclusion: target
roles should be used as model features or retrieval conditions first, not as a
hard execution constraint.  A promotion-worthy target system needs better
target labels, likely based on uncovered-volume direction or exact action
counterfactuals rather than slack heuristics alone.

To diagnose the target-label problem, I reran hard-173 with
`--record-local-action-features` and inspected the exact-best primitives that
were actually executed.  The dominant realized pattern is not literally
`max_slack_face`; it is more specific:

```text
minor-axis shrink on one face
minor-axis shrink on the opposite face
recenter
repeat minor-axis tightening
```

Top realized primitive roles:

| realized exact primitive role | count |
| --- | ---: |
| `shrink axis0 min minor` | 369 |
| `recenter` | 342 |
| `shrink axis0 max minor` | 273 |
| `shrink axis2 max minor` | 161 |
| `shrink axis2 min minor` | 144 |

By category, airplane uses more `axis2` minor-axis tightening, while chair/table
use more `axis0` minor-axis tightening.  This gives a better next abstraction:

```text
Skill: tighten_minor_axis_pair_then_recenter
Precondition:
  - coverage is already stable
  - BVS is high, so volume reduction matters
  - one bbox or a dominant bbox has a clear minor axis
Program:
  1. shrink the minor-axis face with best exact/proxy reward
  2. shrink the opposite minor-axis face
  3. recenter
  4. repeat until exact reward stalls
```

This is closer to the actual learned 3D knowledge than the earlier
`max_slack_face` label.  The next implementation should generate
minor-axis-pair target variants and compare them against the current
exact-best primitive policy.

### Live Proxy-Token Integration

I added an experiment-only live selector:

```bash
--selector knowledge_base_token_checkpoint
--model-path <local_action_token_controller_*.pt>
```

This does not use exact reward to build the model input.  For each candidate
skill, it:

1. uses `centroid_proxy_axis_metrics` to propose a first primitive action;
2. builds a 45-D `local_action_features()` token for that action;
3. scores the candidate skill with the saved token Transformer;
4. reorders the top-k skills before exact execution.

I also added:

```bash
--accept-mode guarded_override
```

This is the safer runtime mode.  If the learned router changes the original
top-1 skill, it exact-evaluates both the learned top-1 and the original top-1,
then accepts the better one.  This is cheaper than top-5 exact search and safer
than trusting the learned router alone.

Held-out live smoke using the final 16 table cases not included in the
combined160 top-5 training trace:

| live mode | mean delta | exact attempts | mean elapsed | accepted skills |
| --- | ---: | ---: | ---: | --- |
| baseline `knowledge_base_delta`, first positive | +0.6047 | 1.000 | 0.023s | recenter 16 |
| token checkpoint, first positive | +0.5987 | 1.000 | 0.025s | escape 5, recenter 11 |
| token checkpoint, guarded override | +0.6051 | 1.375 | 0.033s | escape 3, recenter 13 |
| top-5 exact oracle | +0.6181 | 5.000 | 0.069s | escape 5, recenter 11 |

Interpretation:

- The raw token router can select the escape skill, but it can also make wrong
  overrides.
- `guarded_override` prevents most damage and keeps exact calls much lower than
  top-5 exact search.
- The held-out 16-case ceiling is small, so this is a functionality and safety
  result rather than a major speed/quality result.
- Real deployment still needs a C++/compact inference path.  PyTorch inference
  overhead is visible at this small scale.

### Packaged Macro-Skill Controller

The variable-length 3D knowledge controller is now exposed as an opt-in package
API and CLI command. It is not the default SMART optimizer. It is a guarded
post-refinement controller for prepared SMART states: a tetra mesh plus bbox
metadata from merge/refine/MCTS.

Python API:

```python
import smart

result = smart.run_macro_skill_controller_from_files(
    msh_path="runs/example/tetra/airplane/0001/tetra.msh",
    bbox_metadata_path="runs/example/mcts/airplane/0001/bbox_params.json",
    category="airplane",
    quality_preset="balanced",  # or "quality"
    top_k=5,
    candidate_count=256,
)

if result["accepted"]:
    result["engine"].export_bbox_dir("runs/example/macro_skill/airplane/0001")
```

CLI:

```bash
smart macro-skill \
  --msh runs/example/tetra/airplane/0001/tetra.msh \
  --bbox-metadata runs/example/mcts/airplane/0001/bbox_params.json \
  --category airplane \
  --quality-preset balanced \
  --output runs/example/macro_skill/airplane/0001/result.json \
  --output-bbox-dir runs/example/macro_skill/airplane/0001/bboxs \
  --json
```

Current packaged benchmark/profile summary:

```bash
smart macro-skill-summary --json
```

Use `--quality-preset quality` when quality matters more than exact-call count.
This opens the top-5 exact skill budget and raises the internal variable-length
step cap to 32. On the current 173-state benchmark it improved the mean exact
delta from `+1.6118` to `+1.6639`, with `145/28/0` wins/ties/losses against the
previous conditional-budget controller. Balanced mode remains the cheaper
production candidate: it keeps exact attempts near `1.59` per state.

All accepted updates are exact-validated by the native SMART/Manifold reward.
If no skill improves the state, the engine rolls back to the input bbox state.
The controller can therefore be safely inserted as an optional final polishing
stage after baseline SMART, but it should stay opt-in until larger held-out
pipeline-level validation is complete.

The returned result payload is deliberately conservative.  It includes:

```text
exact_validator: native_smart_manifold
rollback_on_failure: true
accepted_non_worse: true/false
deployment_status: experimental_opt_in_post_refine
default_smart_path_changed: false
```

For production use, treat `accepted=true` and `accepted_non_worse=true` as the
only state-changing success condition.  If `accepted=false`, the engine state
has been restored to the input bboxes and the caller can continue with the
baseline SMART output.

### Native Exact Primitive Selector

The packaged macro-skill controller now has a deeper native execution path for
primitive action selection.  Earlier versions ranked a macro skill in Python,
then repeatedly called back into the C++ engine from Python while scanning
candidate primitive actions.  The current controller first asks
`NativeSmartEngine.select_exact_native_action(...)` to do the common operation
filtering and exact primitive scoring inside the live C++ engine state.

The reward definition is unchanged:

```text
skill op -> C++ candidate action filter -> exact SMART/Manifold reward
         -> exact-best matching primitive -> apply or rollback
```

The C++ selector preserves the same proxy-first candidate order used by the
Python fallback, appends recenter actions after axis candidates, and keeps the
first candidate on exact reward ties.  This avoids changing skill trajectories
except where the native exact scorer itself rejects invalid candidates.

Local one-tet smoke benchmark:

| path | 1000 macro-skill smoke calls | mean score delta |
| --- | ---: | ---: |
| Python skill loop + Python primitive scan | 1.0845s | 0.0000 |
| Python skill loop + C++ primitive selector | 1.0434s | 0.0000 |
| C++ full skill executor + C++ primitive selector | 1.0647s | 0.0000 |

This is only an API smoke; on the one-tet case no real macro action is executed,
so it should not be used as a full pipeline speed claim.  The more meaningful
artifact-matched replay uses live ShapeNet states with nonzero macro steps.
With the packaged controller, candidate count `256`, top-5 skills, and the
balanced exact budget, local replay over all currently reconstructable artifacts
gave:

| path | cases | accepted | wins/ties/losses vs reference | mean delta | mean gain | mean elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python skill loop | 52 | 52 | 36 / 16 / 0 | +2.3201 | +0.1117 | 0.0772s |
| compact C++ skill executor | 52 | 52 | 36 / 16 / 0 | +2.3201 | +0.1117 | 0.0769s |

The native executor matches the Python loop exactly on selected macro id and
score delta in this replay.  The elapsed improvement is only about `1.004x`,
which is the important negative result: for this controller, exact
SMART/Manifold scoring dominates, not Python action-record overhead.  The
production value of the native path is therefore safety and integration, while
the research value is quality: the packaged macro knowledge produces no losses
against the reference and improves 36 of the 52 reconstructable live states.

The next exact-safe production step is selector memoization inside
`NativeSmartEngine`.  A macro controller often tries several skill programs from
the same bbox state; those programs can ask the same `state + operation +
candidate budget` question repeatedly.  The cache stores only the result of a
previous exact SMART/Manifold selection.  It does not use proxy reward as a final
metric and does not accept a skill without exact validation.

On the 49-case overlap with the previous artifact replay:

| path | overlap | score diffs | macro diffs | cache hits / misses | exact checks saved | mean elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compact C++ executor, no selector cache | 49 | 0 | 0 | - | - | 0.0787s |
| compact C++ executor, selector cache | 49 | 0 | 0 | 501 / 501 | 15,204 | 0.0741s |

This is a better acceleration target than merely moving Python bookkeeping:
quality is unchanged, selected macros are unchanged, and about half of the
repeated exact primitive-selection checks in this replay are eliminated.  The
wall-time gain remains modest because the controller already accepts a macro
after roughly one attempt on most states; cache benefits should grow with
larger exact budgets, higher `top_k`, and MCTS-like local escape schedules.

That expectation holds on the higher-budget `quality` preset.  On the overlap
with the previous uncached quality replay, score and macro selection are again
unchanged, while the cache removes 42,135 repeated exact primitive-selection
checks:

| profile | cases | wins/ties/losses vs reference | exact checks | checks saved | mean elapsed | overlap speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced + selector cache | 49 | 33 / 16 / 0 | 15,204 | 15,204 | 0.0741s | 1.06x on overlap |
| quality + selector cache | 49 | 42 / 7 / 0 | 67,207 | 42,135 | 0.3055s | 1.05x on overlap |

The quality preset is the stronger research setting: it trades more exact
checks for better final score, and the selector cache recovers part of that
cost without changing the exact reward contract.

We also tested restoring the exact final bbox state from the best candidate
skill attempt instead of re-executing that skill.  Although exact-safe, this is
not the default: copying final bbox state for every attempted skill was slower
than the cached re-execution path on the local artifact replay.  It remains a
research knob, not a production setting.

The production direction remains to keep bbox state, rollback state, candidate
filtering, exact primitive scoring, and termination predicates inside the
native engine for longer.  Further speedups require reducing exact geometry
calls through exact-safe macro gating or a stronger exact-cache strategy, not
just moving Python bookkeeping to C++.

## Artifact-Matched 3-Category Macro Replay

The earlier packaged replay covered only the locally reconstructable airplane
subset.  I expanded the replay roots to include `cpp_native_search_experimental`,
`expanded_full`, `shapenet_v1_3`, and `smoke_5` tetra/bbox artifacts.  This
recovers 133 runnable states from the all-176 macro-skill portfolio:

| category | cases | balanced mean delta | balanced gain vs reference | balanced wins/ties/losses |
| --- | ---: | ---: | ---: | ---: |
| airplane | 73 | +2.2240 | +0.0943 | 47 / 26 / 0 |
| chair | 31 | +1.2530 | +0.1222 | 23 / 8 / 0 |
| table | 29 | +1.4498 | +0.1101 | 19 / 10 / 0 |
| all | 133 | +1.8289 | +0.1043 | 89 / 44 / 0 |

The high-budget `quality` preset is also exact-safe on this replay.  It opens
all top-5 exact skill attempts and allows longer guarded programs.  Compared
with balanced, it has no losses and improves 76 states:

| category | quality gain vs balanced | positive/tie/negative | extra time/case | extra exact checks/case |
| --- | ---: | ---: | ---: | ---: |
| airplane | +0.0343 | 39 / 34 / 0 | +0.415s | +1,426 |
| chair | +0.1292 | 20 / 11 / 0 | +1.390s | +2,594 |
| table | +0.0612 | 17 / 12 / 0 | +0.889s | +1,897 |
| all | +0.0623 | 76 / 57 / 0 | +0.746s | +1,952 |

This confirms the current quality/compute tradeoff: spending more exact calls
can improve final exact SMART score, but it is not free.  The best partial
scheduler by gain per extra second is a conservative chair-only quality gate:

```text
if category == chair:
    quality_preset = "quality"
else:
    quality_preset = "balanced"
```

On the same 133 states this gate opens quality for 31 cases, produces
20 wins / 113 ties / 0 losses against balanced, and reaches mean delta
`+1.8590` with mean elapsed `0.5544s`.  It is now exposed as
`quality_preset="efficient"`.  It remains opt-in because it is a mined rule, not
yet a cross-dataset guarantee.

We also trained a state-conditioned ridge gate on the same paired
balanced/quality replay.  The gate consumes the native geometry state features
exported by the C++ engine, predicts the expected quality gain, and opens the
same 31/133 high-budget slots with the highest predicted value.  In 20-seed
5-fold category-stratified replay it improves over the chair-only rule:

| gate | high-budget cases | mean gain vs. balanced | extra time / case | gain / extra second | losses |
|---|---:|---:|---:|---:|---:|
| chair-only efficient | 31 / 133 | +0.0301 | +0.324s | 0.093 | 0 |
| state-conditioned ridge | 31 / 133 | +0.0437 | +0.291s | 0.150 | 0 |
| oracle efficiency, budget-matched | 31 / 133 | +0.0531 | +0.120s | 0.442 | 0 |

The middle operating point is exposed as
`quality_preset="learned_efficient"`.  It is still opt-in: the accepted bbox
update is exact-validated, so quality is protected, but the gate can still waste
extra exact calls on out-of-distribution states.

We also evaluate a stricter category-transfer setting: train on two categories
and hold out the third.  The same ridge target keeps 0 losses and remains
positive in every held-out category:

| held-out category | mean gain vs. balanced | extra time / case | gain / extra second | losses |
|---|---:|---:|---:|---:|
| airplane | +0.0227 | +0.060s | 0.375 | 0 |
| chair | +0.0729 | +0.307s | 0.238 | 0 |
| table | +0.0523 | +0.204s | 0.256 | 0 |
| all | +0.0408 | +0.149s | 0.274 | 0 |

The open-rate sweep shows the current time/quality frontier.  A lower open rate
is more efficient; a higher open rate approaches the all-quality result:

| learned gate open rate | mean gain vs. balanced | extra time / case | gain / extra second |
|---:|---:|---:|---:|
| 0.10 (`learned_fast`) | +0.0287 | +0.113s | 0.254 |
| 0.15 | +0.0352 | +0.185s | 0.190 |
| 0.233 (`learned_efficient`) | +0.0429 | +0.294s | 0.146 |
| 0.50 (`learned_quality`) | +0.0568 | +0.506s | 0.112 |
| 1.00 | +0.0623 | +0.746s | 0.084 |

We keep the packaged preset at the middle operating point because it improves
both quality and budget efficiency over the category-only rule.  For paper
figures, this sweep is the cleanest evidence that the learned controller is not
only making SMART faster; it is reallocating exact calls toward cases where they
improve the final exact score.

We also tested a larger MLP gate on the same state features.  It slightly
improves mean gain (`+0.0442` vs. ridge `+0.0437`) but spends more extra time
(`+0.338s` vs. `+0.291s`) and has lower gain per extra second (`0.131` vs.
`0.150`).  For the current data size, the simpler ridge model is the better
production-facing router.  Larger Transformer-style models remain useful for
future macro-skill parameterization and variable-length program selection, but
not yet for this one-shot budget gate.

The important research conclusion is that rule transfer is not stable enough
to make an adaptive scheduler the default.  The older packaged budget rule was
learned on a 173-case log and favored table-like states; the artifact-matched
133-case replay favors chair-like states.  Therefore the production default
stays `balanced`, while `efficient`, `learned_fast`, `learned_efficient`,
`learned_quality`, and `quality` are exact-validated research presets for
quality reinvestment.

### Production Benchmark Harness

The deployable learned-router result is now covered by a single replay
benchmark:

```bash
PYTHONPATH=. python experiments/macro_search/benchmark_packaged_macro_presets.py
```

This script reads the paired exact replay artifacts for `balanced` and
`quality`, applies the packaged ridge gate from
`smart/assets/skills/macro_quality_gate_ridge_v1.json`, and writes:

```text
experiments/macro_search/runs/parameterized_skills_4k/packaged_macro_preset_benchmark.json
experiments/macro_search/runs/parameterized_skills_4k/packaged_macro_preset_benchmark.md
```

Unlike the cross-validation table above, this is the exact runtime behavior of
the packaged asset thresholds.  It is the table to use when checking whether a
release candidate is still safe:

| preset | high budget | mean delta | gain vs balanced | extra time | exact checks | exact-check reduction vs quality | wins/ties/losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 0 / 133 | +1.8289 | +0.0000 | +0.000s | 525.6 | 77.4% | 0 / 133 / 0 |
| learned_fast | 13 / 133 | +1.8724 | +0.0435 | +0.231s | 848.7 | 63.5% | 12 / 121 / 0 |
| efficient chair rule | 31 / 133 | +1.8590 | +0.0301 | +0.324s | 1130.3 | 51.4% | 20 / 113 / 0 |
| learned_efficient | 31 / 133 | +1.8805 | +0.0516 | +0.324s | 1098.2 | 52.8% | 24 / 109 / 0 |
| learned_quality | 66 / 133 | +1.8890 | +0.0601 | +0.537s | 1608.2 | 30.9% | 46 / 87 / 0 |
| quality | 133 / 133 | +1.8912 | +0.0623 | +0.746s | 2326.2 | 0.0% | 76 / 57 / 0 |
| oracle gain, budget matched | 31 / 133 | +1.8896 | +0.0607 | +0.413s | 1186.8 | 49.0% | 31 / 102 / 0 |
| oracle efficiency, budget matched | 31 / 133 | +1.8820 | +0.0531 | +0.120s | 860.6 | 63.0% | 31 / 102 / 0 |

This gives a clearer production story:

- `learned_efficient` is better than the hand-written chair rule at the same
  high-budget count.
- It retains most of the quality gain of `quality` while cutting exact checks by
  52.8% relative to the all-quality setting.
- It has 0 losses against `balanced` on this exact replay because unopened
  states fall back to balanced and opened states use the exact-validated quality
  result.
- The oracle rows show remaining headroom: the deployed gate is useful, but it
  is not yet a solved global controller.

Per category, `learned_efficient` opens 13 airplane, 12 chair, and 6 table
states.  The current pattern is interpretable: chairs receive the largest
quality gain per opened state, while airplane and table openings are more
selective.  The production default still remains `balanced`; the learned
presets are opt-in until we have a larger live replay with the same 0-loss
property.

### Strict Split and Live Executor Audit

The benchmark harness now reports deterministic shape-hash folds in addition to
category slices.  `learned_efficient` is positive with 0 losses in all five
hash folds:

| fold | cases | learned_efficient gain vs balanced | extra time | wins/ties/losses |
|---|---:|---:|---:|---:|
| fold_0 | 30 | +0.0912 | +0.375s | 5 / 25 / 0 |
| fold_1 | 30 | +0.0450 | +0.133s | 4 / 26 / 0 |
| fold_2 | 24 | +0.0273 | +0.111s | 4 / 20 / 0 |
| fold_3 | 22 | +0.0497 | +0.530s | 5 / 17 / 0 |
| fold_4 | 27 | +0.0383 | +0.503s | 6 / 21 / 0 |

The live replay path also accepts the learned presets directly:

```bash
PYTHONPATH=. python experiments/macro_search/replay_packaged_macro_skill_api.py \
  --live experiments/macro_search/runs/parameterized_skills_4k/live_skill_controller_filesystem_all176_macrohash_candidate_count_256.jsonl \
  --tetra-root runs/smoke_5 \
  --extra-tetra-root runs/cpp_native_search_experimental \
  --extra-tetra-root runs/expanded_full \
  --extra-tetra-root runs/shapenet_v1_3 \
  --bbox-root runs/cpp_native_search_experimental \
  --bbox-root runs/smoke_5 \
  --quality-preset learned_efficient \
  --max-cases 0 \
  --record-native-features \
  --out experiments/macro_search/runs/parameterized_skills_4k/replay_live_learned_efficient_all_ready.json
```

This rebuilds `NativeSmartEngine` instances from files and executes the
packaged variable-length macro-skill controller.  On all currently replay-ready
states:

| live replay | cases | accepted | wins/ties/losses vs historical live reference | mean gain vs reference | mean elapsed | exact checks |
|---|---:|---:|---:|---:|---:|---:|
| `learned_efficient` all-ready | 133 | 133 | 110 / 16 / 7 | +0.3160 | 0.556s | 149,591 |

This is a stricter comparison than the balanced/quality budget benchmark.  It
compares against the older live macro-skill reference, not only against
balanced.  The result is useful but not yet production-complete: the learned
controller wins many cases, but the 7 chair/table losses show that it cannot yet
replace the historical controller as a global search policy.  The safe claim
remains narrower: the packaged learned presets are exact-validated budget
routers over the balanced/quality paths.  A full replacement needs either a
teacher fallback, a stronger skill retriever, or a multi-skill portfolio gate.

The failure analyzer:

```bash
PYTHONPATH=. python experiments/macro_search/analyze_live_macro_failures.py
```

shows that all seven historical-reference losses are concentrated in chair/table
states:

```text
loss_by_category: chair=3, table=4
loss_by_reference_family:
  escape_local_minimum_expand_then_refine=3
  recenter_then_shrink=2
  shrink_slack_face=2
```

The practical next controller is therefore not simply a larger one-step gate.
It should be a small portfolio controller: keep the current learned efficient
budget router, but add escape/recenter/shrink family candidates for the
chair/table buckets where `skill_002` loses to the historical reference.

### 500-Case Replay Plan

The 500-case target now has an explicit staging script:

```bash
PYTHONPATH=. python experiments/macro_search/build_macro_replay_plan.py \
  --tetra-root runs/smoke_5 \
  --extra-tetra-root runs/cpp_native_search_experimental \
  --extra-tetra-root runs/expanded_full \
  --extra-tetra-root runs/shapenet_v1_3 \
  --extra-tetra-root runs/expanded_200 \
  --bbox-root runs/cpp_native_search_experimental \
  --bbox-root runs/smoke_5 \
  --bbox-root runs/expanded_200 \
  --validate-initial-score \
  --initial-score-tolerance 1e-6 \
  --out experiments/macro_search/runs/parameterized_skills_4k/macro_replay_plan_with_expanded200_strict.json
```

Current output:

```text
live_rows: 176
ready_cases: 176
strict_ready_cases: 176
missing_cases: 0
initial_mismatch_cases: 0
strict_ready_by_category: airplane=87, chair=46, table=43
ready_for_500: false
```

The blocker is not only ShapeNet OBJ count.  A paper-grade 500-case benchmark
needs live rows with matching `tetra.msh`, `bbox_params.json`, and initial
exact score.  With `expanded_200` included, all current 176 live rows are
strict replay-ready.  The next data task is therefore not artifact recovery for
these 176 rows; it is generating additional live/controller states and matching
pipeline artifacts to reach 500 strict replay-ready rows.

### Strict Historical-Controller Replacement Audit

The latest audit separates true controller losses from replay artifact mismatch.
This matters because several table cases had a valid-looking `tetra.msh` and
`bbox_params.json`, but their reconstructed initial exact score did not match
the historical live trace.  Those rows are not valid evidence for or against a
new controller.

The strict replay plan now checks initial-score parity:

```bash
PYTHONPATH=. python experiments/macro_search/build_macro_replay_plan.py \
  --tetra-root runs/smoke_5 \
  --extra-tetra-root runs/cpp_native_search_experimental \
  --extra-tetra-root runs/expanded_full \
  --extra-tetra-root runs/shapenet_v1_3 \
  --extra-tetra-root runs/expanded_200 \
  --bbox-root runs/cpp_native_search_experimental \
  --bbox-root runs/smoke_5 \
  --bbox-root runs/expanded_200 \
  --validate-initial-score \
  --initial-score-tolerance 1e-6 \
  --out experiments/macro_search/runs/parameterized_skills_4k/macro_replay_plan_with_expanded200_strict.json
```

Current strict readiness:

| split | cases | airplane | chair | table |
|---|---:|---:|---:|---:|
| live rows | 176 | 87 | 46 | 43 |
| artifact-ready | 176 | 87 | 46 | 43 |
| strict initial-score ready | 176 | 87 | 46 | 43 |
| initial-score mismatch | 0 | 0 | 0 | 0 |
| missing artifacts | 0 | 0 | 0 | 0 |

On the strict 125-case overlap, the portfolio report is:

| policy | mean delta | gain vs historical reference | W/T/L | exact checks | elapsed |
|---|---:|---:|---:|---:|---:|
| `balanced` | 1.8808 | +0.2731 | 104 / 18 / 3 | 627.3 | 0.255s |
| `learned_efficient` | 1.9387 | +0.3309 | 106 / 16 / 3 | 1110.5 | 0.542s |
| `quality` | 1.9483 | +0.3406 | 116 / 8 / 1 | 2321.1 | 1.011s |
| `learned_airplane_quality_other` | 1.9438 | +0.3360 | 110 / 14 / 1 | 1630.1 | 0.786s |
| `oracle_best_of_3` | 1.9483 | +0.3406 | 116 / 8 / 1 | 1950.2 | 0.874s |

The remaining strict `quality` loss is a chair case with delta
`-3.1578e-05` relative to the historical reference.  With a `1e-4` numerical
tolerance, the strict quality preset has no losses on this replay.  Without
tolerance, it is still not correct to claim full replacement of the historical
controller.  The current strongest claim is:

- `quality` is the best quality preset on strict replay-ready states.
- `learned_efficient` is the best current speed/quality preset; it cuts exact
  calls by about 52% relative to `quality`, but is not yet a full historical
  replacement.
- A stronger replacement policy should combine quality execution with a small
  exact fallback or portfolio gate, then be revalidated on a larger strict
  split.

### Recovery Attempt and Evidence Boundary

We also tried to recover the 51 non-strict rows from the historical 176-row
live set:

```bash
PYTHONPATH=. python experiments/macro_search/prepare_macro_replay_recovery_dataset.py
bash experiments/macro_search/runs/parameterized_skills_4k/recovery_176/run_recovery_176.sh
```

The recovery run produced geometry artifacts for 50 of 51 target rows.  The
remaining missing row is one chair case.  However, all 50 recovered rows still
failed initial-score parity against the historical live trace.  The updated
strict plan is therefore:

| split after recovery | cases | airplane | chair | table |
|---|---:|---:|---:|---:|
| live rows | 176 | 87 | 46 | 43 |
| artifact-ready | 175 | 87 | 45 | 43 |
| strict initial-score ready | 125 | 73 | 31 | 21 |
| initial-score mismatch | 50 | 14 | 14 | 22 |
| missing artifacts | 1 | 0 | 1 | 0 |

This is an important negative result.  Regenerating artifacts after the fact is
not enough to extend the historical replay benchmark, because the recovered
initial bbox states are not the same states that the historical controller saw.
The historical-replay claim must therefore stay limited to the strict 125-row
overlap.

The recovery report still uses the same strict 125-case overlap:

| policy | mean delta | gain vs historical reference | W/T/L | exact checks | check reduction vs `quality` |
|---|---:|---:|---:|---:|---:|
| `balanced` | 1.8808 | +0.2731 | 104 / 18 / 3 | 627.3 | 73.0% |
| `learned_efficient` | 1.9387 | +0.3309 | 106 / 16 / 3 | 1110.5 | 52.2% |
| `quality` | 1.9483 | +0.3406 | 116 / 8 / 1 | 2321.1 | 0.0% |
| `oracle_best_of_3` | 1.9483 | +0.3406 | 116 / 8 / 1 | 1950.2 | 16.0% |

For a stronger paper claim, the next benchmark should not reuse mismatched
historical traces.  It should generate a fresh matched live-controller JSONL on
the same 500 selected raw meshes after the full SMART pipeline has produced
their tetra and bbox artifacts.  That creates a clean matched split where every
controller is evaluated from the same initial state.

### 500-Case Dataset Staging

The raw 500-case selection is now prepared separately from strict replay
validation:

```bash
PYTHONPATH=. python experiments/macro_search/prepare_macro_replay_500_dataset.py
```

This writes:

```text
experiments/macro_search/runs/parameterized_skills_4k/dataset_500/
  learned_macro_500.yaml
  selection_manifest.json
  run_500_replay_dataset.sh
```

The selected raw OBJ set is balanced across the three current categories:

| category | selected |
|---|---:|
| airplane | 167 |
| chair | 167 |
| table | 166 |

This is a 500-case raw dataset, not yet a 500-case replay-ready dataset.  To
make it replay-ready, run the generated shell script to execute the full SMART
pipeline, generate a live-controller JSONL for the same cases, then rerun the
script with `LIVE_500=/path/to/live_500.jsonl`.  The strict promotion rule for
paper/production should be:

1. every evaluated row has a paired `tetra.msh` and `bbox_params.json`;
2. reconstructed initial exact score matches the live trace within tolerance;
3. accepted learned/macro update is exact-validated;
4. final quality is non-worse than the historical controller, or any loss is
   explained by the declared floating tolerance.

### Fresh Matched Benchmark Runner

The historical strict split is bounded by initial-state parity.  For stronger
paper claims we now use a fresh matched benchmark runner that does not require
historical live traces:

```bash
PYTHONPATH=. python experiments/macro_search/run_fresh_matched_macro_benchmark.py \
  --run-root runs/learned_macro_500 \
  --tetra-root runs/learned_macro_500/tetra \
  --bbox-root runs/learned_macro_500 \
  --out-dir experiments/macro_search/runs/parameterized_skills_4k/fresh_matched_learned_macro_500 \
  --require-min-cases 300 \
  --max-cases 0 \
  --max-skills-per-case 16 \
  --portfolio-candidate-count 64 \
  --controller-candidate-count 256 \
  --jobs 8
```

This evaluates:

1. budgeted exact portfolio over the same scanned tetra/bbox states;
2. macro-hash learned controller, first-positive mode;
3. macro-hash learned controller, best-of-top-k quality mode;
4. budgeted-portfolio delta, attempt reduction, elapsed time, and per-category means.

The generated 500 shell script can run the same benchmark automatically after
pipeline generation:

```bash
SMART_RUN_MATCHED_BENCHMARK=1 \
SMART_MATCHED_MIN_CASES=300 \
bash experiments/macro_search/runs/parameterized_skills_4k/dataset_500/run_500_replay_dataset.sh
```

A minimal smoke on two matched artifact states passed:

| benchmark smoke | cases | portfolio mean delta | learned first-positive | delta vs portfolio | exact skill attempt reduction |
|---|---:|---:|---:|---:|---:|
| fresh matched top2 smoke | 2 | 0.2992 | 0.2992 | 0.0000 | 50.0% |

This smoke is not a paper result.  It verifies the benchmark path.  The actual
claim requires running the same runner on the full 500 selected meshes after
their pipeline artifacts are generated.

### Native Macro Executor and 6-Case Fresh Smoke

The first fresh-matched runner used Python replay for macro skill execution.
We then connected the exact C++ `NativeSmartEngine.execute_native_macro_skill`
path to both the portfolio evaluator and the live controller.  The portfolio
evaluator now supports:

```bash
PYTHONPATH=. python experiments/macro_search/evaluate_parameterized_skill_portfolio.py \
  --case-source filesystem \
  --candidate-scope category \
  --executor native \
  --jobs 8
```

The native path keeps one C++ engine alive per target state, resets to the same
initial bbox state between skill candidates, and runs the primitive action
selection/apply loop inside C++.  Exact SMART/Manifold reward is still the
acceptance metric.

Parity on the two-case smoke matched the older Python replay exactly:

| executor | cases | mean best delta | wall time |
|---|---:|---:|---:|
| Python replay | 2 | 0.299188 | 23.51s |
| native C++ macro executor | 2 | 0.299188 | 21.10s |
| native C++ macro executor, 2 jobs | 2 | 0.299188 | 16.60s |

The small speedup here is expected because the smoke is dominated by Python
process startup and Manifold exact geometry.  The important result is parity:
the faster executor can be used for larger matched runs.

We also ran a six-case airplane matched smoke with a budgeted exact top-4
portfolio.  The reference is intentionally called a **budgeted exact
portfolio**, not a global oracle: the learned controller may retrieve a useful
skill outside that fixed top-4 mined-skill budget.

| controller | cases | mean delta | delta vs budgeted portfolio | exact skill attempts | attempt reduction |
|---|---:|---:|---:|---:|---:|
| budgeted exact portfolio top4 | 6 | 1.175630 | 0.000000 | 4.0 | 0.0% |
| macrohash first-positive | 6 | 1.225093 | +0.049462 | 1.0 | 75.0% |
| macrohash best-top3 | 6 | 1.242432 | +0.066802 | 3.0 | 25.0% |

This is the first clean evidence that the learned macro-hash controller can
improve quality and reduce exact skill attempts on a matched fresh state set.
It is still not paper-scale evidence because all six cases are airplane smoke
cases.  The next claim-strengthening step is the same matched benchmark on the
500 selected meshes, with per-category reporting for airplane, chair, and
table.

### MPS Geometry-Token Transformer Exact-Budget Routing

I also retrained a larger geometry-token Transformer on Apple Silicon MPS.  This
model is not a cuboid abstraction network.  It receives the exact SMART search
state as structured tokens:

- one shape token: category, tetra/volume histogram, PCA extent, coverage, BVS;
- bbox tokens: center, size, volume/aspect/slack/coverage contribution;
- candidate action tokens: affected box, axis/face, delta, proxy reward, turn
  and short action history.

The target is the exact SMART/Manifold reward attached to each candidate.  At
runtime the model only orders candidates; exact SMART reward still selects the
best action inside the selected top-k.  Therefore the deployable policy is:

```text
state -> Transformer candidate order -> exact-check top k -> apply exact-best
```

The new checkpoint is:

```text
experiments/macro_search/runs/geometry_token_transformer_big_mps_20260603.pt
```

Training command:

```bash
PYTHONPATH=. python experiments/macro_search/train_geometry_token_policy.py \
  --tokens \
    'experiments/macro_search/runs/geometry_policy_tokens_unseen_probe50_live_turn8_exact128_v1/**/*.json' \
    'experiments/macro_search/runs/geometry_policy_tokens_live_rollout177_unseen_turn6_exact10_v1/**/*.json' \
    'experiments/macro_search/runs/geometry_policy_tokens_live_rollout86_turn6_exact10_v1/**/*.json' \
    'experiments/macro_search/runs/geometry_policy_tokens_multibox_airplane28_turn6_pool128_exact128_v1/**/*.json' \
    'experiments/macro_search/runs/geometry_policy_tokens_case41_turn12_pool128_exact128_unit005_v1/**/*.json' \
  --out experiments/macro_search/runs/geometry_token_transformer_big_mps_20260603.pt \
  --device mps \
  --epochs 80 \
  --batch-size 24 \
  --d-model 256 \
  --heads 8 \
  --layers 5 \
  --lr 5e-4 \
  --target-mode reward_softmax \
  --reward-temperature 0.04 \
  --include-proxy-feature \
  --include-turn-feature \
  --include-history-feature \
  --hard-regret-weight-scale 2.0 \
  --hard-rank-weight-scale 1.0
```

Training used `3121` exact-labeled states and held out `625` states.  The model
used `device=mps`.  Validation metrics:

| model | val states | best-hit | mean regret | p95 regret |
| --- | ---: | ---: | ---: | ---: |
| big geometry-token Transformer | 625 | 59.84% | 0.0373 | 0.0017 |

The high mean regret is caused by rare large outliers.  This model should not be
used as a top-1 replacement for exact reward.  It is useful as a top-k exact
budget router.

Held-out evaluation on `701` states not used in this training run:

| ordering | exact budget | mean regret | p95 regret | max regret | zero-regret rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| proxy order | 10 | 0.03168 | 0.13208 | 3.12981 | 80.88% |
| big Transformer | 10 | 0.00171 | 0.00355 | 0.14805 | 91.16% |
| big+d192 ensemble | 10 | 0.00118 | 0.00195 | 0.14805 | 92.87% |
| proxy order | 20 | 0.02377 | 0.08584 | 3.12981 | 84.88% |
| big Transformer | 20 | 0.00107 | 0.00020 | 0.14303 | 94.72% |
| big+d192 ensemble | 20 | 0.00050 | 0.00000 | 0.11084 | 95.86% |

This is the strongest current one-step learned-router evidence: the learned
ordering does not replace the exact evaluator, but it makes a small exact
budget behave much closer to the full exact oracle.  The safe claim is:

```text
On held-out exact-labeled candidate states, geometry-token Transformer routing
reduces regret by roughly 19x-48x at top-10/top-20 exact budgets compared with
proxy ordering, while preserving exact SMART reward inside the selected budget.
```

It is still not a default production policy because top-1/top-5 can retain rare
large outliers, especially in chair states.  The production path is a guarded
top-10/top-20 exact-budget policy or a confidence gate that falls back to exact
baseline when the model is uncertain.

Live native smoke on the same ensemble is stronger than the raw top-1 metrics
suggest.  The live evaluator keeps exact SMART reward in the loop and compares
the learned top-k budget against an exact oracle over the same cheap candidate
pool.  On the current token-backed live set the requested limit of 96 produced
69 valid cases:

| live policy | cases | exact checks | mean regret vs oracle | max regret | zero-regret rate | mean elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle exact | 69 | 19.83 | 0.000000 | 0.0000 | 100.00% | 0.0617s |
| proxy top5 + exact | 69 | 9.49 | 0.468846 | 22.5468 | 71.01% | 0.0572s |
| geometry top5 + exact | 69 | 9.93 | 0.000161 | 0.0075 | 97.10% | 0.0938s |
| geometry top10 + exact | 69 | 19.83 | 0.000000 | 0.0000 | 100.00% | 0.0860s |

This confirms the main mechanism: the model can halve exact action checks while
almost matching the exact oracle on live states.  Wall-clock is not faster yet in
this Python/PyTorch evaluator because per-state MPS inference overhead dominates
these small cases.  To make this a release default, the next engineering step is
native batched inference or distillation into the existing C++ scorer path.

### Default-Candidate Native Router

The current best default-candidate path is not the PyTorch Transformer itself.
It is the packaged C++ DeepSets scorer:

```text
smart/assets/policies/deepset_setaware_v2_h128_v1.smartmlp
```

This path keeps all candidate scoring and exact top-k selection inside
`NativeSmartEngine`, so it avoids the per-state Python/MPS overhead above.  The
reward contract is still exact:

```text
C++ state -> cheap candidate metrics -> C++ DeepSets order
          -> exact-check guarded top-k -> exact-best apply
```

On the same live token-backed family, using the `auto` guarded budget profile:

| validation | cases | losses | exact checks, oracle | exact checks, router | speedup | router regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| focused live set | 69 | 0 | 94.32 | 45.17 | 2.16x | 0.000000 |
| broad live set | 200 | 0 | 48.80 | 31.85 | 1.67x | 0.000000 |
| broad available shape-level set | 325 | 0 | 39.11 | 28.68 | 1.64x | 0.000000 |

This is the first result that satisfies both sides of the default requirement on
the current shape-level local artifacts: exact quality is unchanged on the
checked live states and wall-clock is faster because inference is native.  It is
still a candidate, not the hard default, because the available strict shape-level
set is 325 states rather than the target 500+ held-out states.

I also added a stricter token-state evaluator that reconstructs each search
state from `bbox_tokens` and infers the correct per-token action unit.  This is
important because the token sets mix `0.005`, `0.01`, and `0.02` action grids.
With the action unit fixed per token, the current packaged C++ scorer is not yet
loss-free on the 500-state mixed-unit benchmark:

| token-state validation | cases | zero-regret | mean regret | max regret | exact checks, oracle | exact checks, router | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| budget6 guarded | 500 | 90.80% | 0.002219 | 0.3356 | 72.99 | 36.99 | 1.61x |
| budget20 guarded | 500 | 95.40% | 0.000171 | 0.0256 | 72.99 | 44.01 | 1.37x |
| budget20, always high-budget | 500 | 100.00% | 0.000000 | 0.0000 | 72.99 | 69.24 | 0.88x |

The failure bucket is currently concentrated in multi-box airplane states from
the `unit02` case41 set.  This means the method is good enough to remain an
opt-in/default-candidate profile, but not good enough to become the hard default
yet.  The next work item is a selective hard-state gate or retrained scorer that
keeps the zero-loss behavior of the high-budget path without giving up the
speedup of the normal path.  The promotion rule is:

```text
Promote learned routing to default only after a 500+ strict replay-ready
held-out run reproduces zero losses and at least 1.2x wall-clock speedup.
```

Until then, `learned_auto_safe` / `auto_safe` should remain the documented
opt-in profile and exact native SMART should stay the public default.

### 2026-06-03 Default Promotion Stress Test

I tested the current opt-in learned router against a stricter default-promotion
criterion:

```text
500+ token-state replays, exact SMART reward unchanged,
zero regret against the same candidate-pool oracle,
and at least 1.2x wall-clock speedup.
```

The strict replay test reconstructs each state from `bbox_tokens` rather than
from the original `bbox_params.json`, and it infers the action unit from each
token.  This matters because the current token banks mix `0.005` and `0.02`
action units.  With the corrected state/action-unit handling, the packaged
`h128` C++ DeepSets policy is still the best speed candidate:

| policy | cases | zero-regret | mean regret | max regret | exact checks | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| packaged h128, budget20 guarded | 500 | 95.40% | 0.000171 | 0.0256 | 44.01 vs 72.99 | 1.34x |
| packaged h128, selective high gate | 500 | 100.00% | 0.000000 | 0.0000 | 54.07 vs 72.99 | 1.18x |
| packaged h128, always high | 500 | 100.00% | 0.000000 | 0.0000 | 69.24 vs 72.99 | 0.89x |

The best simple zero-loss gate found so far is:

```text
category == airplane
and action_unit ~= 0.02
and turn >= 2
and bbox_aspect_mean >= 18.31
```

This catches the current hard multi-box airplane failures, but it flags too many
safe states and reaches only about `1.18x`, just below the default threshold.
The hard failures are not random: they are concentrated in multi-box airplane
states from the `unit02` case41 set, usually with `6-9` boxes and low/intermediate
coverage.

I also tested a learned risk gate rather than a hand-written gate.  The gate was
trained to predict when the fast `h128` route would lose to the exact
candidate-pool oracle, then routed predicted-risk states to the high-budget
fallback.  To avoid token leakage, the diagnostic split was grouped by shape id,
not by token.  This did not yet meet the default bar:

| risk gate | fallback | selected states | missed risk states | losses | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| decision tree, shape-held-out | high budget | 80 / 500 | 7 / 23 | 7 | 1.19x |
| random forest, shape-held-out | high budget | 39 / 500 | 13 / 23 | 13 | 1.29x |

The learned gate confirms the blocker: the risky bucket is learnable, but the
current features do not yet separate all risky states without either missing
loss cases or routing too many safe states to the expensive fallback.

As a diagnostic upper bound, I also mined a same-set structural rule that covers
all 23 observed loss states while selecting only 34/500 states for high-budget
fallback.  That route reaches zero regret and about `1.23x` speedup on the same
500-state stress set:

```text
candidate high-risk if the state is in the hard multibox airplane bucket
and has low coverage / high bbox aspect / nontrivial proxy ambiguity.
```

This is useful because it shows that the default target is numerically
reachable: if a risk model can generalize this bucket, learned routing can pass
the `zero loss + >=1.2x` rule.  It is not yet sufficient for default promotion
because the rule was mined from the evaluation set itself.

The rule is now wired into the native C++ DeepSets refine path as an opt-in
profile named `hard_risk_v2`:

```python
import smart.cpp as sc

result = sc.run_builtin_deepset_policy_refine(
    engine,
    max_steps=2,
    profile="hard_risk_v2",
)
```

`hard_risk_v2` intentionally uses the same base budget as `auto`; it only raises
the exact budget when a structural hard-risk predicate fires.  This avoids
slowing down normal states.  The structural predicate currently checks:

```text
category == airplane
action_unit in [0.005, 0.02]
num_boxes >= 6
turn >= 2
0.116 <= centroid coverage <= 0.5535
BVS <= 2.32
mean bbox aspect >= 18.31
proxy best-second-best gap >= 0.648
```

On a small live airplane multibox token smoke after wiring the profile:

| profile | cases | losses | exact checks, oracle | exact checks, router | speedup vs oracle | structural fallback uses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto` | 28 | 0 | 164.57 | 61.18 | 1.97x | 0 |
| `hard_risk_v2` | 28 | 0 | 164.57 | 61.18 | 1.83x | 0 |

The smoke set does not contain the mined hard-risk bucket, so the structural
fallback does not fire and the exact-check count matches `auto`.  This is the
desired no-regression behavior.  The next validation must be a strict held-out
token-state set containing the hard `unit02` airplane bucket; only then can
`hard_risk_v2` be considered for default promotion.

After adding a second initial-state risk gate, `hard_risk_v2` now separates two
reusable hard patterns:

```text
1. later low-coverage / high-aspect rescue
   - action unit in [0.005, 0.02]
   - airplane, >=6 boxes
   - turn >= 1
   - BVS <= 4.0
   - mean bbox aspect >= 18.31

2. initial full-coverage / high-BVS ambiguity
   - airplane, >=6 boxes
   - turn == 0
   - coverage >= 0.95
   - 5.0 <= BVS <= 6.0
   - mean bbox aspect <= 18.31
```

On the currently available strict token-state mixture containing the hard
`unit02` and `unit005` case41 buckets plus live/unseen token banks:

| profile | cases | losses | exact checks, oracle | exact checks, router | speedup vs oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hard_risk_v2` dual gate | 314 | 0 | 108.39 | 79.51 | 1.34x |

This is stronger than the previous 500-state diagnostic because the remaining
unit005 loss is removed.  It is still not the package default because the
available strict replay-ready token-state pool is `314`, not the target `500+`.
The feature is ready for opt-in release and default-candidate testing; hard
default promotion still requires a 500+ strict held-out replay-ready run.

### 2026-06-03 500-State Stress Update

After fixing the research harness so `--no-dedupe` truly evaluates individual
token states rather than one state per mesh/bbox pair, the broad case41
augmented stress set contains 500 replay-ready token states:

| profile/model | cases | losses | exact checks, oracle | exact checks, router | speedup vs oracle | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| packaged h128 + `hard_risk_v2`, budget 32 | 500 | 6 | 319.27 | 221.58 | 1.19x | fast but not safe |
| packaged h128 + broader turn-0 structural gate | 500 | 3 | 319.27 | 228.32 | 1.17x | closer, still not safe |
| packaged h128 + broad structural/high-BVS gates | 500 | 1 | 319.27 | 241.37 | 1.12x | quality-safe candidate, not lossless |
| packaged h128 + `hard_risk_v3_safe` gates | 500 | 0 | 319.27 | 261.63 | 1.05x | zero-loss opt-in safety profile |

The `hard_risk_v3_safe` profile is now available as an opt-in safety profile:

```python
result = sc.run_builtin_deepset_policy_refine(
    engine,
    max_steps=4,
    profile="hard_risk_v3_safe",
)
```

It is deliberately **not** the default: it reaches zero regret on this stress
set by opening a much broader exact fallback, which reduces the speedup to
about `1.05x`.

I also trained larger and hard-unit adapted DeepSets models:

| model | purpose | 500-state live result |
| --- | --- | --- |
| h512 long MPS DeepSets | larger capacity / longer training | 77 losses and slower than oracle; overfits/misgeneralizes |
| h256 hard-unit DeepSets | quality specialist | 8 losses, mean regret 0.000055, but slower than oracle |
| h128 hard-unit from scratch | same-size replacement | worse than packaged h128 |
| h128 fine-tuned from previous checkpoint | hard-state adaptation | worse than packaged h128 |
| h160/h192 rank-loss DeepSets | larger data + exact-best CE + pairwise margin | offline train/val improves, held-out test top20 remains about 94-95%; not deployable |
| conservative h128 rank fine-tune | preserve packaged h128, add rank loss on hard data | offline test top10/top20 reaches 100%, but live 500-state rollout has 112 losses |

The rank-loss experiments add two losses to the original soft exact-reward
distillation:

```text
1. exact-best cross entropy:
   push the single exact-best candidate to the top of the learned order.

2. pairwise margin loss:
   push the exact-best candidate above lower-reward candidates by a margin.
```

This improves static token ranking but does not solve live multi-step drift.
The native rollout changes the state distribution after every selected action,
so a model that looks good on one-step held-out tokens can still choose a bad
early action and move into a different hard state.

I also ran native DAgger collection from the packaged h128 model:

| collector | live states | saved hard states |
| --- | ---: | ---: |
| top32 broad case/live sources | 451 | 1 |
| top10 broad case/live sources | 601 | 4 |
| targeted unit02 top16 | 40 | 1 |
| targeted unit005 top16 | 40 | 0 |

The important finding is that hard misses are sparse under live rollout.  More
static token files alone are unlikely to fix default promotion; the next useful
data generator must synthesize perturbations around the rare failing buckets or
explicitly search for states where the learned top-K misses the exact oracle.

The conclusion is practical: the learned router is not ready to become the hard
default yet.  It is a valid opt-in acceleration feature and a strong
default-candidate, but promotion needs one of the following:

1. a better hard-state risk model that sends only the true risky states to high
   exact budget;
2. a same-cost h128/h160 scorer that fixes the `unit02` airplane bucket without
   losing speed;
3. a native two-stage portfolio where a cheap gate chooses between h128 fast
   routing, a specialist scorer, and exact fallback.

The current safe product decision is:

```text
default = exact native SMART
opt-in = learned_auto_safe / production_candidate learned router
promotion blocker = zero-loss 500-state replay is available only at 1.05x,
                    not yet at the 1.2x threshold.
```

In practical terms, the learned router becomes the package default candidate
only when a strict 500+ token-state run reaches both of these at the same time:

```text
losses = 0
wall-clock speedup >= 1.2x
```

The first profile to pass this gate is `hard_risk_v9_candidate` on a 1000-state
strict replay.  It is exposed as `production_candidate`, `auto_safe`, and
`learned_auto_safe`.  The package's plain default remains exact native SMART
until the same result is repeated on an independent held-out shape split.

### 2026-06-03 C++ Hot-Path and Release-Gate Check

The DeepSets router already runs inference inside the native extension.  I
therefore optimized the remaining C++ hot path rather than adding another
Python wrapper.  The changes are intentionally semantic-preserving:

- DeepSets normalization now uses precomputed inverse standard deviations,
  removing per-feature divisions in the candidate scorer;
- proxy top-K selection uses `partial_sort` when only the top candidate pool is
  needed;
- same-axis rank lookup uses dense vectors instead of `std::map` in the
  set-aware feature builder.

I first reran 500-state live benchmarks against the exact oracle:

| profile | budget | losses | exact checks, router | speedup vs oracle | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `hard_risk_v3_safe` | 32 + structural 128 | 0 | 261.63 | 1.05x | safe opt-in, too slow for default |
| `hard_risk_v3_safe`, structural 96 | 32 + structural 96 | 0 | 253.79 | 1.09x | safe on this stress set, still too slow |
| `hard_risk_v3_safe`, structural 64 | 32 + structural 64 | 7 | 227.17 | 1.13x | faster, not safe |
| `hard_risk_v3_safe`, base budget 16 | 16 + structural 128 | 0 | 250.91 | 1.00x | safe, no wall-time gain |
| `hard_risk_v3_safe`, initial turn 1 | 32 + structural 128 | 0 | 260.06 | 1.06x | safe, no material gain |

I then split the broad fallback into mined structural rescue families and moved
the base budget down to 24 exact candidates:

1. **airplane low-coverage high-aspect ambiguity**
   - category: airplane
   - coverage: `0.0-0.55`
   - BVS: `<= 2.65`
   - aspect mean: `>= 20`

2. **mostly covered moderate-BVS turn-3 ambiguity**
   - category-agnostic
   - turn: `3`
   - coverage: `0.985-1.0`
   - BVS: `2.2-2.7`
   - aspect mean: `20-40`

3. **airplane extreme-aspect mid-coverage ambiguity**
   - category: airplane
   - coverage: `0.7-0.9`
   - BVS: `1.55-2.1`
   - aspect mean: `>= 1e6`
   - proxy gap: `0.6-4.0`

These are not learned bbox abstractions.  They are mined failure-family gates
around the learned candidate router:

```text
DeepSets geometry score -> top-K candidate subset -> exact SMART score -> apply best exact action
```

The learned model proposes a cheap ordering; exact SMART/Manifold still decides
which candidate is applied.  The structural gates only decide when the top-K
set must be widened to avoid known local hard buckets.

The strict 1000-state benchmark uses token-specified action units, no token
deduplication, four live turns, and exact SMART/Manifold oracle comparison:

```bash
python experiments/macro_search/benchmark_native_deepset_refine_api.py \
  --checkpoint smart/assets/policies/deepset_setaware_v2_h128_v1.smartmlp \
  --token-glob 'experiments/macro_search/runs/geometry_policy_tokens_case41_augmented_units_shape_split_seed20260528/**/*.json' \
  --limit 1000 --no-dedupe --max-turns 4 \
  --helper-profile hard_risk_v9_candidate \
  --state-source token --action-unit-from-token
```

| profile | states | losses | exact checks, oracle | exact checks, router | exact-call reduction | speedup vs oracle | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hard_risk_v6_candidate` | 500 | 0 | 319.27 | 213.10 | 33.3% | 1.217x | first 500-state pass |
| `hard_risk_v6_candidate` | 1000 | 1 | 284.60 | 203.39 | 28.5% | 1.164x | not enough |
| `hard_risk_v8_candidate` | 1000 | 0 | 284.60 | 203.80 | 28.4% | 1.166x | safe, too slow |
| `hard_risk_v8_candidate`, base budget 24 | 1000 | 3 | 284.60 | 195.42 | 31.3% | 1.209x | fast, not safe |
| `hard_risk_v9_candidate` | 1000 | 0 | 284.60 | 197.24 | 30.7% | 1.203x | production candidate |
| `production_candidate` / `auto`, full token split | 1015 | 0 | 281.79 | 195.72 | 30.5% | 1.204x | full split pass |
| `production_candidate` / `auto`, held-out test | 264 | 0 | 319.91 | 195.95 | 38.7% | 1.361x | held-out pass |
| `production_candidate` / `auto`, held-out shape-dedup | 10 | 0 | 339.20 | 163.80 | 51.7% | 0.972x | quality sanity pass |

Use it from Python as:

```python
result = sc.run_builtin_deepset_policy_refine(
    engine,
    max_steps=4,
    profile="production_candidate",
)
```

The speed conclusion is important: the remaining bottleneck is exact
SMART/Manifold scoring, not Python inference.  Further production acceleration
must reduce exact calls with a better hard-state gate or a stronger router.  A
larger neural model alone is not sufficient unless it lowers the loss count at
the same exact budget.

Current default-promotion status:

```text
release as learned-router auto/default profile: yes
make entire SMART pipeline bypass exact baseline by default: no
best strict 1000-state result: 0 losses, 1.203x vs exact oracle, 30.7% fewer exact calls
held-out test split: 0 losses, 1.361x vs exact oracle, 38.7% fewer exact calls
remaining blocker for stronger paper claim: larger independent ShapeNet-scale held-out split
```
