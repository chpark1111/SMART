# SMART Python Package

The Python package is the public wrapper around the native C++ SMART core. It
provides config loading, CLI/API entrypoints, manifests, rendering helpers, and
release packaging. The SMART merge/refine/MCTS core is exposed through
`smart._cpp` and the `smart-cpp-native` executable.

## Install

From source:

```bash
python -m pip install -e ".[pipeline]"
python -m smart --config configs/smoke_5.yaml build-tools
```

`build-tools` prepares Mesh2Tet tools, CoACD CLI runtime, the fixed Manifold
runtime, and the local SMART C++ extension/executable. `build-cpp` is only
needed later if you want to rebuild the SMART wrapper without rebuilding the
external tools. The setup is idempotent: SMART probes an existing CoACD CLI
before trying slow source installation, and falls back to the PyPI CoACD runtime
if the source editable install is not usable on the local platform.

From a release wheel:

```bash
python -m pip install "smart-bbox[pipeline]"
```

The wheel includes the Python package, public configs, the native SMART C++
extension, and `smart-cpp-native`. The `[pipeline]` extra installs Python
dependencies such as CoACD, trimesh, scipy/sklearn, and torch. Full raw-mesh
reproduction still needs Mesh2Tet/fTetWild/ManifoldPlus runtime tools and
data. Prepare those external C++ tools in a writable location:

```bash
export SMART_TOOLS_ROOT="$PWD/.smart-tools"
smart --config smoke_5.yaml build-tools
```

This keeps `pip install` reliable: package installation does not download and
compile large external geometry repositories, but the follow-up `build-tools`
command makes that setup a single SMART-managed step.

Users do not need to manually clone ManifoldPlus, fTetWild, CoACD, or the fixed
Manifold runtime. `smart build-tools` owns that setup, and existing binaries can
still be supplied through `SMART_MANIFOLDPLUS_BIN`, `SMART_FTETWILD_BIN`,
`SMART_COACD_BIN`, and `SMART_MANIFOLD_PYTHON`.

## CLI

```bash
smart --config configs/smoke_5.yaml doctor
smart --config configs/smoke_5.yaml run
smart --config configs/smoke_5.yaml summary
smart --config configs/smoke_5.yaml evaluate
smart --config configs/smoke_5.yaml render
```

Native executable:

```bash
smart-cpp-native run-pipeline \
  --input data/shapenet_airplane/<model_id>/model.obj \
  --work_dir runs/native_one/<model_id> \
  --manifoldplus_bin external/mesh2tet/ManifoldPlus/build/manifold \
  --ftetwild_bin external/mesh2tet/fTetWild/build/FloatTetwild_bin \
  --coacd_bin external/CoACD/python/package/bin/coacd \
  --epsilon 0.002 \
  --edge_length 0.1 \
  --refine_max_step 2000 \
  --mcts_iter 3000
```

## API

```python
import smart

cfg = smart.load_config("configs/smoke_5.yaml")
print(smart.doctor(cfg))
records = smart.run(cfg)
metrics = smart.evaluate(cfg, stage="mcts")
```

Run one mesh through the native executable from Python:

```python
import smart

record = smart.run_native_pipeline(
    input_mesh="data/shapenet_airplane/<model_id>/model.obj",
    work_dir="runs/native_one/<model_id>",
    manifoldplus_bin="external/mesh2tet/ManifoldPlus/build/manifold",
    ftetwild_bin="external/mesh2tet/fTetWild/build/FloatTetwild_bin",
    coacd_bin="external/CoACD/python/package/bin/coacd",
    epsilon=0.002,
    edge_length=0.1,
    refine_max_step=2000,
    mcts_iter=3000,
)
print(record)
```

Opt-in learned candidate routing is available for research runs that want to
reduce exact action checks inside native refine. The bundled DeepSets router
orders candidate actions, but SMART still exact-scores the checked candidates
with the native Manifold reward before applying an action:

```python
import smart.cpp as sc

engine = sc.NativeSmartEngine(...)
print(sc.native_deepset_route_diagnostics(engine, profile="auto"))
result = sc.run_builtin_deepset_policy_refine(
    engine,
    max_steps=200,
    profile="auto",
)
```

For the current research portfolio, let SMART choose the exact-native or
learned-router route based on box count:

```python
result = sc.run_builtin_deepset_portfolio_refine(engine, mode="speed")
```

`mode="speed"` uses native exact C++ refine for one-box states and learned
`auto` routing for multibox states.  `mode="balanced"` uses native exact C++
refine for one-box states and learned `hard` routing for multibox states.

The same packaged policy can be used as a native MCTS action prior:

```python
mcts = sc.run_builtin_deepset_prior_mcts(
    engine,
    mode="guarded",
    transposition_table=True,
)
```

Available MCTS-prior modes are `speed`, `balanced`, `quality`, `frontier`, and
`guarded`.  They are research presets that prune/order MCTS actions with the
packaged DeepSets scorer while final state rewards still come from exact native
SMART/Manifold scoring.  `guarded` is the recommended research mode: it uses the
aggressive frontier route when the state looks reliable and falls back to exact
shallow MCTS for low-confidence multibox states.

The same route can be enabled in YAML:

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

The repository also ships `configs/learned_frontier.yaml`, which applies this
profile to the packaged three-category example layout.

For deeper research there is a state-refreshed node-prior variant:

```python
mcts = sc.run_builtin_deepset_dynamic_prior_mcts(engine, num_iter=100)
```

This refreshes DeepSets candidate scores inside the C++ MCTS tree when each
node is created.  It is currently research-only: on the 114-state local check,
dynamic top6/top15 was safe but slower than the tuned static-root prior, and
dynamic top4/top15 introduced quality-loss cases.

This is also opt-in research code.  In local probes, top6 was safe for hard
airplane multibox states, while one-box states needed top15/top20 to avoid
quality loss.  On a combined 114-state local check, the tuned box-count top-k
rule cut MCTS wall time by 70.3% with zero quality-loss cases at fixed MCTS
budget.  With budget reinvestment, the current strongest validation preset is
`mode="guarded"`: multibox top1, one-box top15, depth4, 25 iterations, and
`transposition_table=True`, with an exact fallback for multibox states whose
initial score is above `-0.5`.  Against exact MCTS 50/depth2 it preserves the
frontier gain on the 114-state pool and removes the quality-loss cases observed
on the larger unseen multibox probe.
The fallback uses exact depth2/35 iterations, with a faster 30-iteration tier
for very near-zero risky states.
Passing `transposition_table=True` gave a small additional runtime gain in the
same local check without changing quality, but the main acceleration came from
DeepSets top-k pruning.

The same router is also wired into the normal pipeline as an opt-in refine
setting:

```yaml
refine:
  backend: cpp_native
  learned_router:
    enabled: true
    policy: default
    profile: auto
    overrides: {}
```

or from the CLI:

```bash
smart --config configs/smoke_5.yaml refine \
  --set refine.learned_router.enabled=true \
  --set refine.learned_router.profile=auto
```

The release default remains exact native SMART. Use the learned router only
when you are benchmarking the acceleration/quality tradeoff.
The bundled `auto` profile sends small one-box states to exact native refine and
uses the learned router on larger multibox states with a conservative adaptive
exact-check budget. It also exact-scores small candidate pools to avoid known
low-pool routing misses inside the C++ router path. For more aggressive
research runs, use `profile="hard"` or override `small_pool_exact_threshold`
after validating on your split.

Built-in learned-router profiles:

| profile | intent |
| --- | --- |
| `auto` | safest opt-in profile; exact route for one-box states and small candidate pools |
| `mixed` | balanced research profile for mixed-category multibox validation |
| `hard` | faster hard-case research profile; fewer small-pool exact fallbacks |
| `fast` | most aggressive probe profile; use only when measuring quality drift |

Current local validation snapshot for the bundled `default` policy
(`max_turns=4`):

| split | profile | quality | exact-call change | wall-time change |
| --- | --- | --- | --- | --- |
| unseen probe50, 120 states | `auto` | zero regret, no oracle-loss cases | 67.1% fewer exact checks | 1.38x vs oracle pool |
| mixed case41, 39 states | `auto` | zero regret, no oracle-loss cases | 55.5% fewer exact checks | 1.59x vs oracle pool |
| hard airplane multibox, 28 states | `hard` | zero regret, no oracle-loss cases | 79.9% fewer exact checks | 2.02x vs oracle pool |

Treat these numbers as a research baseline, not a paper metric.  Re-run the
benchmark on your category split before promoting a learned-router profile.
See [`docs/LEARNED_ROUTER.md`](LEARNED_ROUTER.md) for the current research
summary and quality reinvestment results.

The same saved exact-call budget can be reinvested into more refinement turns.
In local 4-turn exact baseline probes, hard airplane states improved by +0.4010
mean score with 58.2% fewer exact checks and 18.4% less wall time using hard
router 6-turn.  The mixed case41 split improved by +0.3449 mean score with
55.2% fewer exact checks and 9.6% less wall time using hard router 6-turn.  A
more conservative auto router 5-turn mixed run improved by +0.1817 mean score
with 40.6% fewer exact checks and 20.8% less wall time.  This is still research
evidence, so it remains opt-in.

Latest portfolio validation separates learned and exact-native routes.  On 28
hard airplane multibox states, `mode="speed"` selected learned `auto` 5-turn
and improved mean score by +0.8821 with 25.3% fewer exact checks and 3.3%
lower wall time.  `mode="balanced"` selected learned `hard` 6-turn and improved
mean score by +1.0744 with 45.1% fewer exact checks, but was 4.1% slower.  On
86 one-box live states the portfolio uses native exact C++ refine, not the
learned router, because learned-only routing was slower there.

## Config Profiles

Packaged public profiles:

- `demo.yaml`
- `example_3x3.yaml`
- `expanded_full.yaml`
- `paper_like.yaml`
- `smoke_5.yaml`

List installed profiles:

```bash
smart configs --json
```

Every important pipeline parameter can be changed through YAML or `--set`:

```bash
smart --config configs/smoke_5.yaml \
  --set mcts.mcts_iter=100 \
  --set refine.max_step=200 \
  run
```

## Compatibility

The top-level `pymesh.py` file is not the external PyMesh package. It is a small
compatibility shim so required legacy SMART modules that still do `import
pymesh` route to `smart.pymesh_compat`. New code should import
`smart.pymesh_compat` directly.

Experimental RL priors, pruning gates, and benchmark configs are kept in the
local ignored `experiments/` tree, not in release wheels.
