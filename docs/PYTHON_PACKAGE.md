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
