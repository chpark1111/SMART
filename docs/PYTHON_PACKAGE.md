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
python -m smart --config configs/smoke_5.yaml build-cpp
```

From a release wheel:

```bash
python -m pip install "smart-bbox[pipeline]"
```

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

