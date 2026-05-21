# SMART Quickstart

This guide is the shortest path for a new user to install SMART, verify the
native C++ package, and reproduce a small paper-style run.

For the full source tree and package/local-only directory layout, see
[`REPOSITORY_STRUCTURE.md`](REPOSITORY_STRUCTURE.md).

## 1. Install

From a released wheel:

```bash
python -m pip install "smart-bbox[pipeline]"
```

From a source checkout:

```bash
git clone https://github.com/chpark1111/SMART.git
cd SMART
python -m pip install -e ".[pipeline]"
```

The package exposes:

- `smart`: Python CLI/API/config wrapper;
- `smart-cpp-native`: packaged C++ native SMART executable;
- `smart._cpp`: native C++ extension used by the Python API.

Python is intentionally still used for packaging, YAML configs, manifests, and
high-level orchestration. SMART's default algorithm path is `engine=cpp_native`.

## 2. Verify The Install

```bash
smart --config configs/smoke_5.yaml doctor
smart-cpp-native --help
smart-smoke-console-scripts
```

If `smart` is not on `PATH`, use:

```bash
python -m smart --config configs/smoke_5.yaml doctor
```

## 3. Build External Reproduction Tools

For full paper reproduction from meshes, SMART needs Mesh2Tet tools and the
fixed Manifold runtime. In a source checkout:

```bash
smart --config configs/smoke_5.yaml build-tools
```

This prepares ManifoldPlus, fTetWild, CoACD source setup, and the fixed
vendored Manifold binding. The fixed Manifold source is authoritative and must
not be pulled, replaced, or rewritten.

Prebuilt binaries can be supplied instead:

```bash
export SMART_MANIFOLDPLUS_BIN=/path/to/ManifoldPlus/build/manifold
export SMART_FTETWILD_BIN=/path/to/fTetWild/build/FloatTetwild_bin
export SMART_COACD_BIN=/path/to/coacd
export SMART_MANIFOLD_PYTHON=/path/to/smart/vendor/manifold/build/bindings/python
```

## 4. Prepare ShapeNet Data

SMART expects:

```text
data/shapenet_airplane/<model_id>/model.obj
data/shapenet_chair/<model_id>/model.obj
data/shapenet_table/<model_id>/model.obj
```

For the three paper categories, the ShapeNet synsets are:

```text
airplane  02691156
chair     03001627
table     04379243
```

If you have the zipped category archives:

```bash
python scripts/prepare_shapenet_samples.py \
  --archive-dir /path/to/shapenet_zips \
  --output-root data/expanded \
  --categories airplane chair table \
  --limit 100000 \
  --normalize preserve
```

Then verify:

```bash
smart --config configs/expanded_full.yaml check-data
```

SMART normalizes into `runs/<profile>/normalized/`; it does not mutate the
downloaded ShapeNet meshes.

For a tiny source-checkout example using existing local `data/` meshes:

```bash
bash examples/prepare_sample_shapes.sh
bash examples/run_example_3x3.sh
```

## 5. Run A Smoke Reproduction

The easiest source-checkout helper is:

```bash
python scripts/quickstart_reproduce.py \
  --config configs/smoke_5.yaml \
  --build-tools \
  --build-cpp \
  --run-smoke \
  --category airplane
```

After installation from a wheel, the same helper is available as:

```bash
smart-quickstart \
  --config configs/smoke_5.yaml \
  --run-smoke \
  --category airplane
```

For direct C++ execution without the Python stage loop:

```bash
smart-cpp-native run-batch \
  --data_root data \
  --categories shapenet_airplane,shapenet_chair,shapenet_table \
  --limit_per_category 1 \
  --output_root runs/native_batch \
  --manifoldplus_bin external/mesh2tet/ManifoldPlus/build/manifold \
  --ftetwild_bin external/mesh2tet/fTetWild/build/FloatTetwild_bin \
  --coacd_bin external/CoACD/python/package/bin/coacd \
  --jobs auto \
  --partition_threads auto \
  --reuse_existing \
  --resume_success

smart-cpp-native batch-summary \
  --manifest runs/native_batch/native_pipeline.jsonl
```

Use `batch-summary` after every optimization run. It reports success/failure
counts, reuse counts, the slowest mesh, and the slowest stage; it also accepts
older `smart native-run` manifests by finding
`output_path/native_pipeline_stats.json` automatically.

## 6. Tune Parameters

All major pipeline parameters are config-driven:

```bash
smart --config configs/demo.yaml run \
  --set tetra.airplane.epsilon=0.002 \
  --set tetra.airplane.edge_length=0.1 \
  --set refine.max_step=2000 \
  --set mcts.mcts_iter=3000 \
  --set render.transparent=true
```

Common profiles:

- `configs/example_3x3.yaml`: 3 local example meshes per paper category;
- `configs/smoke_5.yaml`: fast local smoke;
- `configs/demo.yaml`: small demo data;
- `configs/paper_like.yaml`: closer paper-style settings;
- `configs/expanded_full.yaml`: larger prepared ShapeNet layout.

## 7. Build Release Artifacts

Maintainers can build and validate a local release candidate with:

```bash
smart-release-preflight \
  --dist-dir /private/tmp/smart_release_check \
  --venv-dir /private/tmp/smart_release_venv \
  --recreate-venv \
  --run-asan-smoke
```

This builds wheel/sdist, audits packaged native files, runs `twine check`,
installs the wheel into a fresh venv, verifies console scripts, and runs a
short AddressSanitizer smoke for the native executable.

## Current Performance Notes

On the local Apple Silicon smoke runs, the monolithic C++ path improved a
single airplane smoke from about `23.8s` to `8.9s` and a three-category
sequential smoke from about `48.3s` to `30.2s`. Native batch execution with
`--jobs auto` is the recommended dataset-scale path. Remaining wall time is
mostly external tetrahedralization/pre-segmentation and exact boolean geometry,
not Python loop overhead.
