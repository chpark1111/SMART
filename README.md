# SMART: Split, Merge, and Refine

Official implementation of **Split, Merge, and Refine: Fitting Tight Bounding
Boxes via Over-Segmentation and Iterative Search**, 3DV 2024.

![SMART teaser](docs/teaser.png)

[Paper](https://arxiv.org/pdf/2304.04336) |
[arXiv](https://arxiv.org/abs/2304.04336) |
[Quickstart](docs/QUICKSTART.md) |
[Pipeline](docs/PIPELINE.md) |
[Package Docs](docs/PYTHON_PACKAGE.md)

[Chanhyeok Park](https://chpark1111.github.io/) and
[Minhyuk Sung](https://mhsung.github.io/)

SMART fits a compact set of tight 3D bounding boxes to a mesh without human
supervision. The package uses a Python CLI/API with a native C++ SMART core and
the fixed vendored Manifold backend.

## Highlights

- Paper-style pipeline: normalize mesh, tetrahedralize, pre-segment, merge,
  refine, run MCTS, render, and evaluate.
- Native C++ backend for the SMART core through `smart._cpp` and
  `smart-cpp-native`.
- Fixed vendored Manifold source is kept unchanged for exact geometry
  operations.
- Python API and CLI for reproducible experiments and package use.
- ShapeNet airplane/chair/table reproduction configs are included.

## Installation

Install the package:

```bash
python -m pip install "smart-bbox[pipeline]"
```

Install from source:

```bash
git clone https://github.com/chpark1111/SMART.git
cd SMART
python -m pip install -e ".[pipeline]"
```

Verify the install:

```bash
smart --config configs/smoke_5.yaml doctor
smart-cpp-native --help
```

For the complete install and reproduction path, see
[`docs/QUICKSTART.md`](docs/QUICKSTART.md).

## Local Example

If this checkout already has local ShapeNet meshes under `data/`, create a
3-per-category example set and run it:

```bash
bash examples/prepare_sample_shapes.sh
bash examples/run_example_3x3.sh
```

The example meshes are copied to `examples/sample_shapes/`, which is ignored by
git and excluded from packages.

## Data

SMART expects ShapeNet-style mesh folders:

```text
data/shapenet_airplane/<model_id>/model.obj
data/shapenet_chair/<model_id>/model.obj
data/shapenet_table/<model_id>/model.obj
```

Paper category synsets:

```text
airplane  02691156
chair     03001627
table     04379243
```

Prepare zipped category archives:

```bash
python scripts/prepare_shapenet_samples.py \
  --archive-dir /path/to/shapenet_zips \
  --output-root data/expanded \
  --categories airplane chair table \
  --limit 100000 \
  --normalize preserve
```

SMART writes normalized meshes under `runs/<profile>/normalized/`; it does not
modify the downloaded meshes in `data/`.

## Build External Tools

Full reproduction from raw meshes requires Mesh2Tet tools, CoACD, and the fixed
Manifold runtime. In a source checkout:

```bash
smart --config configs/smoke_5.yaml build-tools
smart --config configs/smoke_5.yaml build-cpp
```

Prebuilt binaries can also be supplied:

```bash
export SMART_MANIFOLDPLUS_BIN=/path/to/ManifoldPlus/build/manifold
export SMART_FTETWILD_BIN=/path/to/fTetWild/build/FloatTetwild_bin
export SMART_COACD_BIN=/path/to/coacd
export SMART_MANIFOLD_PYTHON=/path/to/smart/vendor/manifold/build/bindings/python
```

## Run SMART

Smoke run through the Python pipeline:

```bash
smart --config configs/smoke_5.yaml run
smart --config configs/smoke_5.yaml summary
```

Run one mesh through the native C++ executable:

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

Run a native batch:

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
  --reuse_existing \
  --resume_success

smart-cpp-native batch-summary \
  --manifest runs/native_batch/native_pipeline.jsonl
```

## Evaluate And Render

Evaluate bbox outputs with the paper metrics:

```bash
smart --config configs/smoke_5.yaml evaluate
```

Render final boxes:

```bash
smart --config configs/smoke_5.yaml render \
  --set render.transparent=true \
  --set render.joint_mesh=false
```

The default renderer is the packaged software preview renderer so macOS does
not launch Blender during normal runs. The adapted paper Blender renderer is
still packaged under `smart/legacy/renderer` and can be enabled explicitly:

```bash
smart --config configs/smoke_5.yaml --set render.backend=blender render
```

## Python API

```python
import smart

cfg = smart.load_config("configs/smoke_5.yaml")
records = smart.run(cfg)
print(records[-1])
```

Package/API details are in [`docs/PYTHON_PACKAGE.md`](docs/PYTHON_PACKAGE.md).

## Repository Layout

```text
smart/        Python package, CLI/API, configs, pipeline wrappers
cpp/          Native C++ SMART core and smart-cpp-native executable
configs/      Source-checkout YAML profiles
examples/     Public shell examples; local sample meshes are ignored
scripts/      Supported data prep, release, and reproduction utilities
tests/        Package/native/release tests
docs/         User docs, paper assets, and release notes
experiments/  Ignored local research configs, scripts, assets, and tests
data/         Local ShapeNet data only; not packaged
runs/         Local outputs only; not packaged
external/     Downloaded Mesh2Tet/CoACD tools; not packaged
past_codes/   Original research archive; reference only
```

See [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md) for more
detail.

## Configs

Recommended public configs:

- `configs/smoke_5.yaml`: fast local smoke test.
- `configs/example_3x3.yaml`: 3 meshes per paper category from local example data.
- `configs/demo.yaml`: small demo profile.
- `configs/paper_like.yaml`: paper-style parameters.
- `configs/expanded_full.yaml`: larger local ShapeNet layout.

Experimental RL, pruning, and acceleration profiles are local-only under
`experiments/configs/`; they are ignored by git and excluded from release
packages.

## Compatibility Notes

`pymesh.py` is a compatibility shim, not the external PyMesh package. It keeps
legacy SMART code that imports `pymesh` working by forwarding to
`smart.pymesh_compat`. New code should import `smart.pymesh_compat` directly.

## Release

Build and validate release artifacts:

```bash
smart-release-preflight \
  --dist-dir /private/tmp/smart_release_check \
  --venv-dir /private/tmp/smart_release_venv \
  --recreate-venv \
  --run-asan-smoke
```

Release notes and publishing steps are in [`docs/RELEASE.md`](docs/RELEASE.md).

## Citation

```bibtex
@inproceedings{park2024smart,
  title = {Split, Merge, and Refine: Fitting Tight Bounding Boxes via Over-Segmentation and Iterative Search},
  author = {Park, Chanhyeok and Sung, Minhyuk},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2024}
}
```

## License

This project is released for non-commercial research under
CC BY-NC-SA 4.0. See [`LICENSE`](LICENSE).
