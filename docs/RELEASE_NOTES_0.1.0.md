# SMART 0.1.0 Release Notes

First public `smart-bbox` release candidate for reproducing SMART 3DV 2024
results and using the pipeline as a Python package.

## Highlights

- Package name: `smart-bbox`; import name: `smart`.
- Main commands:
  - `smart`
  - `smart-cpp-native`
  - `smart-quickstart`
  - `smart-release-preflight`
- Native C++ release path:
  - packaged `smart._cpp` extension;
  - packaged `smart/bin/smart-cpp-native` executable;
  - bundled fixed-Manifold `pymanifold` runtime under
    `smart/pymanifold_runtime`;
  - Python remains CLI/API/config/package orchestration.
- C++ native pipeline coverage:
  - OBJ normalization;
  - Gmsh `.msh` inspection and tet utilities;
  - CoACD/BSP part-to-tet partition metadata;
  - SMART BAVF merge;
  - greedy refine;
  - MCTS;
  - one-mesh `run-pipeline`;
  - dataset `discover-meshes`, `run-batch`, and `batch-summary`.
- Default release engine is `cpp_native`; `legacy_python` remains available
  for debugging and comparison.
- Fixed vendored Manifold source is kept authoritative and unchanged.

## Installation

After PyPI publication:

```bash
python -m pip install "smart-bbox[pipeline]"
smart --config configs/smoke_5.yaml doctor
smart-cpp-native --help
```

From a source checkout:

```bash
python -m pip install -e ".[pipeline]"
smart --config configs/smoke_5.yaml build-tools
smart --config configs/smoke_5.yaml doctor
```

## Reproduction Smoke

```bash
smart-quickstart \
  --config configs/smoke_5.yaml \
  --run-smoke \
  --category airplane
```

For direct C++ batch execution:

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

## Local Verification

The current local Apple Silicon release candidate was verified on
May 20, 2026:

- artifacts:
  - `/private/tmp/smart_pkg_ready/smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl`
  - `/private/tmp/smart_pkg_ready/smart_bbox-0.1.0.tar.gz`
- `python3 -m pytest -q`: `285 passed`;
- `smart-release-preflight --recreate-venv`: passed;
- release artifact audit: passed;
- `twine check`: passed;
- fresh-venv wheel install: passed;
- installed console-script smoke: passed;
- source `smart --config configs/smoke_5.yaml doctor --json`: passed.

## Known Limits

- External Mesh2Tet/fTetWild and CoACD binaries are still required for full
  mesh-to-box pipeline reproduction.
- GPU/Metal acceleration is not enabled. Manifold CUDA support is NVIDIA-only;
  local Apple Silicon builds use CPU.
- Windows wheels are not part of the first release matrix.
- Learned policy/gate assets are kept out of the public wheel and remain
  local-only under `experiments/`.
- ShapeNet data is not packaged.

## Release Action

Before publishing:

1. Confirm PyPI Trusted Publishing is configured for package `smart-bbox`,
   environment `pypi`, workflow `.github/workflows/wheels.yml`.
2. Run `smart-release-preflight --dist-dir /private/tmp/smart_pkg_ready --venv-dir /private/tmp/smart_pkg_ready_venv --recreate-venv`.
3. Push a version tag after committing the release state:

   ```bash
   git tag -a v0.1.0 -m "SMART 0.1.0"
   git push origin v0.1.0
   ```

4. Confirm the GitHub `Wheels` workflow builds all wheels, audits artifacts,
   runs installed smoke, and publishes to PyPI.
