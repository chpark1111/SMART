# SMART Repository Structure

This repository is organized around a packaged Python interface with a native
C++ SMART core. Python remains responsible for CLI/API/config orchestration and
packaging. The algorithm hot path and exact Manifold-backed helpers live in C++.

## Source Tree

```text
SMART/
├── smart/                  # Installable Python package
│   ├── cli.py              # `smart` command line entrypoint
│   ├── api.py              # Public Python API
│   ├── pipeline/           # YAML config, manifests, and stage orchestration
│   ├── cpp.py              # C++ build/integration helpers
│   ├── native.py           # Python wrapper for `smart._cpp`
│   ├── native_runner.py    # Native executable orchestration
│   ├── native_compat.py    # Native/fallback compatibility facade
│   ├── legacy/             # Required adapted legacy runtime and renderer
│   ├── vendor/manifold/    # Fixed Manifold source; keep unchanged
│   ├── configs/            # Config files bundled into wheels
│   └── bin/                # Packaged native executable location
├── cpp/                    # Native C++ SMART implementation
│   ├── smart_native_core.* # Geometry, bbox, tet, merge, and search helpers
│   ├── smart_native_engine.* # Stateful native SMART engine
│   ├── manifold_bridge.cpp # Bridge to the fixed Manifold C++ source
│   ├── smart_cpp_module.cpp # Python extension module
│   └── smart_native_cli.cpp # `smart-cpp-native` executable
├── configs/                # Source-checkout config profiles
├── examples/               # Public shell examples; local sample meshes ignored
├── scripts/                # Supported maintainer/release/reproduction tools
├── experiments/            # Local ignored research configs, scripts, assets, tests
├── tests/                  # Pytest suite for package/native behavior
├── docs/                   # User docs, release notes, and paper assets
├── cmake/                  # Build compatibility helpers
├── patches/                # External tool patches used by setup/build scripts
├── setup.py                # Package build entrypoint
├── setup_cpp.py            # Native C++ extension/executable build logic
├── pyproject.toml          # Python package metadata
├── MANIFEST.in             # Source distribution include/exclude rules
└── pymesh.py               # Lightweight compatibility shim for legacy imports
```

## Local-Only Directories

These paths are useful in a source checkout but are intentionally excluded from
release wheels and normal git commits.

```text
data/                    # Local ShapeNet samples and prepared datasets
runs/                    # Pipeline outputs, benchmarks, traces, renders
external/mesh2tet/       # Downloaded/built Mesh2Tet tools
external/CoACD/          # Downloaded/built CoACD source/binaries
past_codes/              # Original research code archive for reference only
experiments/             # Local benchmark/RL/trace-mining scripts; ignored
examples/sample_shapes/   # Local copied example meshes; ignored
examples/runs/            # Local example outputs; ignored
smart/vendor/manifold/build/
build/, dist/, *.egg-info/, target/, __pycache__/
```

`data/README.md` documents the expected mesh layout. SMART normalizes inputs
into `runs/<profile>/normalized/`; it should not mutate downloaded ShapeNet
meshes in `data/`.

Detailed development logs and RL/acceleration research notes are local-only
under `experiments/docs/`. The root README is intentionally kept short for
public paper-repo readability.

## Official Backend Direction

- `engine=cpp_native` is the intended release backend.
- `legacy_python` remains available for debugging and parity checks.
- Rust is no longer part of the official source tree or package path.
- The fixed Manifold source under `smart/vendor/manifold` is authoritative and
  should not be pulled, replaced, or rewritten.

## Build Artifacts

Do not commit generated native binaries from a local checkout:

```text
smart/_cpp*.so
smart/pymanifold_runtime/pymanifold*.so
smart/bin/smart-cpp-native
```

They are created by `python -m pip install -e .`, `smart build-cpp`, or the
release build process and are audited before wheel publication.

## Configuration Layout

The root `configs/` directory is the source-checkout copy used by examples and
benchmarks. `smart/configs/` mirrors the same profiles so installed wheels can
run without a repository checkout.

Recommended entry configs:

- `configs/smoke_5.yaml`: fast local smoke and CI checks.
- `configs/example_3x3.yaml`: 3-per-category local example.
- `configs/demo.yaml`: small prepared ShapeNet demo.
- `configs/paper_like.yaml`: paper-style parameters.
- `configs/expanded_full.yaml`: larger local ShapeNet training/evaluation data.

Experimental configs live under `experiments/configs/` and are not part of the
public reproduction package.
