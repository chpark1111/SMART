# SMART Native C++ Core

This directory contains the native SMART implementation used by the Python
package.

## Files

- `smart_native_core.hpp` / `smart_native_core.cpp`: geometry, bbox, tet,
  scoring, merge, and search utilities.
- `smart_native_engine.hpp` / `smart_native_engine.cpp`: stateful native SMART
  engine.
- `manifold_bridge.cpp`: bridge to the fixed vendored Manifold C++ source.
- `smart_cpp_module.cpp`: Python extension module exposed as `smart._cpp`.
- `smart_native_cli.cpp`: standalone `smart-cpp-native` executable.

## Build

```bash
smart --config configs/smoke_5.yaml build-tools --only-manifold-binding
smart --config configs/smoke_5.yaml build-cpp
```

The fixed Manifold source lives in `smart/vendor/manifold` and should not be
pulled, replaced, or rewritten.
