#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MESH_ID="${1:-1f5537f4747ec847622c69c3abc6f80}"

smart-cpp-native run-pipeline \
  --input "examples/sample_shapes/shapenet_airplane/${MESH_ID}/model.obj" \
  --work_dir "examples/runs/native_one/${MESH_ID}" \
  --manifoldplus_bin "external/mesh2tet/ManifoldPlus/build/manifold" \
  --ftetwild_bin "external/mesh2tet/fTetWild/build/FloatTetwild_bin" \
  --coacd_bin "external/CoACD/python/package/bin/coacd" \
  --epsilon 0.002 \
  --edge_length 0.1 \
  --refine_max_step 50 \
  --mcts_iter 10 \
  --mcts_max_step 5
