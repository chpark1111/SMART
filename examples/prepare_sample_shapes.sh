#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/examples/sample_shapes"

rm -rf "${DEST}"

copy_mesh() {
  local category="$1"
  local mesh_id="$2"
  local src="${ROOT}/data/${category}/${mesh_id}/model.obj"
  local dst="${DEST}/${category}/${mesh_id}/model.obj"
  if [[ ! -f "${src}" ]]; then
    echo "missing local source mesh: ${src}" >&2
    echo "prepare data first, or edit configs/example_3x3.yaml for your mesh ids" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  cp "${src}" "${dst}"
}

copy_mesh shapenet_airplane 1f5537f4747ec847622c69c3abc6f80
copy_mesh shapenet_airplane 260305219d81f745623cba1f26a8e885
copy_mesh shapenet_airplane 172764bea108bbcceae5a783c313eb36

copy_mesh shapenet_chair 11b7c86fc42306ec7e7e25239e7b8f85
copy_mesh shapenet_chair 17883ea5a837f5731250f48219951972
copy_mesh shapenet_chair 23acbdfee13b407ce42d6c2ea750090e

copy_mesh shapenet_table 1692563658149377630047043c6a0c50
copy_mesh shapenet_table 1804dd6f5c827c1a4bf8d5f43e57b138
copy_mesh shapenet_table 2e2894138df855b26f88aa1b7f7cc6c6

echo "prepared example meshes in ${DEST}"
