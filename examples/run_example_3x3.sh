#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

rm -rf examples/runs/example_3x3

python3 -m smart --config configs/example_3x3.yaml check-data
python3 -m smart --config configs/example_3x3.yaml doctor
python3 -m smart --config configs/example_3x3.yaml run
python3 -m smart --config configs/example_3x3.yaml summary
