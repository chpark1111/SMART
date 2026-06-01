#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

rm -rf examples/runs/learned_frontier

python3 -m smart --config configs/learned_frontier.yaml check-data
if [[ ! -x build/smart-cpp-native ]]; then
  python3 -m smart --config configs/learned_frontier.yaml build-cpp
fi
python3 -m smart --config configs/learned_frontier.yaml doctor
python3 -m smart --config configs/learned_frontier.yaml run
python3 -m smart --config configs/learned_frontier.yaml summary
