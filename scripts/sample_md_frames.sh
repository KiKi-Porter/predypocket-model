#!/usr/bin/env bash
set -euo pipefail

SYSTEM_DIR=${1:-examples/dynamic_data_example/TOY/TOY_TOY_1.LIG_B_101}
OUT_PREFIX=${2:-outputs/toy_sample}

mkdir -p "$(dirname "${OUT_PREFIX}")"

PYTHONPATH=src python -u src/sample_predypocket_frames.py \
  --system-dir "${SYSTEM_DIR}" \
  --out-prefix "${OUT_PREFIX}"
