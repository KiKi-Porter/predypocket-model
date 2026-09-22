#!/usr/bin/env bash
set -euo pipefail

SYSTEM_DIR=${1:-dynamic_data/4KVK/4KVK_4KVK_1.PG4_A_703}
OUT_PREFIX=${2:-outputs/query_predypocket}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-weights/predypocket_model}

mkdir -p "$(dirname "${OUT_PREFIX}")"

PYTHONPATH=src python -u src/predypocket_predict.py \
  --system-dir "${SYSTEM_DIR}" \
  --checkpoint "${MODEL_CHECKPOINT}" \
  --out-prefix "${OUT_PREFIX}"
