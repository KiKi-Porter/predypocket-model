#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEVICE="${DEVICE:-/GPU:0}"
GPU_IDS="${GPU_IDS:-2,3,4,5}"

cd "$REPO_ROOT"
python scripts/predypocket/run_protocol_v2_5fold.py \
  --config configs/predypocket_protocol_v2.json \
  --device "$DEVICE" \
  --gpu-ids "$GPU_IDS" \
  --execute
