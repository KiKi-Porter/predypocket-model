#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPU_ID="${GPU_ID:-5}"
RUN_DIR="$REPO_ROOT/outputs/predypocket_protocol_v2/training/fold0/anchor_matched"

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/training.pid" ]]; then
  existing_pid=$(sed -n '1p' "$RUN_DIR/training.pid")
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Protocol-v2 anchor fold0 is already running as PID $existing_pid" >&2
    exit 3
  fi
fi
if [[ -e "$RUN_DIR/history.json" || -e "$RUN_DIR/best_checkpoint.index" ]]; then
  echo "Refusing to overwrite existing protocol-v2 anchor fold0 results" >&2
  exit 2
fi

nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python scripts/predypocket/train_protocol_v2.py \
    --config configs/predypocket_protocol_v2.json \
    --fold 0 \
    --model-variant anchor-matched \
    --stage 1 \
    --device /GPU:0 \
    --seed 42 \
    >"$RUN_DIR/training.log" 2>&1 < /dev/null &

printf '%s\n' "$!" > "$RUN_DIR/training.pid"
echo "Started protocol-v2 fold0 anchor-matched PID $! on physical GPU $GPU_ID"
