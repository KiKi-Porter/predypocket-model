#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$REPO_ROOT/outputs/predypocket_model/training/fold0"
PID_FILE="$RUN_DIR/training.pid"
LOG_FILE="$RUN_DIR/training.log"
test -f "$RUN_DIR/last_checkpoint.index"
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
nohup python scripts/predypocket/train_predypocket.py \
  --config configs/predypocket_1ns_gap1ns_future20ns.json \
  --fold 0 \
  --stage 1 \
  --resume auto \
  >>"$LOG_FILE" 2>&1 &
echo "$!" >"$PID_FILE"
echo "Resumed Dynamic PreDyPocket fold 0 with PID $(<"$PID_FILE")"
