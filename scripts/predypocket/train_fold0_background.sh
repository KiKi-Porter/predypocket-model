#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$REPO_ROOT/outputs/predypocket_model/training/fold0"
PID_FILE="$RUN_DIR/training.pid"
LOG_FILE="$RUN_DIR/training.log"

test -f "$REPO_ROOT/outputs/atlas_full_10proteins/FINAL_DATASET_REPORT.md"
test -f "$REPO_ROOT/data/atlas/atlas_dynamic_manifest_10proteins_1ns_gap1ns_future20ns.csv"
test -f "$REPO_ROOT/data/atlas/atlas_10protein_5fold_splits_seed42.json"
mkdir -p "$RUN_DIR"
if test -f "$PID_FILE" && ps -p "$(<"$PID_FILE")" >/dev/null 2>&1; then
  echo "Dynamic PreDyPocket fold 0 is already running"
  exit 1
fi
cd "$REPO_ROOT"
nohup python scripts/predypocket/train_predypocket.py \
  --config configs/predypocket_1ns_gap1ns_future20ns.json \
  --fold 0 \
  --stage 1 \
  >"$LOG_FILE" 2>&1 &
echo "$!" >"$PID_FILE"
echo "Started Dynamic PreDyPocket fold 0 with PID $(<"$PID_FILE")"
