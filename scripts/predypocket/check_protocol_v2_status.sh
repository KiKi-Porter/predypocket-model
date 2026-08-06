#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="$REPO_ROOT/outputs/predypocket_protocol_v2/training"

for fold in 0 1 2 3 4; do
  for variant in anchor_matched dynamic; do
    dir="$ROOT/fold${fold}/$variant"
    state=not_started
    epochs=0
    if [[ -f "$dir/history.json" ]]; then
      epochs=$(grep -c '"epoch":' "$dir/history.json" || true)
      state=training_or_stopped
    fi
    if [[ -f "$dir/best_checkpoint.index" ]]; then
      state=checkpoint_available
    fi
    if [[ -f "$dir/training.pid" ]]; then
      pid=$(sed -n '1p' "$dir/training.pid")
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        state=running
      fi
    fi
    printf 'fold%s %-14s state=%-20s epochs=%s\n' "$fold" "$variant" "$state" "$epochs"
  done
done
