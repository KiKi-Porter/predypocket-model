#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

root="outputs/predypocket_model/training"
config="configs/predypocket_1ns_gap1ns_future20ns.json"
specs=("1:2" "2:3" "3:4" "4:5")

mkdir -p "$root"

exec 9>"$root/.remaining_folds_parallel.lock"
flock -n 9 || {
  echo "$(date -Is) ERROR: launcher already running"
  exit 3
}

for spec in "${specs[@]}"; do
  fold="${spec%%:*}"
  dir="$root/fold${fold}"

  if [[ -d "$dir" ]] &&
     [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "$(date -Is) ERROR: $dir is not empty; refusing to overwrite"
    exit 4
  fi
done

pids=()

for spec in "${specs[@]}"; do
  fold="${spec%%:*}"
  gpu="${spec##*:}"
  dir="$root/fold${fold}"

  mkdir -p "$dir"

  (
    set -euo pipefail

    echo "$(date -Is) Training fold $fold on physical GPU $gpu"

    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$gpu" \
      python scripts/predypocket/train_predypocket.py \
        --config "$config" \
        --fold "$fold" \
        --stage 1 \
        --batch-size 1 \
        --gradient-accumulation 8 \
        --device /GPU:0 \
        >"$dir/training.log" 2>&1

    echo "$(date -Is) Evaluating fold $fold on physical GPU $gpu"

    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$gpu" \
      python scripts/predypocket/evaluate_predypocket.py \
        --config "$config" \
        --fold "$fold" \
        --batch-size 1 \
        --device /GPU:0 \
        >"$dir/evaluation.log" 2>&1

    echo "$(date -Is) Completed fold $fold"
  ) >"$dir/pipeline.log" 2>&1 &

  pids+=("$!")
  echo "$(date -Is) Launched fold $fold on physical GPU $gpu as PID $!"
done

rc=0

for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done

if [[ "$rc" -eq 0 ]]; then
  echo "$(date -Is) All folds completed successfully"
else
  echo "$(date -Is) One or more folds failed"
fi

exit "$rc"
