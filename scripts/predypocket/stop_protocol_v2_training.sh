#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="$REPO_ROOT/outputs/predypocket_protocol_v2/training"
found=0

while IFS= read -r pid_file; do
  pid=$(sed -n '1p' "$pid_file")
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  command=$(ps -p "$pid" -o args= 2>/dev/null || true)
  if [[ "$command" == *"scripts/predypocket/train_protocol_v2.py"* ]]; then
    kill "$pid"
    echo "Sent TERM to protocol-v2 training PID $pid ($pid_file)"
    found=1
  fi
done < <(find "$ROOT" -type f -name training.pid -print 2>/dev/null | sort)

if [[ "$found" -eq 0 ]]; then
  echo "No live protocol-v2 training PID found"
fi
