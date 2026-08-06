#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$REPO_ROOT/outputs/predypocket_model/training/fold0"
PID_FILE="$RUN_DIR/training.pid"
LOG_FILE="$RUN_DIR/training.log"
if ! test -f "$PID_FILE"; then
  echo "No Dynamic PreDyPocket training PID file"
  exit 0
fi
PID="$(<"$PID_FILE")"
if ps -p "$PID" -o pid=,etime=,stat=,args=; then
  test ! -f "$LOG_FILE" || tail -n 30 "$LOG_FILE"
else
  echo "Dynamic PreDyPocket PID $PID is not running"
fi
