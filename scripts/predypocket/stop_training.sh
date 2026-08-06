#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_FILE="$REPO_ROOT/outputs/predypocket_model/training/fold0/training.pid"
if ! test -f "$PID_FILE"; then
  echo "No Dynamic PreDyPocket training PID file"
  exit 0
fi
PID="$(<"$PID_FILE")"
COMMAND="$(tr '\0' ' ' <"/proc/$PID/cmdline" 2>/dev/null || true)"
case "$COMMAND" in
  *scripts/predypocket/train_predypocket.py*) ;;
  *)
    echo "Refusing to signal PID $PID: it is not Dynamic PreDyPocket training"
    exit 1
    ;;
esac
kill -TERM "$PID"
echo "Requested graceful stop for Dynamic PreDyPocket PID $PID"
