#!/usr/bin/env bash
# P2 — console as a browser tab: start the fusion GUI backend + open it.
# (The vr_* tools spawn this same server lazily; this script is for humans.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
[ -f "$REPO/.env" ] && { set -a; . "$REPO/.env"; set +a; }
export SKILL_SCRIPTS="$HERE/engine"
export FUSION_CONFIG="$HERE/config"
export FUSION_CASES="$HERE/data/cases.jsonl"
export FUSION_JOBS="$HERE/data/jobs"
export FUSION_UPLOADS="$HERE/data/uploads"
export FUSION_WEB="$HERE/server/web"
export BENCH_DATA="$REPO/dataset"
PORT="${VTL_PORT:-8788}"
mkdir -p "$HERE/data/jobs" "$HERE/data/uploads"
VENV_PY="$REPO/video/bin/python"
if curl -sf "http://127.0.0.1:$PORT/health" | grep -q "$HERE/data"; then
  echo "[serve] backend already running on :$PORT"
else
  echo "[serve] backend on http://127.0.0.1:$PORT (log: $HERE/data/server.log)"
  nohup "$VENV_PY" "$HERE/server/server.py" --host 127.0.0.1 --port "$PORT" --warmup --watch 30 \
    >> "$HERE/data/server.log" 2>&1 &
  echo $! > "$HERE/data/server.pid"
  sleep 1
fi
open "http://127.0.0.1:$PORT" 2>/dev/null || true
