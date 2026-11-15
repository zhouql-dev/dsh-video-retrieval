#!/usr/bin/env bash
# P2 — console as a browser tab: start the fusion GUI backend + open it.
# (The vr_* tools spawn this same server lazily; this script is for humans.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
[ -f "$REPO/.env" ] && { set -a; . "$REPO/.env"; set +a; }
# P0-2：唯一状态根（插件默认根，plugins 与 standalone 共用）——必须在任何 $STATE 使用之前
STATE="${VTL_STATE_DIR:-$HOME/.dsh/video-retrieval}"
export SKILL_SCRIPTS="$HERE/packages/vr-backend/vr_backend/engines"
export FUSION_CONFIG="$STATE/config"   # 运行态配置跟状态走，不写包内（P1-10 的地基）
export FUSION_CASES="$STATE/cases.jsonl"
export FUSION_JOBS="$STATE/jobs"
export FUSION_UPLOADS="$STATE/uploads"
export FUSION_WEB="$HERE/packages/vr-plugin/client"
export BENCH_DATA="$REPO/dataset"
PORT="${VTL_PORT:-8788}"
mkdir -p "$STATE/jobs" "$STATE/uploads" "$STATE/config"
VENV_PY="$REPO/video/bin/python"
if curl -sf "http://127.0.0.1:$PORT/health" | grep -q "$STATE"; then
  echo "[serve] backend already running on :$PORT"
else
  echo "[serve] backend on http://127.0.0.1:$PORT (state: $STATE, log: $STATE/server.log)"
  nohup "$VENV_PY" "$HERE/packages/vr-backend/vr_backend/server.py" --host 127.0.0.1 --port "$PORT" --warmup --watch 30 \
    >> "$STATE/server.log" 2>&1 &
  echo $! > "$STATE/server.pid"
  sleep 1
fi
open "http://127.0.0.1:$PORT" 2>/dev/null || true
