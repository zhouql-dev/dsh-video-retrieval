#!/usr/bin/env bash
# One-way vendoring refresh (developer tool): canonical sources → this package.
#   engine/  : never --delete (local self-edits survive a sync)
#   server/  : a generated mirror of fusion/ (--delete keeps it exact)
#   weights/ : a copy (fully portable, ~0.5 GB)
#
# This script is only for the PLUGIN DEVELOPER syncing from their private
# fusion/ checkout and skill source; end users never run it. Configure the
# source locations via env vars.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${VTL_REPO_DIR:-}"
SKILL="${VTL_SKILL_DIR:-}"

# 1) engine scripts (canonical skill)
if [ -n "$SKILL" ]; then
  [ -d "$SKILL/scripts" ] || { echo "canonical skill missing: $SKILL/scripts" >&2; exit 1; }
  mkdir -p "$HERE/engine"
  rsync -a --exclude '__pycache__' "$SKILL/scripts"/ "$HERE/engine"/
  echo "engine synced"
else
  echo "SKIP engine/ (set VTL_SKILL_DIR to the skill source dir)"
fi

# 2) fusion harness → server/ (mirror, minus runtime state/tests/caches)
if [ -n "$REPO" ]; then
  [ -d "$REPO/fusion" ] || { echo "fusion dir missing: $REPO/fusion" >&2; exit 1; }
  mkdir -p "$HERE/server"
  rsync -a --delete \
    --exclude '__pycache__' --exclude 'jobs' --exclude 'uploads' --exclude 'outputs' \
    --exclude 'output' --exclude 'bench_out' --exclude 'config' --exclude 'cases.jsonl' \
    --exclude 'test_*.py' \
    "$REPO/fusion"/ "$HERE/server"/
  echo "server synced (fusion mirror)"
else
  echo "SKIP server/ (set VTL_REPO_DIR to the fusion repo root)"
fi

# 3) weights (copy into the package; skip unless the source repo is provided)
if [ -n "$REPO" ]; then
  mkdir -p "$HERE/weights/osnet" "$HERE/weights/clip"
  rsync -a --exclude '__pycache__' "$REPO/weights/osnet"/ "$HERE/weights/osnet"/ 2>/dev/null || true
  rsync -a --exclude '__pycache__' "$REPO/weights/clip"/ "$HERE/weights/clip"/ 2>/dev/null || true
  cp -f "$REPO/yolov8s-worldv2.pt" "$HERE/yolov8s-worldv2.pt" 2>/dev/null || true
  echo "weights synced"
else
  echo "SKIP weights/ (set VTL_REPO_DIR)"
fi
