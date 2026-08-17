#!/usr/bin/env bash
# One-way vendoring refresh (design decisions 2+3): canonical sources → dsh/.
#   engine/  : never --delete (local self-edits survive a sync)
#   server/  : a generated mirror of fusion/ (--delete keeps it exact)
#   weights/ : a copy (fully portable, ~0.5 GB)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SKILL=/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize

# 1) engine scripts (canonical skill)
[ -d "$SKILL/scripts" ] || { echo "canonical skill missing: $SKILL/scripts" >&2; exit 1; }
mkdir -p "$HERE/engine"
rsync -a --exclude '__pycache__' "$SKILL/scripts"/ "$HERE/engine"/

# 2) fusion harness → dsh/server/ (mirror, minus runtime state/tests/caches)
mkdir -p "$HERE/server"
rsync -a --delete \
  --exclude '__pycache__' --exclude 'jobs' --exclude 'uploads' --exclude 'outputs' \
  --exclude 'output' --exclude 'bench_out' --exclude 'config' --exclude 'cases.jsonl' \
  --exclude 'test_*.py' \
  "$REPO/fusion"/ "$HERE/server"/

# 3) weights (decision 2: copy into dsh/)
mkdir -p "$HERE/weights/osnet" "$HERE/weights/clip"
rsync -a --exclude '__pycache__' "$REPO/weights/osnet"/ "$HERE/weights/osnet"/
rsync -a --exclude '__pycache__' "$REPO/weights/clip"/ "$HERE/weights/clip"/
cp -f "$REPO/yolov8s-worldv2.pt" "$HERE/yolov8s-worldv2.pt"

echo "synced: engine/ + server/ (fusion mirror) + weights/ → $HERE"
