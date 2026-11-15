#!/usr/bin/env bash
# 单向 vendoring 刷新（开发者工具）：canonical 源 → 本包。
#   engine/  ：从 skill 源同步（不 --delete，保留本地自改）
#   server/  ：fusion/ 的生成镜像（--delete 保证完全一致）
#   weights/ ：拷贝（~0.5 GB）
#
# 用法：
#   bash scripts/sync.sh            实际同步
#   bash scripts/sync.sh --check    只校验（内容比对），有差异 exit 1（CI 用）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${VTL_REPO_DIR:-$(cd "$HERE/.." && pwd)}"
SKILL="${VTL_SKILL_DIR:-}"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1

EXCLUDES=(--exclude '__pycache__' --exclude 'jobs' --exclude 'uploads' --exclude 'outputs'
          --exclude 'output' --exclude 'bench_out' --exclude 'config' --exclude 'cases.jsonl'
          --exclude 'tests/' --exclude 'test_*.py')

if [ "$CHECK" = 1 ]; then
  if [ ! -d "$REPO/fusion" ]; then echo "SKIP check (no fusion at $REPO/fusion)"; exit 0; fi
  out=$(diff -r -q \
        -x '__pycache__' -x 'jobs' -x 'uploads' -x 'outputs' -x 'output' \
        -x 'bench_out' -x 'config' -x 'cases.jsonl' -x 'tests' -x 'test_*.py' \
        "$REPO/fusion" "$HERE/server" 2>&1 || true)
  if [ -n "$out" ]; then
    echo "$out" | head -20
    echo "FAIL fusion 与 server 不一致（运行 bash scripts/sync.sh 修复）"
    exit 1
  fi
  echo "PASS fusion 与 server 一致"
  exit 0
fi

if [ -n "$SKILL" ]; then
  [ -d "$SKILL/scripts" ] || { echo "canonical skill missing: $SKILL/scripts" >&2; exit 1; }
  mkdir -p "$HERE/engine"
  rsync -a --exclude '__pycache__' "$SKILL/scripts"/ "$HERE/engine"/
  echo "engine synced"
else
  echo "SKIP engine/ (set VTL_SKILL_DIR)"
fi

if [ -d "$REPO/fusion" ]; then
  mkdir -p "$HERE/server"
  rsync -a --delete "${EXCLUDES[@]}" "$REPO/fusion"/ "$HERE/server"/
  echo "server synced (fusion mirror)"
else
  echo "SKIP server/ (fusion not found at $REPO/fusion)"
fi

if [ -d "$REPO/weights" ]; then
  mkdir -p "$HERE/weights/osnet" "$HERE/weights/clip"
  rsync -a --exclude '__pycache__' "$REPO/weights/osnet"/ "$HERE/weights/osnet"/ 2>/dev/null || true
  rsync -a --exclude '__pycache__' "$REPO/weights/clip"/ "$HERE/weights/clip"/ 2>/dev/null || true
  cp -f "$REPO/yolov8s-worldv2.pt" "$HERE/yolov8s-worldv2.pt" 2>/dev/null || true
  echo "weights synced"
else
  echo "SKIP weights/ (no weights dir)"
fi
