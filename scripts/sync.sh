#!/usr/bin/env bash
# sync.sh — 引擎源同步 + 『一份代码』校验（P1-8/P1-18 之后）。
#
# 历史上的行为：把 ../fusion/ rsync 到 server/（fusion 是上游源、server 是镜像）。
# 2026-11-16 起 fusion/ 已归档（archive/fusion-legacy/ + fusion/README-ARCHIVE.md），
# 代码只有一份：packages/vr-backend/vr_backend/。**本脚本不再写后端代码**，
# 只做两件事：
#
#   1. 设置了 VTL_SKILL_DIR 时，把外部 skill 的 engines/ 同步进包（不 --delete，
#      保留本地自改，这是唯一允许的反向同步）；
#   2. 校验"一份代码"不变量：同名模块只允许出现一次、没有残留副本。
#
# 用法：
#   bash scripts/sync.sh            同步引擎（若 VTL_SKILL_DIR 存在）+ 校验
#   bash scripts/sync.sh --check    只校验（CI 用，有违规 exit 1）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SKILL="${VTL_SKILL_DIR:-}"
BACKEND="$HERE/packages/vr-backend/vr_backend"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
FAIL=0
bad() { echo "  FAIL $*"; FAIL=1; }
ok()  { echo "  OK   $*"; }

if [ "$CHECK" = 0 ] && [ -n "$SKILL" ]; then
  [ -d "$SKILL/scripts" ] || { echo "canonical skill missing: $SKILL/scripts" >&2; exit 1; }
  mkdir -p "$BACKEND/engines"
  rsync -a --exclude '__pycache__' "$SKILL/scripts"/ "$BACKEND/engines"/
  echo "  engines synced from $SKILL"
elif [ "$CHECK" = 0 ]; then
  echo "  SKIP engines sync (set VTL_SKILL_DIR)"
fi

# ── 校验 ─────────────────────────────────────────────────────────────────────
echo "== 一份代码（I1）"
# 1a. 会被 import 的同名模块只允许一份
for name in wevisone_local.py provider.py config.py paths.py; do
  n=$(find "$HERE/packages" -name "$name" -not -path '*/__pycache__/*' | wc -l | tr -d ' ')
  [ "$n" = 1 ] && ok "$name ×1" || bad "$name ×$n"
done
# 1b. 字节相同的重复 .py（排除 __pycache__ 与 archive）
dupes=$(find "$HERE/packages" -name '*.py' -not -path '*/__pycache__/*' -exec md5 -q {} \; \
        | sort | uniq -d | wc -l | tr -d ' ')
[ "$dupes" = 0 ] && ok "无内容重复的 .py" || bad "$dupes 个 .py 内容重复"

echo "== 源码不再出现在归档目录"
for d in fusion; do
  n=$(find "$REPO/$d" -maxdepth 2 -name '*.py' -not -path '*/bench_out/*' 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = 0 ] && ok "$d/ 无 .py（已归档）" || bad "$d/ 仍有 $n 个 .py"
done

echo "== 包内没有可写运行态"
for d in data jobs uploads work outputs config; do
  [ -e "$BACKEND/$d" ] && bad "packages/vr-backend/vr_backend/$d 存在（运行态必须在状态根）" \
                       || ok "无 $d/"
done

echo
[ "$FAIL" = 0 ] && { echo "PASS 一份代码（I1）"; exit 0; } || { echo "FAIL 见上"; exit 1; }
