#!/usr/bin/env bash
# 一致性门禁（P0-18）：仓库 ↔ profile、fusion ↔ server、engine ↔ server、工作区
# 用法：bash scripts/check-consistency.sh [--quiet]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
QUIET=0; [ "${1:-}" = "--quiet" ] && QUIET=1
FAIL=0
h() { if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
ok() { [ "$QUIET" = 1 ] || echo "  OK   $*"; }
bad() { echo "  FAIL $*"; FAIL=1; }

echo "== 1/4 profile ↔ 仓库（JS 宿主与客户端）"
for P in web desktop; do
  P_DIR="$HOME/.dsh/profiles/$P/node_modules/dsh-video-retrieval"
  for f in src/index.js dist/index.js dist-client/index.js; do
    if [ ! -f "$P_DIR/$f" ]; then bad "$P/$f 缺失"; continue; fi
    a=$(h "$HERE/packages/vr-plugin/$f"); b=$(h "$P_DIR/$f")
    [ "$a" = "$b" ] && ok "$P/$f" || bad "$P/$f（仓库 ${a:0:8} / profile ${b:0:8}）"
  done
done

echo "== 2/4 fusion ↔ server（Python 镜像）"
miss=0
while IFS= read -r rel; do
  a="$REPO/fusion/$rel"; b="$HERE/packages/vr-backend/vr_backend/$rel"
  [ -f "$b" ] || { bad "server/$rel 缺失（fusion 有）"; continue; }
  [ "$(h "$a")" = "$(h "$b")" ] && ok "fusion=server $rel" || { bad "fusion≠server $rel"; miss=1; }
done < <(cd "$REPO/fusion" 2>/dev/null && find . -name '*.py' -not -path './bench_out/*' -not -path './__pycache__/*' -not -name 'test_*.py' | sed 's|^\./||' | sort)
[ "$miss" = 0 ] && ok "镜像一致" || true

echo "== 3/4 engine ↔ server（同名文件）"
for f in provider.py wevisone_local.py; do
  [ -f "$HERE/packages/vr-backend/vr_backend/engines/$f" ] && [ -f "$HERE/packages/vr-backend/vr_backend/$f" ] || continue
  [ "$(h "$HERE/packages/vr-backend/vr_backend/engines/$f")" = "$(h "$HERE/packages/vr-backend/vr_backend/$f")" ] && ok "$f" || bad "engine/$f ≠ server/$f"
done

echo "== 4/4 工作区"
case "$(git -C "$HERE" status --porcelain 2>/dev/null | head -1)" in
  "") ok "工作区干净" ;;
  *)  bad "工作区有未提交改动（$(git -C "$HERE" status --porcelain | wc -l | tr -d ' ') 项）" ;;
esac

echo
[ "$FAIL" = 0 ] && { echo "PASS 一致性通过"; exit 0; } || { echo "FAIL 存在不一致（见上）"; exit 1; }
