#!/usr/bin/env bash
# check-consistency.sh — 一致性门禁（P0-18 起，P1-15…18 后按新结构改写）
#
# 四段：
#   1/4 profile ↔ 仓库（桌面 profile 直接软链仓库；其它 profile 说明状态）
#   2/4 一份代码（I1）：sync.sh --check
#   3/4 路径单点（I3）：python 与 node 解析器输出一致
#   4/4 工作区：干净
#
# 用法：bash scripts/check-consistency.sh [--quiet]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUIET=0; [ "${1:-}" = "--quiet" ] && QUIET=1
FAIL=0
ok() { [ "$QUIET" = 1 ] || echo "  OK   $*"; }
bad() { echo "  FAIL $*"; FAIL=1; }
PY="${VTL_PY:-python3}"

echo "== 1/4 profile ↔ 仓库（插件宿主）"
PLUGIN="$HERE/packages/vr-plugin"
for P in desktop web; do
  P_DIR="$HOME/.dsh/profiles/$P/node_modules/dsh-video-retrieval"
  [ -e "$P_DIR" ] || { ok "$P: 未安装，跳过"; continue; }
  # 桌面 profile 是软链到本仓库 —— 它跑的就是仓库代码；别的 profile 是独立快照。
  TARGET="$(cd "$P_DIR" 2>/dev/null && pwd -P || echo "")"
  REAL_HERE="$(cd "$HERE" && pwd -P)"
  case "$TARGET" in
    "$REAL_HERE"|"$REAL_HERE"/*)
      for f in packages/vr-plugin/src/index.js packages/vr-plugin/src/paths.js; do
        [ -f "$HERE/$f" ] || bad "仓库缺 $f"
      done
      ok "$P → 仓库（软链，运行即仓库代码）" ;;
    *)
      echo "  NOTE $P: 独立快照安装（$TARGET）"
      echo "       └ 与仓库不同源；要同步请：dsh plugin --profile $P add file:$HERE" ;;
  esac
done

echo "== 2/4 一份代码（I1）"
if bash "$HERE/scripts/sync.sh" --check >/tmp/vtl-i1.log 2>&1; then
  ok "sync.sh --check 通过"
else
  bad "sync.sh --check 失败：$(tail -3 /tmp/vtl-i1.log | tr '\n' ' ')"
fi

echo "== 3/4 路径单点（I3）"
PY_PATH=$(cd "$HERE" && PYTHONPATH="$HERE/packages/vr-backend" "$PY" -m vr_backend.paths state_root 2>/dev/null)
NODE_BIN="$(command -v node || true)"
if [ -n "$NODE_BIN" ]; then
  JS_PATH=$("$NODE_BIN" "$PLUGIN/src/paths.js" state_root 2>/dev/null)
  [ "$PY_PATH" = "$JS_PATH" ] && ok "state_root 两端一致（$PY_PATH）" \
                             || bad "state_root 漂移：py=$PY_PATH js=$JS_PATH"
else
  ok "node 不可用，跳过（tests/test_paths_contract.py 覆盖此判定）"
fi

echo "== 4/4 工作区"
case "$(git -C "$HERE" status --porcelain 2>/dev/null | head -1)" in
  "") ok "工作区干净" ;;
  *)  bad "工作区有未提交改动（$(git -C "$HERE" status --porcelain | wc -l | tr -d ' ') 项）" ;;
esac

echo
[ "$FAIL" = 0 ] && { echo "PASS 一致性通过"; exit 0; } || { echo "FAIL 存在不一致（见上）"; exit 1; }
