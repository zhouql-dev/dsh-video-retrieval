#!/usr/bin/env bash
# install.sh — one-shot install for the dsh-video-retrieval plugin.
#
# What it does:
#   1. Builds the host half (dist/index.js + dist/index.<hash>.js).
#   2. Resolves the installed package's absolute path from the profile.
#   3. Rewrites the preset's path placeholders with those absolutes.
#   4. Deploys the preset into ~/.dsh/.agent-presets/video-retrieval/.
#   5. Installs the package into the web profile via `dsh plugin add`.
#   6. Downloads weights if missing (via setup.sh).
#   7. Smoke-tests the build and client bundle.
#
# Run after cloning the repo or after editing src/ / server/ / engine/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSH_HOME_DIR="${DSH_HOME:-$HOME/.dsh}"
PROFILE_DIR="$DSH_HOME_DIR/profiles/web"
PATCH_FILE="$PROFILE_DIR/cordis.patch.yml"
DSH_BIN="${DSH_BIN:-$(command -v dsh 2>/dev/null || echo npx @deepseek-ai/dsh)}"
VENV_PY="${VTL_PY:-python3}"
BENCH_DATA="${BENCH_DATA:-}"
DATA_DIR="${DATA_DIR:-$DSH_HOME_DIR/video-retrieval}"
cd "$HERE"

echo "[1/7] building host half..."
node build.mjs

echo "[2/7] resolving package path..."
PKG_ROOT=$(node -e "const {createRequire}=require('node:module'); const r=createRequire('$PROFILE_DIR/package.json'); console.log(r.resolve('dsh-video-retrieval/package.json').replace(/\\/package\\.json$/, ''))" 2>/dev/null || echo "")
if [ -z "$PKG_ROOT" ]; then
  # Package not yet installed in the profile — install first, then resolve.
  echo "       (package not in profile yet; installing now...)"
  "$DSH_BIN" plugin --profile web add "file:$HERE" >/dev/null 2>&1 || true
  PKG_ROOT=$(node -e "const {createRequire}=require('node:module'); const r=createRequire('$PROFILE_DIR/package.json'); console.log(r.resolve('dsh-video-retrieval/package.json').replace(/\\/package\\.json$/, ''))" 2>/dev/null)
fi
if [ -z "$PKG_ROOT" ]; then
  echo "ERROR: could not resolve dsh-video-retrieval package path." >&2
  exit 1
fi
echo "       package root: $PKG_ROOT"

ENGINE_DIR="$PKG_ROOT/engine"
SERVER_DIR="$PKG_ROOT/server"
CONFIG_DIR="$PKG_ROOT/config"

echo "[3/7] rewriting preset placeholders (into a deploy copy)..."
STAGE="$(mktemp -d)"
rsync -a preset/video-retrieval/ "$STAGE/video-retrieval/"
sed -i '' \
  -e "s|__VENV_PY__|$VENV_PY|g" \
  -e "s|__ENGINE_DIR__|$ENGINE_DIR|g" \
  -e "s|__SERVER_DIR__|$SERVER_DIR|g" \
  -e "s|__CONFIG_DIR__|$CONFIG_DIR|g" \
  -e "s|__DATA_DIR__|$DATA_DIR|g" \
  -e "s|__DATASET_DIR__|$BENCH_DATA|g" \
  "$STAGE/video-retrieval/agent.cordis.yml"

echo "[4/7] deploying preset to $DSH_HOME_DIR/.agent-presets/video-retrieval/"
DEPLOY="$DSH_HOME_DIR/.agent-presets/video-retrieval"
mkdir -p "$(dirname "$DEPLOY")"
rsync -a --delete --exclude '__pycache__' "$STAGE/video-retrieval/" "$DEPLOY/"
rm -rf "$STAGE"

echo "[5/7] ensuring package is in the web profile..."
"$DSH_BIN" plugin --profile web add "file:$HERE" >/dev/null 2>&1 || true

echo "[6/7] downloading weights if needed..."
bash scripts/setup.sh || echo "       (weight download skipped — run scripts/setup.sh after uploading weights to GitHub Releases)"

echo "[7/7] smoke tests..."
mkdir -p node_modules
ln -sfn "$PROFILE_DIR/node_modules/@deepseek-ai" node_modules/@deepseek-ai 2>/dev/null || true

VIDEO="${VTL_TEST_VIDEO:-}"
if [ -n "$VIDEO" ] && [ -f "$VIDEO" ]; then
  node --input-type=module -e "
const m = await import('file://$HERE/dist/index.js')
const registered = []
m.apply({ tools: { register: (t) => registered.push(t) } }, {
  python: '$VENV_PY', engineDir: '$ENGINE_DIR', serverDir: '$SERVER_DIR',
  configDir: '$CONFIG_DIR', dataDir: '$DATA_DIR', datasetDir: '$BENCH_DATA',
  backendUrl: 'http://127.0.0.1:8788',
})
console.log('registered tools (' + registered.length + '):', registered.map(t => t.name).join(', '))
const pf = registered.find(t => t.name === 'vr_preflight')
if (!pf) throw new Error('vr_preflight missing')
const res = await pf.execute({ video: '$VIDEO' }, { signal: undefined })
function assertLossless(v, p='root') {
  if (v === undefined) throw new Error(p + ': undefined')
  if (typeof v === 'function' || typeof v === 'symbol' || typeof v === 'bigint') throw new Error(p + ': non-JSON type')
  if (typeof v === 'number' && !Number.isFinite(v)) throw new Error(p + ': non-finite number')
  if (v && typeof v === 'object') {
    if (v instanceof Map || v instanceof Set || v instanceof Date) throw new Error(p + ': exotic object')
    for (const [k, x] of Object.entries(v)) assertLossless(x, p + '.' + k)
  }
}
assertLossless(res)
console.log('lossless-JSON boundary: OK')
"
else
  node --input-type=module -e "
const m = await import('file://$HERE/dist/index.js')
const registered = []
m.apply({ tools: { register: (t) => registered.push(t) } }, {
  python: '$VENV_PY', engineDir: '$ENGINE_DIR', serverDir: '$SERVER_DIR',
  configDir: '$CONFIG_DIR', dataDir: '$DATA_DIR', datasetDir: '$BENCH_DATA',
  backendUrl: 'http://127.0.0.1:8788',
})
console.log('registered tools (' + registered.length + '):', registered.map(t => t.name).join(', '))
console.log('(skipped vr_preflight execution — set VTL_TEST_VIDEO to a sample mp4 to run it)')
"
fi

node -e "
const fs = require('fs')
let def
global.window = { __ModuleLoader__: { load: (d) => { def = d } } }
require('$HERE/dist-client/index.js')
if (!def) throw new Error('client bundle did not register with __ModuleLoader__')
const fakeReact = { createElement: (t,p,...c)=>({t}), useSyncExternalStore: (s,g)=>g(), useEffect: (f)=>{f()}, useRef: ()=>({current:null}), useState: (i)=>[i,()=>{}] }
const regs = []
const api = def.factory((id) => id === 'react' ? fakeReact : undefined)
api.apply({ get: (k) => k === 'slots' ? { inject: (n,cb)=>{cb();return()=>{}}, register: (o)=>{regs.push(o.name)} } : undefined })
if (!regs.includes('shell.overlay') || !regs.includes('sidebar.footer.action')) throw new Error('missing slot registrations')
console.log('client bundle smoke: OK')
"

echo
echo "✅ install complete."
echo "   RESTART dsh web (bundle patch is a boot-time fact):"
echo "     npx @deepseek-ai/dsh web"
echo "   Then pick 视频检索模式 in the preset picker."
