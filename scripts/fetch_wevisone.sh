#!/usr/bin/env bash
# fetch_wevisone.sh — download the WeVisOne-VL-2B snapshot into
#   $DEST/models/hf-home/hub/models--tencent--WeVisOne-VL-2B/
# (the exact layout wevisone_local._resolve_model_path() looks up, so the
# runtime stays fully offline afterwards). Tries huggingface.co first, then
# the hf-mirror.com fallback for restricted networks.
set -euo pipefail
DEST="${DEST:?set DEST to the package dir}"
PY="${VTL_PYTHON:-python3}"
SNAP="$DEST/models/hf-home"
mkdir -p "$SNAP"

download_from() {
  local endpoint="$1"
  WEVISONE_DEST="$DEST" HF_HOME="$SNAP" HF_ENDPOINT="$endpoint" "$PY" - <<'EOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("tencent/WeVisOne-VL-2B")
print("wevisone weights at", p)
EOF
}

if download_from https://huggingface.co; then
  echo "wevisone weights ready"
else
  echo "huggingface.co unreachable -> retrying via hf-mirror.com ..."
  download_from https://hf-mirror.com
  echo "wevisone weights ready"
fi
