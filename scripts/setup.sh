#!/usr/bin/env bash
# setup.sh — download model weights into the package's weights/ directory.
# Skips files that already exist (idempotent).
#
# Weights are ~0.5 GB and not shipped in git/npm; this script pulls them from
# GitHub Releases (or a local mirror via WEIGHTS_URL_PREFIX env).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${WEIGHTS_URL_PREFIX:-https://github.com/zhouql1978_1/dsh-video-retrieval/releases/latest/download}"
mkdir -p "$HERE/weights/osnet" "$HERE/weights/clip"

download() {
  local name="$1" url="$2" dest="$3"
  if [ -e "$dest" ]; then echo "skip $name (exists)"; return; fi
  echo "downloading $name..."
  curl -fL --retry 3 -o "$dest" "$url"
}

download yolov8s-worldv2.pt "$PREFIX/yolov8s-worldv2.pt" "$HERE/weights/../yolov8s-worldv2.pt"
download osnet_x0_25.pth   "$PREFIX/osnet_x0_25_msmt17.pth" "$HERE/weights/osnet/osnet_x0_25_msmt17.pth"
download ViT-B-32.pt        "$PREFIX/ViT-B-32.pt"           "$HERE/weights/clip/ViT-B-32.pt"

echo "weights ready in $HERE/weights/"
