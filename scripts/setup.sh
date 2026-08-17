#!/usr/bin/env bash
# setup.sh — download model weights into the INSTALLED package's weights/ dir.
# Skips files that already exist (idempotent).
#
# Weights are ~0.5 GB and not shipped in git/npm; this script pulls them from
# GitHub Releases (or a local mirror via WEIGHTS_URL_PREFIX env).
#
# Destination: by default this script's own package dir (for a standalone clone);
# install.sh overrides it with VTL_INSTALL_DIR so weights land where the engine
# actually looks (the pnpm-installed package under ~/.dsh/profiles/web/node_modules).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${VTL_INSTALL_DIR:-$HERE}"
PREFIX="${WEIGHTS_URL_PREFIX:-https://github.com/zhouql-dev/dsh-video-retrieval/releases/latest/download}"
mkdir -p "$DEST/weights/osnet" "$DEST/weights/clip"

download() {
  local name="$1" url="$2" dest="$3"
  if [ -e "$dest" ]; then echo "skip $name (exists)"; return; fi
  echo "downloading $name..."
  curl -fL --retry 3 -o "$dest" "$url"
}

# Release asset filenames (must match the GitHub Release uploads exactly).
OSNET_X0_25="osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
OSNET_X1_0="osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"

download yolov8s-worldv2.pt "$PREFIX/yolov8s-worldv2.pt" "$DEST/yolov8s-worldv2.pt"
download osnet_x0_25.pth        "$PREFIX/$OSNET_X0_25"      "$DEST/weights/osnet/$OSNET_X0_25"
download osnet_x1_0.pth         "$PREFIX/$OSNET_X1_0"       "$DEST/weights/osnet/$OSNET_X1_0"
download ViT-B-32.pt            "$PREFIX/ViT-B-32.pt"        "$DEST/weights/clip/ViT-B-32.pt"

echo "weights ready in $DEST/weights/"
