#!/usr/bin/env bash
# Fetch a stock libcudart into third_party/cudart.
#
# This is the CUDA runtime PyTorch itself ships. Running a program against it
# with our shim underneath is the real phase-2 test: it exercises how cudart
# discovers driver entry points, which is the part that plain symbol
# interposition does not cover. About a megabyte, versus gigabytes for a
# PyTorch image.
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION=${CUDART_VERSION:-12.8.90}
ARCH=${ARCH:-$(uname -m)}
case "$ARCH" in
  aarch64|arm64) WHEEL_PLATFORM=manylinux2014_aarch64 ;;
  x86_64|amd64)  WHEEL_PLATFORM=manylinux2014_x86_64 ;;
  *) echo "unsupported arch $ARCH" >&2; exit 1 ;;
esac

DEST=third_party/cudart
if [[ -f "$DEST/libcudart.so.12" ]]; then
  echo "$DEST/libcudart.so.12 already present"
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "downloading nvidia-cuda-runtime-cu12==$VERSION for $WHEEL_PLATFORM ..."
python3 -m pip download "nvidia-cuda-runtime-cu12==$VERSION" \
  --platform "$WHEEL_PLATFORM" --only-binary=:all: --no-deps -d "$tmp" >/dev/null
python3 -m zipfile -e "$(echo "$tmp"/*.whl)" "$tmp/out"

mkdir -p "$DEST"
find "$tmp/out" -name 'libcudart.so*' -exec cp {} "$DEST/" \;
# The linker looks for the unversioned name; the loader uses the SONAME.
ln -sf libcudart.so.12 "$DEST/libcudart.so"
ls -la "$DEST"
