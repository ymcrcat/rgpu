#!/usr/bin/env bash
# Extract the CUDA headers we build against into third_party/cuda_include.
#
# They come from the nvidia-cuda-runtime pip wheel, which is about a megabyte,
# rather than the CUDA devel container image, which is several gigabytes. Only
# headers are needed: the shim replaces the driver instead of linking it, and
# the server is built on the GPU host where the toolkit already exists.
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION=${CUDA_HEADER_VERSION:-12.8.90}
NVCC_VERSION=${NVCC_HEADER_VERSION:-12.8.93}
DEST=third_party/cuda_include

if [[ -f "$DEST/cuda.h" ]]; then
  echo "$DEST/cuda.h already present"
  grep -E '^#define CUDA_VERSION' "$DEST/cuda.h" || true
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "downloading nvidia-cuda-runtime-cu12==$VERSION ..."
python3 -m pip download "nvidia-cuda-runtime-cu12==$VERSION" \
  --platform manylinux2014_x86_64 --only-binary=:all: --no-deps -d "$tmp/whl" \
  >/dev/null

# python3 -m zipfile avoids depending on unzip, which slim images lack.
python3 -m zipfile -e "$(echo "$tmp"/whl/*.whl)" "$tmp/out"
src="$tmp/out/nvidia/cuda_runtime/include"
if [[ ! -f "$src/cuda.h" ]]; then
  echo "cuda.h not found in the wheel; layout may have changed" >&2
  exit 1
fi

mkdir -p "$DEST"
cp -R "$src"/. "$DEST"/

# cuda_runtime_api.h includes crt/host_defines.h, which lives in the nvcc
# wheel rather than the runtime one. Only the headers are taken.
echo "downloading nvidia-cuda-nvcc-cu12==$NVCC_VERSION for crt headers ..."
python3 -m pip download "nvidia-cuda-nvcc-cu12==$NVCC_VERSION" \
  --platform manylinux2014_x86_64 --only-binary=:all: --no-deps -d "$tmp/nvcc" \
  >/dev/null
python3 -m zipfile -e "$(echo "$tmp"/nvcc/*.whl)" "$tmp/nvcc_out"
nvcc_inc="$tmp/nvcc_out/nvidia/cuda_nvcc/include"
if [[ -d "$nvcc_inc/crt" ]]; then
  cp -R "$nvcc_inc/crt" "$DEST/"
else
  echo "warning: crt headers not found in the nvcc wheel" >&2
fi

echo "extracted $(ls "$DEST" | wc -l | tr -d ' ') headers into $DEST"
grep -E '^#define CUDA_VERSION' "$DEST/cuda.h"
