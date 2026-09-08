#!/usr/bin/env bash
# Build the test kernel into a fatbin. Needs nvcc, so this runs on the GPU
# host; the resulting fatbin is copied to the client, which cannot compile
# CUDA.
#
# The nvidia-cuda-nvcc pip wheel ships only ptxas and NVVM, not the nvcc
# driver, so there is no lightweight way to compile a .cu on a client with no
# CUDA toolkit. That is fine: the cross-architecture question this was meant to
# probe, whether an aarch64-built fatbin loads on an x86_64 driver, gets a
# real answer in phase 3, where PyTorch's own aarch64 libraries supply the
# fatbins. If it fails there, the client container switches to linux/amd64.
set -euo pipefail

cd "$(dirname "$0")/.."

NVCC=${NVCC:-nvcc}
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found. Run this on the GPU host, or set NVCC=/path/to/nvcc" >&2
  exit 1
fi

# sm_75 (Turing, T4) through sm_90 (Hopper) covers anything likely to be
# rented. Trailing PTX lets a newer GPU compile on the spot.
GENCODE=${GENCODE:-"-gencode arch=compute_75,code=sm_75 \
                    -gencode arch=compute_86,code=sm_86 \
                    -gencode arch=compute_89,code=sm_89 \
                    -gencode arch=compute_90,code=sm_90 \
                    -gencode arch=compute_90,code=compute_90"}

mkdir -p build
"$NVCC" -fatbin $GENCODE -o build/vecadd.fatbin tests/cuda/vecadd_kernel.cu

ls -la build/vecadd.fatbin
echo "built on $(uname -m)"
