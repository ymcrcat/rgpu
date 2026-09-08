#!/usr/bin/env bash
# Build the client shim (libcuda.so.1) and the host-side tests.
#
#   ./scripts/build_client.sh
#
# Produces build/libcuda.so.1. Point LD_LIBRARY_PATH at build/ and any CUDA
# program will call the shim instead of a driver.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=rgpu-build
PLATFORM=${PLATFORM:-linux/arm64}

if [[ ! -f third_party/cuda_include/cuda.h ]]; then
  ./scripts/fetch_headers.sh
fi
if [[ ! -f client/generated/client_stubs.cpp ]]; then
  ./codegen/run.sh
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build --platform "$PLATFORM" -t "$IMAGE" -f docker/build.Dockerfile docker/
fi

docker run --rm --platform "$PLATFORM" -v "$PWD:/src" -w /src "$IMAGE" bash -c '
  set -e
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCUDA_INCLUDE_DIR=/src/third_party/cuda_include
  cmake --build build -j"$(nproc)"
  ctest --test-dir build --output-on-failure
'

ls -la build/libcuda.so* 2>/dev/null || true
