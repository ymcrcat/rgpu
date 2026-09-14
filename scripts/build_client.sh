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

# Everything this build needs, so one command is enough from a fresh clone.
if [[ ! -f third_party/cuda_include/cuda.h ]]; then
  ./scripts/fetch_headers.sh
fi

# Regenerate when an input is newer than what it produced, not only when the
# output is missing. Editing codegen/annotations.py and getting a build of the
# code generated before the edit is a silent wrong answer, and the edit is the
# whole reason anyone touches the generator.
STUBS=client/generated/client_stubs.cpp
regen=0
if [[ ! -f $STUBS ]]; then
  regen=1
else
  for src in codegen/*.py third_party/cuda_include/cuda.h; do
    if [[ -f $src && $src -nt $STUBS ]]; then
      echo "$src is newer than the generated code; regenerating"
      regen=1
      break
    fi
  done
fi
if (( regen )); then
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
