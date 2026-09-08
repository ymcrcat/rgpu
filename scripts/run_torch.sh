#!/usr/bin/env bash
# Run the PyTorch ladder in the client container against a remote GPU server.
#
#   ./scripts/run_torch.sh                       # server reachable on the host
#   RGPU_SERVER=1.2.3.4:9713 ./scripts/run_torch.sh
#
# The container has no GPU and no NVIDIA driver. Stock PyTorch runs in it; only
# libcuda.so.1 is ours.
set -euo pipefail

cd "$(dirname "$0")/.."

PLATFORM=${PLATFORM:-linux/arm64}
IMAGE=${IMAGE:-rgpu-client}
# host.docker.internal reaches a server (or an ssh tunnel) on the Mac itself.
SERVER=${RGPU_SERVER:-host.docker.internal:9713}

if [[ ! -f build/libcuda.so.1 ]]; then
  echo "build/libcuda.so.1 missing; run ./scripts/build_client.sh first" >&2
  exit 1
fi

# Rebuild the image only when the shim is newer than it, since the torch layer
# is large and rarely changes.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1 \
   || [[ build/libcuda.so.1 -nt docker/libcuda.so.1 ]]; then
  cp build/libcuda.so.1 docker/libcuda.so.1
  docker build --platform "$PLATFORM" -t "$IMAGE" -f docker/client.Dockerfile docker/
fi

docker run --rm --platform "$PLATFORM" \
  -e RGPU_SERVER="$SERVER" \
  -e RGPU_VERBOSE="${RGPU_VERBOSE:-}" \
  -e RGPU_TRACE="${RGPU_TRACE:-}" \
  -v "$PWD/tests:/work/tests:ro" \
  "$IMAGE" \
  python3 /work/tests/torch/ladder.py
