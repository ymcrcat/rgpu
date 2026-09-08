#!/usr/bin/env bash
# Regenerate the API model and the generated sources.
#
#   ./codegen/run.sh
#
# Parses cuda.h in a container and writes codegen/api.json plus the generated
# C++ under */generated/.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=rgpu-codegen
PLATFORM=${PLATFORM:-linux/arm64}
HEADERS=third_party/cuda_include

if [[ ! -f "$HEADERS/cuda.h" ]]; then
  echo "cuda.h missing; running scripts/fetch_headers.sh first"
  ./scripts/fetch_headers.sh
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE ..."
  docker build --platform "$PLATFORM" -t "$IMAGE" codegen/
fi

run() {
  docker run --rm --platform "$PLATFORM" -v "$PWD:/src" -w /src "$IMAGE" "$@"
}

run codegen/parse.py --header "$HEADERS/cuda.h" -o codegen/api.json
run codegen/emit.py --api codegen/api.json

echo
echo "coverage summary (codegen/report.txt):"
head -1 codegen/report.txt
