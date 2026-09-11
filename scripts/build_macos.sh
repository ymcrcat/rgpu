#!/usr/bin/env bash
# Build the client natively on macOS, no Docker.
#
#   ./scripts/build_macos.sh            # into build-macos/
#   ./scripts/build_macos.sh --test     # and run the GPU-free tests
#
# This is the driver API only: build-macos/libcuda.dylib, which any C or C++
# program written against cuda.h can link to and have its CUDA calls run on a
# remote GPU. It is not PyTorch. The macOS torch wheel contains libtorch_cpu
# and no CUDA backend at all, so there is nothing for this library to sit
# underneath; PyTorch on a Mac still means the Linux client container.
#
# ponytail: a script rather than CMake, because the CMake build is full of
# Linux-only choices (ELF version scripts, SONAMEs, the maths-library shims)
# that macOS does not need. The source list is duplicated here; fold it into
# CMake if macOS becomes a first-class target.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=build-macos
mkdir -p "$OUT"
FLAGS=(-std=c++17 -O2 -I. -Ithird_party/cuda_include)

if [[ ! -f third_party/cuda_include/cuda.h ]]; then ./scripts/fetch_headers.sh; fi

clang++ "${FLAGS[@]}" -dynamiclib -fPIC -install_name @rpath/libcuda.dylib \
  client/shim.cpp client/rpc.cpp \
  client/generated/client_stubs.cpp common/generated/api_names.cpp \
  -o "$OUT/libcuda.dylib"

# A server backed by a fake driver, so the client can be tested with no GPU.
# The maths-library dispatchers are optional weak symbols, which the macOS
# linker has to be told may be missing.
clang++ "${FLAGS[@]}" \
  server/main.cpp server/generated/dispatch.cpp common/generated/api_names.cpp \
  tests/generated/fake_driver.cpp tests/fake_cuda.cpp \
  -Wl,-undefined,dynamic_lookup -o "$OUT/rgpu-server-fake"

for t in rpc_smoke launch_smoke reconnect_smoke bench; do
  clang++ "${FLAGS[@]}" "tests/$t.cpp" -L"$OUT" -lcuda -Wl,-rpath,@executable_path \
    -o "$OUT/$t"
done
echo "built $(file -b "$OUT/libcuda.dylib")"

[[ "${1:-}" == "--test" ]] || exit 0

cd "$OUT"
./rgpu-server-fake 9801 > server.log 2>&1 &
S1=$!
RGPU_DROP_AFTER=12 RGPU_SESSION_GRACE=30 ./rgpu-server-fake 9802 > server-drop.log 2>&1 &
S2=$!
trap 'kill $S1 $S2 2>/dev/null' EXIT
for p in 9801 9802; do
  for _ in $(seq 1 50); do nc -z 127.0.0.1 "$p" 2>/dev/null && break; sleep 0.1; done
done
rc=0
RGPU_SERVER=127.0.0.1:9801 ./rpc_smoke || rc=1
RGPU_SERVER=127.0.0.1:9801 ./launch_smoke || rc=1
RGPU_SERVER=127.0.0.1:9802 ./reconnect_smoke || rc=1
exit $rc
