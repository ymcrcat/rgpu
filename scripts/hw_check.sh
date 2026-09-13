#!/usr/bin/env bash
# Real-hardware verification of the driver behaviours rgpu relies on, and of
# the real server end to end. Run ON THE GPU POD, from the repo root, after
# scripts/deploy_server.sh has copied the source and built build/:
#
#   ./scripts/hw_check.sh
#
# 1. builds tests/hw/driver_probe.cpp against the real libcuda and runs it
#    (no rgpu involved), see that file for what each check means;
# 2. runs tests/hw/e2e_check.sh against build/rgpu-server with the client shim.
#
# Everything goes to one log file, whose path is printed at the start and the
# end. Safe to run again. Needs no RunPod API key and touches nothing outside
# build/hw and the log. Exit status: 0 if both parts passed, 1 otherwise (a
# probe that only skipped checks counts as passed, and says so).
#
# Settings: HW_LOG (log path), NVCC, CXX, HW_IMAGE (a fatbin or PTX for the
# kernel checks; built from tests/cuda/vecadd_kernel.cu if absent), RGPU_NVRTC
# (a libnvrtc for the probe to compile the kernel itself), plus the HW_*
# settings of tests/hw/e2e_check.sh.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=$ROOT/build/hw
mkdir -p "$OUT"
LOG=${HW_LOG:-$OUT/hw_check-$(date +%Y%m%d-%H%M%S).log}

# Everything below, stdout and stderr, into the log as well as the terminal.
exec > >(tee -a "$LOG") 2>&1
TEE_PID=$!

CHILDREN=()
descendants() {
  local p
  for p in $(pgrep -P "$1" 2>/dev/null); do
    echo "$p"
    descendants "$p"
  done
}
cleanup() {
  local p i left
  # TERM first, so tests/hw/e2e_check.sh runs its own trap and stops its
  # servers; then anything still below this script, whatever started it.
  for p in ${CHILDREN[@]+"${CHILDREN[@]}"}; do
    kill -TERM "$p" 2>/dev/null
  done
  for i in $(seq 1 30); do
    left=0
    for p in ${CHILDREN[@]+"${CHILDREN[@]}"}; do
      kill -0 "$p" 2>/dev/null && left=1
    done
    ((left)) || break
    sleep 0.1
  done
  for p in $(descendants $$); do
    [[ "$p" == "$TEE_PID" ]] || kill -9 "$p" 2>/dev/null
  done
  echo
  echo "log: $LOG"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "rgpu hardware check, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "log: $LOG"
echo "host: $(uname -m) $(uname -sr)"
git -C "$ROOT" rev-parse --short HEAD 2>/dev/null | sed 's/^/commit: /' ||
  echo "commit: unknown (not a git checkout; deploy_server.sh copies a tarball)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total \
    --format=csv,noheader
else
  echo "WARNING: no nvidia-smi"
fi
if [[ ":${LD_LIBRARY_PATH:-}:" == *":$ROOT/build:"* || ":${LD_LIBRARY_PATH:-}:" == *":build:"* ]]; then
  echo "WARNING: LD_LIBRARY_PATH names build/, where the client shim is; the probe and the server are run without it"
fi
if pgrep -x rgpu-server >/dev/null 2>&1; then
  echo "NOTE: an rgpu-server is already running (pid $(pgrep -x rgpu-server | paste -sd, -)); it is left alone, but it holds GPU memory the E1 baseline will include"
fi

# --- toolchain ------------------------------------------------------------------

CXX=${CXX:-}
if [[ -z "$CXX" ]]; then
  for c in g++ c++ clang++; do
    if command -v "$c" >/dev/null 2>&1; then
      CXX=$c
      break
    fi
  done
fi
if [[ -z "$CXX" ]]; then
  echo "FAIL: no C++ compiler (apt-get install -y build-essential)"
  exit 1
fi
export CXX

# The driver API header. The repo's copy (12.8, what the server builds
# against) first; a toolkit's own cuda.h only if the repo's is missing.
CUDA_INC=""
for d in "$ROOT/third_party/cuda_include" /usr/local/cuda/include /usr/include; do
  if [[ -f "$d/cuda.h" ]] && grep -q "cuDevicePrimaryCtxRetain" "$d/cuda.h"; then
    CUDA_INC=$d
    break
  fi
done
if [[ -z "$CUDA_INC" ]]; then
  echo "FAIL: no driver-API cuda.h (expected third_party/cuda_include; run scripts/fetch_headers.sh)"
  exit 1
fi
export CUDA_INC
echo "cuda.h: $CUDA_INC ($(grep -m1 -E '^#define CUDA_VERSION' "$CUDA_INC/cuda.h"))"

# The real driver library, never the shim in build/.
DRIVER_LIB=""
while read -r path; do
  if [[ -e "$path" && "$path" != "$ROOT"/* ]]; then
    DRIVER_LIB=$path
    break
  fi
done < <(
  ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF}'
  ls /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/aarch64-linux-gnu/libcuda.so.1 \
    /usr/lib64/libcuda.so.1 /usr/local/nvidia/lib64/libcuda.so.1 \
    /usr/local/cuda/compat/libcuda.so.1 2>/dev/null
)
LINK_LIB=$DRIVER_LIB
if [[ -z "$LINK_LIB" ]]; then
  for s in /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so; do
    [[ -e "$s" ]] && LINK_LIB=$s && break
  done
fi
if [[ -z "$LINK_LIB" ]]; then
  echo "FAIL: no libcuda.so.1 found (is this a GPU host?)"
  exit 1
fi
echo "driver library: ${DRIVER_LIB:-none found; linking the toolkit stub $LINK_LIB}"

NVCC=${NVCC:-}
if [[ -z "$NVCC" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC=$(command -v nvcc)
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    NVCC=/usr/local/cuda/bin/nvcc
  fi
fi

# A libnvrtc, so the probe can compile its kernel if there is no fatbin.
if [[ -z "${RGPU_NVRTC:-}" ]]; then
  for n in /usr/local/cuda/lib64/libnvrtc.so* \
    "$ROOT"/venv/lib/python3*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*; do
    if [[ -e "$n" && "$n" != *builtins* ]]; then
      export RGPU_NVRTC=$n
      break
    fi
  done
fi

# --- kernel image -----------------------------------------------------------------

IMAGE=${HW_IMAGE:-$ROOT/build/vecadd.fatbin}
if [[ ! -f "$IMAGE" && -n "$NVCC" ]]; then
  echo
  echo "== building the vecadd fatbin with $NVCC"
  if ! NVCC=$NVCC ./scripts/build_fatbin.sh; then
    # An older toolkit may not know every architecture build_fatbin.sh names.
    echo "build_fatbin.sh failed; trying the toolkit's default architectures"
    "$NVCC" -fatbin -o "$ROOT/build/vecadd.fatbin" tests/cuda/vecadd_kernel.cu ||
      echo "nvcc could not build the fatbin either"
  fi
fi
PROBE_IMAGE_ARGS=()
if [[ -f "$IMAGE" ]]; then
  PROBE_IMAGE_ARGS=(--image "$IMAGE")
  echo "kernel image: $IMAGE"
elif [[ -n "${RGPU_NVRTC:-}" ]]; then
  echo "kernel image: none; the probe compiles one with $RGPU_NVRTC"
else
  echo "kernel image: none, and no nvcc or libnvrtc; kernel checks will SKIP"
fi
export HW_IMAGE=$IMAGE

# --- part 1: the driver probe --------------------------------------------------------

echo
echo "################ part 1: driver probe (real libcuda, no rgpu) ################"
probe_rc=1
if "$CXX" -std=c++17 -O2 -g -Wall -Wextra -Wno-unused-parameter \
  -Wno-deprecated-declarations -I"$CUDA_INC" tests/hw/driver_probe.cpp \
  -o "$OUT/driver_probe" "$LINK_LIB" -ldl -lpthread; then
  # No LD_PRELOAD, and no build/ on the library path: nothing of rgpu.
  ldpath=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' |
    grep -vxF "$ROOT/build" | grep -vxF "build" | paste -sd: -)
  env -u LD_PRELOAD LD_LIBRARY_PATH="$ldpath" \
    "$OUT/driver_probe" ${PROBE_IMAGE_ARGS[@]+"${PROBE_IMAGE_ARGS[@]}"} &
  CHILDREN+=($!)
  wait $!
  probe_rc=$?
else
  echo "FAIL: the driver probe did not compile"
fi
case $probe_rc in
0) probe_verdict="passed" ;;
2) probe_verdict="passed, with checks skipped (see SKIP lines)" ;;
*) probe_verdict="FAILED (exit $probe_rc)" ;;
esac

# --- part 2: end to end ------------------------------------------------------------

echo
echo "################ part 2: real rgpu-server end to end ################"
e2e_rc=1
if [[ -x build/rgpu-server && -e build/libcuda.so.1 ]]; then
  tests/hw/e2e_check.sh &
  CHILDREN+=($!)
  wait $!
  e2e_rc=$?
else
  echo "FAIL: build/rgpu-server or build/libcuda.so.1 is missing; run scripts/deploy_server.sh first"
  ls -la build/rgpu-server build/*.so* 2>&1 | sed 's/^/  /'
fi

echo
echo "################ summary ################"
echo "part 1, driver probe: $probe_verdict"
echo "part 2, end to end:   $([[ $e2e_rc == 0 ]] && echo passed || echo "FAILED (exit $e2e_rc)")"
if [[ $probe_rc == 0 || $probe_rc == 2 ]] && [[ $e2e_rc == 0 ]]; then
  exit 0
fi
exit 1
