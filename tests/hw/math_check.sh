#!/usr/bin/env bash
# Real cuBLAS, cuBLASLt and cuDNN math through the rgpu shims, on a GPU host.
# Run on the pod from the repo root, after scripts/deploy_server.sh has built
# build/rgpu-server and the shims:
#
#   tests/hw/math_check.sh
#
# Builds tests/hw/cublas_check.cu and tests/hw/cudnn_check.cu, runs each once
# against the real libraries (to show the program itself is right) and once
# through the shims and a real rgpu-server on loopback. Prints PASS / FAIL
# lines and exits 1 if anything failed.
#
# Settings: NVCC (default nvcc, else /usr/local/cuda/bin/nvcc), MATH_VENV
# (where pip puts the real libraries, default ~/mathvenv), MATH_PORT (9831).
set -uo pipefail

cd "$(dirname "$0")/../.."
ROOT=$PWD
OUT=$ROOT/build/hw
PORT=${MATH_PORT:-9831}
VENV=${MATH_VENV:-$HOME/mathvenv}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
mkdir -p "$OUT"

# Matches the vendored headers: cuBLAS 12.8, cuDNN 9.
if [[ ! -x $VENV/bin/python ]]; then
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q \
    "nvidia-cublas-cu12==12.8.*" "nvidia-cudnn-cu12==9.*" || exit 1
fi
NV=$(echo "$VENV"/lib/python3*/site-packages/nvidia)

# nvcc will not take a library path ending in .so.N, so the linker gets them.
build() {
  "$NVCC" -cudart shared -Wno-deprecated-gpu-targets "$@" || exit 1
}
build -I third_party/cublas_include -o "$OUT/cublas_check" tests/hw/cublas_check.cu \
  -Xlinker "$NV/cublas/lib/libcublas.so.12" -Xlinker "$NV/cublas/lib/libcublasLt.so.12"
build -I third_party/cudnn_include -o "$OUT/cudnn_check" tests/hw/cudnn_check.cu \
  -Xlinker "$NV/cudnn/lib/libcudnn.so.9"

failed=0
real_libs="$NV/cublas/lib:$NV/cudnn/lib"
shims="$ROOT/build/libcudart.so.12:$ROOT/build/libcublas.so.12:$ROOT/build/libcublasLt.so.12:$ROOT/build/libcudnn.so.9:$ROOT/build/libcuda.so.1"

echo "== natively, against the real libraries =="
for prog in cublas_check cudnn_check; do
  LD_LIBRARY_PATH=$real_libs "$OUT/$prog" || failed=1
done

echo
echo "== through the shims and rgpu-server =="
RGPU_CUBLAS=$NV/cublas/lib/libcublas.so.12 \
RGPU_CUBLASLT=$NV/cublas/lib/libcublasLt.so.12 \
RGPU_CUDNN=$NV/cudnn/lib/libcudnn.so.9 \
LD_LIBRARY_PATH=$real_libs \
  "$ROOT/build/rgpu-server" "$PORT" > "$OUT/math_server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
for _ in $(seq 50); do
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  sleep 0.1
done

for prog in cublas_check cudnn_check; do
  # No real libraries on the client's path: if the shims did not take every
  # call, the program fails to load instead of quietly using the local GPU.
  # LD_DEBUG shows which files were actually loaded.
  LD_DEBUG=files LD_DEBUG_OUTPUT=$OUT/$prog.loads LD_PRELOAD=$shims \
    RGPU_SERVER=127.0.0.1:$PORT "$OUT/$prog" || failed=1
  if grep -h "file=.*nvidia/\(cublas\|cudnn\)/lib" "$OUT/$prog.loads".* >/dev/null 2>&1; then
    echo "FAIL $prog loaded a real cuBLAS or cuDNN on the client"
    failed=1
  fi
  rm -f "$OUT/$prog.loads".*
done

if grep -q "unimplemented\|not implemented" "$OUT/math_server.log"; then
  echo "note: the server log names unimplemented calls:"
  grep "unimplemented\|not implemented" "$OUT/math_server.log" | sort -u | head
fi
echo
[[ $failed == 0 ]] && echo "all math checks passed" || echo "some math checks FAILED"
exit $failed
