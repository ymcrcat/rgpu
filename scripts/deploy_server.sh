#!/usr/bin/env bash
# Copy the source to a rented GPU box, build the server there, and bring the
# test fatbin back.
#
#   ./scripts/deploy_server.sh user@host
#   ./scripts/deploy_server.sh user@host -p 2222        # extra ssh options
#
# Assumes the box has an NVIDIA driver and the CUDA toolkit, which every GPU
# rental image ships. Reports clearly if either is missing.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 user@host [ssh options...]" >&2
  exit 1
fi
TARGET=$1
shift
SSH_OPTS=("$@")
REMOTE_DIR=${REMOTE_DIR:-rgpu}

ssh_run() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }

echo "== checking the remote box =="
ssh_run bash -s <<'EOF'
set -e
echo "host: $(uname -m) $(uname -sr)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "ERROR: nvidia-smi not found; is this a GPU host?" >&2
  exit 1
fi
if command -v nvcc >/dev/null; then
  nvcc --version | tail -2 | head -1
else
  echo "ERROR: nvcc not found; install the CUDA toolkit" >&2
  exit 1
fi
command -v cmake >/dev/null || echo "WARNING: cmake missing; will try to install"
EOF

echo
echo "== copying source =="
# Only what the server build needs. Generated sources go too, so the remote
# does not need libclang.
tar czf - \
  CMakeLists.txt common server client tests codegen scripts docs README.md \
  --exclude='__pycache__' --exclude='*.pyc' \
  | ssh_run "mkdir -p $REMOTE_DIR && tar xzf - -C $REMOTE_DIR"

echo
echo "== building on the GPU host =="
ssh_run bash -s <<EOF
set -e
cd $REMOTE_DIR
command -v cmake >/dev/null || sudo apt-get update -qq && sudo apt-get install -y -qq cmake build-essential || true
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
cmake --build build -j\$(nproc) 2>&1 | tail -5
echo
ls -la build/rgpu-server
./scripts/build_fatbin.sh
EOF

echo
echo "== bringing the fatbin back =="
mkdir -p build
scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_DIR/build/vecadd.fatbin" build/vecadd.fatbin

echo
cat <<EOF
Done. To run:

  1. Start the server on the GPU host and forward its port:

       ssh ${SSH_OPTS[*]} -L 9713:localhost:9713 $TARGET \\
         '$REMOTE_DIR/build/rgpu-server'

     The protocol has no authentication, so keep it on the tunnel and do not
     open port 9713 to the internet.

  2. From the client, with no GPU present:

       LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./build/rpc_smoke
       LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./build/vecadd build/vecadd.fatbin

     From inside a container on a Mac, use host.docker.internal:9713 instead.
EOF
