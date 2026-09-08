#!/usr/bin/env bash
# Run the PyTorch ladder on the GPU host, against a server on the same box.
#
#   ./scripts/deploy_server.sh user@host     # first, builds the shims there
#   ./scripts/remote_torch.sh   user@host
#
# No container is needed. The host has a real driver, but LD_LIBRARY_PATH puts
# our libcuda.so.1 and libcudart.so.12 ahead of it, so PyTorch talks to the
# shims, which reach the server over loopback, which uses the real driver. The
# script checks that the shims really are what got loaded.
#
# This is the fastest way to a working PyTorch: it takes the network and the
# client's architecture out of the picture while the CUDA surface is still
# being filled in. Moving the client to another machine afterwards is a change
# of RGPU_SERVER.
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
TORCH_VERSION=${TORCH_VERSION:-2.9.1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

echo "== copying the ladder =="
scp "${SSH_OPTS[@]}" tests/torch/ladder.py "$TARGET:$REMOTE_DIR/ladder.py"

echo
echo "== preparing a torch environment on the GPU host =="
ssh "${SSH_OPTS[@]}" "$TARGET" bash -s <<EOF
set -e
cd $REMOTE_DIR
if [[ ! -x venv/bin/python ]]; then
  python3 -m venv venv
fi
# torchvision is only needed for the ResNet rung.
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX"
./venv/bin/python -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
EOF

echo
echo "== running =="
ssh "${SSH_OPTS[@]}" "$TARGET" bash -s <<EOF
set -e
cd $REMOTE_DIR

# Start the server if one is not already listening.
if ! (exec 3<>/dev/tcp/127.0.0.1/9713) 2>/dev/null; then
  echo "starting rgpu-server"
  ./build/rgpu-server >server.log 2>&1 &
  for _ in \$(seq 1 50); do
    (exec 3<>/dev/tcp/127.0.0.1/9713) 2>/dev/null && break
    sleep 0.2
  done
fi

export LD_LIBRARY_PATH=\$PWD/build
export RGPU_SERVER=127.0.0.1:9713

echo
echo "libraries actually loaded by the ladder:"
LD_LIBRARY_PATH=\$PWD/build ldd ./venv/lib/python*/site-packages/torch/lib/libtorch_cuda.so \
  2>/dev/null | grep -E "libcuda\.so|libcudart" || true

echo
./venv/bin/python ladder.py
EOF
