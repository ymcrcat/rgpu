#!/usr/bin/env bash
# Build the IFRT proxy server on a Linux box, against the XLA revision that
# matches the client's jaxlib. Run it ON the build host, not from the Mac.
#
#   bash build_on_host.sh            # cpu backend, the cheap proof
#   BACKEND=CUDA bash build_on_host.sh
#
# The revision is pinned deliberately. jaxlib 0.11.1 was built from XLA
# dcf304bc (2026-08-17), and the IFRT proxy performs a version handshake
# between client and server, so a server built from a different revision may
# refuse a 0.11.1 client even when it compiles. Pinning also avoids whatever
# HEAD happens to be broken on: HEAD of 2026-09-15 failed inside gRPC, on a
# target that includes protobuf's upb headers without declaring them, before
# reaching any of our code.
#
# To move to another jaxlib, read the commit out of the matching jax tag:
#   gh api repos/jax-ml/jax/contents/third_party/xla/revision.bzl?ref=jax-vX.Y.Z \
#     -q .content | base64 -d | grep XLA_COMMIT
set -euo pipefail

XLA_COMMIT=${XLA_COMMIT:-dcf304bc5dca1932b99f740b911dbd73631a1a69}   # jaxlib 0.11.1
BACKEND=${BACKEND:-CPU}
SRC=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-/root}

echo "== toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl build-essential python3 python3-dev >/dev/null 2>&1
if ! command -v bazel >/dev/null; then
  curl -sSL -o /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
  chmod +x /usr/local/bin/bazel
fi

echo "== xla at $XLA_COMMIT"
cd "$ROOT"
if [ ! -d xla ]; then
  # A shallow clone cannot check out an arbitrary commit, so fetch just that one.
  mkdir xla && cd xla && git init -q
  git remote add origin https://github.com/openxla/xla.git
  git fetch -q --depth 1 origin "$XLA_COMMIT"
  git checkout -q FETCH_HEAD
else
  cd xla
fi
echo "   $(git rev-parse --short HEAD), $(nproc) cores"

echo "== our server into the tree"
mkdir -p xla/python/ifrt_proxy/rgpu
cp "$SRC/rgpu_ifrt_server.cc" "$SRC/BUILD" xla/python/ifrt_proxy/rgpu/

echo "== configure ($BACKEND)"
python3 configure.py --backend="$BACKEND" >"$ROOT/configure.log" 2>&1

# Upstream's own target first: a failure there is XLA's, not ours, and that
# distinction is worth the two minutes it costs on a warm cache.
echo "== upstream grpc_server (proves the toolchain)"
bazel build -c opt --jobs="$(nproc)" \
  //xla/python/ifrt_proxy/server:grpc_server 2>&1 | tail -5

echo "== rgpu_ifrt_server"
bazel build -c opt --jobs="$(nproc)" \
  //xla/python/ifrt_proxy/rgpu:rgpu_ifrt_server 2>&1 | tail -15
ls -la bazel-bin/xla/python/ifrt_proxy/rgpu/rgpu_ifrt_server 2>/dev/null \
  || echo "not built; see the errors above"
