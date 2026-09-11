#!/usr/bin/env bash
# Install the client libraries so CUDA programs on this machine find them.
#
#   ./scripts/install.sh                     # into /opt/rgpu, plus an `rgpu` command
#   ./scripts/install.sh --system            # and preload for every process
#   ./scripts/install.sh --prefix DIR --from BUILD_DIR
#
# Then:
#
#   RGPU_SERVER=gpuhost:9713 rgpu python train.py
#
# Why preloading rather than LD_LIBRARY_PATH: our libcuda.so.1 would be found
# the ordinary way, since a client has no real driver, but PyTorch finds the
# runtime and the maths libraries through RPATHs baked into its own libraries
# and loads them by absolute path from the nvidia pip packages. Both win over
# LD_LIBRARY_PATH. A preloaded library with the same SONAME wins over both.
#
# --system writes /etc/ld.so.preload, so every process gets the libraries with
# nothing to remember. That is the right thing in a container built for this.
# It is the wrong thing on a machine with a real NVIDIA GPU, where it would
# hide the real driver from everything, so it refuses there.
set -euo pipefail

PREFIX=/opt/rgpu
FROM=build
SYSTEM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX=$2; shift 2 ;;
    --from) FROM=$2; shift 2 ;;
    --system) SYSTEM=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

# libcuda last: the others are what PyTorch asks for by name, and the driver
# is what they all end up calling.
LIBS=(libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcudnn.so.9 libcuda.so.1)

# A library built for the wrong architecture fails later with a message about
# something else entirely, so check here. The ELF machine field says which.
want=$(uname -m)
for lib in "${LIBS[@]}"; do
  f="$FROM/$lib"
  if [[ ! -f "$f" ]]; then
    echo "missing $f; build the client first (scripts/build_client.sh)" >&2
    exit 1
  fi
  machine=$(od -An -tx1 -j18 -N2 "$f" | tr -d ' \n')
  case "$machine" in
    3e00) arch=x86_64 ;;
    b700) arch=aarch64 ;;
    *) arch="unknown ($machine)" ;;
  esac
  if [[ "$arch" != "$want" ]]; then
    echo "$f is built for $arch but this machine is $want." >&2
    echo "Rebuild with PLATFORM=linux/$( [[ $want == x86_64 ]] && echo amd64 || echo arm64 ) ./scripts/build_client.sh" >&2
    exit 1
  fi
done

install -d "$PREFIX/lib" "$PREFIX/bin"
for lib in "${LIBS[@]}"; do
  install -m 0755 "$FROM/$lib" "$PREFIX/lib/$lib"
  ln -sf "$lib" "$PREFIX/lib/${lib%%.so*}.so"
done

PRELOAD=$(printf "$PREFIX/lib/%s:" "${LIBS[@]}")
PRELOAD=${PRELOAD%:}

cat > "$PREFIX/bin/rgpu" <<EOF
#!/bin/sh
# Run a command with its CUDA calls going to the GPU host in RGPU_SERVER.
if [ \$# -eq 0 ]; then
  echo "usage: RGPU_SERVER=host:port rgpu COMMAND [ARGS...]" >&2
  exit 2
fi
export LD_PRELOAD="$PRELOAD\${LD_PRELOAD:+:\$LD_PRELOAD}"
export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "\$@"
EOF
chmod 0755 "$PREFIX/bin/rgpu"
if [[ -w /usr/local/bin ]]; then
  ln -sf "$PREFIX/bin/rgpu" /usr/local/bin/rgpu
fi

if [[ $SYSTEM -eq 1 ]]; then
  if [[ -e /proc/driver/nvidia/version ]]; then
    echo "refusing --system: this machine has a real NVIDIA driver, and" >&2
    echo "preloading for every process would hide it. Use the rgpu command." >&2
    exit 1
  fi
  # Preserve anything already there, and do not add ours twice.
  touch /etc/ld.so.preload
  grep -vF "$PREFIX/lib/" /etc/ld.so.preload > /etc/ld.so.preload.new || true
  printf "%s\n" "${LIBS[@]/#/$PREFIX/lib/}" >> /etc/ld.so.preload.new
  mv /etc/ld.so.preload.new /etc/ld.so.preload
  echo "every process on this machine now preloads the rgpu libraries"
fi

echo "installed into $PREFIX"
echo "  run:  RGPU_SERVER=gpuhost:9713 ${PREFIX}/bin/rgpu python your_script.py"
