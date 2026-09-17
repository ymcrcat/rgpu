#!/usr/bin/env bash
# Start rgpu-opserver on a GPU pod, detached, and prove it is listening.
#
#   bash scripts/opserver_pod.sh 2.14        # the Mac's torch major.minor
set -euo pipefail
cd "$(dirname "$0")/.."
want=${1:?give the client torch version, e.g. 2.14}
venv=${RGPU_OPSERVER_VENV:-$HOME/opvenv}
log=${RGPU_OPSERVER_LOG:-$HOME/opserver.log}
if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv "$venv"
fi
if ! "$venv/bin/python" -c "import torch,sys; sys.exit(torch.__version__.split('.')[:2] != '$want'.split('.')[:2])" 2>/dev/null; then
  for cu in cu128 cu126 cu130; do
    echo "installing Torch $want from the $cu index (the CUDA wheel is several GB)"
    "$venv/bin/pip" install "torch==$want.*" \
      --index-url "https://download.pytorch.org/whl/$cu" && break
  done
fi
"$venv/bin/python" -c "import torch,sys; sys.exit(torch.__version__.split('.')[:2] != '$want'.split('.')[:2])" || {
  echo "could not install Torch $want with a compatible CUDA wheel" >&2
  exit 1
}
"$venv/bin/pip" install -q -e python --no-deps
pkill -f "rgpu.server" 2>/dev/null || true
sleep 1
setsid nohup "$venv/bin/python" -m rgpu.server --device cuda > "$log" 2>&1 < /dev/null &
sleep 5
if (exec 3<>/dev/tcp/127.0.0.1/9720) 2>/dev/null; then echo LISTENING; else echo DEAD; tail -20 "$log"; fi
