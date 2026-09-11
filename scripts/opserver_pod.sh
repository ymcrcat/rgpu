#!/usr/bin/env bash
# Start rgpu-opserver on a GPU pod, detached, and prove it is listening.
#
#   bash scripts/opserver_pod.sh 2.14        # the Mac's torch major.minor
set -euo pipefail
cd "$(dirname "$0")/.."
want=${1:?give the client torch version, e.g. 2.14}
if [[ ! -x /root/opvenv/bin/python ]]; then
  python3 -m venv /root/opvenv
fi
if ! /root/opvenv/bin/python -c "import torch,sys; sys.exit(not torch.__version__.startswith('$want'))" 2>/dev/null; then
  for cu in cu128 cu126 cu130; do
    /root/opvenv/bin/pip install -q "torch==$want.*" --index-url "https://download.pytorch.org/whl/$cu" && break
  done
fi
/root/opvenv/bin/pip install -q -e python --no-deps
pkill -f "rgpu.server" 2>/dev/null || true
sleep 1
setsid nohup /root/opvenv/bin/python -m rgpu.server --device cuda > /root/opserver.log 2>&1 < /dev/null &
sleep 5
if (exec 3<>/dev/tcp/127.0.0.1/9720) 2>/dev/null; then echo LISTENING; else echo DEAD; tail -20 /root/opserver.log; fi
