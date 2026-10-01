#!/usr/bin/env bash
# Run the Section 7.2 native and colocated legs on the GPU host, then leave
# the rgpu server up for a WAN leg from the client. Run ON the GPU host:
#
#   bash run_on_gpu_host.sh
#
# Writes native.json and colocated.json next to gpu_benchmark.py, plus
# env.txt recording exactly what torch and driver produced them.
#
# Torch: try the version the client runs (2.14.0, whose PyPI wheel bundles
# CUDA 13) first, because a matched pair removes the client/server version
# mismatch that Section 7.1 lists as a threat to validity. It only works on a
# CUDA-13 driver; on an older one cuda.is_available() is False and we fall
# back to 2.11.0+cu128, and the WAN leg then needs the mismatch override.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/root/rgpu_pkg

echo "== venv"
python3 -m venv /root/paperenv 2>/dev/null || true
PIP=/root/paperenv/bin/pip; PY=/root/paperenv/bin/python
$PIP install -q --upgrade pip >/dev/null 2>&1
$PIP install -q numpy >/dev/null 2>&1

echo "== torch: try the client's version first"
$PIP install -q "torch==2.14.0" >/dev/null 2>&1 || true
if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "   2.14.0 cannot see the GPU on this driver; falling back to 2.11.0+cu128"
  $PIP uninstall -q -y torch >/dev/null 2>&1 || true
  $PIP install -q "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128 >/dev/null 2>&1
fi
$PY - <<'PY' | tee env.txt
import torch, subprocess
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("driver", subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
      capture_output=True, text=True).stdout.strip())
PY
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'"

echo "== native leg"
$PY gpu_benchmark.py --mode native > native.json
echo "== colocated leg"
$PY gpu_benchmark.py --mode colocated > colocated.json

echo "== summaries"
$PY - <<'PY'
import json
for f in ("native.json", "colocated.json"):
    d = json.load(open(f)); e = d["environment"]
    print(f, "| torch", e["torch"], "| gpu", e["gpu"],
          "| check %.5f vs %.5f" % (e["numerical_check"]["cpu_ref_loss"], e["numerical_check"]["device_loss"]))
    for k, v in d["summary"].items():
        print("   %-28s med=%9.2f ms  msgs=%6d waits=%3d" % (k, v["median_ms"], v["messages"], v["waits"]))
PY

echo "== leaving an rgpu server up for the WAN leg (127.0.0.1:9720, tunnel in)"
pkill -f "rgpu.server" 2>/dev/null || true
export RGPU_TF32=0
setsid nohup $PY -m rgpu.server --device cuda --bind 127.0.0.1 --port 9720 > /root/opserver.log 2>&1 < /dev/null &
sleep 8; ss -tlnp 2>/dev/null | grep -q 9720 && echo "   server LISTENING" || { echo "   server NOT listening"; tail -5 /root/opserver.log; }
