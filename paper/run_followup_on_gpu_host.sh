#!/usr/bin/env bash
# Follow-up experiments on the GPU host (see gpu_experiments.py). Run ON the host:
#
#   bash run_followup_on_gpu_host.sh
#
# Client, server and delay proxy are pinned to disjoint cores, so the client
# and server cannot compete for a core, and latency is injected by a proxy on
# loopback, so the round trip changes while the client CPU does not. Writes
# one JSON per condition into followup/, plus host.txt. Leaves a pinned
# server on 127.0.0.1:9720 for a leg from the laptop.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/root/rgpu_pkg RGPU_TF32=0
PIP=/root/paperenv/bin/pip; PY=/root/paperenv/bin/python
OUT=followup; mkdir -p $OUT

python3 -m venv /root/paperenv 2>/dev/null || true
$PIP install -q --upgrade pip >/dev/null 2>&1
$PIP install -q numpy "torch==2.14.0" >/dev/null 2>&1
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'" || exit 1

# The first 18 cores this container may use: 8 server, 8 client, 2 proxy.
read -r SERVER CLIENT PROXY SHARED <<<"$($PY - <<'PY'
import os
c = sorted(os.sched_getaffinity(0))
j = lambda xs: ",".join(map(str, xs))
print(j(c[0:8]), j(c[8:16]), j(c[16:18]), j(c[0:4]))
PY
)"

{
  echo "date $(date -u +%FT%TZ)"
  echo "package_sha256 $(sha256sum /root/rgpu_pkg.tgz | cut -d' ' -f1)"
  for f in gpu_experiments.py gpu_benchmark.py delay_proxy.py run_followup_on_gpu_host.sh; do
    echo "sha256 $f $(sha256sum $f | cut -d' ' -f1)"; done
  $PY -c "import torch,sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))"
  echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  echo "nproc $(nproc)"
  echo "cgroup_cpu_max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)"
  echo "cpuset $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null)"
  echo "cores server=$SERVER client=$CLIENT proxy=$PROXY shared=$SHARED"
  lscpu | grep -E "Model name|Socket|Core\(s\)|Thread|NUMA node[0-9]"
  echo "loadavg $(cat /proc/loadavg)"
} | tee $OUT/host.txt

SPID=
server() {   # server <cores or empty>
  [ -n "$SPID" ] && kill $SPID 2>/dev/null && sleep 2
  if [ -n "$1" ]; then taskset -c "$1" $PY -m rgpu.server --device cuda --bind 127.0.0.1 --port 9720 > /root/opserver.log 2>&1 &
  else $PY -m rgpu.server --device cuda --bind 127.0.0.1 --port 9720 > /root/opserver.log 2>&1 & fi
  SPID=$!
  for _ in $(seq 60); do ss -tln | grep -q ":9720 " && return; sleep 1; done
  echo "server did not start"; tail -5 /root/opserver.log; exit 1
}
PPID_=
proxy() {    # proxy <one-way ms>
  [ -n "$PPID_" ] && kill $PPID_ 2>/dev/null
  taskset -c "$PROXY" $PY delay_proxy.py --listen 9731 --target 127.0.0.1:9720 --one-way-ms "$1" > /root/proxy.log 2>&1 &
  PPID_=$!; sleep 1
}
run() {      # run <name> <client cores or empty> <port> <cases> <label>
  echo "== $1  ($(date +%T))"
  local pin=""; [ -n "$2" ] && pin="taskset -c $2"
  $pin $PY gpu_experiments.py --exp train --server 127.0.0.1:$3 --server-pid $SPID \
    --cases "$4" --label "$5" > $OUT/$1.json 2> $OUT/$1.err || { echo "   FAILED"; tail -3 $OUT/$1.err; }
  echo "loadavg after $1: $(cat /proc/loadavg)" >> $OUT/host.txt
}
ALL=eager,eager_loss_read,eager_foreach,graph_eager
NET=eager,eager_loss_read,graph_eager

echo "== optimizer (native)"
server "$SERVER"
taskset -c "$CLIENT" $PY gpu_experiments.py --exp optimizer --device cuda --server 127.0.0.1:9720 \
  --label "client pinned $CLIENT" > $OUT/optimizer.json 2> $OUT/optimizer.err || tail -3 $OUT/optimizer.err
echo "== correctness"
taskset -c "$CLIENT" $PY gpu_experiments.py --exp correctness --device cuda --server 127.0.0.1:9720 \
  > $OUT/correctness.json 2> $OUT/correctness.err || tail -3 $OUT/correctness.err

run pinned_direct   "$CLIENT" 9720 $ALL "disjoint cores, no proxy"
# RTT conditions in a shuffled order, so drift is not confounded with delay.
for ms in 25 0 58 5; do
  proxy $ms
  run proxy_oneway_${ms}ms "$CLIENT" 9731 $NET "disjoint cores, proxy one-way ${ms} ms"
done
server "$SHARED"
run shared_cores    "$SHARED" 9720 eager "client and server on the same 4 cores"
server ""
run unpinned        ""        9720 eager,eager_loss_read "no pinning, as in colocated.json"
server "$SERVER"
run pinned_direct_repeat "$CLIENT" 9720 eager "disjoint cores, no proxy, end of session"

echo "== summaries"
$PY - <<'PY'
import json, glob
for f in sorted(glob.glob("followup/*.json")):
    try: d = json.load(open(f))
    except Exception as e: print(f, "UNREADABLE", e); continue
    print(f, "|", d["environment"].get("label"))
    if "summary" in d:
        for k, v in d["summary"].items():
            print("   %-18s med=%9.2f  [%8.2f..%8.2f] msgs=%6s waits=%3s client_cpu=%8.1f server_cpu=%s" % (
                k, v["median_ms"], v["min_ms"], v["max_ms"], v["messages"], v["waits"],
                v["client_cpu_ms"], v["server_cpu_ms"]))
        print("   adamw:", d["environment"].get("adamw_selects"))
    else:
        r = d["result"]; print("   ref", r["reference"], "grad scale", r["reference_max_abs_first_grad"])
        for k, v in r["variants"].items():
            print("   %-15s" % k, {a: b for a, b in v.items() if a != "losses"})
PY
echo "== server left pinned on 127.0.0.1:9720 (pid $SPID) for the laptop leg"
echo "=== FOLLOWUP FINISHED ==="
