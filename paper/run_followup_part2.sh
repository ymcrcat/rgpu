#!/usr/bin/env bash
# Native-only and correctness runs, in processes of their own (see
# gpu_experiments.py). Run ON the host after run_followup_on_gpu_host.sh,
# which leaves a pinned server on 127.0.0.1:9720.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/root/rgpu_pkg
PY=/root/paperenv/bin/python; OUT=followup; C=8-15
echo "sha256 gpu_experiments.py (part 2) $(sha256sum gpu_experiments.py | cut -d' ' -f1)" >> $OUT/host.txt
echo "== optimizer"
taskset -c $C $PY gpu_experiments.py --exp optimizer --device cuda --label "native, pinned $C" \
  > $OUT/optimizer.json 2> $OUT/optimizer.err || tail -3 $OUT/optimizer.err
echo "== correctness, native"
taskset -c $C $PY gpu_experiments.py --exp correctness --device cuda --save-ref /root/ref.pt \
  > $OUT/correctness_native.json 2> $OUT/correctness_native.err || tail -3 $OUT/correctness_native.err
echo "== correctness, rgpu"
taskset -c $C $PY gpu_experiments.py --exp correctness --device cuda --server 127.0.0.1:9720 --load-ref /root/ref.pt \
  > $OUT/correctness_rgpu.json 2> $OUT/correctness_rgpu.err || tail -3 $OUT/correctness_rgpu.err
rm -f $OUT/correctness.json $OUT/correctness.err
echo "loadavg after part 2: $(cat /proc/loadavg)" >> $OUT/host.txt
echo "=== PART2 FINISHED ==="
