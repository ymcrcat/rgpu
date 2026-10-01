"""CPU loopback mechanism benchmark; no GPU or network-performance claim.

PYTHONPATH=/path/to/rgpu/python python benchmark.py > results.json
Uses the existing rgpu and torch installation. Starts only a local CPU server.
"""
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import rgpu
from rgpu import session


def chain(x):
    for _ in range(10):
        x = x * 1.001 + 0.01
    return x


def main():
    torch.set_num_threads(1)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ["RGPU_OPSERVER"] = f"127.0.0.1:{port}"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "rgpu.server", "--device", "cpu",
                             "--port", str(port)], env=env, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.1):
                    break
            except OSError:
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("CPU server failed to start")
                time.sleep(.1)
        local = torch.linspace(-1, 1, 1024)
        remote = local.to("rgpu")
        compiled = torch.compile(chain, backend=rgpu.compile_backend(compiler="eager"),
                                 dynamic=False, fullgraph=True)
        expected = chain(local)
        cases = ["native_cpu", "eager_flush1", "eager_flush64",
                 "eager_sync_each_call", "graph_eager"]
        rows = []
        rng = random.Random(0)
        with torch.no_grad():
            for trial in range(7):
                order = cases[:]
                rng.shuffle(order)
                for name in order:
                    native = name == "native_cpu"
                    fn = compiled if name == "graph_eager" else chain
                    x = local if native else remote
                    session.get().flush_ops = 1 if name == "eager_flush1" else 64
                    for _ in range(2):
                        out = fn(x)
                    torch.rgpu.synchronize()
                    before = rgpu.stats()
                    start = time.perf_counter_ns()
                    for _ in range(20):
                        out = fn(x)
                        if name == "eager_sync_each_call":
                            torch.rgpu.synchronize()
                    if not native:
                        torch.rgpu.synchronize()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    after = rgpu.stats()
                    torch.testing.assert_close(out if native else out.cpu(), expected)
                    rows.append({"case": name, "trial": trial, "elapsed_ms": elapsed,
                                 **{key: after[key] - before[key] for key in before}})
        summary = {}
        for name in cases:
            selected = [r for r in rows if r["case"] == name]
            times = [r["elapsed_ms"] for r in selected]
            summary[name] = {"median_ms": statistics.median(times),
                             "min_ms": min(times), "max_ms": max(times),
                             **{k: statistics.median([r[k] for r in selected])
                                for k in ("messages", "waits", "bytes_out", "bytes_in")}}
        assert summary["eager_flush64"]["waits"] == 1
        assert summary["eager_flush1"]["waits"] == 1
        assert summary["eager_sync_each_call"]["waits"] == 21
        assert summary["graph_eager"]["messages"] < summary["eager_flush64"]["messages"]
        repo = Path(rgpu.__file__).resolve().parents[2]
        print(json.dumps({"environment": {"platform": platform.platform(),
                         "python": sys.version, "torch": torch.__version__,
                         "source": str(repo), "commit": subprocess.check_output(
                             ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                         "cuda_available": torch.cuda.is_available(),
                         "mps_available": torch.backends.mps.is_available(),
                         "client_torch_threads": torch.get_num_threads()},
                         "method": {"trials": 7, "warmup_calls_per_case": 2,
                                    "timed_calls": 20, "ops_per_call": 20,
                                    "elements": 1024, "dtype": "float32",
                                    "server": "separate CPU process over TCP loopback",
                                    "graph_compiler": "eager; no kernel fusion",
                                    "seed_for_case_order": 0},
                         "summary": summary, "samples": rows}, indent=2))
    finally:
        session.reset()
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
