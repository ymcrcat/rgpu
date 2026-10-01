"""Same-GPU controlled experiment for Section 7.2: native vs colocated vs WAN.

    # on the GPU host, both of these (same torch, same GPU, same inputs):
    PYTHONPATH=/path/to/rgpu/python python gpu_benchmark.py --mode native    > native.json
    PYTHONPATH=/path/to/rgpu/python python gpu_benchmark.py --mode colocated > colocated.json
    # on the client, through an ssh tunnel to that same server:
    RGPU_OPSERVER=127.0.0.1:9720 python gpu_benchmark.py --mode wan       > wan.json

The point is the pairing. "native" runs PyTorch on the GPU directly with no
rGPU. "colocated" runs rGPU's client and server on that same host over
loopback, which isolates the cost of the remoting path from the cost of the
network. "wan" runs the client elsewhere against the same server, adding only
the network. Native-vs-colocated on one machine uses matching torch versions,
which the historical A40-vs-MPS comparison did not.

Two workloads, mirroring benchmark.py so the CPU and GPU artifacts read the
same way: the 1,024-element elementwise chain, and a training step of the
pinned transformer (10.8M parameters: n_embd 384, 6 heads, 6 layers, block
256, batch 64). Every timed block is bracketed by a device barrier; on rGPU
that is torch.rgpu.synchronize(), whose server handler calls
torch.cuda.synchronize(), so remote timings include GPU completion.

TF32 is disabled natively because the rGPU server disables it by default;
otherwise "native" would silently run matmuls at lower precision.

This measures one GPU, one host, one link. It does not compare against other
remoting systems, and the chain workload is deliberately small: it is the
adverse case for remoting, not a representative one.
"""
import argparse
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
import torch.nn as nn
from torch.nn import functional as F

# --- workloads ---------------------------------------------------------------

BATCH, BLOCK, N_EMBD, N_HEAD, N_LAYER, VOCAB = 64, 256, 384, 6, 6, 65


def chain(x):
    for _ in range(10):
        x = x * 1.001 + 0.01
    return x


class Head(nn.Module):
    def __init__(self, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK, BLOCK)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class Block(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        hs = N_EMBD // N_HEAD
        self.heads = nn.ModuleList([Head(hs, dropout) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.ffwd = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.ReLU(),
                                  nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.drop(self.proj(torch.cat([hd(h) for hd in self.heads], dim=-1)))
        return x + self.ffwd(self.ln2(x))


class GPT(nn.Module):
    """The pinned tutorial transformer, with dropout as a parameter so the
    numerical check can run it deterministically."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.Sequential(*[Block(dropout) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB)

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        return F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))


# --- harness ----------------------------------------------------------------

def start_local_server(device):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ["RGPU_OPSERVER"] = f"127.0.0.1:{port}"
    proc = subprocess.Popen([sys.executable, "-m", "rgpu.server", "--device", device,
                             "--port", str(port)], stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 60
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.1):
                return proc
        except OSError:
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("server failed to start")
            time.sleep(.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "colocated", "wan"], required=True)
    ap.add_argument("--device", default="cuda", help="the real device under test")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--train-steps", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    # Match the server, which turns TF32 off so results match a CPU reference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    remote = args.mode != "native"
    proc = start_local_server(args.device) if args.mode == "colocated" else None
    if remote:
        import rgpu
        from rgpu import session
        dev = "rgpu"
        barrier = torch.rgpu.synchronize
        stats = rgpu.stats
    else:
        dev = args.device
        barrier = (torch.cuda.synchronize if args.device == "cuda" else lambda: None)
        stats = lambda: {"messages": 0, "waits": 0, "bytes_out": 0, "bytes_in": 0}

    try:
        rows = []
        rng = random.Random(0)

        # -- workload 1: the elementwise chain, as in benchmark.py ------------
        local = torch.linspace(-1, 1, 1024)
        x_dev = local.to(dev)
        expected_chain = chain(local)
        if remote:
            compiled = torch.compile(chain, backend=rgpu.compile_backend(compiler="eager"),
                                     dynamic=False, fullgraph=True)
            chain_cases = ["eager_flush64", "eager_sync_each_call", "graph_eager"]
        else:
            chain_cases = ["native"]

        with torch.no_grad():
            for trial in range(args.trials):
                order = chain_cases[:]
                rng.shuffle(order)
                for name in order:
                    fn = compiled if name == "graph_eager" else chain
                    if remote:
                        session.get().flush_ops = 64
                    for _ in range(2):
                        out = fn(x_dev)
                    barrier()
                    before = stats()
                    t0 = time.perf_counter_ns()
                    for _ in range(20):
                        out = fn(x_dev)
                        if name == "eager_sync_each_call":
                            barrier()
                    barrier()
                    ms = (time.perf_counter_ns() - t0) / 1e6
                    after = stats()
                    torch.testing.assert_close(out.cpu(), expected_chain)
                    rows.append({"workload": "chain", "case": name, "trial": trial,
                                 "elapsed_ms": ms,
                                 **{k: after[k] - before[k] for k in before}})

        # -- workload 2: a training step of the pinned transformer ------------
        # Deterministic numerical check first: eval mode, dropout 0, fixed input.
        g = torch.Generator().manual_seed(1)
        xb = torch.randint(0, VOCAB, (BATCH, BLOCK), generator=g)
        yb = torch.randint(0, VOCAB, (BATCH, BLOCK), generator=g)
        torch.manual_seed(2)
        ref_model = GPT(dropout=0.0).eval()
        with torch.no_grad():
            ref_loss = ref_model(xb, yb).item()          # CPU reference
        torch.manual_seed(2)
        check_model = GPT(dropout=0.0).to(dev).eval()
        with torch.no_grad():
            dev_loss = check_model(xb.to(dev), yb.to(dev)).item()
        torch.testing.assert_close(torch.tensor(dev_loss), torch.tensor(ref_loss),
                                   rtol=1e-4, atol=1e-4)
        del check_model

        # Timed training at the pinned dropout. Trajectories are not compared
        # across devices because dropout draws from per-device RNGs.
        train_cases = (["eager_flush64", "eager_sync_each_step"] if remote else ["native"])
        for trial in range(args.trials):
            order = train_cases[:]
            rng.shuffle(order)
            for name in order:
                torch.manual_seed(3)
                model = GPT(dropout=0.2).to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
                xd, yd = xb.to(dev), yb.to(dev)
                if remote:
                    session.get().flush_ops = 64

                def step():
                    loss = model(xd, yd)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    return loss

                for _ in range(2):
                    step()
                barrier()
                before = stats()
                t0 = time.perf_counter_ns()
                last = None
                for _ in range(args.train_steps):
                    loss = step()
                    if name == "eager_sync_each_step":
                        last = loss.item()             # a host observation per step
                barrier()
                ms = (time.perf_counter_ns() - t0) / 1e6
                after = stats()
                final = loss.item()
                assert final == final and final < 10, f"loss diverged: {final}"
                rows.append({"workload": "train", "case": name, "trial": trial,
                             "elapsed_ms": ms, "steps": args.train_steps,
                             "final_loss": final,
                             **{k: after[k] - before[k] for k in before}})
                del model, opt, xd, yd

        summary = {}
        for wl in ("chain", "train"):
            for name in sorted({r["case"] for r in rows if r["workload"] == wl}):
                sel = [r for r in rows if r["workload"] == wl and r["case"] == name]
                t = [r["elapsed_ms"] for r in sel]
                summary[f"{wl}/{name}"] = {
                    "median_ms": statistics.median(t), "min_ms": min(t), "max_ms": max(t),
                    **{k: statistics.median([r[k] for r in sel])
                       for k in ("messages", "waits", "bytes_out", "bytes_in")}}
        if remote:
            assert summary["chain/eager_flush64"]["waits"] == 1
            assert summary["chain/eager_sync_each_call"]["waits"] == 21
            assert summary["train/eager_sync_each_step"]["waits"] > \
                summary["train/eager_flush64"]["waits"]

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env = {"mode": args.mode, "device": args.device, "platform": platform.platform(),
               "python": sys.version.split()[0], "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpu": gpu,
               "server": ("(local)" if args.mode == "colocated" else
                          os.environ.get("RGPU_OPSERVER") if remote else None),
               "numerical_check": {"cpu_ref_loss": ref_loss, "device_loss": dev_loss},
               "trials": args.trials, "train_steps": args.train_steps,
               "tf32": False}
        if remote:
            env["rgpu_commit"] = subprocess.run(
                ["git", "-C", str(Path(rgpu.__file__).resolve().parents[2]),
                 "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        print(json.dumps({"environment": env, "summary": summary, "samples": rows},
                         indent=2))
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
