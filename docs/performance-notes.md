# Performance notes

Measurements taken on 2026-09-13/14, with what each one does and does not
show. These are single runs on one Mac against one GPU, not a benchmark suite:
treat them as evidence for the specific claims made beside them, and re-measure
before relying on any of them.

Client throughout: MacBook, Apple M3, macOS 26.5, torch 2.14.0. Where a figure
names a remote GPU it is an NVIDIA A40 rented from RunPod, reached over an ssh
tunnel.

## 1. A remote GPU beat the local one, across an ocean

nanoGPT (the `yangpt` build-along: 10.8M parameters, 6 layers, 6 heads,
`n_embd` 384), batch 64, block 256, 20 training iterations. Eager, no
`torch.compile`.

| | local MPS, loopback | **A40 in EU-SE-1, 116 ms RTT** |
|---|---|---|
| train loop | 60.8 s | **20.8 s** |
| per iteration | 3.04 s | **1.04 s** |
| messages | 139,931 | 139,931 |
| **round trips (`waits`)** | 5 | **5** |
| bytes out | 70.9 MB | 70.9 MB |

About 3x faster on a GPU 1,500 km away than on the machine's own. The reason
is the wait count: 5 round trips at 116 ms is ~0.6 s of network time across the
whole run, out of 20.8 s. The queue absorbs roughly 7,000 operations per
iteration into 5 synchronisation points, so latency barely enters into it.

**If a workload looks latency-bound, count `waits` first** — `rgpu.stats()`
reports them.

Caveats: the server ran torch 2.11 against a 2.14 client under
`RGPU_ALLOW_VERSION_MISMATCH=1`, because no 2.14 wheel exists for that host's
CUDA 12.8 driver. 20 iterations is short — the 60.8 s *total* wall time is
dominated by startup, the one-time weight upload and the closing `generate`,
which is why the train loop is quoted separately.

### Correctness alongside it

Step 0 loss was identical across local and remote (`4.2896` train / `4.2847`
val), and identical again on a CPU server. Later steps diverge slightly
(step 19: `3.2923` remote against `3.2926` local) because dropout draws from a
per-device RNG. An earlier A40 run at 44.7 ms RTT survived the tunnel being
killed and restored across 200 steps with bit-identical losses.

### Native MPS could not run this model at all

`YG_DEVICE=mps` died in the autograd engine on `loss.backward()`:

```
RuntimeError: opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED
  at torch/csrc/autograd/engine.cpp:1141
```

A torch 2.14.0 bug, nothing to do with rgpu — but worth recording, because
`train.py`'s own fallback is `'cuda' if available else 'mps'`, so the script
cannot train on this machine without rgpu.

## 2. Graph shipping: 3.1x fewer messages, not 10x

nanoGPT against a local MPS server, batch 8, block 64, 20 iterations, eager
against `torch.compile(model, backend="rgpu", dynamic=False)`.

| | messages | total bytes | op-stream bytes | train loop |
|---|---|---|---|---|
| eager | 139,931 | 55.7 MB | 12.5 MB | 7.2 s |
| graph-shipped | 44,880 | 48.0 MB | 4.8 MB | 8.3 s |
| reduction | **3.1x** | 1.16x | **2.6x** | *slower* |

Losses identical. Op-stream is total minus the 43.2 MB one-time float32 weight
upload, which graph shipping does not touch.

Two things to take from this:

- The "31 messages per iteration to 3" figure in the performance proposal
  (#10) is for a simple 20-op chain. On a real model with autograd it is 3.1x.
- **It was net slower in wall time** at this scale, because one-time
  compilation is not amortised over 20 iterations. Graph shipping reduces bytes
  and messages; it does not automatically reduce time. Over hundreds of
  iterations that flips.

A smaller check, a 3-layer MLP, 10 forward calls, CPU server: eager 169
messages, graph-shipped 50. Same ~3x.

## 3. What is actually on the wire, and what compresses

`zlib.compress(level=1)` on real `rgpu.wire` payloads, measured on the M3.

| payload | share of the 70.9 MB run | ratio | compress speed |
|---|---|---|---|
| float32 weights (one-time `.to("rgpu")`) | 43.2 MB, 61% | **1.08x** | 45 MB/s |
| int64 token batches | 5.2 MB, 7% | 5.38x | 268 MB/s |
| operation stream | 22.5 MB, 32% | **10.12x** | 612 MB/s |

- **Float32 tensor data is incompressible** — 1.08x, and the slowest to
  attempt. IEEE-754 mantissas are high-entropy. Compressing the 43 MB of
  weights costs about a second of CPU to save 3 MB.
- **The operation stream is the compressible part** — 10x at 612 MB/s, because
  those messages repeat the same `aten::` names and structure thousands of
  times per iteration.
- The int64 figure is a dtype problem in disguise: it compresses only because
  8 bytes are carrying values under 65.

Bandwidth was not the bottleneck in any of this. The train loop moved ~27.7 MB
over 20.8 s — about **1.3 MB/s, or 10 Mbps**. See #11.

## 4. Driver-path batching

Not re-measured here; carried from earlier work on an RTX 3090 over loopback,
2000 real kernel launches.

| | round trip per call | batched |
|---|---|---|
| kernel launch, issue only | 20.3 us | 0.2 us |
| kernel launch, including the synchronize | 20.3 us | 3.2 us |

The second row is the honest one: it waits for all 2000 kernels to run.
Loopback is the least favourable case, since a round trip costs almost nothing
there; over a network the saving is a full round trip per launch.

## 5. Launcher startup

`rgpu-run` as a module inside the `rgpu` package took **4.35 s** to print
`--help`, because importing the package registers the device and pulls in
torch. As a top-level module it is **0.36 s**. The child imports torch itself
either way.

## How to reproduce

The nanoGPT runs used a generated variant of `yangpt/src/train.py` with
`device` and the hyperparameters made overridable, run against
`rgpu-opserver`. The compression ratios are
`zlib.compress(wire.encode_element(...), 1)` on the three payload shapes above.
Message and byte counts come from `rgpu.stats()`, which reports `messages`,
`waits`, `bytes_out` and `bytes_in`.

For a remote GPU: `scripts/opserver_pod.sh <major.minor>` sets up the server on
a fresh box, and `rgpu-run --host user@gpuhost python train.py` opens the
tunnel and runs the workload.

## Open threads

- #10 — performance proposal; item 1 (queue byte accounting) done.
- #11 — selective op-stream compression, gated on measuring the link.
- #13 — nothing tells a user graph shipping exists, so nobody turns it on.
