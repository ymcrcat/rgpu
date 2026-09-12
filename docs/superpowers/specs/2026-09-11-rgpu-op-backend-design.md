# An `rgpu` device for PyTorch: remoting at the level of operations

## Why this exists

The current system remotes CUDA underneath PyTorch: our `libcuda`, `libcudart`
and maths-library shims replace NVIDIA's, and stock PyTorch's CUDA backend runs
on top of them. That works on Linux and cannot work on macOS. The macOS torch
wheel ships `libtorch_cpu` and no CUDA backend at all - checked by downloading
it, not assumed - so there is nothing for a replacement driver to sit
underneath.

This design adds a second path that works above CUDA instead of below it: a
device PyTorch knows as `rgpu`, whose tensors live on a remote GPU host. It
runs on the stock CPU-only wheel, so a Mac can drive a remote NVIDIA GPU
natively, without Docker.

The CUDA-level stack stays. It remains the Linux path, needing no code
changes in the application. This is the macOS path.

## Goals

**Primary:** stock macOS PyTorch drives a remote NVIDIA GPU, natively.

**First milestone**, from native Python on a Mac against a rented GPU over an
ssh tunnel, every result compared against a CPU reference:

- inference: the existing ladder, up to ResNet-18
- training: the eight existing training rungs, checking gradients
- torch.compile: the five existing compile tests, with `backend="rgpu"`
- a training run survives the tunnel being killed and restored

**Not in the first milestone:** authentication, multiple GPUs, making the
default `torch.compile` backend select us automatically, and support on
Linux clients that also have a local GPU.

## What was verified before this design

Every claim the design rests on was tried on the target platform - torch
2.14, macOS arm64, CPU-only wheel - before being written down.

| Question | Result |
|---|---|
| Does the macOS wheel have a CUDA backend? | No: `libtorch_cpu` only, no CUDA libraries |
| Can pure Python register a device? | Yes, via `torch.utils.backend_registration._setup_privateuseone_for_python_backend` (marked experimental) |
| Do the device's own tensors work? | Partly: `create_empty_tensor` gives storage of size zero, so strided views fail a bounds check. Rejected. |
| Does a tensor subclass reporting `device=rgpu` work? | Yes: matmul, transpose, indexing, broadcasting, and backward through autograd |
| Do factory functions (`torch.randn(device="rgpu")`) work? | Yes, once `empty.memory_format` and `empty_strided` are registered for the device |
| Can output shapes be known without the data? | Yes: running the op on meta tensors gives shape, dtype and strides |
| Can torch.compile trace through the subclass? | Only if it is traceable. Without `__tensor_flatten__` it breaks into one graph per op. With it, the forward and backward graphs are one each, identical to plain tensors. |

## Architecture

```
Mac (stock CPU-only PyTorch)                    GPU host
  your code                                       rgpu-opserver (Python)
    ↓ torch ops on device "rgpu"                    real PyTorch on CUDA
  rgpu package                                      id → tensor, per session
    RemoteTensor: id + meta tensor                        ↑
    dispatch: meta inference, ids, queue ── TCP 9720 ─────┘
```

### Client: the `rgpu` Python package

In `python/rgpu/`, standard library only.

- **`device.py`** - `import rgpu` registers the device through the Python
  backend hook, registers the two factory kernels, and installs a
  `torch.rgpu` module with `is_available`, `device_count`, `synchronize` and
  `manual_seed`, the same shape as `torch.cuda`.
- **`tensor.py`** - `RemoteTensor`, a wrapper subclass that reports
  `device=rgpu:0`. It holds a tensor id and a meta tensor with the shape,
  dtype and strides. There is never any data on the client. It is traceable:
  `__tensor_flatten__` exposes the meta tensor.
- **`dispatch.py`** - `__torch_dispatch__` receives every op on an `rgpu`
  tensor, after autograd. It infers outputs on meta, mints ids, classifies
  the op, and queues it.
- **`wire.py`** - framing and the argument encoding.
- **`session.py`** - connection, handshake, replay after reconnect.
- **`compile.py`** - the `rgpu` torch.compile backend.

### Server: `rgpu-opserver`

A Python process on the GPU host running real PyTorch. It keeps, per session,
a table from tensor id to real tensor, runs ops by qualified `aten` name with
ids replaced by tensors, stores outputs under the ids the client chose, moves
bytes in and out, and compiles graphs.

`--device` chooses where ops run: `cuda` in production, `cpu` or `mps` for
testing on a machine with no NVIDIA GPU.

### Identity and views

Every output gets its own id, and view ops run on the server as well. When the
client does `x.t()`, the server does the same to its real tensor, so the
server's aliasing is exactly PyTorch's: an in-place op on a view changes the
base, and nothing on the client tracks aliases. The client runs the same view
on its meta tensor to get the new metadata.

A tensor's id is stored against its meta tensor, not against the wrapper,
because compiled code works on the unwrapped meta tensors. Rewrapping a meta
tensor recovers its id; a meta tensor produced by compiled code gets a new one.

When a `RemoteTensor` is garbage-collected, a `FREE` for its id joins the next
batch.

## How an op flows

Every op falls into one of four kinds, and only two of them wait.

| Kind | Examples | Behaviour |
|---|---|---|
| Stream | `mm`, `conv`, `relu`, `add_`, views, `_foreach_*` | Outputs inferred on meta, ids minted, queued. No wait. |
| Upload | `x.to("rgpu")`, `copy_` from CPU | Bytes copied into the message at the call, so later changes to the CPU tensor cannot leak in. No wait. |
| Download | `.cpu()`, `.item()`, `.tolist()`, printing, `bool(t)` | Flush and wait for bytes. One round trip. |
| Shape unknown | `nonzero`, boolean masks, `unique`, `masked_select`, anything without a meta kernel | The server runs the op and returns output metadata. One round trip. |

Autograd runs on the client, above the dispatch point, so forward, backward
and the optimizer step are all streaming ops. A training step waits only where
the user's code pulls data back, typically `loss.item()`. **Plain eager
training should cost about one round trip per step.** That is the claim the
milestone most needs to confirm.

For the shape-unknown kind, the number of outputs is still known from the
op's schema - only their sizes are not - so the client mints the output ids as
usual and the server replies with the shapes and strides it produced.

The queue is flushed at every wait, and also every 64 ops or 256 KB, so the
GPU works while the client is still issuing a step. Both limits are
environment variables.

Meta kernels perform the same shape and dtype checks as real ones, so most
mistakes raise immediately on the line that made them, before anything is
sent. Only failures the client cannot predict, such as running out of memory,
happen on the server.

`torch.manual_seed` also seeds the server's generator through the device
module. Runs are reproducible, but not bit-identical to local CUDA.

## Wire protocol

TCP on port 9720, separate from the CUDA-level protocol on 9713. A handshake
carries a session id, then length-prefixed frames. A frame is a batch of
messages, each with a sequence number.

Messages from the client:

- `RUN` - op name and overload, encoded arguments, output ids
- `UPLOAD` - id, dtype, shape, strides, bytes
- `DOWNLOAD` - id; the reply carries the bytes
- `FREE` - a list of ids
- `SEED`, `SYNC`
- `COMPILE` - graph id and a serialized graph
- `CALL` - graph id, input ids, output ids

The argument encoding is a small tagged binary format. It carries None, bool,
int, float, str, lists, tensor ids, dtype, device, layout and memory format.
Any other type raises on the client, at the call.

## The compile path

`torch.compile(model, backend="rgpu")`, optionally with
`mode="reduce-overhead"`.

The backend wraps `aot_autograd` and receives forward and backward graphs of
`aten` ops. Each graph is serialized with the same allowlist and argument
encoding as eager ops, and sent once as `COMPILE`. The server rebuilds an FX
graph and compiles it for its device with inductor; in reduce-overhead mode it
also captures CUDA graphs on the GPU, where replaying them costs nothing. Each
step is then one `CALL` for the forward and one for the backward, both
streaming, because the output shapes are known from the traced graph.

Graphs are specialized to shapes and recompile when shapes change, as with
ordinary torch.compile. The server caches compiled graphs per session.

A graph the wire cannot carry whole - a constant tensor inside it, dynamic
shapes, a node that is not a call, a plain Python callable - is not shipped.
It runs eagerly through the normal dispatch path, correct and slower, with a
log line naming what forced it.

A custom op is different: it cannot run on rgpu at all, compiled or eager.
The server resolves ops by name and runs only `aten` ops, so falling back to
eager would post the same op and fail there. A graph containing one raises at
compile time, naming the op.

Plain `torch.compile(model)` without `backend="rgpu"` would try to generate
code for a device inductor does not know. In the first milestone,
`backend="rgpu"` is the supported way to compile.

## Failures and sessions

Sessions follow the CUDA-level server. The server keeps a session for a grace
period after its connection drops, 120 seconds by default; the client keeps
unacknowledged messages and replays them after reconnecting; the server keeps
its last reply, so a request that ran but whose reply was lost is not run
twice. An upload is one message, so a large one can exceed the 64 MB replay
limit; the session is then marked unrecoverable until the server acknowledges
it, and the log says so.

When a streamed op fails on the server, its output ids are poisoned. Any later
op using a poisoned id is skipped and poisons its own outputs. At the next
wait the user gets one exception naming the first failure -
`rgpu.RemoteError: aten::mm failed: CUDA out of memory ...` - rather than a
cascade. Using a poisoned tensor raises again.

A dead server loses every tensor. The client raises a clear error rather than
continuing with ids that point at nothing.

## Security

There is no authentication in this design, as in the rest of the project.
Until there is:

- the server binds 127.0.0.1 by default, so an ssh tunnel is the only way in;
  binding other interfaces takes an explicit flag
- only `aten` ops are run, looked up by name, with a blocklist for the few
  that reach outside the GPU, notably `aten::from_file`, which reads files
  from the server's disk
- there is no pickle, no `eval`, no general attribute lookup on `torch`
- decoding enforces limits on list length, nesting depth and upload size

## Testing

Because `rgpu-opserver --device cpu` runs real PyTorch ops, nearly everything
can be tested on the Mac with no GPU, against a CPU reference that should
match almost exactly. `--device mps` gives a second real backend.

1. **Unit** - argument encoding round-trips for every type; op classification;
   meta inference agrees with real outputs; poisoning stops a cascade.
2. **End to end on the Mac, server on CPU** - the ladder rewritten for
   `device="rgpu"`, the training rungs, the compile tests, data-dependent ops,
   and a dropped-connection test using a drop hook in the server. Every check
   compares values, not status codes.
3. **Real hardware** - the same suites from native macOS Python over the
   tunnel to a RunPod GPU.

## Done when

From native Python on a Mac, against a RunPod GPU over an ssh tunnel:

- the ladder, training rungs and compile tests pass against CPU references
- this table is filled in with measured round trips per step:

  | | CUDA-level (measured) | op-level |
  |---|---|---|
  | eager inference | 49 | |
  | eager training step | many | |
  | compiled training step | 2 | |

- a training run survives the tunnel being killed and restored, with
  bit-identical losses

## Risks

- **The Python backend hook is experimental.** `_setup_privateuseone_for_python_backend`
  could change between torch releases. The fallback is a small C++ extension
  providing only the device guard and hooks; the rest of the design is
  unaffected.
- **Python per-op overhead on the server**, roughly 20 to 50 microseconds, a
  few milliseconds per eager ResNet-18 step. Small next to a 34 ms round trip,
  and gone on the compile path. If it matters, the dispatch loop is what to
  move to C++.
- **Ops whose meta kernels are missing or wrong.** Missing ones fall back to a
  round trip and are logged. Wrong ones would give the client the wrong
  shape; the unit test comparing meta outputs against real ones is the guard,
  and the ops in the ladder are the ones checked first.
