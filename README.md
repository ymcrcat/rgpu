# rgpu

Run PyTorch on a machine with no GPU. The tensors and the work live on a remote
GPU host; your code runs locally and looks ordinary.

## Product documentation

The Fumadocs site lives in [`website/`](website/README.md). It includes setup,
training, deployment, configuration, troubleshooting, and performance.
To preview it locally:

```sh
cd website
npm ci
npm run dev
```

Open <http://localhost:3000>. `npm run build` produces a static site in
`website/out/` for hosting at a domain root.

## Which of the two paths do you want?

They are independent. Pick one before reading further.

| | **`rgpu` device** | **CUDA shim** |
|---|---|---|
| You write | `device="rgpu"` | `device="cuda"`, unchanged |
| Client needs | stock CPU-only `pip install torch` | CUDA runtime + math libraries |
| Runs on a Mac | **yes, directly** | no, needs a Linux container |
| Anything to build | no | yes |
| What crosses the wire | PyTorch operations | CUDA driver calls |
| GPU host runs | `rgpu-opserver` (a PyTorch process) | `rgpu-server` (C++, no PyTorch) |
| Port | 9720 | 9713 |

**Start with the `rgpu` device.** It is the lighter path, it needs nothing
built, and it is the only one that runs from macOS directly. The CUDA shim is
the one to use when the code must say `cuda` and cannot be changed.

Neither protocol has any authentication. Anyone who can reach the port can run
work on that GPU and read its memory, so bind to localhost and use an ssh
tunnel. This is not a deployment you expose.

---

# Path 1: the `rgpu` device

A PyTorch device registered from pure Python through `PrivateUse1`. It runs on
the stock CPU-only `torch` wheel — no CUDA libraries, no Docker, no driver,
nothing to compile. A tensor made with `device="rgpu"` is a small stand-in
locally; its data and every operation on it are on the GPU host.

## Quickstart

**On the GPU host.** It runs PyTorch, so it needs its own install, and the
major.minor version must match the client's — op schemas differ between
versions and the client checks at connect time.

```sh
pip install torch          # must match the client's major.minor
pip install -e python      # this repo
rgpu-opserver              # binds 127.0.0.1:9720, no authentication
```

**On your machine.**

```sh
pip install -e python
ssh -N -L 9720:localhost:9720 user@gpuhost
```

```python
import rgpu, torch

x = torch.randn(1024, 1024, device="rgpu")
y = (x @ x).relu().sum().item()      # one round trip, at .item()
```

`RGPU_OPSERVER=host:port` points the client somewhere other than
`127.0.0.1:9720`.

### One command instead of the tunnel

`rgpu-run` opens the tunnel, points the client at it, runs your command and
closes the tunnel afterwards:

```sh
rgpu-run --host user@gpuhost python train.py
rgpu-run --host user@gpuhost -i ~/.ssh/key --ssh-port 22050 python train.py
rgpu-run --server 127.0.0.1:9720 python train.py   # a tunnel you already have
```

It moves the plumbing, not the device: the script still has to `import rgpu`
and ask for `device="rgpu"`. It exits with the command's own status, and both
ends of the forward are bound to 127.0.0.1, so an unauthenticated GPU is never
reachable from the rest of your network.

On a fresh GPU box, `scripts/opserver_pod.sh 2.14` (the major.minor your client
runs) makes a venv, installs a matching torch, and starts the server detached.

## What it costs

Operations are queued and sent in one batch ahead of the next call that needs
an answer, so a long chain of work costs one round trip rather than one per
operation. This is what makes a distant GPU practical.

Measured from a Mac to a rented A40, nanoGPT (10.8M parameters, batch 64,
block 256), 20 training iterations:

| | local MPS | **A40 in Sweden, 116 ms RTT** |
|---|---|---|
| per iteration | 3.04 s | **1.04 s** |
| messages | 139,931 | 139,931 |
| **round trips** | 5 | **5** |

A GPU across an ocean beat the local machine by about 3x. Five round trips at
116 ms is 0.6 s of network time across the whole run: the queue is absorbing
roughly 7,000 operations per iteration into 5 synchronisation points, so
latency barely enters into it. If a workload ever looks latency-bound, count
round trips first — `rgpu.stats()` reports them as `waits`.

That run used torch 2.11 on the server against 2.14 on the client with
`RGPU_ALLOW_VERSION_MISMATCH=1`, because no 2.14 wheel exists for that host's
CUDA 12.8 driver. Matching versions is the supported configuration.

`docs/performance-notes.md` collects these measurements with their methodology
and caveats, along with what compresses on the wire and what graph shipping is
worth.

An earlier run against an A40 at 44.7 ms RTT: a ResNet-18 training step costs
one round trip at 90–106 ms depending on mode (eager inference is 152.0 ms),
and a 200-step run survived the tunnel being killed and restored with
bit-identical losses. `docs/PRODUCT_SPEC.md` has the numbers. This is one Mac
against one A40, not a benchmark suite.

TF32 is off by default on a CUDA device so results match a CPU reference; set
`RGPU_TF32=1` to trade that for speed.

## Settings

| Variable | Meaning |
|---|---|
| `RGPU_OPSERVER` | `host:port` of the op server, default `127.0.0.1:9720` |
| `RGPU_TF32` | `1` to allow TF32 matmuls and convolutions on the server |
| `RGPU_ALLOW_VERSION_MISMATCH` | `1` to connect to a server whose torch differs |
| `RGPU_ADVISE_AFTER` | messages before suggesting graph shipping, default `50000` |
| `RGPU_SESSION_GRACE` | seconds the server keeps a session whose connection dropped, default `120` |

## Limits

- **Move a model to `rgpu` before any grad-tracking forward pass.** Because an
  rgpu tensor is a wrapper subclass, `nn.Module.to("rgpu")` installs parameters
  with `torch.utils.swap_tensors`, which refuses a parameter something else
  still holds — and the autograd graph of an earlier forward holds exactly
  that. Moving afterwards raises `_apply(): Couldn't swap <Module>.<param>`. A
  real CUDA device has no such rule. Under `torch.no_grad()`, or before the
  first forward, rgpu behaves the same as CUDA.
- **An operation whose output size depends on the data costs a round trip**,
  and cannot write into an `out=` tensor: only the data can say how big the
  result is, so there is nothing to allocate ahead of time. Call it without
  `out=`.
- **A graph containing a non-`aten` custom op cannot run on rgpu at all**,
  compiled or eager. `rgpu-opserver` runs aten ops only, so a custom op has
  nowhere to run.
- **`torch.compile(model, backend=rgpu.compile_backend())` is the supported way
  to compile.** The graph is shipped once and compiled on the server; a call is
  one message after that.

---

# Path 2: the CUDA shim

`torch.device("cuda")` behaves as if a local GPU existed. The device is spelled
`cuda`, not `rcuda`: the shim replaces the driver library underneath PyTorch
rather than adding a backend inside it, so stock PyTorch wheels work unmodified
and no PyTorch source changes are needed.

```
client (no GPU, no driver)
  python + torch → libtorch_cuda → libcudart / libcublas / libcudnn   (stock)
                                        ↓ driver API
                                 our libcuda.so.1
                                        ↓ TCP
────────────────────────────────────────────────────────────────────────
  rgpu-server → real libcuda.so.1 → GPU                    (GPU host)
```

Interception happens at two levels. `libcuda.so.1` is replaced and forwards
driver calls to the GPU host. `libcudart.so.12` is replaced too, and translates
the CUDA runtime API into driver calls locally.

The second is not optional. Stock `libcudart` calls `cuGetExportTable`
immediately after `cuInit` and refuses to start without a table of undocumented
internal driver function pointers. Those are addresses inside the driver's own
process, so they cannot be forwarded anywhere. Every working system in this
space replaces the runtime for this reason.

## Build and test, no GPU required

```sh
./scripts/build_client.sh
```

That is the whole thing, from a fresh clone. It fetches `cuda.h` from the
nvidia pip wheel if it is missing, regenerates the client and server code when
`cuda.h` or anything in `codegen/` is newer than what it produced, builds the
container image if it is not there, then builds and runs the test suite.

The steps are also runnable on their own when you want just one of them:

```sh
./scripts/fetch_headers.sh   # cuda.h from the nvidia pip wheel, about 1 MB
./codegen/run.sh             # parse cuda.h, generate client and server code
```

`build/libcuda.so.1` is the shim.

The GPU-free tests are the meaningful ones: real client stubs, real wire format,
real server dispatch, with a fake driver at the bottom. They cover a byte-exact
memory round trip, the runtime API path PyTorch sits on, and the kernel launch
path, where the client must ask the server for a kernel's parameter layout and
pack arguments into it. That last test includes a launch with deliberately wrong
arguments, which must be rejected: without it, a server that accepted anything
would pass.

## Run against a real GPU

On the GPU host, with the CUDA toolkit installed:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/rgpu-server
```

Then, from the client:

```sh
ssh -L 9713:localhost:9713 gpu-host
LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./build/rpc_smoke
```

`LD_LIBRARY_PATH` is **not** enough to put the shims in front of PyTorch: its
libraries use `RPATH`, which takes precedence, and it preloads the CUDA
libraries by absolute path. Use `LD_PRELOAD`, or install without the stock
`nvidia-cuda-runtime` package.

### The PyTorch ladder

`tests/torch/ladder.py` climbs from "is CUDA available" through tensor
allocation, elementwise math, cuBLAS matmul and cuDNN convolution to ResNet-18
inference, comparing every rung against a CPU reference. Every rung runs even
when an earlier one fails, so one run shows the whole picture.

**On the GPU host** (`scripts/remote_torch.sh user@host`) is the fastest loop.
The host has a real driver, but the shims go ahead of it and reach a server on
the same box over loopback. The script prints which libraries actually loaded,
so this is verifiable rather than assumed.

**From a client with no GPU** (`scripts/run_torch.sh`) is the real target. It
builds a container with stock PyTorch and both shims. PyTorch plus the CUDA
math libraries is several gigabytes; on a Mac, Docker Desktop's disk allocation
may need raising first.

No settings are required: cuBLAS, cuBLASLt and cuDNN are all forwarded, so
matmul and convolution take their normal paths.

Two knobs on the ladder script itself help tell a library problem apart from a
problem underneath it. `RGPU_NO_CUDNN=1` makes it set
`torch.backends.cudnn.enabled = False`, falling back to PyTorch's own
convolution kernels; `RGPU_NO_CUBLASLT=1` makes it set
`DISABLE_ADDMM_CUDA_LT=1`, routing `addmm` through plain cuBLAS. Both are read
by `tests/torch/ladder.py`, not by the shim.

## What a call costs

Calls whose effect is only visible at a later synchronization go without a
reply, queued and written in one batch ahead of the next call that needs an
answer. Measured on an RTX 3090 over loopback, 2000 real kernel launches:

| | round trip per call | batched |
|---|---|---|
| kernel launch, issue only | 20.3 us | 0.2 us |
| kernel launch, including the synchronize | 20.3 us | 3.2 us |

The second row is the honest one: it waits for all 2000 kernels to actually
run. Reporting only the first would measure how fast work can be deferred
rather than how fast it happens.

Loopback is the least favourable case, since a round trip costs almost nothing
there. Over a network the saving is a full round trip per launch, so the gap
widens with latency. Calls that return data still round trip.

`RGPU_BATCH=0` turns batching off, which is slower but makes a failing call
report itself where it happened rather than at the next synchronization.

## Settings

| Variable | Meaning |
|---|---|
| `RGPU_SERVER` | `host:port` of the GPU server, default `127.0.0.1:9713` |
| `RGPU_PORT` | port the server listens on, default `9713` |
| `RGPU_VERBOSE` | log every forwarded call |
| `RGPU_BATCH` | `0` to make every call a round trip, for debugging |
| `RGPU_CUBLAS`, `RGPU_CUBLASLT`, `RGPU_CUDNN` | paths the server opens for the real maths libraries |
| `RGPU_SESSION_GRACE` | seconds a dropped session is kept, default `120` |
| `RGPU_MAX_SESSIONS` | most sessions kept at once, grace included, default `64` |
| `RGPU_RECONNECT_SECONDS` | seconds the client keeps retrying, default `60` |
| `RGPU_MAX_CLIENT_THREADS` | most live client threads per session, default `4096` |
| `RGPU_MAX_CONTEXT_STACK` | deepest context stack per client thread, default `64` |

## Limits

**No NVIDIA math library can run on the client.** Each initialises through the
driver's undocumented export tables, exactly as the stock runtime does, so each
has to be forwarded to the GPU host. cuBLAS, cuBLASLt and cuDNN are all
forwarded.

The real cuBLASLt is worth a note: over a remoted driver it does not report an
error, it segfaults inside `cublasLtMatmulAlgoGetHeuristic`. Anything unshimmed
that reaches the driver may fail that way rather than cleanly.

Managed memory and zero-copy host mapping cannot work across a network and are
refused explicitly. So are the deprecated `cuCtxAttach` and the green-context
calls (`cuGreenCtx*`, `cuCtxFromGreenCtx`), which would hand a client a context
the server cannot account for; `cuCtxDetach` works, and destroys the context as
`cuCtxDestroy` does. Kernel launches with `cuLaunchKernelEx` launch
configurations are not marshalled yet, and `cuLaunchCooperativeKernel` is
refused: the launch message has no way to say "cooperative", and running it as
an ordinary launch would drop the guarantees the kernel was written around.

`cuDevicePrimaryCtxReset` destroys what every session on the server holds in
that context, so it is refused with `CUDA_ERROR_NOT_SUPPORTED` while any other
session is live, including one waiting out its grace period.

Each client thread keeps its own current context, context stack and stream
capture mode on the server, but calls are served one at a time, not in
parallel. Past `RGPU_MAX_CLIENT_THREADS` or `RGPU_MAX_CONTEXT_STACK` the call
is refused. A call sent without a reply that fails has its error handed to the
same thread's next call that replies; if that thread never makes one, the error
shows up only in the server log. PyTorch's autograd threads are real threads,
but they also allocate and call cuBLAS, which reply, so in practice their
errors still arrive.

### Surviving a dropped connection

A dropped connection does not lose the GPU state. The server keeps the session
for `RGPU_SESSION_GRACE` seconds; the client reconnects, resends what the
server never acknowledged, and carries on. A call that still gets no reply
after that one retry fails with `CUDA_ERROR_UNKNOWN` and is then treated as
answered: it is never sent again, so it ran at most once — whether it ran at
all depends on whether it reached the server — and later calls carry on
normally.

If the client does not get back in time the session expires and the server
releases what it held, as it does if the server restarts. The client is then
told its session is gone: it says so, sends nothing more, and every later call
fails. It never carries on against an empty GPU.

Expiry ends any stream capture the session left open, releasing the graph it
yields, and releases the allocations, contexts, modules, loaded libraries,
streams, events, graphs and maths-library handles a session made. It does not
track `cuMemAddressReserve` ranges, `cuGraphConditionalHandleCreate` handles,
user-object retains, cuDNN descriptors minted on the client, or texture
references (`cuTexRefCreate`, deprecated and needing a current context). Those
stay until the server exits.

A server keeps at most `RGPU_MAX_SESSIONS` sessions, counting those waiting out
their grace period. Past that a new client is refused at the handshake. A
client returning to a session the server still has is never refused.

---

# Status

| Piece | State |
|---|---|
| `rgpu` device (`python/`) | working, 181 tests passing |
| Wire format and transport | working |
| Code generation from `cuda.h` | 248 generated, 23 hand-written, 176 stubbed |
| Client shim `libcuda.so.1` | working |
| Client shim `libcudart.so.12` | ~55 runtime calls translated |
| Client shims `libcublas`, `libcublasLt`, `libcudnn` | forwarded to the host |
| Server | runs on the GPU host |
| Driver and runtime API end-to-end | passing, fake driver and real GPU |
| Kernel launch marshalling | passing, fake driver and real GPU |
| **PyTorch, all nine rungs including ResNet-18** | **passing on real hardware** |
| Asynchronous batching | working, 6x on real hardware |

On a rented RTX A4000, every rung of `tests/torch/ladder.py` passes. ResNet-18
inference runs on the remote GPU and its logits match a CPU reference to
3.8e-06, with identical top-5 predictions. `ldd` confirms the client process
has our shims in front of the driver.

Anything the generator cannot marshal becomes a stub that logs its name and
returns `CUDA_ERROR_NOT_SUPPORTED`, so gaps show up as loud failures rather
than wrong answers. `codegen/report.txt` lists what is generated, hand-written
and stubbed.

# Renting a GPU to test against

`scripts/runpod.sh create` rents one from RunPod, reading the API key from
1Password so it never lands in a file, and prints the ssh and deploy commands.
`status`, `stop` and `delete` do what they say, and `status` lists every pod on
the account. A running pod bills by the hour and a stopped one still bills for
its disk, so delete it when you are done.

`scripts/deploy_server.sh user@host` copies the source over and builds the
server there, checking for a driver and a toolkit first.
`scripts/remote_torch.sh user@host` then runs the PyTorch ladder on that box.
`scripts/gcp_up.sh` and `scripts/gcp_down.sh` do the same job on GCP.

# How the code is organized

| Path | What it does |
|---|---|
| `python/rgpu/` | the `rgpu` PyTorch device: wire, session, tensor, dispatch, compile |
| `python/rgpu/server/` | `rgpu-opserver`, which runs the operations |
| `codegen/parse.py` | `cuda.h` to a JSON API model, via libclang |
| `codegen/annotations.py` | what the header cannot express: buffer sizes, direction, execution class |
| `codegen/emit.py` | generates client stubs, server dispatch, entry table, fake driver |
| `common/wire.h` | frame format and serialization |
| `client/shim.cpp` | `cuGetProcAddress`, kernel launch marshalling, host allocations |
| `client/cudart_impl.cpp` | the runtime API translated into driver calls |
| `client/cublas_impl.cpp`, `cublaslt_impl.cpp`, `cudnn_impl.cpp` | maths calls forwarded to the host |
| `server/main.cpp` | accept loop, dispatch, parameter layout lookup |
| `server/inventory.cpp` | what each session owns, released when it expires |
| `tests/fake_cuda.cpp` | a driver backed by host memory, for testing without a GPU |
| `.claude/skills/rgpu-runpod` | how to rent, use and release a test GPU without wasting money |

`docs/superpowers/specs/2026-09-07-cuda-api-remoting-design.md` has the design,
the prior art it draws on, and the three hard problems it has to solve.
