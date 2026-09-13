# rgpu

Run PyTorch on a machine with no GPU. `torch.device("cuda")` behaves as if a
local GPU existed; the CUDA work executes on a remote GPU host.

The device is spelled `cuda`, not `rcuda`. The shim replaces the driver library
underneath PyTorch rather than adding a backend inside it, so stock PyTorch
wheels work unmodified and no PyTorch source changes are needed.

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

The second one is not optional. Stock `libcudart` calls `cuGetExportTable`
immediately after `cuInit` and refuses to start without a table of undocumented
internal driver function pointers. Those are addresses inside the driver's own
process, so they cannot be forwarded anywhere. Every working system in this
space replaces the runtime for this reason.

See `docs/superpowers/specs/2026-09-07-cuda-api-remoting-design.md` for the
design, the prior art it draws on, and the three hard problems it has to solve.

## Status

| Phase | State |
|---|---|
| Wire format and transport | working |
| Code generation from `cuda.h` | working, 259 of 435 functions generated |
| Client shim `libcuda.so.1` | builds, 436 exported entry points |
| Client shim `libcudart.so.12` | ~55 runtime calls translated |
| Client shim `libcublas.so.12` | matmul family forwarded to the host |
| Client shim `libcublasLt.so.12` | matmul path forwarded to the host |
| Server | runs on the GPU host |
| Driver API end-to-end | passing, fake driver and real GPU |
| Runtime API end-to-end | passing, fake driver and real GPU |
| Kernel launch marshalling | passing, fake driver |
| A real CUDA kernel on a remote GPU | passing |
| **PyTorch, all nine rungs including ResNet-18** | **passing** |
| Asynchronous batching | working, 6x on real hardware |
| End-to-end on a real GPU | needs a GPU host |
| PyTorch client image and test ladder | written, not yet run |

On a rented RTX A4000, every rung of `tests/torch/ladder.py` passes. ResNet-18
inference runs on the remote GPU and its logits match a CPU reference to
3.8e-06, with identical top-5 predictions. The client process has our shims in
front of the driver, which `ldd` confirms.

One setting is required and the ladder sets it:
`torch.backends.cudnn.enabled = False`, which keeps convolution on PyTorch's
own kernels because cuDNN is not forwarded yet. `addmm` runs on its default
cuBLASLt path. Set `RGPU_NO_CUBLASLT=1` to send it through plain cuBLAS
instead, which is useful for telling the two paths apart when something
breaks.

The GPU-free tests are the meaningful ones so far: real client stubs, real wire
format, real server dispatch, with a fake driver at the bottom. They cover a
byte-exact memory round trip, the runtime API path that PyTorch sits on, and
the kernel launch path, where the client must ask the server for a kernel's
parameter layout and pack arguments into it. That last test includes a launch
with deliberately wrong arguments, which must be rejected: without it, a server
that accepted anything would pass.

## Build and test, no GPU required

```sh
./scripts/fetch_headers.sh   # cuda.h from the nvidia pip wheel, about 1 MB
./codegen/run.sh             # parse cuda.h, generate client and server code
./scripts/build_client.sh    # build the shim and run the test suite
```

`build/libcuda.so.1` is the shim. Put it on `LD_LIBRARY_PATH` and any CUDA
program will call it instead of a driver.

## Run against a real GPU

On the GPU host, with the CUDA toolkit installed:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/rgpu-server
```

The protocol has no authentication. Anyone who can reach the port can run
arbitrary kernels on that GPU, so bind it to a trusted network and reach it
over an SSH tunnel:

```sh
ssh -L 9713:localhost:9713 gpu-host
```

Then from the client:

```sh
LD_LIBRARY_PATH=build RGPU_SERVER=127.0.0.1:9713 ./build/rpc_smoke
```

### On a rented GPU box

`scripts/runpod.sh create` rents one from RunPod, reading the API key from
1Password so it never lands in a file, and prints the ssh and deploy commands.
`status`, `stop` and `delete` do what they say. A running pod bills by the
hour, around twenty cents for the cards this asks for, and a stopped one still
bills for its disk, so delete it when you are done.

`scripts/deploy_server.sh user@host` copies the source over and builds the
server there. It checks for a driver and a toolkit first and says which is
missing. `scripts/remote_torch.sh user@host` then runs the PyTorch ladder on
that box.

`scripts/gcp_up.sh` and `scripts/gcp_down.sh` do the same job on GCP, if you
would rather use Compute Engine. Either way the instance bills while running.

### PyTorch

`tests/torch/ladder.py` climbs from "is CUDA available" through tensor
allocation, elementwise math, cuBLAS matmul and cuDNN convolution to ResNet-18
inference, comparing every rung against a CPU reference. Every rung runs even
when an earlier one fails, so one run shows the whole picture.

Two ways to run it:

**On the GPU host** (`scripts/remote_torch.sh user@host`) is the fastest loop
while the CUDA surface is still being filled in. The host has a real driver,
but `LD_LIBRARY_PATH` puts our shims ahead of it, so PyTorch talks to them and
they reach a server on the same box over loopback. The script prints which
libraries actually got loaded so this is verifiable rather than assumed.

**From a client with no GPU** (`scripts/run_torch.sh`) is the real target. It
builds a container with stock PyTorch and both shims. Note that PyTorch plus
the CUDA math libraries is several gigabytes; on a Mac, Docker Desktop's disk
allocation may need raising before this image will build.

## Cost of a call

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

Loopback is the least favourable case for this, since a round trip costs
almost nothing there. Over a network the saving is a full round trip per
launch, so the gap widens with latency. Calls that return data still round
trip and are unaffected.

`RGPU_BATCH=0` turns batching off, which is slower but makes a failing call
report itself where it happened rather than at the next synchronization.

## Environment variables

| Variable | Meaning |
|---|---|
| `RGPU_SERVER` | `host:port` of the GPU server, default `127.0.0.1:9713` |
| `RGPU_PORT` | port the server listens on, default `9713` |
| `RGPU_VERBOSE` | log every forwarded call |
| `RGPU_BATCH` | `0` to make every call a round trip, for debugging |
| `RGPU_CUBLAS`, `RGPU_CUBLASLT` | paths the server opens for the real maths libraries |
| `RGPU_SESSION_GRACE` | seconds the server keeps a session whose connection dropped, default `120` |
| `RGPU_RECONNECT_SECONDS` | seconds the client keeps trying to reach the server again, default `60` |
| `RGPU_MAX_CLIENT_THREADS` | server: most client threads a session may have live at once, default `4096` |
| `RGPU_MAX_CONTEXT_STACK` | server: deepest context stack a client thread may push, default `64` |

## How the code is organized

| Path | What it does |
|---|---|
| `codegen/parse.py` | `cuda.h` to a JSON API model, via libclang |
| `codegen/annotations.py` | what the header cannot express: buffer sizes, direction, execution class |
| `codegen/emit.py` | generates client stubs, server dispatch, entry table, fake driver |
| `common/wire.h` | frame format and serialization |
| `client/shim.cpp` | `cuGetProcAddress`, kernel launch marshalling, host allocations |
| `client/cudart_impl.cpp` | the runtime API translated into driver calls, plus kernel registration |
| `client/cublas_impl.cpp`, `client/cublaslt_impl.cpp` | the maths calls PyTorch makes, forwarded to the host |
| `.claude/skills/rcuda-runpod` | how to rent, use and release a test GPU without wasting money |
| `server/main.cpp` | accept loop, dispatch, parameter layout lookup |
| `tests/fake_cuda.cpp` | a driver backed by host memory, for testing without a GPU |

Anything the generator cannot marshal becomes a stub that logs its name and
returns `CUDA_ERROR_NOT_SUPPORTED`, so gaps show up as loud failures rather
than wrong answers. `codegen/report.txt` lists what is generated, hand-written
and stubbed.

## Known limits

**No NVIDIA math library can run on the client.** Each one initialises through
the driver's undocumented export tables, exactly as the stock runtime does, so
each has to be forwarded to the GPU host instead.

cuBLAS and cuBLASLt are forwarded, covering the matmul paths PyTorch uses.
cuDNN is not, and has a stand-in that reports every call rather than running,
hence the one setting above. Forwarding cuDNN is the next piece of work and is
much the largest of the three, because of its graph API.

The real cuBLASLt is worth a note: over a remoted driver it does not report an
error, it segfaults inside `cublasLtMatmulAlgoGetHeuristic`. Anything unshimmed
that reaches the driver may fail that way rather than cleanly.

Note also that `LD_LIBRARY_PATH` is not enough to put these shims in front of
PyTorch: its libraries use `RPATH`, which takes precedence, and it preloads the
CUDA libraries by absolute path. Use `LD_PRELOAD`, or install without the stock
`nvidia-cuda-runtime` package.

Managed memory and zero-copy host mapping cannot work across a network and are
refused explicitly. So are the deprecated `cuCtxAttach` and the green-context
calls (`cuGreenCtx*` and `cuCtxFromGreenCtx`), which would hand a client a
context the server cannot account for; `cuCtxDetach` works, and destroys the
context as `cuCtxDestroy` does. Kernel launches with `cuLaunchKernelEx` launch
configurations are not marshalled yet, and `cuLaunchCooperativeKernel` is
refused: the launch message has no way to say "cooperative", and running it as
an ordinary launch would drop the guarantees the kernel was written around.

A dropped connection does not lose the GPU state. The server keeps the session
for `RGPU_SESSION_GRACE` seconds, and the client reconnects, sends again what
the server never acknowledged, and carries on. If the client does not get back
in time, the session expires and the server releases what it held. The same
happens if the server restarts. The client is then told its session is gone:
it says so, sends nothing more, and every later call fails. It never carries on
against an empty GPU. Expiry releases the allocations, contexts, modules,
streams, events, graphs and maths-library handles a session made. It does not
track `cuMemAddressReserve` ranges, `cuGraphConditionalHandleCreate` handles,
user-object retains, cuDNN descriptors minted on the client,
`cuLibraryLoadData` libraries or texture references. Those stay until the
server exits.

`cuDevicePrimaryCtxReset` destroys what every session on the server holds in
that context, so the server refuses it with `CUDA_ERROR_NOT_SUPPORTED` while
any other session is live, including one waiting out its grace period.

Each client thread keeps its own current context, context stack and stream
capture mode on the server, but calls are still served one at a time, not in
parallel. A session may have at most `RGPU_MAX_CLIENT_THREADS` live client
threads, and a thread may push at most `RGPU_MAX_CONTEXT_STACK` contexts. Past
either limit the call is refused. A call sent without a reply that fails has
its error handed to the same thread's next call that replies. If that thread
never makes one, the error shows up only in the server log. PyTorch's autograd
threads are real threads, but they also allocate and call cuBLAS, which reply,
so in practice their errors still arrive.

## The macOS path: `rgpu` as a PyTorch device

Everything above replaces the driver underneath CUDA, which needs a container
with CUDA's own runtime and math libraries. There is a second, lighter path in
`python/`: `rgpu` as a PyTorch device name, registered from pure Python
through PyTorch's own `PrivateUse1` backend hook. It runs on the stock,
CPU-only `torch` wheel you get from `pip install torch` on a Mac - no CUDA
libraries, no Docker, no driver, nothing to build. A tensor made with
`device="rgpu"` is a small stand-in on this side; its data and every op on it
live on a remote GPU host running `rgpu-opserver`.

Three steps:

```sh
# on the Mac
pip install -e python

# on the GPU host: binds 127.0.0.1 by default and has no authentication
# at all, so an ssh tunnel is the only supported way in
rgpu-opserver
ssh -N -L 9720:localhost:9720 user@gpuhost
```

```python
# on the Mac
import rgpu, torch

x = torch.randn(1024, 1024, device="rgpu")
y = (x @ x).relu().sum().item()   # one round trip, at .item()
```

`RGPU_OPSERVER=host:port` points the client at the tunnel if it isn't the
default `127.0.0.1:9720`. The client and server must run the same torch
major.minor version - op schemas can differ between versions, and the client
checks this at connect time.

Measured once from a Mac over an ssh tunnel to a rented NVIDIA A40 (44.7 ms
round trip): the full suite passes (152 passed, 4 skipped), a ResNet-18
training step costs one round trip and runs in 90-106 ms depending on mode
(eager inference is a separate figure, 152.0 ms), and a 200-step training
run survived the tunnel being killed and restored, with bit-identical
losses. See `docs/PRODUCT_SPEC.md` for the numbers; this is one Mac against
one A40, not a benchmark suite. The server
also turns TF32 off by default on a CUDA device so results match a CPU
reference - set `RGPU_TF32=1` to trade that for speed.

### Known limitations

- **Move a model to `rgpu` before any grad-tracking forward pass.** Because an
  rgpu tensor is a wrapper subclass, `nn.Module.to("rgpu")` installs
  parameters with `torch.utils.swap_tensors`, which refuses a parameter that
  something else still holds - and the autograd graph of an earlier forward
  holds exactly that. Moving afterwards raises `_apply(): Couldn't swap
  <Module>.<param>`. A real CUDA device has no such rule. Under
  `torch.no_grad()`, or before the first forward, rgpu behaves the same as
  CUDA.
- **An op whose output size depends on the data costs a round trip**, and
  cannot write into an `out=` tensor - only the data can say how big the
  result is, so there is no size to allocate it at ahead of time. Call it
  without `out=`.
- **A graph containing a non-`aten` custom op cannot run on rgpu at all**,
  compiled or eager: `rgpu-opserver` runs aten ops only, so a custom op has
  nowhere to run.
- **`torch.compile(model, backend="rgpu")` is the supported way to compile.**
  The graph is shipped once and compiled on the server; after that, a call is
  one message.
