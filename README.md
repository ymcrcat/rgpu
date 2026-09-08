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
| Client shim `libcudart.so.12` | builds, 38 translated and 278 stubs |
| Server | compiles; not yet run against a real GPU |
| Driver API end-to-end, fake driver | passing |
| Kernel launch marshalling, fake driver | passing |
| Runtime API end-to-end, fake driver | passing |
| End-to-end on a real GPU | needs a GPU host |
| PyTorch client image and test ladder | written, not yet run |

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

`scripts/deploy_server.sh user@host` copies the source over, builds the server
there, and brings the test fatbin back. It checks for a driver and a toolkit
first and says which is missing.

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

## Environment variables

| Variable | Meaning |
|---|---|
| `RGPU_SERVER` | `host:port` of the GPU server, default `127.0.0.1:9713` |
| `RGPU_PORT` | port the server listens on, default `9713` |
| `RGPU_VERBOSE` | log every forwarded call |

## How the code is organized

| Path | What it does |
|---|---|
| `codegen/parse.py` | `cuda.h` to a JSON API model, via libclang |
| `codegen/annotations.py` | what the header cannot express: buffer sizes, direction, execution class |
| `codegen/emit.py` | generates client stubs, server dispatch, entry table, fake driver |
| `common/wire.h` | frame format and serialization |
| `client/shim.cpp` | `cuGetProcAddress`, kernel launch marshalling, host allocations |
| `client/cudart_impl.cpp` | the runtime API translated into driver calls, plus kernel registration |
| `server/main.cpp` | accept loop, dispatch, parameter layout lookup |
| `tests/fake_cuda.cpp` | a driver backed by host memory, for testing without a GPU |

Anything the generator cannot marshal becomes a stub that logs its name and
returns `CUDA_ERROR_NOT_SUPPORTED`, so gaps show up as loud failures rather
than wrong answers. `codegen/report.txt` lists what is generated, hand-written
and stubbed.

## Known limits

Managed memory and zero-copy host mapping cannot work across a network and are
refused explicitly. Kernel launches with `cuLaunchKernelEx` launch
configurations are not marshalled yet. A dropped connection loses all
server-side GPU state, so the client fails subsequent calls rather than
silently reconnecting to an empty GPU.
