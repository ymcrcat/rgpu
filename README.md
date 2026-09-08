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

Interception happens at the CUDA driver API. Replacing `libcuda.so.1` means the
stock runtime and every math library run unmodified on the client and funnel
through one API, so cuBLAS and cuDNN come free from a single surface.

See `docs/superpowers/specs/2026-09-07-cuda-api-remoting-design.md` for the
design, the prior art it draws on, and the three hard problems it has to solve.

## Status

| Phase | State |
|---|---|
| Wire format and transport | working |
| Code generation from `cuda.h` | working, 259 of 435 functions generated |
| Client shim `libcuda.so.1` | builds, 436 exported entry points |
| Server | compiles; not yet run against a real GPU |
| End-to-end with a fake driver | passing |
| End-to-end on a real GPU | needs a GPU host |
| PyTorch | not started |

The GPU-free test is the meaningful one so far: real client stubs, real wire
format, real server dispatch, with a fake driver at the bottom. A byte-exact
memory round trip through it means the marshalling is right.

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

`scripts/gcp_up.sh` and `scripts/gcp_down.sh` provision and stop a GCP GPU
instance. It bills while running.

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
