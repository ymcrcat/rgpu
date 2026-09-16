# The missing IFRT proxy server

`jax/pjrt-precondition.md` establishes that jaxlib already ships the IFRT proxy
**client** (`jaxlib/_ifrt_proxy.so`, exposing `get_client`), and that
`jax.extend.backend` re-exports it as a supported extension point. If the
matching server runs on a GPU host, a Mac running stock JAX gets a remote GPU
through ordinary `jax.Array` placement — no hand-written PJRT plugin.

OpenXLA ships that server as **libraries only**. `xla/python/ifrt_proxy/server`
has `grpc_server.{h,cc}` and `ifrt_backend.{h,cc}`, its `BUILD` declares eight
`cc_library` targets and no `cc_binary`, and a code search for
`grpc_server_main` across the whole repository returns nothing. So the server
exists and is maintained, but nobody ships a runnable one.

`rgpu_ifrt_server.cc` is that missing `main()`. It is ~40 lines around
`GrpcServer::CreateFromIfrtClientFactory`, modelled on
`xla/python/ifrt_proxy/integration_tests/scoped_pjrt_cpu_via_proxy.cc`, the only
in-tree example of standing the server up over a PJRT client.

## Status

**It works.** A Mac running stock JAX placed an array on a remote machine and
ran a jitted computation there, through `jax.device_put` and `jax.jit`, with no
plugin and no code generation.

```
connected to grpc://127.0.0.1:12345
  devices: [CpuDevice(id=0)]          <- the remote machine's device
  placed  : (3, 4) on CpuDevice(id=0)
  computed: [[28.0, 76.0, 124.0], [76.0, 252.0, 428.0], [124.0, 428.0, 732.0]]
  matches a local numpy reference: True
```

Server side, for the same run:

```
grpc_service_impl.cc:150] Cleaning up host buffer store for session 8
grpc_service_impl.cc:152] Done with IFRT session 8
```

That is the plan's milestone 3 acceptance shape — real remote-backed arrays,
`device_put`, jitted arithmetic — reached **without writing a PJRT plugin**.

### GPU backend: most of the way, stops at JIT

The CUDA build works and the remote A40 is fully visible from the Mac. What
does not yet work is executing a compiled computation.

Working:

```
StreamExecutor [0]: NVIDIA A40, Compute Capability 8.6 (Driver: 13.0.0[580.159.4])
backend has 1 device(s), platform cuda
Addressable PjRt-IFRT device: CudaDevice(id=0)
```
```
connected to grpc://127.0.0.1:12345
  devices: [CudaDevice(id=0)]      <- the remote A40, seen from macOS
  placed  : (3, 4) on cuda:0       <- device_put to the remote GPU
```

So the handshake, device discovery and host-to-remote-GPU transfer all work.
`jax.jit` then dies: the server logs
`se_gpu_pjrt_compiler.cc:382 Performing a JIT compilation` and the session
ends, the client seeing `Stream removed (Socket closed)`. **No fatal error is
logged and the server process stays alive**, so a worker thread is going down
without reporting. Diagnosing it needs a core dump or gdb rather than logs.

Ruled out: XLA autotuning (`--xla_gpu_autotune_level=0` changes nothing),
memory (503 GB, no OOM), and missing ptxas.

Worth a look next: the box's CUDA is inconsistent — `ptxas` on disk is 12.8
while XLA reports `Runtime: 13.2.0; Toolkit: 13.2.0` from its hermetic CUDA,
against a 13.0 driver. A pinned XLA compiled for one CUDA and JIT-compiling
against another is a plausible cause, and a pod image whose CUDA matches the
pin would test it cheaply.

### Two BUILD deps the compiler cannot tell you about

Both are `alwayslink` static-initialiser registries: omit them and the binary
builds and starts perfectly, then fails at runtime.

| dep | symptom without it |
|---|---|
| `//xla/stream_executor/cuda:all_runtime` | `Could not find registered platform with name: "cuda"` |
| `//xla/backends/gpu/collectives:gpu_collectives_plugin` | allocates 33 GiB on the GPU, then `Check failed: No collectives registered for platform: cuda` |

Neither is reachable by reading headers, which is how the rest of the dep list
was written. They are in `BUILD` now.

The CUDA build took **4h22m** for 15,382 actions and a 349 MB binary, against
74 minutes for CPU. Incremental rebuilds after a dep change were 84-137 s.

### What it cost

Pinned XLA `dcf304bc` on a 96-core box, `--backend=CPU`:

| | |
|---|---|
| upstream `grpc_server` | 2352 s, 9,040 actions |
| our `rgpu_ifrt_server` | 4443 s total, 11,798 actions |
| binary | 292 MB |

**The pin was the whole fix.** No `--features` workarounds were needed: XLA
HEAD had failed three times inside gRPC, and `dcf304bc` simply built. Our
`main()` and BUILD file compiled and linked on the first attempt.

An earlier note here said a full build takes about fifteen minutes. That was
measured from a build that *failed partway* and never ran the full action
graph; the honest cold number is about 40 minutes for upstream and 74 for
ours, still only a few cents of pod time.

### Three things that each cost an afternoon

1. **Import `jax` before calling `get_client`.** Without it the interpreter
   segfaults — exit 139, no traceback, no message. The jax runtime has to be
   initialised first. This is the one that looked like a version mismatch and
   was not.
2. **`IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true`, on both sides**, spelled
   exactly like that. Anything else silently leaves ALTS credentials in force
   and every connection is refused as `Invalid credentials`.
3. **Match the server's XLA revision to the client's jaxlib**, because of the
   proxy's version handshake. `build_on_host.sh` pins it.

See `client_example.py` for the working client.

## Building it

It has to be built inside an OpenXLA checkout, because it depends on XLA
targets that are not exposed by any released package.

Copy `build_on_host.sh` to the build machine and run it there. It installs
bazelisk, fetches the pinned XLA revision, drops our source into the tree,
configures, and builds upstream's target before ours:

```sh
bash build_on_host.sh              # CPU backend, the cheap proof
BACKEND=CUDA bash build_on_host.sh
```

About fifteen minutes cold on a 96-core box, under two minutes warm, and tens
of gigabytes of disk.

**Do the CPU backend first.** `--backend=cpu` needs no CUDA in the build at
all, and proves the part that is actually in question: that a Mac client can
drive a remote IFRT proxy server transparently. Adding CUDA afterwards is a
build concern, not a design one.

## Running it

```sh
# on the GPU host
./rgpu_ifrt_server --port=12345 --backend=gpu
# from the client
ssh -N -L 12345:127.0.0.1:12345 user@gpuhost
IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true python your_script.py
```

The client needs `grpc://127.0.0.1:12345` and that environment variable set to
exactly lowercase `true`. `1`, `TRUE`, `True` and `yes` are silently ignored and
leave ALTS credentials in force, which fails every connection with
`INVALID_ARGUMENT: Invalid credentials` and no hint that a setting was dropped.

No authentication, as everywhere else in rgpu: bind localhost, tunnel in.
Anyone who can reach the port can run arbitrary computations on the GPU and
read its memory.

## Why this is still the cheaper path

Writing a PJRT C-API plugin means implementing device discovery, compilation,
buffers, execution, transfers, completion events and deletion, then keeping up
with the C API. This is a `main()` over an implementation Google maintains and
tests. The cost here is a build, not a design.
