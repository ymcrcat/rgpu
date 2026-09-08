# rgpu — CUDA API remoting for PyTorch

Design document. Written 2026-09-07.

## Goal

Run PyTorch on a machine with no GPU. `torch.device("cuda")` behaves as if a local
GPU existed; the CUDA work executes on a remote GPU host.

The device is spelled `cuda`, not `rcuda`. Because the shim replaces the driver
library underneath PyTorch rather than adding a backend inside it, stock PyTorch
wheels work unmodified. A separate `rcuda` device would require a PyTorch fork and a
new `PrivateUse1` backend: far more work, worse result. No PyTorch source changes
are needed at any point.

## Prior art and what we take from each

| System | Layer | Lesson taken |
|---|---|---|
| rCUDA | Runtime API + per-library wrappers | Pipelining and batching are what make remoting usable |
| GVirtuS | Runtime API | Pluggable transport backends |
| qCUDA | Runtime API over virtio | Runtime-level interception needs `-cudart=shared` and covered only 32 functions. Evidence for choosing the driver layer instead |
| mrCUDA | Runtime API | Remote-to-local migration by replaying recorded state-establishing calls. We reserve a `record` annotation so this stays possible |
| AvA (OSDI '20) | Generated from a spec | The LAPIS annotation vocabulary. Our annotation schema borrows it directly |
| SCUDA / LUPINE | Driver API, `libcuda.so.1` | Confirms driver-level shim plus NVML shim is a viable production shape |

AvA's LAPIS is the direct model for our generator. Its vocabulary maps onto what we
need: `ava_buffer(n)` for sized buffers with the size expressed in terms of other
parameters, `ava_input`/`ava_output` for direction, `ava_handle` for opaque values
that must never be dereferenced, and `ava_sync`/`ava_async`/`ava_flush` for
execution semantics. We adopt a reduced form of it as a Python dictionary rather
than a new DSL with its own compiler, because we target one API rather than an
open-ended set.

## Interception layer

Replace `libcuda.so.1`. Stock `libcudart`, `libcublas`, `libcudnn`, `libcusparse`
and `libcufft` then run unmodified on the client and funnel through one API, so
every math library comes free from a single surface. Wrapping the runtime API plus
each math library by hand would mean maintaining a cuBLAS wrapper, a cuDNN wrapper
and so on, all of which PyTorch exercises. qCUDA's 32-function scope shows how that
path narrows.

The client has no NVIDIA driver, so there is no real `libcuda.so.1` to `LD_PRELOAD`
over. The shim **is** `libcuda.so.1`: built with that `SONAME`, placed on
`LD_LIBRARY_PATH` in the client container.

## The three hard problems

1. **`cuGetProcAddress` must be intercepted.** Since CUDA 11.3 `libcudart` resolves
   driver entry points through `cuGetProcAddress`, not `dlsym`, so exported symbols
   alone are not enough. We intercept `cuGetProcAddress` and `cuGetProcAddress_v2`
   and hand back our own pointers, honouring the requested CUDA version.

2. **Kernel arguments carry no sizes.** `cuLaunchKernel`'s `kernelParams` is an
   array of N pointers with no lengths; the driver recovers sizes from module
   metadata the shim does not have. After loading a module the server calls
   `cuFuncGetParamInfo`, or `cuKernelGetParamInfo` for the CUDA 12 library API, to
   get each parameter's offset and size, caches the layout, and returns it to the
   client on first use of a function handle. When the caller uses the `extra` path
   with `CU_LAUNCH_PARAM_BUFFER_SIZE`, the size is given and no query is needed.

3. **Fatbin images arrive without a length.** `cuModuleLoadData` and
   `cuLibraryLoadData` take a bare `void*`. The shim parses the fatbin header, magic
   `0xBA55ED50`, to compute the blob size, and handles the cubin ELF and
   NUL-terminated PTX cases.

## Handles and pointers

Device pointers and opaque handles are the server's real values, passed through
verbatim. The client never dereferences them, so there is no translation table. This
keeps the shim stateless about object identity.

## Architecture

```
client container (linux/arm64, no GPU, no driver)
  python + torch → libtorch_cuda → libcudart / libcublas / libcudnn   (stock)
                                        ↓ driver API
                                 our libcuda.so.1
                                        ↓ TCP, length-prefixed frames
────────────────────────────────────────────────────────────────────────
  rgpu-server → real libcuda.so.1 → GPU                (cloud GPU VM)
```

One TCP connection per client process, requests carrying an id so client threads
multiplex over it. Each connection owns a server-side session holding contexts,
modules, streams and the parameter-layout cache.

## Protocol, built for batching

Frames carry a flags field with a `NO_REPLY` bit. Each API is tagged with an
execution class, following LAPIS:

- **sync** — returns data or an immediately observable status. Round trip.
- **async** — effect observable only at a later synchronization point.
  `cuLaunchKernel`, async copies, `cuEventRecord`.
- **flush** — returns immediately but forces queued work to be submitted.

Version one round-trips everything. Enabling batching later means letting the async
class go fire-and-forget and flushing at sync points: a policy change, not a
protocol rewrite. Deferring errors this way matches CUDA's own sticky error
reporting.

## Code generation

The driver API is roughly 700 functions, so we generate. `codegen/` parses `cuda.h`
with libclang into a JSON model, then emits client stubs, server dispatch and
serialization. An annotations file carries what the parser cannot infer: pointer
direction, buffer lengths, execution class, and which functions need hand-written
bodies. Anything unannotated and non-trivial gets a stub that logs the function name
and returns `CUDA_ERROR_NOT_SUPPORTED`, so gaps are discovered rather than silently
wrong. That log is the worklist for later phases.

Codegen runs in a CUDA devel container, which supplies both `cuda.h` and clang, so
it is reproducible and adds no host dependencies.

## Phases

- **0. Cross-architecture check.** arm64 client, x86_64 server. Fatbin selection is
  driven by device architecture, so this should work, but it is unverified and
  decides the client topology. Build a fatbin on the client, load and launch it on
  the server. On failure, switch the client container to `linux/amd64` under
  emulation; the client does only host-side work, so the cost is iteration speed.
- **1. Skeleton.** Codegen, wire format, transport, ~30 functions. A hand-written
  driver-API vector-add runs remotely. No `libcudart` yet.
- **2. The runtime API.** Our own `libcudart.so.12`, translating to driver
  calls: contexts, memory, streams, events, kernel registration and launch. A
  program written against the runtime API runs correctly.
- **3. PyTorch. Milestone 1.** Up the ladder filling gaps: `is_available`, tensor
  allocation, add, matmul via cuBLAS, convolution via cuDNN, plus a minimal NVML
  shim. Done when ResNet-18 inference matches a CPU reference within tolerance.
- **4. Performance.** Fire-and-forget for the async class, coalescing, a bulk
  transfer path, and measurement.

Later, explicitly not milestone 1: training, multi-GPU and NCCL, `torch.compile` and
Triton, shared-memory transport, RDMA, and the mrCUDA-style migration that the
`record` annotation keeps open.

## Known unsupported

Managed memory, `cuMemAllocManaged`, cannot work transparently across a network.
Zero-copy mapped host memory, `cuMemHostGetDevicePointer`, likewise. Both return a
clear error rather than corrupting silently. PyTorch needs neither by default.

## Verification

- Round-trip serialization tests per generated parameter type.
- Each CUDA sample runs natively on the GPU host and through the shim from the
  client. Outputs must match byte for byte.
- Torch operations compared against CPU references within tolerance.
- Any call reaching an unimplemented stub logs its name and fails loudly, so a clean
  run proves nothing was silently skipped.

## Finding, 2026-09-08: the driver-only design cannot carry the stock runtime

Testing the shim under a stock `libcudart` 12.8 showed that `cuGetProcAddress`
interception works exactly as intended: cudart asks us for entry points and we
answer. It then calls `cuGetExportTable` immediately after `cuInit`, and
refuses to initialise when it does not get one. `cudaGetDeviceCount` fails.

`cuGetExportTable` returns a table of undocumented internal driver function
pointers keyed by UUID. cudart 12.8 asks for two:

    6bd5fb6c-5bf4-e74a-8987-d93912fd9df9
    a094798c-2e74-2e74-93f2-0800200c0a66

The refusal is not sensitive to the error code returned. `CUDA_ERROR_NOT_FOUND`,
`CUDA_ERROR_NOT_SUPPORTED` and even `CUDA_SUCCESS` with a null table all fail;
success with a null table simply makes cudart ask for the second table before
giving up.

The table cannot be forwarded, because its contents are function pointers into
the server's address space and the client would call straight into them.

This is a known result rather than a mistake in our implementation. The Cricket
paper states that these hidden functions are used extensively by the runtime
API and that their undocumented nature means a virtualization layer at the
driver API level does not allow the use of the original runtime API on top of
it. LUPINE, which advertises a `libcuda.so.1` shim, also ships its own CUDA
runtime translation stubs rather than running the stock runtime. ZLUDA does
implement the tables, as reverse-engineered work against an undocumented
interface.

### Consequence

The premise that stock `libcudart`, cuBLAS and cuDNN could sit unmodified on a
driver-level shim is false. Reaching PyTorch requires replacing `libcudart` as
well, which is what every working system in this space does.

What phase 1 built still stands: the code generator, the wire format and
transport, the server framework, the fake-driver test harness, and the
`libcuda.so.1` shim itself, which is still needed for direct driver API users
and for the driver calls the math libraries make. The generator reads any
header, so pointing it at `cuda_runtime_api.h` reuses the same machinery.

## Revised architecture

The runtime is replaced by a **translation** layer rather than a forwarding
one. `libcudart.so.12` expresses the runtime API in driver API calls on the
client; `libcuda.so.1` remotes those. Nothing about the server changes, and it
stays purely driver-level.

```
client (no GPU, no driver)
  python + torch → libtorch_cuda → libcublas / libcudnn      (stock)
                        ↓ runtime API        ↓ runtime + driver API
                  our libcudart.so.12  ──────┘
                        ↓ driver API
                  our libcuda.so.1
                        ↓ TCP
──────────────────────────────────────────────────────────────────
  rgpu-server → real libcuda.so.1 → GPU                 (GPU host)
```

Translating locally rather than forwarding the runtime API keeps one wire
protocol and one server, and means the math libraries still ride along for
free: they sit on our runtime and our driver, both of which are ours.

Three parts of the runtime have no driver-API equivalent and are implemented
directly:

- **Per-thread current device and lazy primary contexts.** The runtime binds a
  primary context per device on first use; the driver does not.
- **Last-error state**, which the runtime remembers per thread and clears on
  read.
- **Kernel registration.** `__cudaRegisterFatBinary` and
  `__cudaRegisterFunction` are how a host function pointer comes to stand for a
  kernel, which is what `cudaLaunchKernel` receives. We keep that registry on
  the client, load each module on first launch rather than at registration
  because PyTorch registers far more device code than a given process uses, and
  pass launch arguments straight to the driver shim, which already recovers
  their sizes from the server's parameter layout.

`cudaMemcpyDefault` needs to know whether each pointer is host or device.
Remotely we cannot probe a device pointer, since it is an address in the
server's process, so the client remembers the ranges it handed out.


## Finding, 2026-09-08: the export table problem is not limited to the runtime

Every NVIDIA library initialises the same way. cuBLAS calls `cuGetExportTable`
during `cublasCreate` and returns `CUBLAS_STATUS_NOT_INITIALIZED` without it.
cuBLASLt is worse: it does not report an error at all but segfaults inside
`cublasLtMatmulAlgoGetHeuristic`, which is where PyTorch's `addmm` lands by
default.

So the rule is general. A library that talks to the driver through the dark API
has to run on the GPU host, and the client gets a shim that forwards its calls.
That is the rCUDA shape, arrived at from the other direction.

cuBLAS is now forwarded. Its calls are hand-written rather than generated,
because the parameters need judgement the header does not carry: matrix
arguments are device pointers passed through untouched, while alpha and beta
are host values or device pointers depending on the handle's pointer mode,
which the client therefore tracks. Getting that wrong would produce quietly
wrong arithmetic rather than an error.

### Milestone 1 reached

ResNet-18 inference on a remote RTX A4000 matches a CPU reference to 3.8e-06
with identical top-5 predictions, over the full stack: PyTorch on stock wheels,
our runtime shim, our cuBLAS shim, our driver shim, TCP, and the real driver on
the host.

Two PyTorch settings are needed until cuBLASLt and cuDNN are forwarded:
`DISABLE_ADDMM_CUDA_LT=1` and `torch.backends.cudnn.enabled = False`.

### On library precedence

`LD_LIBRARY_PATH` is not enough to put a shim in front of PyTorch. Its
libraries carry `RPATH`, which the loader consults before `LD_LIBRARY_PATH`,
and PyTorch additionally preloads the CUDA libraries by absolute path.
`LD_PRELOAD` wins over both, because symbol lookup finds preloaded objects
first regardless of which file was loaded. A client that never installs the
stock libraries does not have this problem.
