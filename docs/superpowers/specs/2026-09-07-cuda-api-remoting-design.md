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
- **2. Stock `libcudart`.** The `cuGetProcAddress` double-wrapper, primary contexts,
  CUDA 12 lazy loading, parameter layouts, streams, events, pinned memory. An
  `nvcc`-compiled program with `cudaMalloc` and a `<<<>>>` launch runs correctly.
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
