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

cuBLAS and cuBLASLt are now forwarded. Its calls are hand-written rather than generated,
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

One PyTorch setting is needed until cuDNN is forwarded:
`torch.backends.cudnn.enabled = False`. With cuBLASLt forwarded, `addmm` runs
on its default path and all nine rungs pass.

### On library precedence

`LD_LIBRARY_PATH` is not enough to put a shim in front of PyTorch. Its
libraries carry `RPATH`, which the loader consults before `LD_LIBRARY_PATH`,
and PyTorch additionally preloads the CUDA libraries by absolute path.
`LD_PRELOAD` wins over both, because symbol lookup finds preloaded objects
first regardless of which file was loaded. A client that never installs the
stock libraries does not have this problem.


## Phase 4 result, 2026-09-09: batching

Calls in the async class now go without a reply, queued on the client and
written in one batch ahead of the next call that needs an answer. TCP keeps the
order, so the server sees the sequence the application issued.

Measured on an RTX 3090 with a real kernel, 2000 launches, over loopback:

| | round trip | batched |
|---|---|---|
| issue only | 20.3 us | 0.2 us |
| issue and synchronize | 20.3 us | 3.2 us |

Six times faster end to end on the least favourable transport there is. The
protocol reservation made in the original design, a NO_REPLY flag plus an
execution class per function, turned out to be the whole of what was needed:
enabling this was a policy change in the generator, not a protocol rewrite.

A call with no reply has nowhere to report a failure. The server holds the
first one and returns it from the next call that does reply, and logs it either
way. That is how CUDA reports asynchronous failures too, so the semantics are
not a compromise. The generator refuses to make a call fire-and-forget if it
returns data.

Verified on real hardware: all nine ladder rungs pass with batching on, and
identically with it off.


## Measured, 2026-09-09: ResNet-18 on an RTX A4000

Native and remoted on the same box, so the network is loopback and every
figure below is the floor, not an estimate of a real deployment.

| | native | remoted | remoted, RGPU_BATCH=0 |
|---|---|---|---|
| batch 1 | 2.37 ms | 24.99 ms | 23.14 ms |
| batch 32 | 11.26 ms | 24.58 ms | 23.59 ms |

Two things stand out. Remoted time barely moves with batch size, so this is
not about how much data crosses the wire. And batching makes almost no
difference, which says the calls being made are not the ones the async class
covers.

The round trip counter explains both: **638 round trips per inference**, and
27 one-way calls. At loopback's roughly 40 microseconds that is 25 ms, which
is the whole measurement. Where they go, per inference:

| calls | what |
|---|---|
| 289 | `cuDevicePrimaryCtxGetState` |
| 60 | `cudnnBackendSetAttribute` |
| 40 each | `cudnnCreateTensorDescriptor`, `cudnnSetTensorNdDescriptor`, `cudnnDestroyTensorDescriptor`, `cudnnSetStream` |
| 20 each | `cudnnBackendCreateDescriptor`, `Finalize`, `DestroyDescriptor`, `Execute`, `cudnnBatchNormalizationForwardInference` |
| 9 | cuBLAS |

So the ceiling is not bandwidth and not the kernel launch path that phase 4
addressed. It is that PyTorch asks the same cheap questions hundreds of times
per inference, and that cuDNN's descriptor churn is a round trip per
descriptor.

Three fixes follow directly, in order of what they buy:

1. **Cache `cuDevicePrimaryCtxGetState` on the client.** Its answer only
   changes when the client itself retains or releases the primary context, so
   it can be answered locally. Removes 45% of all round trips.
2. **Make the status-only cuDNN calls asynchronous.** `SetAttribute`,
   `Finalize`, `Destroy`, `Execute`, `SetStream` and the batch norm forward
   return nothing but a status, which is exactly what the deferred class is
   for. Another 200 per inference.
3. **Handles are what is left.** The `Create` calls have to answer with a
   value, so removing those 60 means minting handles on the client and
   teaching the server to map them, which is a real change rather than an
   annotation.

The first two together should take 638 to roughly 65, and they are policy and
annotation rather than new machinery. Worth doing before any transport work:
a shared memory transport makes each round trip cheaper, but there are 638 of
them, and ten times fewer round trips beats a faster one.


## Measured again, 2026-09-10: after cutting the round trips

Same box, same model, after caching the primary context on the client and
letting the status-only cuDNN calls go without waiting.

| | native | before | after |
|---|---|---|---|
| batch 1 | 2.2-2.4 ms | 24.99 ms | 5.7-6.5 ms |
| batch 32 | 11.27 ms | 24.58 ms | 11.30 ms |

At batch 32 remoting is now free: 11.30 against 11.27 native, which is inside
the run-to-run spread. At batch 1 it is 2.6 times native, down from 10.5.

Round trips per inference went from 638 to 109, and what is left is almost
entirely calls that have to answer with a value:

| calls | what |
|---|---|
| 40 | `cudnnCreateTensorDescriptor` |
| 20 | `cudnnBackendCreateDescriptor` |
| 20 | `cudnnBackendFinalize` |
| ~20 | cuBLAS, stream synchronisation, the rest |

`cuDevicePrimaryCtxGetState` no longer appears at all.

Two things this predicts. On a 0.1 ms datacentre link the added latency is
about 11 ms per inference rather than 64 ms, so batch 32 stays practical and
batch 1 does not. And the next round trip to remove is a handle: the create
calls have to answer, so going further means minting handles on the client and
mapping them on the server, which is the change the earlier note described.

Finalize stays synchronous on purpose. A cuDNN frontend uses its failure to
decide an engine is unsupported and try the next one, which is control flow
rather than an error, and deferring it would change which kernels run.


## torch.compile, 2026-09-11: the overhead goes away entirely

torch.compile needed almost nothing. Inductor's kernels reach the GPU through
the same driver calls this already forwards - Triton emits PTX, ptxas makes a
cubin on the client, then it is `cuModuleLoadData` and `cuLaunchKernel` - so
four of the five modes tested passed against eager on the first attempt. Only
`reduce-overhead` failed, for want of stream capture.

Capture turned out to need no new thinking. It happens on the server, and our
calls arrive there in the order the application made them, on the stream it
named, so the graph the driver builds is the graph the application described.
Nothing in the shim has to understand what is being captured.

ResNet-18, batch 1, on an RTX A4000:

| | native | remoted | round trips per inference |
|---|---|---|---|
| eager | 1.89 ms | 3.97 ms | 49 |
| torch.compile | 1.50 ms | 4.51 ms | 86 |
| torch.compile, reduce-overhead | 1.18 ms | **1.18 ms** | **2** |

With graphs the remoting overhead is not reduced, it is gone: the remoted
figure matches native to the hundredth of a millisecond, because an iteration
is one replay rather than several hundred calls. This is the answer to the
batch-size question as well. Batch 32 hid the overhead behind compute; graphs
remove it, so batch 1 is free too.

Plain `torch.compile` without graphs is *worse* than eager over the wire, 86
round trips against 49, even though it runs fewer kernels. Triton's launcher
asks the driver for the device pointer behind every kernel argument on every
launch, which was 144 round trips per inference before the client started
answering from its own allocation table, and is most of what remains.

### A quiet bug the generator would have shipped

`cuGraphGetNodes` takes a count that is the caller's capacity going in and the
number written coming out. The generator has no way to express that: it
classified the array as a single handle and never sent the capacity, so the
call would have reported zero nodes rather than failing. It is hand-written
now, but the shape is not unique to it, and the others of that shape should be
audited before something depends on one.
