# JAX on a remote GPU: what was tried and what came of it

Results from 2026-09-15/16, against `jax-extension-plan.md`. That document is
the proposal; this one is what happened, and it supersedes parts of it.

**Headline: transparent remote JAX works, and needs no PJRT plugin.** A Mac
with no GPU, running stock `pip install jax`, executed a JAX computation on a
remote machine through ordinary `jax.device_put` and `jax.jit`. No plugin, no
shim, no code generation.

Everything below was run rather than reasoned about, except where it says
otherwise.

## 1. Cross-platform export works (milestone 0)

The plan's gating question: can a CPU-only Mac export JAX that a remote CUDA
host runs, with matching answers?

Yes. Four cases — a matmul, a differentiated MLP update, random number
generation, and `lax.scan` — exported on an M3 with no GPU, run on a rented
A40:

```
  matmul     ok    max|diff| 0.000e+00
  mlp_grad   ok    max|diff| 0.000e+00
  prng       ok    max|diff| 0.000e+00
  scan       ok    max|diff| 0.000e+00
```

Bit-identical to native JAX on that GPU. jax/jaxlib 0.11.1 both sides.

Incidental findings: `jax.export` needs `flatbuffers`, which it imports lazily
and dies without at `serialize()`. And `jax.export` takes the platform as
`"cuda"` while `jax.devices()[0].platform` reports `"gpu"` — comparing them
directly refuses a run that should proceed.

Detail: `jax/README.md`, tool: `jax/export_probe.py`.

## 2. A custom platform needs CUDA's lowering rules

The plan gates PJRT work on proving JAX can lower for a registered remote
platform without a client GPU. It can — but not by itself, and **the obvious
test gives the wrong answer**.

An unregistered platform name does not fail. It lowers generically, and for a
matmul produces MLIR structurally identical to CUDA's. Probing with a matmul
would declare success.

What breaks is the **41 primitives holding CUDA-specific rules** (25 CUDA-only):
convolution, attention, FFT, the linear-algebra family, the collectives. `eigh`
and `svd` raise `NotImplementedError` for an unknown platform.

Copying CUDA's rules onto the platform fixes it, and the emitted MLIR is
identical — the only difference across the module was a source-location line
number. The mechanism is a private registry, since the public
`register_lowering` refuses already-wrapped entries.

**This section is now mostly moot**, because the proxy route below does not use
a custom platform name at all. It is kept because it is the answer the plan
asked for, and it would matter again if anyone revisits a PJRT plugin.

Detail: `jax/pjrt-precondition.md`.

## 3. The proxy route, which is the actual result

`jaxlib` already ships the IFRT proxy **client** (`jaxlib/_ifrt_proxy.so`), and
`jax.extend.backend` re-exports it as a supported extension point. OpenXLA
ships the **server** as libraries only — eight `cc_library` targets, no
`cc_binary`, and no `grpc_server_main` anywhere in the repository.

So the missing piece was a ~40-line `main()`. With it:

```
connected to grpc://127.0.0.1:12345
  devices: [CpuDevice(id=0)]          <- the remote machine's device
  placed  : (3, 4) on CpuDevice(id=0)
  computed: [[28.0, 76.0, 124.0], [76.0, 252.0, 428.0], [124.0, 428.0, 732.0]]
  matches a local numpy reference: True
```

On GPU it goes most of the way: the A40 appears on the Mac as
`CudaDevice(id=0)` and `device_put` moves arrays onto it, but `jax.jit` ends
the session during compilation with no error logged and the server still alive.
Unresolved.

Detail and the full build log: `jax/ifrt_server/README.md`.

## What this supersedes in the plan

| plan says | result |
|---|---|
| Milestone 1: build an explicit `jax.export` API whose handles are not `jax.Array` | **skip it.** The transparent route worked directly, so the intermediate API need not be built. |
| Milestone 3: write a PJRT C-API plugin — device discovery, buffers, execution, completion events | **not needed.** jaxlib ships the client; only a server `main()` was missing. |
| "Prove JAX can lower for the registered remote platform" | Answered, and made moot by the proxy, which uses ordinary device names. |

The plan's caution that the first deliverable must not be advertised as
transparent backend support turned out not to bind, because the thing built
*is* the transparent one.

## Costs, measured

| | |
|---|---|
| XLA CPU build (upstream + ours) | 74 min, 11,798 actions, 292 MB |
| XLA CUDA build | **4h22m**, 15,382 actions, 349 MB |
| Incremental rebuild after a dep change | 84-137 s |
| Pod time across the whole effort | ~7 h, roughly $3.50 |

A 96-core box. Note RunPod's API reported `vcpuCount: 9` for it; trust `nproc`.

## Traps worth knowing

Each presents as something other than what it is.

1. **Import `jax` before calling `get_client`.** Otherwise the interpreter
   segfaults — exit 139, no traceback. This reads exactly like a version
   mismatch and is not.
2. **`IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true`**, on both sides, spelled
   exactly. `1`, `TRUE`, `True`, `yes` are silently ignored and leave ALTS
   credentials in force, failing every connection with `Invalid credentials`.
3. **Match the server's XLA revision to the client's jaxlib.** jaxlib 0.11.1 is
   XLA `dcf304bc`; read it from `third_party/xla/revision.bzl` at the matching
   jax tag. XLA HEAD failed three times inside gRPC before pinning fixed it
   outright.
4. **Two `alwayslink` deps the compiler cannot warn you about**:
   `stream_executor/cuda:all_runtime` and
   `backends/gpu/collectives:gpu_collectives_plugin`. Omit either and the binary
   builds and starts perfectly, then fails at runtime.

## Where this leaves things

Two costs decide whether this is a product or an experiment:

- **A ~4.5 hour CUDA rebuild of XLA per jaxlib version**, because the proxy
  handshakes on version. This is the serious one and has no answer yet.
- **The SSH question.** The server is a full JAX+CUDA install on the GPU host,
  so if running the whole program there over SSH is acceptable, this buys
  nothing. Its value is narrow and specific: local REPL, data and plotting,
  with compute and state remote.

Recommended: treat it as a finished experiment unless that local-orchestration
workflow is actually wanted. If it is, solve the version-coupled rebuild first.

Next step if resumed: the GPU `jit` failure needs a core dump rather than logs.
The cheapest lead is a pod image whose CUDA matches the pin — the box used had
`ptxas` 12.8 on disk while XLA reported toolkit 13.2 against a 13.0 driver.
