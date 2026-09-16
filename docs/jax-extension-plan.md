# Extending rGPU to JAX

Status: **partly superseded by results.** See
[jax-findings.md](jax-findings.md), which reports what was actually built and
run on 2026-09-15/16. In short: transparent remote JAX works through the IFRT
proxy, so milestone 1 (the explicit `jax.export` API) can be skipped and
milestone 3 (writing a PJRT plugin) is unnecessary. The reasoning below is kept
as the proposal it was; where it conflicts with the findings, the findings win.

Original status line: proposed; no implementation or JAX benchmarks yet.
Updated 2026-09-14.
Code inspected at `b87d874`; existing uncommitted connection/test edits are outside this proposal.

## Recommendation

Start with an explicit remote compiled-function API using `jax.export`, with a Python JAX server holding arrays and executables. Ship an entire training step, including gradients and optimizer updates, and keep training state on the GPU. Prove native macOS-to-CUDA execution before building a transparent PJRT device backend.

These are two distinct deliverables. The first runs exported JAX functions remotely; its handles are not `jax.Array` objects. The second would make a remote device available through ordinary JAX device placement and execution. Do not advertise the first as transparent JAX backend support.

If moving the entire training program to the GPU host over SSH meets a workload's needs, that remains the simpler option. This extension is useful when Python orchestration must remain local while compiled computations and state stay remote.

## What the existing code contributes

| Existing component | Reuse decision |
| --- | --- |
| `python/rgpu/compile.py` and server `session.py` | Reuse the compile-once, call-by-handle design. Their FX/ATen graph representation and execution are PyTorch-specific. |
| `python/rgpu/wire.py` | Preserve bounded framing and explicit encoding principles. It imports Torch and carries Torch types, so it cannot be the JAX-only client's dependency unchanged. |
| `python/rgpu/session.py` | Use ordering, error propagation and resource-lifetime behavior as references. Do not undertake a shared-session refactor to start this feature. |
| `python/rgpu_run.py` | Reuse the SSH forwarding approach. It currently selects `RGPU_OPSERVER` and port 9720; JAX needs explicit service selection before this launcher can support it. |
| CUDA shim | Keep independent. Linux JAX through intercepted CUDA is a separate compatibility experiment and does not provide a native Mac client. |

Use a separate `jax/` Python distribution with an `rgpu_jax` package; installing it must not install or import Torch. Start with a small client, server and bounded codec, plus one integration test module and one example. Extract existing neutral helpers only when both callers demonstrably benefit. A manual SSH tunnel is sufficient for the first milestone.

## Execution boundary

JAX supports exporting a jitted function for a selected target platform even when that accelerator is absent locally. Use the public export API, explicitly targeting CUDA; do not assume a CPU-lowered module is GPU-portable. Preserve its calling-convention metadata and platform safety checks. [JAX export API](https://docs.jax.dev/en/latest/_autosummary/jax.export.export.html)

The proposed flow is:

1. Locally transform a pure function with `grad`/`value_and_grad`, `vmap`, or `lax.scan` as needed, then jit and export it against input specifications.
2. Send the serialized export once. The server deserializes it, places inputs on the selected GPU, and compiles a reusable callable for that signature.
3. Upload initial parameters, optimizer state, PRNG state and inputs. Return opaque, session-scoped handles.
4. Execute by executable and buffer handles. Store resulting arrays remotely and return their handles.
5. Download only explicitly requested results, metrics or checkpoints; release obsolete state explicitly.

The public export serialization includes more than bare StableHLO. Prefer `Exported.serialize()` and `export.deserialize(...).call(...)` over a private jaxlib compilation interface for the prototype. Serialized exports are trusted executable input; localhost binding and SSH are required deployment boundaries. [JAX exporting and serialization](https://docs.jax.dev/en/latest/export/export.html)

Illustrative proposed API, not currently available:

```python
with rgpu_jax.connect(address) as remote:
    step = remote.compile(train_step, state_spec, batch_spec)
    state = remote.put(initial_state)
    for batch in batches:
        with remote.put(batch) as batch_on_gpu:
            new_state, loss = step(state, batch_on_gpu)
            remote.release(state)
            state = new_state
            print(remote.get(loss))
            remote.release(loss)
    checkpoint = remote.get(state)
```

`train_step` contains the differentiated loss and state update. Transform it before remote compilation. Calling `jax.grad` on this Python RPC wrapper is unsupported. Logging every iteration is a correctness example, not the performance configuration; fetching a scalar still costs a network wait.

## First-release contract

- One local controller, one server process, one selected CUDA GPU, fixed input shapes. CPU server mode exists for tests and must be explicitly selected; never silently fall back from a requested GPU.
- Built-in tuple/list/dict pytrees, supported numeric arrays and scalar leaves. Validate leaf shape, dtype, byte length, tree structure and handle ownership. Static configuration is bound before export; changed signatures require explicit recompilation.
- Begin with float32 and integer inputs, plus a documented uint32 PRNG key representation. Test bfloat16 and typed keys separately before claiming support; reconstructing every dtype through a generic NumPy string is insufficient.
- Keep parameters, optimizer state and PRNG state remote. No global reseeding command: randomness is part of the explicit function state.
- No buffer donation initially. Releasing an input cannot invalidate in-flight GPU use; retain server references through completion. Free executables and arrays on explicit release or session close. Bound active allocations and queued work.
- Reject multi-device exports, effects requiring host callbacks, unsupported layouts/types and arbitrary user custom calls. Compiler-generated CUDA custom calls needed by supported JAX operations remain necessary; validate a tested compatibility set rather than rejecting every custom call.
- Pin a tested client/server JAX and jaxlib release pair initially. Handshake includes protocol version, export compatibility, target device and dtype configuration such as x64. Report incompatibility before accepting work.
- Separate wire magic/version and configurable port from the Torch opserver. Minimal operations: hello, compile, upload, execute, download, release and synchronize. Do not serialize Python functions or use pickle.
- Start with synchronous completion acknowledgments. Connection loss invalidates the session and all handles; never silently replay a training update after an ambiguous disconnect. Resume from an explicit host checkpoint. Reconnect/replay can be added only with an at-most-once execution design and failure tests.

## Milestones and acceptance gates

### 0. Export feasibility — no remoting infrastructure

Create a small script that serializes a CUDA-targeted matmul and a differentiated MLP update on a CPU-only Mac, then deserializes and runs them on a CUDA host. Include random-number generation and `lax.scan` in separate cases. Confirm GPU placement and compare with native JAX on that same GPU using identical inputs and precision settings.

Also run CPU-targeted exports locally to make serialization checks repeatable without GPU access. Record exact versions, artifacts, commands and numerical tolerances. This phase passes only after a real CUDA run; CPU success alone leaves cross-platform feasibility unverified. Resolve unsupported lowerings before building a network service.

### 1. Usable remote training — explicit API

Implement the minimal protocol and API above. Train a small pure-JAX MLP with SGD for 100 deterministic steps. Compare loss, gradients and final parameters to native execution on the same GPU. Instrument uploaded/downloaded bytes and compilation counts: parameters and optimizer state must not travel back each step, and a repeated signature must reuse its executable.

Exercise malformed frames, wrong signatures, cross-session/stale handles, compilation failures, execution failures, release ordering and server termination. Verify repeated steps do not grow retained buffer count without bound. Run existing Torch tests if any shared code changes; otherwise keep its implementation untouched.

### 2. Measure, then reduce waits

Add a small transformer training example after the MLP passes. `../yangpt` supplies a workload shape, but its PyTorch implementation needs a JAX equivalent; this extension cannot run PyTorch source as JAX.

First export a `lax.scan` block of K training updates and return aggregate metrics once per block. This uses JAX's existing compiler and reduces RPCs without a new scheduler. Measure the increased batch-buffer memory and latency tradeoff.

Only if measured workloads need independently queued calls, add asynchronous execution: pending output handles, ordered dependencies, bounded backpressure and completion/error reporting at explicit waits. Define upload snapshot ownership and retain inputs until completion. JAX itself dispatches asynchronously, so server dispatch return must never be confused with device completion. [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)

### 3. Transparent device — separately scoped PJRT milestone

Proceed when normal `jax.Array` placement and integration with existing JAX programs justify a native plugin. PJRT is the appropriate device integration boundary. Implement a proxy client with device discovery, compilation, remote buffers, execution, transfers, completion events and deletion; use the PJRT C API and its tests. Package through JAX plugin discovery with an explicitly tested jaxlib/API version pair. [OpenXLA PJRT integration](https://openxla.org/xla/pjrt/pjrt_integration)

Before committing to this phase, prove that JAX can lower representative computations for the registered remote platform using suitable CUDA lowerings without a client GPU. Device discovery alone is insufficient. Establish how the server accepts PJRT compilation programs: a PJRT compile request is not automatically a serialized `jax.export.Exported` object. Reuse buffer/executable ownership and transport where practical, but budget for a different compilation entry point.

Acceptance: real remote-backed JAX arrays, `device_put`, jitted arithmetic and gradient updates, correct host reads/readiness, deletion and failure behavior, all from stock supported Mac JAX plus the plugin. No silent local execution. Avoid promises about broad Flax/Optax compatibility until representative programs pass.

Multi-GPU support follows this work: first multiple GPUs within one server and supported JAX sharding/collectives, then multiple GPU hosts if needed. It requires explicit topology, placement and collective semantics; exposing extra device names alone does not split training.

## Benchmark plan

There are no measured JAX before/after results for this proposal. Collect these configurations separately:

| Configuration | What it answers |
| --- | --- |
| Native JAX on the selected server GPU | Compute and framework baseline |
| Same exported function executed locally on that server | Export-path overhead |
| rGPU JAX over loopback, same GPU | Service/protocol overhead |
| Mac-to-server RPC, one update per call | Network and transfer cost |
| Same RPC path with K updates per exported scan | Measured benefit of fewer calls |

Measure matmul, MLP training and transformer training. Separate trace/export, upload, compile, warm execution and download. Warm up and synchronize device completion at measurement boundaries; do not time only asynchronous submission. Report median/p95 latency, steps or tokens per second, bytes transferred, host waits, compile count and peak GPU memory, with repeated trials and sample counts.

Hold GPU, JAX versions, shapes, precision, seeds and metric frequency constant. Record RTT and bandwidth. The proposed scan optimization's “before” is the implemented one-update RPC path; its “after” is the same implementation executing K updates per call. Native GPU and loopback are baselines, not claimed patch speedups.

For a serialized step, use compute + serialization + transfer + required network waits as a diagnostic cost model, not a speedup prediction. Set a throughput target only after measuring this baseline. JAX compiler fusion and remote GPUs may help, but neither removes WAN latency.

## Immediate next task

Implement milestone 0 only, producing a reproducible export script and CPU/CUDA parity evidence. Its results determine the tested version pair and supported lowering subset for milestone 1. No PJRT scaffolding, distributed runtime, persistent compilation cache or automatic replay is needed to answer that first question.
