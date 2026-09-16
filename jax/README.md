# JAX experiments

This directory contains research prototypes, not a supported rGPU product
backend. Keep JAX material here and in `docs/`; it is intentionally absent from
the Fumadocs product site.

## Results

- [`../docs/jax-findings.md`](../docs/jax-findings.md) is the current outcome.
  A stock JAX client successfully ran remote CPU work through OpenXLA's IFRT
  proxy. Remote CUDA device discovery and `device_put` work, but GPU `jax.jit`
  still terminates the client session during compilation.
- [`../docs/jax-extension-plan.md`](../docs/jax-extension-plan.md) is the
  original proposal. The findings supersede its custom-PJRT direction.

## Contents

| Path | Purpose |
| --- | --- |
| [`export_probe.py`](export_probe.py) | Export CUDA-targeted JAX programs on a CPU-only client for execution on a GPU host |
| [`pjrt-precondition.md`](pjrt-precondition.md) | Tests of JAX lowering behavior for a custom platform |
| [`ifrt_server/`](ifrt_server/) | Minimal executable wrapper around OpenXLA's IFRT proxy server |

If the experiment resumes, the next useful step is diagnosing the GPU JIT
failure described in the findings, preferably on a host whose CUDA versions
match the pinned XLA revision.
