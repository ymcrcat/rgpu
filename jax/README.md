# rgpu for JAX

Work towards `docs/jax-extension-plan.md`. Nothing here is a JAX backend yet:
this is milestone 0, which asks one question and builds no remoting.

> Can a CPU-only Mac export JAX functions that a remote CUDA host runs, and do
> the answers match native JAX on that host?

The plan is explicit that this is answered before a network service exists, and
that CPU success alone does not answer it.

## Status

| | |
|---|---|
| CUDA-targeted export from a CPU-only Mac | **works** |
| Serialize / deserialize round trip | **works** |
| Platform safety check refuses a CPU run | **works** |
| All four cases run on a real CUDA host and match native JAX | **works** |

Milestone 0 passes. Cross-platform feasibility is established: JAX exported on
a machine with no GPU runs correctly on a remote one.

## Running it

Nothing to install but `jax` and `flatbuffers`; no torch, and no rgpu.
`jax.export` imports flatbuffers lazily and fails at `serialize()` without it.

The repeatable check, which needs no GPU at all:

```sh
python export_probe.py export --platform cpu --out /tmp/cpu
python export_probe.py run --dir /tmp/cpu
```

The real one:

```sh
# on the Mac
python export_probe.py export --platform cuda --out artifacts
scp -r artifacts export_probe.py gpuhost:
# on the GPU host
python export_probe.py run --dir artifacts
```

`run` compares each exported function against native JAX tracing the same
Python source on that same machine. The reference is computed there rather than
shipped, because an artifact that matched only itself would prove nothing.

## The four cases

Taken from the plan, which asks for random number generation and `lax.scan`
separately rather than folded into one example.

| case | what it covers |
|---|---|
| `matmul` | the simplest possible export |
| `mlp_grad` | a differentiated update: `value_and_grad` plus an SGD step, the shape of a training step |
| `prng` | random numbers generated inside the exported function |
| `scan` | `lax.scan`, the loop the plan later wants to ship K updates inside |

## What this does not claim

- **Typed PRNG keys as inputs.** The `prng` case takes a uint32 seed and builds
  its key inside the function, so no threefry key crosses the boundary. Whether
  one survives export as an input is a separate question the plan says to test
  on its own.
- **Anything about dtypes beyond float32 and uint32.** bfloat16 and typed keys
  are called out in the plan as needing their own tests before being claimed.
- **Any performance result.** There are no JAX benchmarks yet, here or in the
  plan.
- **Dynamic shapes.** Every case is exported against fixed input specifications.

## Observed

jax 0.11.1 / jaxlib 0.11.1, exported on macOS arm64 (CPU only).

| case | cpu bytes | cuda bytes |
|---|---|---|
| `matmul` | 1288 | 1292 |
| `mlp_grad` | 3988 | 3992 |
| `prng` | 4904 | 5652 |
| `scan` | 2888 | 2892 |

`prng` differing by ~750 bytes between platforms is the useful signal: the CUDA
export really is lowered for CUDA, rather than CPU code wearing a label.

All four CPU cases match native JAX exactly (`max|diff|` of 0.0).

### The CUDA run

Exported on an M3 Mac (CPU only), run on an NVIDIA A40 rented from RunPod,
jax/jaxlib 0.11.1 on both sides — the tested version pair the plan asks for.

```
artifacts exported by jax 0.11.1 on ['cpu'] for 'cuda'
running on jax 0.11.1, devices [CudaDevice(id=0)]

  matmul     ok    max|diff| 0.000e+00  outputs 1  ran on ['gpu']
  mlp_grad   ok    max|diff| 0.000e+00  outputs 5  ran on ['gpu']
  prng       ok    max|diff| 0.000e+00  outputs 2  ran on ['gpu']
  scan       ok    max|diff| 0.000e+00  outputs 2  ran on ['gpu']

PASS: all 4 cases match native JAX within rtol=1e-06 atol=1e-06
```

Every case was bit-identical to native JAX on that GPU, the differentiated
update and the PRNG included. Nothing needed a fallback, and no lowering was
refused.

One wrinkle worth knowing: `jax.export` takes the platform as `"cuda"`, but
`jax.devices()[0].platform` reports `"gpu"`. A check comparing them directly
refuses a run that should have gone ahead.
