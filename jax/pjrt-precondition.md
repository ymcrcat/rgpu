# PJRT precondition: can JAX lower for a custom platform without a client GPU?

`docs/jax-extension-plan.md` gates its PJRT milestone on this:

> Before committing to this phase, prove that JAX can lower representative
> computations for the registered remote platform using suitable CUDA lowerings
> without a client GPU. Device discovery alone is insufficient.

Probed on an M3 Mac with no GPU, jax/jaxlib 0.11.1. **Answer: yes, but only
after adopting CUDA's lowering rules — it does not happen by itself, and the
failure is not the one you would expect.**

## What happens with a bare custom platform name

Exporting for an unregistered platform name does **not** raise. It silently
lowers down a generic path:

| case | cpu | cuda | rocm | `rgpu` | `totally_made_up` |
|---|---|---|---|---|---|
| matmul | 1172 | 1172 | 1172 | 1172 | — |
| prng | 4516 | 5264 | 5264 | **5264** | 5272 |

`rgpu` matches `cuda`, not `cpu`, and its MLIR is structurally identical to
CUDA's once the embedded platform name is normalised. (`totally_made_up` is 8
bytes larger only because the name string is longer.) So for most primitives a
custom platform already gets CUDA-equivalent lowering.

**This is the trap.** A quick check using a matmul would conclude the
precondition is satisfied and move on. A matmul lowers identically on every
platform, so it proves nothing.

## Where it actually breaks

Primitives that register *platform-specific* lowering rules fail outright:

```
NotImplementedError: MLIR translation rule for primitive 'eigh'
                     not found for platform rgpu
```

`eigh` and `svd` fail for `rgpu` while succeeding for `cuda` and `cpu`. In
jax 0.11.1 there are **41 primitives with a CUDA-specific rule**, of which
**25 are CUDA-only** (no CPU rule at all):

    all_gather*, cholesky, cholesky_update, conv_general_dilated,
    cudnn_fusion, cumsum, debug_callback, debug_print, device_put,
    dot_product_attention_{fwd,bwd,fp8_*}, eig, eigh, empty, fft, geqp3,
    geqrf, householder_product, lu, lu_pivots_to_permutation, ormqr,
    pbroadcast, precv, psend, ragged_dot_general, scaled_matmul,
    scatter-{add,sub}, select_and_{gather_add,scatter_add}, svd,
    symmetric_product, threefry2x32, tridiagonal, tridiagonal_solve

That list is the real answer: convolution, attention, FFT, the linear-algebra
family and the collectives. Precisely what a GPU backend exists to run.

## The fix, and that it works

Adopt CUDA's rules for our platform:

```python
from jax._src.interpreters import mlir
reg = mlir._platform_specific_lowerings
reg["rgpu"].update(reg["cuda"])          # 41 rules
```

After this, `eigh` lowers for `rgpu`, and the emitted MLIR is **identical** to
CUDA's — the only diff across the whole module is a source-location line
number from the two probe scripts.

This is the right adoption rather than a hack: the server *is* a CUDA machine,
so the CUDA custom calls these rules emit are exactly what it can execute.

Two caveats:

- **It uses a private API.** The public `mlir.register_lowering` asserts the
  rule is not already a `LoweringRuleEntry`, and the registry stores wrapped
  entries, so the supported call cannot re-register them. Copying the dict
  works but is unsupported and can break on a jax upgrade. This argues for
  pinning a tested jax/jaxlib pair, which the plan already requires.
- **The rule set is version-specific.** 41 in 0.11.1; a different release will
  differ. Whatever adopts them should copy whatever is there rather than name
  primitives.

## A separate find: jaxlib already ships an IFRT proxy client

`jaxlib/_ifrt_proxy.so` is present, exposing:

```python
jaxlib._ifrt_proxy.get_client(proxy_server_address: str,
                              options: ClientConnectionOptions) -> Client
```

IFRT proxy is OpenXLA's own client/server for proxying execution over a
network, and `jax/extend/backend.py` references it. If the matching **server**
can be obtained or built for the GPU host, transparent remote `jax.Array`
might not need a hand-written PJRT plugin at all — which is the expensive part
of the plan's milestone 3.

Not yet investigated: whether a proxy server binary is distributed, what
transport it speaks, and how its version couples to jaxlib. **This should be
checked before any PJRT plugin is written**, because it could remove most of
that work.

## Status

| question | answer |
|---|---|
| Does a custom platform lower at all without a GPU? | yes, generically |
| Does it match CUDA for ordinary ops? | yes, structurally identical |
| Does it cover conv, attention, FFT, linalg, collectives? | **no, 41 rules missing** |
| Can those be adopted from CUDA? | **yes, and the result is identical MLIR** |
| Is the mechanism supported API? | no, private registry |

The precondition is met. The cost is a pinned jax version and a private-API
dependency, both of which the plan already anticipates.
