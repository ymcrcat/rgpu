# Documentation map

The maintained product documentation lives in [`website/content/docs`](../website/content/docs)
and is built with Fumadocs. Update it whenever installation, configuration,
supported behavior, or operating procedures change.

This directory holds engineering records that are useful beside the product
documentation:

| Document | Role | Status |
| --- | --- | --- |
| [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) | Historical implementation notes and plans | Begun 2026-09-11; not current product status |
| [`performance-notes.md`](performance-notes.md) | Measurements and optimization analysis | Update when benchmarks change |
| [`jax-findings.md`](jax-findings.md) | Results of the JAX experiments | Current experiment outcome |
| [`jax-extension-plan.md`](jax-extension-plan.md) | Original JAX proposal | Partly superseded by `jax-findings.md` |
| [`superpowers/specs/`](superpowers/specs/) | Design decisions behind implemented work | Historical design records |
| [`superpowers/plans/`](superpowers/plans/) | Detailed implementation plans | Historical execution records |

JAX remains experimental and intentionally stays outside the product docs. Its
code and entry points are indexed in [`jax/README.md`](../jax/README.md).
