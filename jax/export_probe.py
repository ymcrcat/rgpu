"""Milestone 0 of docs/jax-extension-plan.md: is a CPU-only Mac able to export
JAX functions that a remote CUDA host can run, and do the answers match?

No remoting, no protocol, no server. One script, run on both machines:

    # on the Mac, with no GPU
    python export_probe.py export --platform cuda --out artifacts
    # copy artifacts/ to the GPU host, then there
    python export_probe.py run --dir artifacts

`run` deserializes each exported function, executes it, and compares against
native JAX tracing the same Python source on that same machine. Matching only
itself would prove nothing, so the reference is computed rather than shipped.

Everything also works with --platform cpu, which is the repeatable check that
needs no GPU at all:

    python export_probe.py export --platform cpu --out /tmp/cpu && \
        python export_probe.py run --dir /tmp/cpu

Four cases, because the plan asks for them separately: a matmul, a
differentiated MLP update, random number generation, and lax.scan.

The PRNG case takes a uint32 seed and builds its key inside the exported
function. Typed PRNG keys crossing the boundary as *inputs* are a separate
question the plan says to test on its own, so this does not claim them.

Needs `flatbuffers` installed: jax.export serialization imports it lazily and
fails at serialize() without it.
"""

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import export, lax

# Same inputs on both machines, built from a fixed seed rather than shipped as
# floats, so an artifact directory stays small and readable.
SEED = 20260915


def _rng(offset=0):
    return np.random.default_rng(SEED + offset)


# --- the cases -----------------------------------------------------------------
#
# Each is (function, inputs). The function is exported on one machine and traced
# natively on the other; the inputs are rebuilt identically on both.


def case_matmul():
    def f(a, b):
        return a @ b
    r = _rng(1)
    a = r.standard_normal((128, 128), dtype=np.float32)
    b = r.standard_normal((128, 128), dtype=np.float32)
    return f, (a, b)


def case_mlp_grad():
    """A differentiated MLP update: the shape of a training step."""
    def f(w1, b1, w2, b2, x, y):
        def loss_of(w1, b1, w2, b2):
            h = jnp.tanh(x @ w1 + b1)
            pred = h @ w2 + b2
            return jnp.mean((pred - y) ** 2)
        loss, grads = jax.value_and_grad(loss_of, argnums=(0, 1, 2, 3))(w1, b1, w2, b2)
        lr = np.float32(0.1)
        updated = tuple(p - lr * g for p, g in zip((w1, b1, w2, b2), grads))
        return (loss, *updated)

    r = _rng(2)
    w1 = r.standard_normal((16, 32), dtype=np.float32) * np.float32(0.1)
    b1 = np.zeros((32,), np.float32)
    w2 = r.standard_normal((32, 4), dtype=np.float32) * np.float32(0.1)
    b2 = np.zeros((4,), np.float32)
    x = r.standard_normal((8, 16), dtype=np.float32)
    y = r.standard_normal((8, 4), dtype=np.float32)
    return f, (w1, b1, w2, b2, x, y)


def case_prng():
    """Random numbers generated inside the exported function.

    The key is built from a uint32 seed here rather than passed in, so no typed
    PRNG key crosses the boundary. Whether a threefry key survives export as an
    input is a separate question, and this case does not answer it.
    """
    def f(seed, x):
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, x.shape, dtype=x.dtype)
        return x + noise, jnp.sum(noise)

    r = _rng(3)
    seed = np.uint32(12345)
    x = r.standard_normal((64,), dtype=np.float32)
    return f, (seed, x)


def case_scan():
    """lax.scan, the loop the plan wants to ship K updates inside later."""
    def f(carry, ws):
        def step(c, w):
            c = jnp.tanh(c @ w)
            return c, jnp.sum(c)
        final, ys = lax.scan(step, carry, ws)
        return final, ys

    r = _rng(4)
    carry = r.standard_normal((8, 8), dtype=np.float32)
    ws = r.standard_normal((5, 8, 8), dtype=np.float32) * np.float32(0.5)
    return carry, ws, f


def _scan_case():
    carry, ws, f = case_scan()
    return f, (carry, ws)


CASES = {
    "matmul": case_matmul,
    "mlp_grad": case_mlp_grad,
    "prng": case_prng,
    "scan": _scan_case,
}

# Exported and native run the same StableHLO on the same device, so they should
# agree very closely. This is not a cross-device tolerance.
RTOL, ATOL = 1e-6, 1e-6


def _specs(inputs):
    return [jax.ShapeDtypeStruct(np.shape(v), np.asarray(v).dtype) for v in inputs]


def _flat(out):
    return [np.asarray(v) for v in jax.tree.leaves(out)]


def do_export(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "jax": jax.__version__,
        "platform": args.platform,
        "exported_on": [d.platform for d in jax.devices()],
        "seed": SEED,
        "cases": {},
    }
    for name, build in CASES.items():
        fn, inputs = build()
        exp = export.export(jax.jit(fn), platforms=[args.platform])(*_specs(inputs))
        blob = exp.serialize()
        (out / f"{name}.exported").write_bytes(blob)
        manifest["cases"][name] = {
            "bytes": len(blob),
            "platforms": list(exp.platforms),
            "in_avals": [str(a) for a in exp.in_avals],
            "out_avals": [str(a) for a in exp.out_avals],
        }
        print(f"  {name:10s} {len(blob):7d} bytes  -> {exp.platforms}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nexported {len(CASES)} cases for '{args.platform}' into {out}/")
    print(f"jax {jax.__version__} on {manifest['exported_on']}")
    return 0


def do_run(args):
    d = pathlib.Path(args.dir)
    manifest = json.loads((d / "manifest.json").read_text())
    here = [dev.platform for dev in jax.devices()]
    print(f"artifacts exported by jax {manifest['jax']} on {manifest['exported_on']} "
          f"for '{manifest['platform']}'")
    print(f"running on jax {jax.__version__}, devices {jax.devices()}\n")
    # jax reports a CUDA device's platform as "gpu", while export takes "cuda".
    aliases = {"cuda": {"cuda", "gpu"}, "cpu": {"cpu"}, "tpu": {"tpu"}}
    runnable = aliases.get(manifest["platform"], {manifest["platform"]})
    if not runnable & set(here):
        print(f"REFUSING: these were exported for '{manifest['platform']}' but this "
              f"machine has {here}. Run them where they were meant to run.")
        return 2

    failures = 0
    for name, build in CASES.items():
        fn, inputs = build()
        exp = export.deserialize(bytearray((d / f"{name}.exported").read_bytes()))
        got = _flat(exp.call(*inputs))
        # The reference: the same Python source, traced natively here. Shipping
        # expected values instead would only prove the artifact matches itself.
        want = _flat(jax.jit(fn)(*inputs))
        placed = {v.device.platform for v in jax.tree.leaves(exp.call(*inputs))
                  if hasattr(v, "device")}
        worst, bad = 0.0, False
        for g, w in zip(got, want):
            if g.shape != w.shape or g.dtype != w.dtype:
                bad = True
                break
            diff = float(np.max(np.abs(g.astype(np.float64) - w.astype(np.float64)))) \
                if g.size else 0.0
            worst = max(worst, diff)
            if not np.allclose(g, w, rtol=RTOL, atol=ATOL):
                bad = True
        status = "FAIL" if bad else "ok"
        if bad:
            failures += 1
        print(f"  {name:10s} {status:4s}  max|diff| {worst:.3e}  "
              f"outputs {len(got)}  ran on {sorted(placed) or here}")
    print()
    if failures:
        print(f"FAILED: {failures} of {len(CASES)} cases disagree with native JAX")
        return 1
    print(f"PASS: all {len(CASES)} cases match native JAX within "
          f"rtol={RTOL:g} atol={ATOL:g}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="export_probe", description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="serialize the cases for a platform")
    e.add_argument("--platform", default="cuda", choices=["cuda", "cpu", "tpu"])
    e.add_argument("--out", required=True)
    e.set_defaults(func=do_export)
    r = sub.add_parser("run", help="deserialize, run, and compare against native JAX")
    r.add_argument("--dir", required=True)
    r.set_defaults(func=do_run)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
