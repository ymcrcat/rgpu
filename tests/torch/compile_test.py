#!/usr/bin/env python3
"""Find out what torch.compile needs from the shim that it does not have.

Inductor's output reaches the GPU through the same driver calls we already
forward: Triton emits PTX, ptxas turns it into a cubin on this machine, and
then it is cuModuleLoadData and cuLaunchKernel. So the question is not whether
the approach works, it is which entry points are missing. The shim names every
one it was asked for and does not have, so running this and reading the log is
the whole discovery step.

Each rung reports rather than stops, because a later one may need entry points
an earlier one never reached.

    LD_PRELOAD=... RGPU_SERVER=host:9713 python3 compile_test.py
"""

import os
import sys
import time
import traceback

import torch

results = []


def rung(name, fn):
    start = time.time()
    try:
        detail = fn()
        results.append(("pass", name, detail, time.time() - start))
    except Exception as e:  # noqa: BLE001 - reporting every failure is the point
        short = "%s: %s" % (type(e).__name__, e)
        results.append(("FAIL", name, short.replace("\n", " ")[:300],
                        time.time() - start))
        if os.environ.get("RGPU_TRACE"):
            traceback.print_exc()


def elementwise():
    def f(x, y):
        return (x * y + x).relu()

    a, b = torch.randn(4096, device="cuda"), torch.randn(4096, device="cuda")
    want = f(a, b)
    got = torch.compile(f)(a, b)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4), "result differs"
    return "4096 elements match eager"


def fused_reduction():
    # A shape Inductor will fuse into one Triton kernel with a reduction, which
    # exercises shared memory and so cuFuncSetAttribute.
    def f(x):
        return (x * 2).softmax(dim=-1).sum(dim=-1)

    x = torch.randn(64, 1024, device="cuda")
    want = f(x)
    got = torch.compile(f)(x)
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3), "result differs"
    return "64x1024 softmax-reduce matches eager"


def matmul_epilogue():
    def f(a, b, c):
        return torch.relu(a @ b + c)

    a = torch.randn(256, 512, device="cuda")
    b = torch.randn(512, 256, device="cuda")
    c = torch.randn(256, device="cuda")
    want = f(a, b, c)
    got = torch.compile(f)(a, b, c)
    assert torch.allclose(got, want, rtol=1e-2, atol=1e-2), "result differs"
    return "matmul with epilogue matches eager"


def resnet_compiled():
    import torchvision.models as models
    torch.manual_seed(0)
    net = models.resnet18(weights=None).eval().cuda()
    x = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.no_grad():
        want = net(x)
        got = torch.compile(net)(x)
    diff = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=1e-2, atol=1e-2), \
        "logits differ by up to %g" % diff
    return "logits match eager, max difference %.2e" % diff


def reduce_overhead():
    # This is the CUDA graphs path: capture once, replay. Expected to need
    # entry points nothing else does, which is exactly what we want to learn.
    def f(x):
        return (x * 3).sin().cos()

    x = torch.randn(1024, device="cuda")
    want = f(x)
    compiled = torch.compile(f, mode="reduce-overhead")
    got = None
    for _ in range(3):  # graphs are captured on a later call, not the first
        got = compiled(x)
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3), "result differs"
    return "captured and replayed, matches eager"


def main():
    print("torch %s, triton %s" % (
        torch.__version__,
        getattr(__import__("triton"), "__version__", "missing")))

    rung("compile an elementwise function", elementwise)
    rung("compile a fused reduction", fused_reduction)
    rung("compile a matmul with an epilogue", matmul_epilogue)
    rung("compile ResNet-18", resnet_compiled)
    rung("compile with reduce-overhead (CUDA graphs)", reduce_overhead)

    width = max(len(n) for _, n, _, _ in results)
    print()
    failed = 0
    for status, name, detail, secs in results:
        if status == "FAIL":
            failed += 1
        print("%-4s %-*s  %7.2fs  %s" % (status, width, name, secs, detail))
    print()
    print("%d of %d passed" % (len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
