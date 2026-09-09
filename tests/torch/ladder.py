#!/usr/bin/env python3
"""Work up from "is there a GPU" to a real model, reporting where it breaks.

Each rung runs even if an earlier one failed, so a single run shows the whole
picture rather than just the first wall. Every rung compares against a CPU
reference: a remoted result that differs is worse than one that errors.

    RGPU_SERVER=host:9713 LD_LIBRARY_PATH=/opt/rgpu/lib python3 ladder.py
"""

import os
import sys
import time
import traceback

# cuBLAS and cuBLASLt are forwarded to the GPU host, so addmm can take its
# default path. Set RGPU_NO_CUBLASLT=1 to route it through plain cuBLAS
# instead, which is useful for telling the two paths apart when something
# breaks.
if os.environ.get("RGPU_NO_CUBLASLT"):
    os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

results = []


def rung(name):
    """Run one step, record the outcome, never stop the ladder."""
    def wrap(fn):
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
        return fn
    return wrap


import torch  # noqa: E402  (imported after the helpers so a failure is visible)

# cuDNN is not forwarded yet. PyTorch's own convolution kernels work over the
# shim, and they are what this falls back to.
torch.backends.cudnn.enabled = False

TOL = dict(rtol=1e-4, atol=1e-4)


@rung("torch imports and reports a CUDA build")
def _():
    assert torch.version.cuda, "this is a CPU-only build of torch"
    return "torch %s, built against CUDA %s" % (torch.__version__,
                                                torch.version.cuda)


@rung("cuda is available")
def _():
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    return "%d device(s)" % torch.cuda.device_count()


@rung("device properties")
def _():
    p = torch.cuda.get_device_properties(0)
    return "%s, sm_%d%d, %.1f GiB" % (p.name, p.major, p.minor,
                                      p.total_memory / (1 << 30))


@rung("allocate and read back a tensor")
def _():
    want = torch.arange(1024, dtype=torch.float32)
    got = want.cuda().cpu()
    assert torch.equal(got, want), "tensor changed in transit"
    return "1024 floats round-tripped exactly"


@rung("elementwise add")
def _():
    a, b = torch.randn(4096), torch.randn(4096)
    got = (a.cuda() + b.cuda()).cpu()
    assert torch.allclose(got, a + b, **TOL), "add result differs from CPU"
    return "4096 elements match CPU"


@rung("reduction")
def _():
    a = torch.randn(8192)
    got = a.cuda().sum().cpu()
    assert torch.allclose(got, a.sum(), rtol=1e-3, atol=1e-3), "sum differs"
    return "sum matches CPU"


@rung("matmul via forwarded cuBLAS")
def _():
    a, b = torch.randn(256, 512), torch.randn(512, 256)
    got = (a.cuda() @ b.cuda()).cpu()
    assert torch.allclose(got, a @ b, rtol=1e-3, atol=1e-3), "matmul differs"
    return "256x512 @ 512x256 matches CPU"


@rung("convolution, PyTorch's own kernels")
def _():
    import torch.nn as nn
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    want = conv(x)
    got = conv.cuda()(x.cuda()).cpu()
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3), "conv differs"
    return "2x3x32x32 conv matches CPU"


@rung("ResNet-18 inference")
def _():
    import torchvision.models as models
    torch.manual_seed(0)
    net = models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        want = net(x)
        got = net.cuda()(x.cuda()).cpu()
    diff = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=1e-2, atol=1e-2), \
        "logits differ by up to %g" % diff
    return "logits match CPU, max difference %.2e" % diff


def main():
    width = max(len(n) for _, n, _, _ in results)
    print()
    failed = 0
    for status, name, detail, secs in results:
        if status == "FAIL":
            failed += 1
        print("%-4s %-*s  %6.2fs  %s" % (status, width, name, secs, detail))
    print()
    if failed:
        print("%d of %d rungs failed" % (failed, len(results)))
        print("Set RGPU_TRACE=1 for tracebacks; the shim logs any entry point "
              "it was asked for and does not have.")
        return 1
    print("all %d rungs passed" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
