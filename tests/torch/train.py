#!/usr/bin/env python3
"""Train over the shim, comparing every step against the same work on CPU.

Inference only has to produce the right numbers once. Training has to produce
the right gradients, apply them, and still be right several steps later, so
each rung here checks a gradient rather than just a forward result: a backward
pass that silently returns zeros would pass a loss check and fail this.

Each rung runs even if an earlier one failed, so a single run names everything
that is missing rather than the first thing.

    LD_PRELOAD=... RGPU_SERVER=host:9713 python3 train.py
"""

import os
import sys
import time
import traceback

import torch
import torch.nn as nn

results = []


def rung(name):
    def wrap(fn):
        start = time.time()
        try:
            detail = fn()
            results.append(("pass", name, detail, time.time() - start))
        except Exception as e:  # noqa: BLE001 - reporting every failure is the point
            short = "%s: %s" % (type(e).__name__, e)
            results.append(("FAIL", name, short.replace("\n", " ")[:250],
                            time.time() - start))
            if os.environ.get("RGPU_TRACE"):
                traceback.print_exc()
        return fn
    return wrap


def grads_match(cpu_model, gpu_model, rtol=1e-3, atol=1e-4):
    """Every parameter's gradient, compared against the CPU's."""
    worst = 0.0
    for (n, a), (_, b) in zip(cpu_model.named_parameters(),
                              gpu_model.named_parameters()):
        if a.grad is None or b.grad is None:
            raise AssertionError("%s has no gradient" % n)
        g = b.grad.cpu()
        if g.abs().sum() == 0 and a.grad.abs().sum() != 0:
            raise AssertionError("%s came back all zeros" % n)
        diff = (g - a.grad).abs().max().item()
        worst = max(worst, diff)
        if not torch.allclose(g, a.grad, rtol=rtol, atol=atol):
            raise AssertionError("%s differs by up to %g" % (n, diff))
    return worst


def paired(make, seed=0):
    """The same model and input on CPU and on the GPU."""
    torch.manual_seed(seed)
    cpu = make()
    torch.manual_seed(seed)
    gpu = make().cuda()
    return cpu, gpu


@rung("backward through a linear layer")
def _():
    cpu, gpu = paired(lambda: nn.Linear(256, 128))
    torch.manual_seed(1)
    x = torch.randn(32, 256)
    cpu(x).sum().backward()
    gpu(x.cuda()).sum().backward()
    return "gradients match CPU, worst %.2e" % grads_match(cpu, gpu)


@rung("backward through a convolution")
def _():
    cpu, gpu = paired(lambda: nn.Conv2d(3, 16, 3, padding=1))
    torch.manual_seed(2)
    x = torch.randn(4, 3, 32, 32)
    cpu(x).pow(2).mean().backward()
    gpu(x.cuda()).pow(2).mean().backward()
    return "gradients match CPU, worst %.2e" % grads_match(cpu, gpu)


@rung("backward through batch norm in training mode")
def _():
    # Batch norm in training mode is the one cuDNN path inference never takes:
    # it computes batch statistics and keeps a reserve space for the backward
    # pass. Different entry points from the inference form entirely.
    cpu, gpu = paired(lambda: nn.BatchNorm2d(16))
    cpu.train()
    gpu.train()
    torch.manual_seed(3)
    x = torch.randn(8, 16, 16, 16)
    cpu(x).pow(2).mean().backward()
    gpu(x.cuda()).pow(2).mean().backward()
    worst = grads_match(cpu, gpu, rtol=1e-3, atol=1e-4)
    rm = (gpu.running_mean.cpu() - cpu.running_mean).abs().max().item()
    if rm > 1e-4:
        raise AssertionError("running mean differs by %g" % rm)
    return "gradients and running statistics match, worst %.2e" % worst


@rung("one optimizer step")
def _():
    cpu, gpu = paired(lambda: nn.Linear(64, 64))
    opt_cpu = torch.optim.SGD(cpu.parameters(), lr=0.1, momentum=0.9)
    opt_gpu = torch.optim.SGD(gpu.parameters(), lr=0.1, momentum=0.9)
    torch.manual_seed(4)
    x = torch.randn(16, 64)
    for _ in range(3):
        opt_cpu.zero_grad()
        cpu(x).sum().backward()
        opt_cpu.step()
        opt_gpu.zero_grad()
        gpu(x.cuda()).sum().backward()
        opt_gpu.step()
    worst = max((b.cpu() - a).abs().max().item()
                for a, b in zip(cpu.parameters(), gpu.parameters()))
    if worst > 1e-3:
        raise AssertionError("weights diverged by %g after three steps" % worst)
    return "weights match CPU after three steps, worst %.2e" % worst


@rung("Adam, which reads and writes state on the device")
def _():
    cpu, gpu = paired(lambda: nn.Linear(64, 64))
    opt_cpu = torch.optim.Adam(cpu.parameters(), lr=1e-2)
    opt_gpu = torch.optim.Adam(gpu.parameters(), lr=1e-2)
    torch.manual_seed(5)
    x = torch.randn(16, 64)
    for _ in range(3):
        opt_cpu.zero_grad()
        cpu(x).sum().backward()
        opt_cpu.step()
        opt_gpu.zero_grad()
        gpu(x.cuda()).sum().backward()
        opt_gpu.step()
    worst = max((b.cpu() - a).abs().max().item()
                for a, b in zip(cpu.parameters(), gpu.parameters()))
    if worst > 1e-3:
        raise AssertionError("weights diverged by %g after three steps" % worst)
    return "weights match CPU after three steps, worst %.2e" % worst


@rung("ResNet-18 training step")
def _():
    import torchvision.models as models
    cpu, gpu = paired(lambda: models.resnet18(weights=None))
    cpu.train()
    gpu.train()
    torch.manual_seed(6)
    x = torch.randn(4, 3, 64, 64)
    y = torch.randint(0, 1000, (4,))
    loss_cpu = nn.functional.cross_entropy(cpu(x), y)
    loss_gpu = nn.functional.cross_entropy(gpu(x.cuda()), y.cuda())
    loss_cpu.backward()
    loss_gpu.backward()
    dl = abs(loss_gpu.item() - loss_cpu.item())
    if dl > 1e-2:
        raise AssertionError("loss differs by %g" % dl)
    # Gradients through eighteen layers drift more than a single layer does,
    # so compare their size rather than demanding element equality.
    for (n, a), (_, b) in zip(cpu.named_parameters(), gpu.named_parameters()):
        na, nb = a.grad.norm().item(), b.grad.norm().cpu().item()
        if nb == 0 and na != 0:
            raise AssertionError("%s gradient came back all zeros" % n)
        if abs(nb - na) > 0.05 * max(na, 1e-6):
            raise AssertionError("%s gradient norm %g against %g" % (n, nb, na))
    return "loss matches to %.2e, every gradient norm within 5%%" % dl


@rung("loss falls over ten steps")
def _():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
    model = model.cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(64, 128).cuda()
    y = torch.randint(0, 10, (64,)).cuda()
    first = last = None
    for step in range(10):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    if not last < first:
        raise AssertionError("loss went from %g to %g" % (first, last))
    return "loss %.3f to %.3f" % (first, last)


@rung("mixed precision with a gradient scaler")
def _():
    torch.manual_seed(8)
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
                          nn.Linear(256, 10)).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(64, 128).cuda()
    y = torch.randint(0, 10, (64,)).cuda()
    first = last = None
    for step in range(5):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step == 0:
            first = loss.item()
        last = loss.item()
    if not last < first:
        raise AssertionError("loss went from %g to %g" % (first, last))
    return "half precision loss %.3f to %.3f" % (first, last)


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
        print("The shim logs any entry point it was asked for and does not "
              "have; RGPU_TRACE=1 adds tracebacks.")
        return 1
    print("all %d rungs passed" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
