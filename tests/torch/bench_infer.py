#!/usr/bin/env python3
"""Time ResNet-18 inference, natively or over the shim.

    python3 bench_infer.py ITERS BATCH

Prints one line: milliseconds per iteration. Run it with and without the
shim preloaded to get the cost of remoting, and with RGPU_STATS=1 to get
the round trip count that predicts what a real network would add.
"""

import sys
import time

import torch
import torchvision.models as models

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 50
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1

torch.manual_seed(0)
net = models.resnet18(weights=None).eval().cuda()
x = torch.randn(batch, 3, 224, 224).cuda()

with torch.no_grad():
    # Warm up: the first pass loads modules and picks algorithms, which over
    # the shim also ships several megabytes of fatbin.
    for _ in range(5):
        net(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        net(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

print("%8.2f ms/iter   (%d iters, batch %d)" % (elapsed / iters * 1000, iters,
                                                batch))
