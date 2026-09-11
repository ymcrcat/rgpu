#!/usr/bin/env python3
"""Time ResNet-18 over whatever link the shim is using, with and without a graph.

Written for the real-network case, where a round trip costs tens of
milliseconds instead of tens of microseconds. The graph is captured with
torch.cuda.graph rather than torch.compile so this needs no Triton: the point
is the number of round trips, not who generated the kernels.
"""

import sys
import time

import torch
import torchvision.models as models

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3

torch.manual_seed(0)
net = models.resnet18(weights=None).eval().cuda()
x = torch.randn(1, 3, 224, 224, device="cuda")

with torch.no_grad():
    for _ in range(2):
        net(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        net(x)
    torch.cuda.synchronize()
    eager = (time.time() - start) / iters
    print("eager:  %8.1f ms/iter" % (eager * 1000))

    # Capture the same work as a graph, then replay it.
    static_in = x.clone()
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            net(static_in)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        static_out = net(static_in)
    torch.cuda.synchronize()

    graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    replay = (time.time() - start) / iters
    print("graph:  %8.1f ms/iter" % (replay * 1000))
    print("speedup: %.0fx" % (eager / replay if replay else 0))
