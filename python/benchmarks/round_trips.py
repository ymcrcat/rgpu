"""Round trips and milliseconds per step, eager and compiled, on ResNet-18.

    RGPU_OPSERVER=127.0.0.1:9720 python benchmarks/round_trips.py [steps]
"""

import sys
import time

import torch
import torch.nn as nn
import torchvision.models as models

import rgpu

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def measure(label, step):
    step()
    step()   # warm up, and compile if compiling
    torch.rgpu.synchronize()
    waits, start = rgpu.stats()["waits"], time.time()
    for _ in range(steps):
        step()
    torch.rgpu.synchronize()
    ms = (time.time() - start) / steps * 1000
    per = (rgpu.stats()["waits"] - waits - 1) / steps   # minus the final synchronize
    print(f"{label:28} {ms:9.1f} ms/step {per:6.2f} round trips/step")


torch.manual_seed(0)
net = models.resnet18(weights=None).to("rgpu")
x = torch.randn(1, 3, 224, 224).to("rgpu")
y = torch.randint(0, 1000, (1,)).to("rgpu")
opt = torch.optim.SGD(net.parameters(), lr=0.01)


def infer(model):
    def step():
        with torch.no_grad():
            model.eval()
            model(x).cpu()
    return step


def train(model):
    def step():
        model.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        loss.item()
    return step


compiled = torch.compile(net, backend=rgpu.compile_backend(), dynamic=False)
measure("eager inference", infer(net))
measure("eager training step", train(net))
measure("compiled training step", train(compiled))
