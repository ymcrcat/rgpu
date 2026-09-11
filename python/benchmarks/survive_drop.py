"""Per-step losses with a fixed seed, to run while the link is broken and restored.

A run that survived the break prints exactly the same losses as one that
never saw it; a reconnect that had quietly started a new session could not.
"""

import sys
import time

import torch
import torch.nn as nn

import rgpu  # noqa: F401 - registers the device

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10)).to("rgpu")
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
x, y = torch.randn(128, 256).to("rgpu"), torch.randint(0, 10, (128,)).to("rgpu")
start = time.time()
for step in range(steps):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    print(f"step {step:2d}  t={time.time() - start:5.1f}s  loss {loss.item():.6f}", flush=True)
