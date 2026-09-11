#!/usr/bin/env python3
"""Train through a dropped connection and check nothing was lost.

Meant to be run while something breaks the link partway through - killing an
ssh tunnel and starting it again is enough. The losses are printed per step
with a seed fixed, so a run that survived the break prints exactly the same
numbers as one that never saw it. A reconnect that quietly started a fresh
session could not do that: the weights would be gone.

    python3 survive_drop.py STEPS
"""

import sys
import time

import torch
import torch.nn as nn

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
x = torch.randn(128, 256).cuda()
y = torch.randint(0, 10, (128,)).cuda()

start = time.time()
for step in range(steps):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    print("step %2d  t=%5.1fs  loss %.6f" % (step, time.time() - start,
                                            loss.item()), flush=True)
