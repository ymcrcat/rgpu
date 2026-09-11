import os

import pytest
import torch
import torch.nn as nn

import rgpu
from rgpu import session
from conftest import start_server

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("RGPU_REAL_SERVER")),
    reason="needs a local server with a drop hook")


def train(steps=8):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4)).to("rgpu")
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x, y = torch.randn(16, 32).to("rgpu"), torch.randint(0, 4, (16,)).to("rgpu")
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@pytest.fixture
def use_server(monkeypatch):
    started = []

    def use(**env):
        proc, address = start_server(env=env)
        started.append(proc)
        monkeypatch.setenv("RGPU_OPSERVER", address)
        session.reset()
        return proc, address

    yield use
    session.reset()
    for p in started:
        p.kill()


def test_training_through_a_drop_gives_identical_losses(use_server):
    use_server()
    clean = train()
    use_server(RGPU_DROP_AFTER="150", RGPU_SESSION_GRACE="30")
    assert train() == clean


def test_a_restarted_server_means_the_session_is_lost(use_server):
    proc, address = use_server()
    t = torch.ones(3, device="rgpu")
    t.cpu()
    proc.kill()
    proc.wait()
    port = int(address.rsplit(":", 1)[1])
    new_proc, _ = start_server(port=port)   # a new server knows nothing of us
    try:
        with pytest.raises(rgpu.SessionLost):
            (t + 1).cpu()
    finally:
        new_proc.kill()
