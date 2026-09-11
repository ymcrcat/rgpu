import os
import socket
import threading
import time

import pytest
import torch
import torch.nn as nn

import rgpu
from rgpu import session, wire
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


def test_a_peer_that_keeps_dying_after_the_handshake_gives_up_in_time(monkeypatch):
    """A fake peer that resumes the handshake and then dies again, every time,
    must not let the client spin forever: the recovery budget has to be shared
    across reconnects, not renewed on each one. Without that fix the client
    never gives up and no error ever reaches the caller (the reviewer's probe
    ran for 10 s and saw 98,649 reconnect attempts with nothing surfacing);
    with it, the operation raises ConnectionError once RGPU_RECONNECT_SECONDS
    has actually elapsed."""
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    accepts = [0]
    stop = [False]

    def serve():
        while not stop[0]:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            accepts[0] += 1
            try:
                wire.recv_exact(conn, len(wire.MAGIC))
                wire.recv_frame(conn)
                wire.send_frame(conn, wire.encode([wire.VERSION, True, 0, torch.__version__]))
                wire.recv_frame(conn)   # swallow whatever it sends, then die without a reply
            except OSError:
                pass
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    monkeypatch.setenv("RGPU_OPSERVER", f"127.0.0.1:{port}")
    monkeypatch.setenv("RGPU_RECONNECT_SECONDS", "2")
    session.reset()

    outcome = []

    def drive():
        try:
            torch.ones(3, device="rgpu").cpu()
        except BaseException as e:   # noqa: BLE001 -- capturing whatever escapes, on purpose
            outcome.append(e)

    t = threading.Thread(target=drive, daemon=True)
    start = time.monotonic()
    t.start()
    t.join(10)
    elapsed = time.monotonic() - start

    stop[0] = True
    srv.close()
    session.reset()

    assert not t.is_alive(), "still retrying after 10s; the recovery deadline was not shared"
    assert outcome and isinstance(outcome[0], ConnectionError), outcome
    assert elapsed < 6   # gave up close to RGPU_RECONNECT_SECONDS=2, not after 10s of spinning
    assert accepts[0] > 0   # retries did happen; it isn't failing for some unrelated reason
