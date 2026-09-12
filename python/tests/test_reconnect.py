import os
import socket
import threading
import time

import pytest
import torch
import torch.nn as nn

import rgpu
from rgpu import session, wire
from conftest import free_port, start_server

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
    must not let the client spin forever: the recovery budget *and* the
    backoff between attempts have to be shared across reconnects, not
    renewed on each one. Without that fix the client hammers the peer at
    full speed and no error ever reaches the caller (the reviewer's probe
    ran for 10 s and saw 98,649 reconnect attempts with nothing surfacing,
    a gap of 0.1-0.3 ms that never grew); with it, the backoff grows across
    calls the same way it does within one, so only a handful of attempts
    happen before the operation raises ConnectionError near
    RGPU_RECONNECT_SECONDS."""
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
    # With backoff growing 0.1s -> 0.2 -> 0.4 -> 0.8 -> 1.6, capped at 3s, a 2s
    # budget allows only a handful of attempts -- nowhere near the thousands
    # per second the bug produced.
    assert 0 < accepts[0] < 30, accepts[0]


def test_giving_up_clears_recovery_state_so_a_later_attempt_gets_a_fresh_budget(monkeypatch):
    """Once _reconnect gives up and raises ConnectionError, the connection
    must not stay wedged: leaving the deadline in the past (and the dead
    socket assigned) would make a later operation fail instantly with zero
    connect attempts, even once a real server is listening again on the
    same address. A peer that keeps completing the handshake and dying
    again (rather than one nothing ever answers) is what leaves self.sock
    non-None at the moment the give-up fires, so this reproduces the
    socket-left-assigned half of the bug too, not just the stale deadline:
    with self.sock still set, _flush's fast path ("if self.sock is None:
    connect") is skipped and the stale _reconnect state is what gets hit.

    had_session is reset between phases: whether a *new* server recognizes
    an *old* session is session identity, a separate concern already
    covered by test_a_restarted_server_means_the_session_is_lost. Resetting
    it isolates the thing this test is actually about -- does the same,
    previously-wedged Connection try again at all, with a fresh budget."""
    port = free_port()
    srv = socket.create_server(("127.0.0.1", port))

    def serve_flapping():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            try:
                wire.recv_exact(conn, len(wire.MAGIC))
                wire.recv_frame(conn)
                wire.send_frame(conn, wire.encode([wire.VERSION, True, 0, torch.__version__]))
                wire.recv_frame(conn)   # swallow whatever it sends, then die without a reply
            except OSError:
                pass
            conn.close()

    threading.Thread(target=serve_flapping, daemon=True).start()
    monkeypatch.setenv("RGPU_OPSERVER", f"127.0.0.1:{port}")
    monkeypatch.setenv("RGPU_RECONNECT_SECONDS", "1")
    session.reset()

    with pytest.raises(ConnectionError):
        torch.ones(3, device="rgpu").cpu()

    srv.close()   # stop the flapping peer and free the port
    session.get().had_session = False   # see docstring: isolate the wedge from session identity

    proc, _ = start_server(port=port)   # a real, working server on the same address
    try:
        result = torch.ones(3, device="rgpu").cpu()   # the same, previously-wedged Connection
        assert torch.equal(result, torch.ones(3))
    finally:
        proc.kill()
        session.reset()
