import os
import socket
import threading

import pytest
import torch

from rgpu import wire
from rgpu.server.__main__ import Drop, Registry, serve_connection
from conftest import start_server


def handshake(address, sid, last_acked=0):
    host, port = address.rsplit(":", 1)
    s = socket.create_connection((host, int(port)))
    s.sendall(wire.MAGIC)
    wire.send_frame(s, wire.encode([wire.VERSION, sid, last_acked, torch.__version__]))
    return s, wire.decode(wire.recv_frame(s))


@pytest.fixture
def server():
    proc, address = start_server()
    yield address
    proc.kill()


def test_a_batch_runs_and_waited_messages_are_answered(server):
    s, (version, resumed, last_seq, server_torch) = handshake(server, os.urandom(16))
    assert version == wire.VERSION and resumed is False and last_seq == 0
    a = torch.arange(6.0)
    wire.send_frame(s, wire.encode([
        [1, wire.RUN, "aten::empty", "memory_format", [[6]],
         {"dtype": torch.float32, "device": wire.Dev("rgpu")}, [1]],
        [2, wire.UPLOAD, 1, wire.Host.of(a)],
        [3, wire.RUN, "aten::mul", "Scalar", [wire.Ref(1), 3.0], {}, [2]],
        [4, wire.DOWNLOAD, 2],
    ]))
    seq, status, host = wire.decode(wire.recv_frame(s))
    assert (seq, status) == (4, wire.OK)
    assert torch.equal(host.tensor(), a * 3)


def test_a_session_outlives_its_connection(server):
    sid = os.urandom(16)
    s, _ = handshake(server, sid)
    wire.send_frame(s, wire.encode([
        [1, wire.RUN, "aten::ones", "default", [[3]],
         {"dtype": torch.float32, "device": wire.Dev("rgpu")}, [1]],
        [2, wire.SYNC],
    ]))
    wire.recv_frame(s)
    s.close()
    s, (_, resumed, last_seq, _) = handshake(server, sid, last_acked=2)
    assert resumed is True and last_seq == 2
    wire.send_frame(s, wire.encode([[3, wire.DOWNLOAD, 1]]))
    _, status, host = wire.decode(wire.recv_frame(s))
    assert status == wire.OK and torch.equal(host.tensor(), torch.ones(3))


def test_a_lost_reply_is_sent_again_on_resume(server):
    sid = os.urandom(16)
    s, _ = handshake(server, sid)
    wire.send_frame(s, wire.encode([
        [1, wire.RUN, "aten::ones", "default", [[2]],
         {"dtype": torch.float32, "device": wire.Dev("rgpu")}, [1]],
        [2, wire.DOWNLOAD, 1],
    ]))
    wire.recv_frame(s)   # the reply arrives, but pretend it was lost
    s.close()
    s, (_, resumed, last_seq, _) = handshake(server, sid, last_acked=0)
    seq, status, host = wire.decode(wire.recv_frame(s))
    assert (seq, status) == (2, wire.OK) and torch.equal(host.tensor(), torch.ones(2))


def test_a_connection_without_the_magic_is_closed(server):
    host, port = server.rsplit(":", 1)
    s = socket.create_connection((host, int(port)))
    s.sendall(b"GET / HTTP/1.1\r\n\r\n")
    with pytest.raises(ConnectionError):
        wire.recv_frame(s)


def test_a_superseded_connection_neither_applies_nor_answers(server):
    """The reviewer's trace: the link drops mid-batch, so the old serving
    thread does not notice; the client reconnects and replays from where the
    server got to. If the old thread is still allowed to apply a waited
    message, it advances last_seq and writes the reply into the dead socket,
    and the replay on the new connection is then deduped away - the client
    waits for an answer nobody will ever send. The old connection must be
    fenced off instead: it applies nothing more and stops serving."""
    sid = os.urandom(16)
    old, (_, resumed, _, _) = handshake(server, sid)
    assert resumed is False
    new, (_, resumed, last_seq, _) = handshake(server, sid)
    assert resumed is True and last_seq == 0

    batch = wire.encode([
        [1, wire.RUN, "aten::ones", "default", [[3]],
         {"dtype": torch.float32, "device": wire.Dev("rgpu")}, [1]],
        [2, wire.DOWNLOAD, 1],
    ])
    old.settimeout(10)
    wire.send_frame(old, batch)
    with pytest.raises(ConnectionError):
        wire.recv_frame(old)   # superseded: nothing applied, nothing answered

    new.settimeout(10)
    wire.send_frame(new, batch)   # the same messages, replayed after reconnecting
    seq, status, host = wire.decode(wire.recv_frame(new))
    assert (seq, status) == (2, wire.OK)
    assert torch.equal(host.tensor(), torch.ones(3))


def test_an_accepted_connection_is_kept_alive_by_the_kernel():
    """Without SO_KEEPALIVE a thread stuck in recv_frame on a partitioned
    link never reaches its finally, so registry.detach never runs and the
    session's GPU memory is held for the life of the process."""
    listener = socket.create_server(("127.0.0.1", 0))
    client = socket.create_connection(listener.getsockname())
    accepted, _ = listener.accept()
    listener.close()
    thread = threading.Thread(target=serve_connection,
                              args=(accepted, Registry("cpu", 1.0), Drop(0)), daemon=True)
    thread.start()
    try:
        client.sendall(wire.MAGIC)
        wire.send_frame(client, wire.encode(
            [wire.VERSION, os.urandom(16), 0, torch.__version__]))
        wire.recv_frame(client)   # the handshake reply: the connection is being served
        assert accepted.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
    finally:
        client.close()
        thread.join(10)
    assert not thread.is_alive()


def test_the_drop_hook_closes_mid_batch():
    proc, address = start_server(env={"RGPU_DROP_AFTER": "2"})
    try:
        s, _ = handshake(address, os.urandom(16))
        wire.send_frame(s, wire.encode([[1, wire.SYNC], [2, wire.SYNC]]))
        wire.recv_frame(s)   # the first is answered
        with pytest.raises(ConnectionError):
            wire.recv_frame(s)
    finally:
        proc.kill()
