import os
import socket

import pytest
import torch

from rgpu import wire
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
