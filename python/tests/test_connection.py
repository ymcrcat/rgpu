import os

import pytest
import torch

from rgpu import session, wire


@pytest.fixture
def conn():
    return session.Connection(os.environ["RGPU_OPSERVER"])


def put(conn, tid, t):
    conn.post(wire.RUN, "aten::empty_strided", "default",
              [list(t.shape), list(t.stride())],
              {"dtype": t.dtype, "device": wire.Dev("rgpu")}, [tid])
    conn.post(wire.UPLOAD, tid, wire.Host.of(t), size=t.numel() * t.element_size())


def test_posts_do_not_wait_and_a_request_does(conn):
    a = torch.randn(10)
    put(conn, 1, a)
    conn.post(wire.RUN, "aten::neg", "default", [wire.Ref(1)], {}, [2])
    assert conn.stats["waits"] == 0 and conn.stats["messages"] == 3
    back = conn.request(wire.DOWNLOAD, 2).tensor()
    assert torch.equal(back, -a)
    assert conn.stats["waits"] == 1


def test_the_queue_flushes_on_count_without_waiting(conn):
    conn.flush_ops = 3
    conn.post(wire.SEED, 1)
    conn.post(wire.SEED, 2)
    assert conn.stats["bytes_out"] == 0
    conn.post(wire.SEED, 3)
    assert conn.stats["bytes_out"] > 0 and conn.stats["waits"] == 0


def test_a_failure_while_streaming_raises_at_the_next_wait(conn):
    put(conn, 1, torch.randn(2, 3))
    put(conn, 2, torch.randn(4, 5))
    conn.post(wire.RUN, "aten::mm", "default", [wire.Ref(1), wire.Ref(2)], {}, [3])
    with pytest.raises(session.RemoteError, match="aten::mm"):
        conn.request(wire.SYNC)
    conn.request(wire.SYNC)   # reported once; the connection is still usable


def test_frees_queued_by_finalizers_go_out_with_the_next_batch(conn):
    put(conn, 1, torch.ones(3))
    session.pending_frees.append(1)
    conn.post(wire.SEED, 0)   # a safe point: the free goes out first
    with pytest.raises(session.RemoteError):
        conn.request(wire.DOWNLOAD, 1)


def test_the_process_connection_is_reused():
    session.reset()
    assert session.get() is session.get()
    session.reset()


def test_versions_must_agree_on_major_and_minor(monkeypatch):
    session._check_versions(torch.__version__)
    with pytest.raises(RuntimeError, match="torch"):
        session._check_versions("1.0.0")
    monkeypatch.setenv("RGPU_ALLOW_VERSION_MISMATCH", "1")
    session._check_versions("1.0.0")
