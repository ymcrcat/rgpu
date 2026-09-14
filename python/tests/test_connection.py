import collections
import gc
import os
import socket
import threading
import time

import pytest
import torch

from rgpu import session, wire
from conftest import free_port


@pytest.fixture
def conn():
    # Tensors freed by an earlier test (on the shared process connection) can
    # still be sitting in this process-wide queue; the next Connection to post
    # anything drains it, so left alone a fresh one here would pick up frees
    # that aren't its own and see an inflated message count. Send them to the
    # process connection first, where they belong.
    gc.collect()
    if session.pending_frees:
        session.get().request(wire.SYNC)
    return session.Connection(os.environ["RGPU_OPSERVER"])


def queued(seq, *fields):
    """A queue entry the way Connection._append makes one: number and bytes."""
    return (seq, wire.encode_element([seq, *fields]))


def put(conn, tid, t):
    conn.post(wire.RUN, "aten::empty_strided", "default",
              [list(t.shape), list(t.stride())],
              {"dtype": t.dtype, "device": wire.Dev("rgpu")}, [tid])
    conn.post(wire.UPLOAD, tid, wire.Host.of(t))


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


def test_an_unsendable_argument_raises_at_the_call_and_leaves_the_queue_usable(conn):
    """A value the wire cannot carry must fail at the call that passed it, and
    must not be left in the queue: a message that can never be encoded would
    otherwise make every later flush raise for the rest of the process."""
    put(conn, 1, torch.ones(3))
    with pytest.raises(wire.EncodeError, match="Generator"):
        conn.post(wire.RUN, "aten::normal", "default",
                  [0.0, 1.0], {"generator": torch.Generator()}, [2])
    back = conn.request(wire.DOWNLOAD, 1).tensor()
    assert torch.equal(back, torch.ones(3))


def test_a_rejected_message_takes_no_sequence_number(conn):
    """Sequence numbers have to stay contiguous: _reconnect reads a gap at the
    front of the replay queue as messages the server can never be sent again."""
    conn.flush_ops, conn.flush_bytes = 1 << 30, 1 << 30   # nothing leaves the queue
    conn.post(wire.SEED, 10)
    with pytest.raises(wire.EncodeError):
        conn.post(wire.SEED, torch.Generator())
    conn.post(wire.SEED, 30)
    seqs = [seq for seq, _ in conn.pending]
    assert seqs == list(range(seqs[0], conn.seq + 1))   # contiguous, no hole
    assert [seq for seq, _ in conn.unacked] == seqs


def test_a_refused_free_puts_its_ids_back(conn, monkeypatch):
    """_queue takes the ids out of the only place that remembers them before
    it queues the FREE, and _append can refuse a message now. If it ever
    refuses this one the ids must come back, in order, or the tensors they
    name leak on the server for the rest of the session."""
    real = session.Connection._append

    def refuse_the_free(self, kind, fields):
        if kind == wire.FREE:
            raise wire.EncodeError("pretend the free could not be encoded")
        return real(self, kind, fields)
    monkeypatch.setattr(session.Connection, "_append", refuse_the_free)

    session.pending_frees.extend([7, 8, 9])
    with pytest.raises(wire.EncodeError):
        conn.post(wire.SEED, 1)
    assert list(session.pending_frees) == [7, 8, 9]
    session.pending_frees.clear()


def test_a_rejected_message_leaves_a_replay_that_still_works(conn, monkeypatch):
    """The replay after a reconnect is built from the same queue, so a message
    that failed to encode must be absent from it and leave no hole either."""
    conn.flush_ops, conn.flush_bytes = 1 << 30, 1 << 30   # nothing leaves the queue
    conn.had_session = True
    conn.post(wire.SEED, 10)
    with pytest.raises(wire.EncodeError):
        conn.post(wire.SEED, torch.Generator())
    conn.post(wire.SEED, 30)
    replayed = list(conn.pending)

    def fake_connect(self):
        self.sock = object()
        self.had_session = True
        return True, 0   # resumed, the server got none of it
    monkeypatch.setattr(session.Connection, "_connect", fake_connect)
    sent = []
    monkeypatch.setattr(wire, "send_frame", lambda sock, frame: sent.append(wire.decode(frame)))

    conn._reconnect()

    assert len(sent) == 1
    # The two seeds, and nothing between them that carries the generator: a
    # finalizer may have slipped a FREE in, so they are found by content.
    seeds = [m for m in sent[0] if m[1] == wire.SEED]
    assert seeds == [[seeds[0][0], wire.SEED, 10], [seeds[1][0], wire.SEED, 30]]
    assert [m[0] for m in sent[0]] == [seq for seq, _ in replayed]


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


def test_the_socket_asks_the_kernel_to_keep_it_alive(conn):
    """Without SO_KEEPALIVE a recv on a partitioned link blocks for hours: no
    FIN ever arrives, so nothing tells this side the peer is gone."""
    conn.request(wire.SYNC)   # opens the connection
    assert conn.sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0


def test_silence_from_the_peer_becomes_a_reconnect_not_a_wedged_call(monkeypatch):
    """A peer that accepts, handshakes and then never answers must not hold
    the caller forever. RGPU_RECV_TIMEOUT turns the silence into an OSError,
    which has to route into _reconnect exactly as a dropped connection does -
    the second connection answers the replayed download, so the caller gets
    its values back and never sees the timeout."""
    srv = socket.create_server(("127.0.0.1", 0))
    accepted = []

    def serve():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            accepted.append(c)   # kept open: closing would send a FIN and cheat
            first = len(accepted) == 1
            try:
                wire.recv_exact(c, len(wire.MAGIC))
                wire.recv_frame(c)
                wire.send_frame(c, wire.encode(
                    [wire.VERSION, not first, 0, torch.__version__]))
                batch = wire.decode(wire.recv_frame(c))
                if first:
                    continue   # handshaken, and now silent forever
                for seq, kind, *fields in batch:
                    if kind == wire.DOWNLOAD:
                        wire.send_frame(c, wire.encode(
                            [seq, wire.OK, wire.Host.of(torch.ones(3))]))
            except OSError:
                pass

    threading.Thread(target=serve, daemon=True).start()
    monkeypatch.setenv("RGPU_OPSERVER", f"127.0.0.1:{srv.getsockname()[1]}")
    monkeypatch.setenv("RGPU_RECV_TIMEOUT", "0.3")
    session.reset()
    try:
        got = torch.ones(3, device="rgpu").cpu()
    finally:
        session.reset()
        srv.close()
        for c in accepted:
            c.close()
    assert torch.equal(got, torch.ones(3))
    assert len(accepted) == 2   # the silence drove exactly one reconnect


def test_a_first_connection_that_is_refused_fails_at_once(monkeypatch):
    """There is no session to recover before the first connect succeeds, so a
    mistyped RGPU_OPSERVER must raise now rather than spend the whole
    reconnect budget retrying a port nothing will ever answer."""
    monkeypatch.setenv("RGPU_RECONNECT_SECONDS", "5")
    conn = session.Connection(f"127.0.0.1:{free_port()}")   # nothing is listening
    start = time.monotonic()
    with pytest.raises(OSError):
        conn.request(wire.SYNC)
    assert time.monotonic() - start < 1


def test_unacked_bytes_drains_on_ack(conn):
    conn.post(wire.SEED, 1)
    assert conn.unacked_bytes == sum(len(b) for _, b in conn.unacked)
    assert len(conn.unacked) == 1
    conn.request(wire.SYNC)
    assert conn.unacked_bytes == 0
    assert len(conn.unacked) == 0
    assert conn.replay_possible is True


# The queues hold the encoded message, so what they cost is len(blob) - not an
# estimate. A RUN carrying an inline host tensor is the case that matters: it
# can be megabytes, and counting it as a fixed overhead lets it slip past both
# the byte-triggered flush and the replay limit that is supposed to bound how
# much is held for a reconnect.
def test_queue_bytes_count_the_encoded_size_of_an_inline_tensor(conn):
    conn.flush_ops, conn.flush_bytes = 1 << 30, 1 << 30   # nothing leaves the queue
    big = torch.zeros(1 << 20, dtype=torch.uint8)         # a megabyte, inline
    conn.post(wire.RUN, "aten::add", "default", [wire.Host.of(big)], {}, [1])
    blob = conn.pending[-1][1]
    assert len(blob) > (1 << 20), "the message really does carry the tensor"
    assert conn.pending_bytes == sum(len(b) for _, b in conn.pending)
    assert conn.unacked_bytes == sum(len(b) for _, b in conn.unacked)


# Acknowledging part of the queue must subtract what it removed, rather than
# only resetting once the queue happens to empty: a long-lived connection that
# is never fully caught up would otherwise keep an estimate that only grows,
# and trip the replay limit on messages the server acknowledged long ago.
def test_acking_one_message_subtracts_only_its_bytes(conn):
    conn.flush_ops, conn.flush_bytes = 1 << 30, 1 << 30
    conn.post(wire.SEED, 1)
    conn.post(wire.SEED, 2)
    assert len(conn.unacked) >= 2
    assert conn.unacked_bytes == sum(len(b) for _, b in conn.unacked)
    conn._ack(conn.unacked[0][0])
    assert conn.unacked, "only the first message was acknowledged"
    assert conn.unacked_bytes == sum(len(b) for _, b in conn.unacked)


# --- _reconnect, against a stubbed socket: no real server involved ---------
#
# Connection.__init__ does no I/O (the socket is only opened lazily by
# _connect), so a plain Connection is a safe, cheap thing to unit-test
# _reconnect against once _connect itself is replaced with a fake.


def test_reconnect_replays_exactly_the_unacknowledged_messages(monkeypatch):
    conn = session.Connection("127.0.0.1:1")
    conn.seq = 3
    conn.unacked = collections.deque([
        queued(1, wire.SEED, 10),
        queued(2, wire.SEED, 20),
        queued(3, wire.SEED, 30),
    ])
    conn.unacked_bytes = 300
    conn.replay_possible = True
    conn.had_session = True
    conn.pending = [queued(3, wire.SEED, 30)]   # already duplicated into unacked
    conn.pending_bytes = 64

    def fake_connect(self):
        self.sock = object()
        self.had_session = True
        return True, 1   # resumed; the server had already applied message 1
    monkeypatch.setattr(session.Connection, "_connect", fake_connect)

    sent = []
    monkeypatch.setattr(wire, "send_frame", lambda sock, frame: sent.append(wire.decode(frame)))

    conn._reconnect()

    assert len(sent) == 1
    assert [m[0] for m in sent[0]] == [2, 3]   # nothing already applied resent, nothing skipped
    assert [m[0] for m in conn.unacked] == [2, 3]
    assert conn.pending == [] and conn.pending_bytes == 0   # already covered by the replay
    expected = wire.encode([[2, wire.SEED, 20], [3, wire.SEED, 30]])
    assert conn.stats["bytes_out"] == len(expected)   # counts what actually went out


def test_reconnect_resets_replay_bookkeeping_once_everything_is_applied(monkeypatch):
    conn = session.Connection("127.0.0.1:1")
    conn.seq = 2
    conn.unacked = collections.deque([queued(1, wire.SEED, 1), queued(2, wire.SEED, 2)])
    conn.unacked_bytes = 200
    conn.replay_possible = False
    conn.had_session = True

    def fake_connect(self):
        self.sock = object()
        self.had_session = True
        return True, 2   # the server has already applied everything we sent
    monkeypatch.setattr(session.Connection, "_connect", fake_connect)
    monkeypatch.setattr(wire, "send_frame",
                         lambda sock, frame: pytest.fail("nothing left to replay"))

    conn._reconnect()

    assert len(conn.unacked) == 0
    assert conn.unacked_bytes == 0
    assert conn.replay_possible is True


def test_reconnect_retries_when_sending_the_replay_itself_fails(monkeypatch):
    """A peer that completes the handshake and then dies before reading the
    replay must be just another failed attempt, not a raw BrokenPipeError out
    of _reconnect (and so out of whatever user call triggered it)."""
    class FakeSocket:
        def close(self):
            pass

    conn = session.Connection("127.0.0.1:1")
    conn.seq = 1
    conn.unacked = collections.deque([queued(1, wire.SEED, 1)])
    conn.unacked_bytes = 64
    conn.replay_possible = True
    conn.had_session = True

    def fake_connect(self):
        self.sock = FakeSocket()
        self.had_session = True
        return True, 0
    monkeypatch.setattr(session.Connection, "_connect", fake_connect)
    monkeypatch.setattr(session, "_sleep", lambda s: None)   # don't actually wait

    sent = []
    calls = {"n": 0}

    def fake_send_frame(sock, frame):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BrokenPipeError("peer died before reading the replay")
        sent.append(wire.decode(frame))
    monkeypatch.setattr(wire, "send_frame", fake_send_frame)

    conn._reconnect()   # must not raise BrokenPipeError

    assert calls["n"] == 2
    assert len(sent) == 1 and [m[0] for m in sent[0]] == [1]


def test_reconnect_shares_one_recovery_deadline_across_calls(monkeypatch):
    """A server that keeps completing the handshake and then dying again must
    not reset the recovery budget on every _reconnect() call: only an actual
    reply (via _ack) proves the connection is alive. Uses a fake clock (via
    session's own module-local _monotonic/_sleep, not the real time module,
    so nothing else in the process is affected) so the test is fast and
    deterministic rather than waiting on a real one."""
    monkeypatch.setenv("RGPU_RECONNECT_SECONDS", "2")
    clock = [0.0]
    monkeypatch.setattr(session, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(session, "_sleep", lambda s: None)

    class FakeSocket:
        def close(self):
            pass

    conn = session.Connection("127.0.0.1:1")
    conn.had_session = True

    def fake_connect(self):
        clock[0] += 0.5   # each handshake attempt "takes" half a second
        self.sock = FakeSocket()
        self.had_session = True
        return True, self.seq   # resumed, fully caught up: nothing to replay
    monkeypatch.setattr(session.Connection, "_connect", fake_connect)

    calls = 0
    with pytest.raises(ConnectionError):
        for _ in range(100):
            calls += 1
            conn._reconnect()
    assert calls < 100   # gave up once the shared deadline passed, not after 100 successes
