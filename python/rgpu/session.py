"""The client's connection to rgpu-opserver.

Messages are queued and sent in batches. Most never wait for an answer: the
server applies them in order, so an op sent without waiting still runs after
everything before it. Only a few things need a reply, and those are the only
places a round trip happens.

There is one connection per process, shared by every thread. Autograd runs
backward on a thread of its own, so everything here is under one lock.
"""

import collections
import os
import socket
import threading
import time

import torch

from . import wire


class RemoteError(RuntimeError):
    """An op failed on the server."""

    def __init__(self, op, message):
        super().__init__(f"{op} failed on the server: {message}")
        self.op = op
        self.remote_message = message


class SessionLost(RuntimeError):
    """The server no longer has this process's session, so its tensors are gone."""


# Ids whose last reference has gone. Finalizers only append here; the ids go
# out as one FREE at the next point where queueing is safe.
pending_frees = collections.deque()

MAX_UNACKED = 64 << 20


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _check_versions(server_torch):
    ours = torch.__version__.split(".")[:2]
    theirs = str(server_torch).split(".")[:2]
    if ours != theirs and not os.environ.get("RGPU_ALLOW_VERSION_MISMATCH"):
        raise RuntimeError(
            f"this client has torch {torch.__version__} but rgpu-opserver has "
            f"{server_torch}; op schemas can differ between torch versions, so "
            "install the same major.minor on both (or set RGPU_ALLOW_VERSION_MISMATCH=1)")


class Connection:
    def __init__(self, address):
        host, _, port = address.rpartition(":")
        self.host = host or "127.0.0.1"
        self.port = int(port)
        self.session_id = os.urandom(16)
        self.lock = threading.RLock()
        self.sock = None
        self.seq = 0
        self.pending = []
        self.pending_bytes = 0
        self.unacked = collections.deque()   # messages written, not yet acknowledged
        self.unacked_bytes = 0
        self.replay_possible = True
        self.last_acked = 0
        self.had_session = False
        self.flush_ops = _env_int("RGPU_FLUSH_OPS", 64)
        self.flush_bytes = _env_int("RGPU_FLUSH_BYTES", 256 * 1024)
        self.stats = {"messages": 0, "waits": 0, "bytes_out": 0, "bytes_in": 0}

    # --- connecting ----------------------------------------------------------

    def _connect(self):
        s = socket.create_connection((self.host, self.port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(wire.MAGIC)
        wire.send_frame(s, wire.encode(
            [wire.VERSION, self.session_id, self.last_acked, torch.__version__]))
        version, resumed, server_seq, server_torch = wire.decode(wire.recv_frame(s))
        if version != wire.VERSION:
            s.close()
            raise RuntimeError(f"rgpu-opserver speaks protocol {version}, this client {wire.VERSION}")
        _check_versions(server_torch)
        if self.had_session and not resumed:
            s.close()
            raise SessionLost("the server no longer has this session; tensors on rgpu are gone")
        self.sock = s
        self.had_session = True
        return resumed, server_seq

    # --- queueing ------------------------------------------------------------

    def _queue(self, kind, fields, size):
        if pending_frees:
            ids = []
            while pending_frees:
                ids.append(pending_frees.popleft())
            self._append(wire.FREE, (ids,), 8 * len(ids))
        return self._append(kind, fields, size)

    def _append(self, kind, fields, size):
        self.seq += 1
        message = [self.seq, kind, *fields]
        self.pending.append(message)
        self.pending_bytes += 64 + size
        self.stats["messages"] += 1
        if self.replay_possible:
            self.unacked.append(message)
            self.unacked_bytes += 64 + size
            if self.unacked_bytes > MAX_UNACKED:
                # Holding gigabytes of uploads to replay would cost more than
                # the recovery is worth. Until the server next acknowledges
                # everything, a dropped connection cannot be recovered.
                self.replay_possible = False
                self.unacked.clear()
                self.unacked_bytes = 0
        return self.seq

    def post(self, kind, *fields, size=0):
        with self.lock:
            self._queue(kind, fields, size)
            if len(self.pending) >= self.flush_ops or self.pending_bytes >= self.flush_bytes:
                self._flush()

    def request(self, kind, *fields):
        with self.lock:
            seq = self._queue(kind, fields, 0)
            self._flush()
            self.stats["waits"] += 1
            return self._await(seq)

    def flush(self):
        with self.lock:
            self._flush()

    # --- the socket ----------------------------------------------------------

    def _reconnect(self):
        """Reach the server again and pick the session back up.

        Worth trying because the state is not on this side: if the server
        still has the session, nothing the application holds is lost.
        """
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        deadline = time.monotonic() + _env_int("RGPU_RECONNECT_SECONDS", 60)
        delay = 0.1
        while True:
            try:
                resumed, server_seq = self._connect()
                break
            except SessionLost:
                raise
            except OSError:
                if time.monotonic() > deadline:
                    raise ConnectionError("could not reach rgpu-opserver again") from None
                time.sleep(delay)
                delay = min(delay * 2, 3.0)
        if server_seq < self.seq and (not self.replay_possible or not self.unacked
                                      or self.unacked[0][0] > server_seq + 1):
            self.sock.close()
            self.sock = None
            raise SessionLost("messages the server never received were too large to keep "
                              "for replay; tensors on rgpu may be stale")
        while self.unacked and self.unacked[0][0] <= server_seq:
            self.unacked.popleft()
        self.pending = []
        self.pending_bytes = 0
        if self.unacked:
            wire.send_frame(self.sock, wire.encode(list(self.unacked)))

    def _flush(self):
        if not self.pending:
            return
        frame = wire.encode(self.pending)
        try:
            if self.sock is None:
                self._connect()
            wire.send_frame(self.sock, frame)
        except SessionLost:
            raise
        except OSError:
            self._reconnect()   # replays everything unacknowledged, this batch included
        self.stats["bytes_out"] += len(frame)
        self.pending = []
        self.pending_bytes = 0

    def _await(self, seq):
        while True:
            try:
                frame = wire.recv_frame(self.sock)
            except SessionLost:
                raise
            except OSError:
                self._reconnect()   # the server resends a lost reply, or runs the replay
                continue
            self.stats["bytes_in"] += len(frame)
            rseq, status, value = wire.decode(frame)
            if rseq < seq:
                continue   # a reply resent after a reconnect that we already had
            self._ack(rseq)
            if status == wire.ERROR:
                raise RemoteError(*value)
            return value

    def _ack(self, seq):
        self.last_acked = seq
        while self.unacked and self.unacked[0][0] <= seq:
            self.unacked.popleft()
        # unacked_bytes is an upper bound on what is held for replay, reset once
        # the server has acknowledged everything (so the 64 MB limit can only
        # trip early, never late).
        if not self.unacked:
            self.unacked_bytes = 0
            self.replay_possible = True


_conn = None
_conn_pid = None
_conn_lock = threading.Lock()


def get():
    global _conn, _conn_pid
    with _conn_lock:
        if _conn is None or _conn_pid != os.getpid():
            _conn = Connection(os.environ.get("RGPU_OPSERVER", "127.0.0.1:9720"))
            _conn_pid = os.getpid()
        return _conn


def reset():
    global _conn
    with _conn_lock:
        if _conn is not None and _conn.sock is not None:
            _conn.sock.close()
        _conn = None


def stats():
    return dict(get().stats)
