"""The client's connection to rgpu-opserver.

Messages are queued and sent in batches. Most never wait for an answer: the
server applies them in order, so an op sent without waiting still runs after
everything before it. Only a few things need a reply, and those are the only
places a round trip happens.

There is one connection per process, shared by every thread. Autograd runs
backward on a thread of its own, so everything here is under one lock.
"""

import collections
import logging
import os
import socket
import threading
import time

import torch

from . import wire

log = logging.getLogger("rgpu")


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

# Module-local names for the two time functions _reconnect needs, so a test
# can fake the clock and the sleep for just this module (monkeypatching the
# real time module would affect pytest's own timing and any other thread for
# as long as the test runs).
_monotonic = time.monotonic
_sleep = time.sleep


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


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
        # Both queues hold (sequence number, encoded message) - see _append.
        self.pending = []                    # encoded, not yet written
        self.pending_bytes = 0
        self.unacked = collections.deque()   # written, not yet acknowledged
        self.unacked_bytes = 0
        self.replay_possible = True
        self.last_acked = 0
        self.had_session = False
        self._recovery_deadline = None   # shared across _reconnect() calls; see _reconnect
        self._recovery_delay = None      # ditto: the backoff also has to persist
        self.flush_ops = _env_int("RGPU_FLUSH_OPS", 64)
        self.flush_bytes = _env_int("RGPU_FLUSH_BYTES", 256 * 1024)
        # Keepalive is the primary detector of a peer that has gone away: it
        # cannot false-positive on a server that is legitimately busy with a
        # long queue. The recv timeout is only a backstop, so it is generous
        # - a single op is allowed to take minutes.
        self.connect_timeout = _env_float("RGPU_CONNECT_TIMEOUT", 10)
        self.recv_timeout = _env_float("RGPU_RECV_TIMEOUT", 300)
        self.stats = {"messages": 0, "waits": 0, "bytes_out": 0, "bytes_in": 0}

    # --- connecting ----------------------------------------------------------

    def _connect(self):
        # The connect timeout is separate and much shorter: a hung SYN that
        # waited out the operating system's own timeout would overshoot the
        # whole reconnect budget on a single attempt.
        s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        s.settimeout(self.recv_timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        wire.set_keepalive(s)
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
        # Encoded here, before anything is committed, so a value the wire
        # cannot carry raises at the call that passed it rather than at some
        # later flush - and leaves the queue exactly as it was, with no gap in
        # the numbering for _reconnect to read as a lost message. The bytes are
        # kept, not thrown away: _flush and the replay join them up, so this
        # costs no extra encoding on the streaming path.
        entry = (self.seq + 1, wire.encode_element([self.seq + 1, kind, *fields]))
        self.seq += 1
        self.pending.append(entry)
        self.pending_bytes += 64 + size
        self.stats["messages"] += 1
        if self.replay_possible:
            self.unacked.append(entry)
            self.unacked_bytes += 64 + size
            if self.unacked_bytes > MAX_UNACKED:
                # Holding gigabytes of uploads to replay would cost more than
                # the recovery is worth. Until the server next acknowledges
                # everything, a dropped connection cannot be recovered.
                self.replay_possible = False
                self.unacked.clear()
                self.unacked_bytes = 0
                log.warning(
                    "rgpu: more than %d MB is waiting to be acknowledged, over the "
                    "replay limit, so this session cannot be recovered if the "
                    "connection drops before the server catches up",
                    MAX_UNACKED >> 20)
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

        _recovery_deadline and _recovery_delay live on the Connection, not as
        state local to this call, so a recovery that spans several
        _reconnect() calls -- a peer that gets through the handshake and
        then dies again, so _connect() itself never raises -- still shares
        one deadline and one growing backoff instead of getting a fresh copy
        of each on every call. The backoff sleep runs before every attempt
        after the very first one of a recovery, whether the previous attempt
        failed to connect at all or connected fine and something else killed
        it afterwards; it is capped at 3 s. Both are cleared once a reply
        actually arrives (see _ack) or once a recovery gives up, so the next
        one starts with a fresh budget instead of failing instantly forever.
        """
        fresh = self._recovery_deadline is None
        if fresh:
            self._recovery_deadline = _monotonic() + _env_int("RGPU_RECONNECT_SECONDS", 60)
            self._recovery_delay = 0.1
        while True:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
            if _monotonic() > self._recovery_deadline:
                self._recovery_deadline = None
                self._recovery_delay = None
                raise ConnectionError("could not reach rgpu-opserver again")
            if not fresh:
                _sleep(self._recovery_delay)
                self._recovery_delay = min(self._recovery_delay * 2, 3.0)
            fresh = False
            try:
                resumed, server_seq = self._connect()
                if server_seq < self.seq and (not self.replay_possible or not self.unacked
                                              or self.unacked[0][0] > server_seq + 1):
                    self.sock.close()
                    self.sock = None
                    raise SessionLost(
                        "messages the server never received were too large to keep "
                        "for replay; tensors on rgpu may be stale")
                while self.unacked and self.unacked[0][0] <= server_seq:
                    self.unacked.popleft()
                if not self.unacked:
                    self.unacked_bytes = 0
                    self.replay_possible = True
                self.pending = []
                self.pending_bytes = 0
                if self.unacked:
                    frame = wire.encode_elements([blob for _, blob in self.unacked])
                    wire.send_frame(self.sock, frame)
                    self.stats["bytes_out"] += len(frame)
                return
            except SessionLost:
                self._recovery_deadline = None
                self._recovery_delay = None
                raise
            except OSError:
                continue

    def _flush(self):
        if not self.pending:
            return
        frame = wire.encode_elements([blob for _, blob in self.pending])
        try:
            if self.sock is None:
                self._connect()
            wire.send_frame(self.sock, frame)
            self.stats["bytes_out"] += len(frame)
        except SessionLost:
            raise
        except OSError:
            if not self.had_session:
                # Nothing to recover: no connection has ever succeeded, so
                # there is no session on the far end to pick back up. A
                # mistyped address must say so now rather than spend the
                # whole reconnect budget retrying a port nothing answers.
                raise
            self._reconnect()   # replays everything unacknowledged, this batch included
        self.pending = []
        self.pending_bytes = 0

    def _await(self, seq):
        while True:
            try:
                frame = wire.recv_frame(self.sock)
            except SessionLost:
                raise
            except OSError:
                # A recv timeout arrives here too - socket.timeout is an
                # OSError - and is treated as a dropped connection rather
                # than raised at the caller: the server resends a lost reply,
                # or runs the replay. It cannot buy unlimited time either,
                # because the recovery deadline is only cleared by a reply
                # actually arriving (see _ack), not by a reconnect.
                self._reconnect()
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
        self._recovery_deadline = None   # a reply arrived: the connection is proven alive
        self._recovery_delay = None
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
