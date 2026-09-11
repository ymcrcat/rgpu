"""rgpu-opserver: runs PyTorch ops on this machine for rgpu clients.

    rgpu-opserver                    # cuda, 127.0.0.1:9720
    rgpu-opserver --device cpu       # for testing on a machine with no GPU
    rgpu-opserver --bind 0.0.0.0     # only on a network you trust

There is no authentication. Anyone who can connect can run GPU work and read
back their own tensors, so the default is to listen on 127.0.0.1 and reach
the server through an ssh tunnel.
"""

import argparse
import os
import socket
import sys
import threading
import time

import torch

from .. import wire
from .session import Session


def log(message):
    print(f"[rgpu-opserver] {message}", file=sys.stderr, flush=True)


class Registry:
    """Sessions by id, kept for a grace period after their connection goes.

    Each attach gets a generation number, so a stale connection noticing its
    own death late cannot mark a session detached while a newer connection is
    using it.
    """

    def __init__(self, device, grace):
        self.device = device
        self.grace = grace
        self.lock = threading.Lock()
        self.sessions = {}   # id -> [Session, generation, detached_at or None]

    def attach(self, sid):
        with self.lock:
            entry = self.sessions.get(sid)
            if entry is not None:
                entry[1] += 1
                entry[2] = None
                return entry[0], True, entry[1]
            session = Session(self.device)
            self.sessions[sid] = [session, 0, None]
            return session, False, 0

    def detach(self, sid, generation):
        with self.lock:
            entry = self.sessions.get(sid)
            if entry is not None and entry[1] == generation:
                entry[2] = time.monotonic()

    def reap(self):
        now = time.monotonic()
        with self.lock:
            for sid, (_, _, since) in list(self.sessions.items()):
                if since is not None and now - since > self.grace:
                    del self.sessions[sid]
                    log(f"session {sid.hex()[:8]} expired; its tensors are gone")


class Drop:
    """Closes a connection on the Nth message, to test that a drop is survivable."""

    def __init__(self, after):
        self.after = after
        self.count = 0
        self.lock = threading.Lock()

    def hit(self):
        if not self.after:
            return False
        with self.lock:
            self.count += 1
            return self.count == self.after


def serve_connection(conn, registry, drop):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sid = generation = None
    try:
        if wire.recv_exact(conn, len(wire.MAGIC)) != wire.MAGIC:
            return
        version, sid, last_acked, client_torch = wire.decode(wire.recv_frame(conn))
        if version != wire.VERSION or not isinstance(sid, bytes) or len(sid) != 16:
            log("refused a client speaking another protocol version")
            return
        session, resumed, generation = registry.attach(sid)
        with session.lock:
            last_seq, cached = session.last_seq, session.last_reply
        wire.send_frame(conn, wire.encode([wire.VERSION, resumed, last_seq, torch.__version__]))
        if resumed:
            log(f"session {sid.hex()[:8]} resumed after message {last_seq}")
            if cached is not None and cached[0] > last_acked:
                wire.send_frame(conn, cached[1])
        else:
            log(f"session {sid.hex()[:8]} started (client torch {client_torch})")
        while True:
            batch = wire.decode(wire.recv_frame(conn))
            for message in batch:
                if drop.hit():
                    log("dropping the connection (RGPU_DROP_AFTER)")
                    return
                with session.lock:
                    reply = session.execute(message)
                if reply is not None:
                    wire.send_frame(conn, reply)
    except (ConnectionError, OSError, wire.DecodeError, ValueError, TypeError) as e:
        if not isinstance(e, ConnectionError):
            log(f"closing a connection: {e}")
    finally:
        conn.close()
        if sid is not None and generation is not None:
            registry.detach(sid, generation)


def _reaper(registry):
    while True:
        time.sleep(1)
        registry.reap()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rgpu-opserver", description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=9720)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args(argv)

    torch.empty(0, device=args.device)   # fail now if the device is unusable
    registry = Registry(args.device, float(os.environ.get("RGPU_SESSION_GRACE", 120)))
    drop = Drop(int(os.environ.get("RGPU_DROP_AFTER", 0)))
    server = socket.create_server((args.bind, args.port))
    log(f"serving {args.device} on {args.bind}:{args.port} "
        "(no authentication: keep it on a trusted network)")
    threading.Thread(target=_reaper, args=(registry,), daemon=True).start()
    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(target=serve_connection, args=(conn, registry, drop),
                             daemon=True).start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
