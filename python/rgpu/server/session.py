"""One client's state on the server: its tensors, and what went wrong.

A session outlives any single connection: after a drop the client reconnects
and carries on with the same tensors. Messages are applied strictly in the
order they were sent, which is what makes streaming correct - an op sent
without waiting still runs after everything sent before it.

A failed op does not raise at the client straight away, because nothing is
waiting for it. Its outputs are poisoned instead: any later op that uses one
is skipped and poisons its own outputs, and the first failure is reported at
the next message the client does wait for. One exception, naming the op that
actually failed, rather than a cascade about missing tensors.
"""

import threading

import torch
from torch.utils._pytree import tree_flatten, tree_map

from .. import wire
from . import ops


class Poison:
    """Stands in for the output of an op that failed, or of one that used it."""

    __slots__ = ("op", "message")

    def __init__(self, op, message):
        self.op = op
        self.message = message


class _Skip(Exception):
    def __init__(self, poison):
        super().__init__(poison.message)
        self.poison = poison


class Session:
    def __init__(self, device):
        self.device = torch.device(device)
        self.tensors = {}
        self.first_error = None      # (op, message), reported at the next reply
        self.last_seq = 0
        self.last_reply = None       # (seq, frame), resent after a reconnect
        self.graphs = {}             # graph id -> callable or Poison
        self.next_server_id = wire.SERVER_ID_BASE
        self.generation = 0          # which attach owns this session now
        self.lock = threading.Lock()

    # --- arguments ---------------------------------------------------------

    def _device(self, name):
        if name.split(":")[0] == "rgpu":
            return self.device
        if name == "cpu":
            return torch.device("cpu")
        raise ValueError(f"device {name!r} is not available on this server")

    def _arg(self, v):
        if isinstance(v, wire.Ref):
            t = self.tensors.get(v.id)
            if t is None:
                raise KeyError(f"no tensor with id {v.id}")
            if isinstance(t, Poison):
                raise _Skip(t)
            return t
        if isinstance(v, wire.Host):
            return v.tensor()
        if isinstance(v, wire.Dev):
            return self._device(v.name)
        return v

    def _args(self, args, kwargs):
        return (tree_map(self._arg, args),
                {k: tree_map(self._arg, v) for k, v in kwargs.items()})

    # --- results -----------------------------------------------------------

    def _store(self, out, ids):
        if not ids:
            return   # nothing to keep; an op returning nothing still flattens to [None]
        flat, _ = tree_flatten(out)
        if len(flat) != len(ids):
            raise RuntimeError(f"op produced {len(flat)} outputs, client expected {len(ids)}")
        for value, tid in zip(flat, ids):
            if tid is not None:
                self.tensors[tid] = value

    def _poison(self, ids, poison):
        for tid in ids:
            if tid is not None:
                self.tensors[tid] = poison

    def _fail(self, op, exc, ids):
        p = Poison(op, f"{type(exc).__name__}: {exc}")
        self._poison(ids, p)
        if self.first_error is None:
            self.first_error = (p.op, p.message)

    # --- messages that nothing waits for -----------------------------------

    def _run(self, name, overload, args, kwargs, out_ids):
        label = f"{name}.{overload}"
        try:
            op = ops.resolve(name, overload)
            a, kw = self._args(args, kwargs)
            out = op(*a, **kw)
            self._store(out, out_ids)
        except _Skip as s:
            self._poison(out_ids, s.poison)
        except Exception as e:  # noqa: BLE001 - reported to the client, not raised here
            self._fail(label, e, out_ids)

    def _upload(self, dst, host):
        try:
            self._arg(wire.Ref(dst)).copy_(host.tensor())
        except _Skip:
            pass
        except Exception as e:  # noqa: BLE001
            self._fail(f"upload to tensor {dst}", e, [dst])

    def _free(self, ids):
        for tid in ids:
            self.tensors.pop(tid, None)

    def _seed(self, seed):
        # Process-global, so two sessions seeding at once interfere with each
        # other's random ops. Giving each session a generator of its own would
        # mean threading it through every factory op; out of scope for now.
        torch.manual_seed(seed)

    def _compile(self, gid, nodes, compiler, mode):
        from . import graphs
        try:
            self.graphs[gid] = graphs.build(nodes, self, compiler, mode)
        except Exception as e:  # noqa: BLE001
            self.graphs[gid] = Poison(f"compile of graph {gid}", f"{type(e).__name__}: {e}")
            if self.first_error is None:
                self.first_error = (self.graphs[gid].op, self.graphs[gid].message)

    def _call(self, gid, in_ids, out_ids):
        fn = self.graphs.get(gid)
        if isinstance(fn, Poison):
            self._poison(out_ids, fn)   # already reported when the compile failed
            return
        if fn is None:
            # A CALL for a gid nothing ever compiled is a failure of its own,
            # not the echo of an earlier one, so it goes through the same
            # bookkeeping as every other failure and is reported at the next
            # wait rather than only if an output happens to be downloaded.
            self._fail(f"graph {gid}", LookupError("was never compiled"), out_ids)
            return
        try:
            inputs = [self._arg(wire.Ref(i)) for i in in_ids]
            outs = fn(*inputs)
            self._store(list(outs), out_ids)
        except _Skip as s:
            self._poison(out_ids, s.poison)
        except Exception as e:  # noqa: BLE001
            self._fail(f"graph {gid}", e, out_ids)

    # --- messages the client waits for --------------------------------------

    def _download(self, tid):
        return wire.Host.of(self._arg(wire.Ref(tid)).detach().cpu())

    def _run_sync(self, name, overload, args, kwargs):
        op = ops.resolve(name, overload)
        a, kw = self._args(args, kwargs)
        out = op(*a, **kw)

        def describe(v):
            if not isinstance(v, torch.Tensor):
                return v
            tid = self.next_server_id
            self.next_server_id += 1
            self.tensors[tid] = v
            return ["__tensor__", tid, v.dtype, list(v.shape), list(v.stride()),
                    v.storage_offset()]

        return tree_map(describe, out)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        return None

    def _ack(self):
        return None

    _ASYNC = {wire.RUN: _run, wire.UPLOAD: _upload, wire.FREE: _free, wire.SEED: _seed,
              wire.COMPILE: _compile, wire.CALL: _call}
    _WAITED = {wire.RUN_SYNC: _run_sync, wire.DOWNLOAD: _download, wire.SYNC: _sync,
               wire.ACK: _ack}

    def execute(self, message):
        seq, kind, *fields = message
        if seq <= self.last_seq:
            return None   # already applied, before a reconnect
        self.last_seq = seq
        if kind in self._ASYNC:
            self._ASYNC[kind](self, *fields)
            return None
        if kind not in self._WAITED:
            self._fail(f"message kind {kind}", ValueError("unknown message"), [])
            return None
        if self.first_error is not None:
            status, value = wire.ERROR, list(self.first_error)
            self.first_error = None
        else:
            try:
                status, value = wire.OK, self._WAITED[kind](self, *fields)
            except _Skip as s:
                status, value = wire.ERROR, [s.poison.op, s.poison.message]
            except Exception as e:  # noqa: BLE001
                status, value = wire.ERROR, [_label(kind, fields), f"{type(e).__name__}: {e}"]
        frame = wire.encode([seq, status, value])
        self.last_reply = (seq, frame)
        return frame


def _label(kind, fields):
    if kind == wire.RUN_SYNC:
        return f"{fields[0]}.{fields[1]}"
    if kind == wire.DOWNLOAD:
        return f"download of tensor {fields[0]}"
    return "sync"
