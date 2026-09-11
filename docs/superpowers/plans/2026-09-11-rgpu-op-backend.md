# rgpu Op-Level Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PyTorch device named `rgpu`, usable from the stock macOS CPU-only torch wheel, whose tensors live on a remote GPU host running `rgpu-opserver`.

**Architecture:** The client is a pure-Python package. `RemoteTensor` is a traceable tensor subclass that reports `device=rgpu:0` and holds only a meta tensor and an id. Every op after autograd lands in `__torch_dispatch__`, where output metadata is inferred on meta tensors, ids are chosen, and the op is streamed to the server without waiting. Waits happen only for downloads and for ops whose output size depends on the data. The server is a Python process that keeps a table from id to real tensor per session and runs `aten` ops by name.

**Tech Stack:** Python ≥ 3.10, torch ≥ 2.14 (standard library only beyond torch), pytest and torchvision for tests.

**Spec:** `docs/superpowers/specs/2026-09-11-rgpu-op-backend-design.md`

## Global Constraints

- Client runs on the stock macOS torch wheel, which has no CUDA backend. Nothing in `rgpu/` except `rgpu/server/` may assume CUDA.
- Standard library only beyond `torch`. Test extras: `pytest`, `torchvision`.
- Device name `rgpu`; server command `rgpu-opserver`; default port **9720**; server binds **127.0.0.1** by default.
- No pickle and no `eval` anywhere. The server runs only `aten` ops looked up by name; `aten::from_file` is blocked.
- Flush the send queue every **64** messages or **256 KB** (`RGPU_FLUSH_OPS`, `RGPU_FLUSH_BYTES`).
- Session grace period **120 s** (`RGPU_SESSION_GRACE`); client replay buffer limit **64 MB**.
- Every test compares values, not status codes.
- The CUDA-level stack (`client/`, `server/`, `common/`, `codegen/`, C++ `tests/`) is not touched.
- Commit messages follow the repo's style: a plain-English subject and an explanatory body, no `feat:` prefixes. Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  ```

## Facts verified before planning (torch 2.14, macOS arm64, CPU wheel)

- `torch.utils.backend_registration._setup_privateuseone_for_python_backend("rgpu", backend_module=mod)` registers the device from Python.
- A `_make_wrapper_subclass` tensor reporting `device=rgpu:0` supports matmul, views and autograd backward.
- torch.compile traces through it only if it defines `__tensor_flatten__`/`__tensor_unflatten__`.
- Meta tensors raise `NotImplementedError` for data-dependent ops (`nonzero`, boolean masks, `unique`, `masked_select`) and `RuntimeError` for genuine shape errors.
- Factory ops such as `arange` build an `empty(0)` and resize it through `out=`, which leaves a fixed-shape wrapper wrong. Registering all factory overloads (52 in this torch) as `PrivateUse1` kernels fixes it.
- Backward runs on a separate autograd thread, so the client connection needs a lock.
- `torch.manual_seed` reaches a custom device through `manual_seed_all` and `_is_in_bad_fork` on the device module.
- Autocast needs `get_amp_supported_dtype` on the device module **and** kernels registered at `AutocastPrivateUse1`; without them every op fails under autocast.

## File Structure

```
python/
  pyproject.toml               package metadata, rgpu-opserver entry point
  rgpu/
    __init__.py                import registers the device; public names
    wire.py                    framing and the tagged encoding (shared)
    session.py                 client connection: batching, waits, reconnect
    tensor.py                  RemoteTensor, ids, meta helpers
    dispatch.py                __torch_dispatch__: classify, infer, stream
    device.py                  device registration, factory kernels, torch.rgpu
    autocast.py                AutocastPrivateUse1 policy
    compile.py                 torch.compile backend
    server/
      __init__.py
      __main__.py              rgpu-opserver: listener, handshake, sessions
      ops.py                   op lookup by name, blocklist
      session.py               per-session execution, poisoning
      graphs.py                rebuild and compile shipped graphs
  tests/
    conftest.py                starts a CPU opserver subprocess
    test_wire.py
    test_server_session.py
    test_server_net.py
    test_connection.py
    test_device.py
    test_aliasing.py
    test_memory.py
    test_ladder.py
    test_training.py
    test_amp.py
    test_compile.py
    test_reconnect.py
  benchmarks/
    round_trips.py             waits and ms per step, eager and compiled
    survive_drop.py            per-step losses, for the killed-tunnel test
```

## Environment

The Mac development environment used by every task:

```bash
python3 -m venv python/.venv
python/.venv/bin/pip install "torch>=2.14" torchvision pytest
python/.venv/bin/pip install -e python --no-deps
```

`python/.venv/` is gitignored (Task 1 adds it). Tests run from the repo root with `python/.venv/bin/pytest python/tests -q`.

---

### Task 1: Package skeleton and wire format

**Files:**
- Create: `python/pyproject.toml`, `python/rgpu/__init__.py`, `python/rgpu/wire.py`, `python/rgpu/server/__init__.py`
- Modify: `.gitignore`
- Test: `python/tests/test_wire.py`

**Interfaces:**
- Produces: `rgpu.wire` with constants `MAGIC`, `VERSION`, `RUN=1`, `RUN_SYNC=2`, `UPLOAD=3`, `DOWNLOAD=4`, `FREE=5`, `SEED=6`, `SYNC=7`, `COMPILE=8`, `CALL=9`, `OK=0`, `ERROR=1`, `SERVER_ID_BASE=1<<62`; classes `Ref(id)`, `NodeRef(index)`, `Dev(name)`, `Host(dtype, shape, data)` with `Host.of(cpu_tensor)` and `Host.tensor()`; exceptions `EncodeError`, `DecodeError`; functions `encode(value) -> bytes`, `decode(bytes) -> value`, `send_frame(sock, payload)`, `recv_frame(sock) -> bytes`, `recv_exact(sock, n) -> bytes`.

- [ ] **Step 1: Create the package skeleton**

`python/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rgpu"
version = "0.1.0"
description = "A PyTorch device whose tensors live on a remote GPU"
requires-python = ">=3.10"
dependencies = ["torch>=2.14"]

[project.optional-dependencies]
test = ["pytest>=8", "torchvision"]

[project.scripts]
rgpu-opserver = "rgpu.server.__main__:main"

[tool.setuptools.packages.find]
include = ["rgpu*"]
```

`python/rgpu/__init__.py` (grows in Task 5):

```python
"""A PyTorch device whose tensors live on a remote GPU."""
```

`python/rgpu/server/__init__.py`:

```python
"""rgpu-opserver: runs PyTorch ops on a GPU host for rgpu clients."""
```

Append to `.gitignore`:

```
python/.venv/
python/*.egg-info/
__pycache__/
```

- [ ] **Step 2: Write the failing tests**

`python/tests/test_wire.py`:

```python
import socket
import struct

import pytest
import torch

from rgpu import wire


def roundtrip(v):
    return wire.decode(wire.encode(v))


@pytest.mark.parametrize("value", [
    None, True, False, 0, -1, 2**62, 3.5, float("inf"), "", "aten::mm",
    b"", b"\x00\x01", [], [1, [2, [3]]], torch.float16, torch.bfloat16,
    torch.int64, torch.bool, torch.strided, torch.channels_last,
    torch.contiguous_format,
])
def test_values_survive(value):
    assert roundtrip(value) == value


def test_tuples_arrive_as_lists():
    assert roundtrip((1, (2, 3))) == [1, [2, 3]]


def test_references_survive():
    assert roundtrip([wire.Ref(7), wire.NodeRef(3)]) == [wire.Ref(7), wire.NodeRef(3)]
    assert roundtrip(wire.Dev("rgpu")).name == "rgpu"


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool, torch.bfloat16])
def test_host_tensor_survives(dtype):
    t = (torch.arange(12) % 3).to(dtype).reshape(3, 4)
    back = roundtrip(wire.Host.of(t)).tensor()
    assert back.dtype == dtype and torch.equal(back, t)


def test_host_takes_a_copy_at_the_call():
    t = torch.zeros(4)
    h = wire.Host.of(t)
    t += 1
    assert torch.equal(roundtrip(h).tensor(), torch.zeros(4))


def test_host_of_non_contiguous_and_empty():
    t = torch.arange(6.0).reshape(2, 3).t()
    assert torch.equal(roundtrip(wire.Host.of(t)).tensor(), t)
    assert roundtrip(wire.Host.of(torch.empty(0, 3))).tensor().shape == (0, 3)


@pytest.mark.parametrize("value", [object(), {1: 2}, {1, 2}, 1j, torch.zeros(1), 2**64])
def test_unsupported_values_raise_where_encoded(value):
    with pytest.raises(wire.EncodeError):
        wire.encode(value)


@pytest.mark.parametrize("data", [
    b"Q",                                     # unknown tag
    b"I\x01",                                 # truncated
    wire.encode(1) + b"x",                    # trailing bytes
    b"L" + struct.pack("<I", 1 << 30),        # list too long
    b"L\x01\x00\x00\x00" * 100 + b"N",        # nested too deeply
    b"Y" + struct.pack("<I", 5) + b"torch",   # not a dtype
])
def test_malformed_frames_raise(data):
    with pytest.raises(wire.DecodeError):
        wire.decode(data)


def test_host_with_inconsistent_length_is_rejected():
    good = wire.encode(wire.Host.of(torch.zeros(4)))
    with pytest.raises(wire.DecodeError):
        wire.decode(good[:-1] + b"")  # shortened payload


def test_frames_over_a_socket():
    a, b = socket.socketpair()
    wire.send_frame(a, b"hello")
    wire.send_frame(a, b"")
    assert wire.recv_frame(b) == b"hello"
    assert wire.recv_frame(b) == b""
    a.close()
    with pytest.raises(ConnectionError):
        wire.recv_frame(b)
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `python/.venv/bin/pytest python/tests/test_wire.py -q`
Expected: collection error, `cannot import name 'wire' from 'rgpu'`.

- [ ] **Step 4: Write `python/rgpu/wire.py`**

```python
"""The wire format shared by the rgpu client and rgpu-opserver.

Frames are length-prefixed. Inside a frame, values use a small tagged
encoding that can carry only these types:

  None, bool, int, float, str, bytes, list (tuples arrive as lists),
  torch.dtype, torch.layout, torch.memory_format,
  Dev     - a device by name; the server decides what "rgpu" means
  Ref     - a tensor held by the server, by id
  Host    - a CPU tensor carried inline: dtype, shape and bytes
  NodeRef - a node of a graph being compiled, by index

Anything else raises EncodeError where it is encoded, so an unsupported
argument fails at the call that passed it. There is deliberately no pickle:
decoding a frame can only ever produce the types above, and decoding enforces
limits so a malformed frame cannot make the reader allocate without bound.
"""

import ctypes
import os
import struct

import torch

MAGIC = b"RGOP"
VERSION = 1

# Message kinds, client to server.
RUN = 1        # op, args, kwargs, output ids; no reply
RUN_SYNC = 2   # op, args, kwargs; the reply describes the outputs
UPLOAD = 3     # destination id, Host; no reply
DOWNLOAD = 4   # id; the reply is a Host
FREE = 5       # list of ids; no reply
SEED = 6       # seed; no reply
SYNC = 7       # the reply comes once everything before it has finished
COMPILE = 8    # graph id, nodes, compiler, mode; no reply
CALL = 9       # graph id, input ids, output ids; no reply

# Reply status.
OK = 0
ERROR = 1

# Ids the server chooses itself, for outputs only it can size. The top bit
# keeps them apart from the client's own.
SERVER_ID_BASE = 1 << 62

MAX_DEPTH = 64
MAX_ITEMS = 1 << 22
MAX_STR = 1 << 20
MAX_FRAME = int(os.environ.get("RGPU_MAX_FRAME", 1 << 34))


class EncodeError(TypeError):
    """A value the wire format cannot carry."""


class DecodeError(ValueError):
    """A frame that is malformed or exceeds a limit."""


class Ref:
    """A tensor held by the server, named by its id."""

    __slots__ = ("id",)

    def __init__(self, id):
        self.id = id

    def __eq__(self, other):
        return isinstance(other, Ref) and other.id == self.id

    def __hash__(self):
        return hash((Ref, self.id))

    def __repr__(self):
        return f"Ref({self.id})"


class NodeRef:
    """A node of a graph being compiled, named by its position."""

    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index

    def __eq__(self, other):
        return isinstance(other, NodeRef) and other.index == self.index

    def __hash__(self):
        return hash((NodeRef, self.index))

    def __repr__(self):
        return f"NodeRef({self.index})"


class Dev:
    """A device by name. The server maps "rgpu" onto whatever it runs on."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Dev({self.name!r})"


class Host:
    """A CPU tensor carried inline."""

    __slots__ = ("dtype", "shape", "data")

    def __init__(self, dtype, shape, data):
        self.dtype = dtype
        self.shape = [int(s) for s in shape]
        self.data = data

    @classmethod
    def of(cls, t):
        c = t.detach().resolve_conj().resolve_neg().contiguous()
        n = c.numel() * c.element_size()
        # Copied now, so a later change to the CPU tensor cannot leak into a
        # message that has not been sent yet.
        data = ctypes.string_at(c.data_ptr(), n) if n else b""
        return cls(c.dtype, c.shape, data)

    def tensor(self):
        if not self.data:
            return torch.empty(self.shape, dtype=self.dtype)
        flat = torch.frombuffer(bytearray(self.data), dtype=self.dtype)
        return flat.reshape(self.shape)


def _table(names):
    out = {}
    for n in names:
        v = getattr(torch, n, None)
        if v is not None:
            out[str(v)] = v
    return out


_DTYPES = _table([
    "float32", "float64", "float16", "bfloat16", "complex64", "complex128",
    "uint8", "int8", "int16", "int32", "int64", "bool", "uint16", "uint32",
    "uint64", "float8_e4m3fn", "float8_e5m2",
])
_LAYOUTS = _table(["strided", "sparse_coo"])
_FORMATS = _table(["contiguous_format", "channels_last", "channels_last_3d",
                   "preserve_format"])


def encode(value):
    out = bytearray()
    _enc(value, out, 0)
    return bytes(out)


def _put_str(tag, s, out):
    b = s.encode()
    out += tag + struct.pack("<I", len(b)) + b


def _enc(v, out, depth):
    if depth > MAX_DEPTH:
        raise EncodeError("value is nested too deeply")
    if v is None:
        out += b"N"
    elif v is True:
        out += b"T"
    elif v is False:
        out += b"F"
    elif isinstance(v, int):
        if not -(1 << 63) <= v < (1 << 63):
            raise EncodeError(f"integer {v} does not fit in 64 bits")
        out += b"I" + struct.pack("<q", v)
    elif isinstance(v, float):
        out += b"D" + struct.pack("<d", v)
    elif isinstance(v, str):
        _put_str(b"S", v, out)
    elif isinstance(v, (bytes, bytearray, memoryview)):
        out += b"B" + struct.pack("<Q", len(v)) + bytes(v)
    elif isinstance(v, (list, tuple)):
        out += b"L" + struct.pack("<I", len(v))
        for x in v:
            _enc(x, out, depth + 1)
    elif isinstance(v, Ref):
        out += b"R" + struct.pack("<Q", v.id)
    elif isinstance(v, NodeRef):
        out += b"G" + struct.pack("<I", v.index)
    elif isinstance(v, Dev):
        _put_str(b"V", v.name, out)
    elif isinstance(v, Host):
        out += b"H"
        _enc(v.dtype, out, depth + 1)
        _enc(v.shape, out, depth + 1)
        _enc(v.data, out, depth + 1)
    elif isinstance(v, torch.dtype):
        _put_str(b"Y", str(v), out)
    elif isinstance(v, torch.layout):
        _put_str(b"A", str(v), out)
    elif isinstance(v, torch.memory_format):
        _put_str(b"M", str(v), out)
    else:
        raise EncodeError(f"cannot send a {type(v).__name__} to the server")


class _Reader:
    def __init__(self, data):
        self.data = memoryview(data)
        self.pos = 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise DecodeError("frame ends early")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.unpack("<I")
        if n > MAX_STR:
            raise DecodeError("string too long")
        return bytes(self.take(n)).decode()


def decode(data):
    r = _Reader(data)
    value = _dec(r, 0)
    if r.pos != len(r.data):
        raise DecodeError("trailing bytes after the value")
    return value


def _lookup(table, name, what):
    try:
        return table[name]
    except KeyError:
        raise DecodeError(f"{name!r} is not a {what}") from None


def _dec(r, depth):
    if depth > MAX_DEPTH:
        raise DecodeError("value is nested too deeply")
    tag = bytes(r.take(1))
    if tag == b"N":
        return None
    if tag == b"T":
        return True
    if tag == b"F":
        return False
    if tag == b"I":
        return r.unpack("<q")
    if tag == b"D":
        return r.unpack("<d")
    if tag == b"S":
        return r.string()
    if tag == b"B":
        return bytes(r.take(r.unpack("<Q")))
    if tag == b"L":
        n = r.unpack("<I")
        if n > MAX_ITEMS:
            raise DecodeError("list too long")
        return [_dec(r, depth + 1) for _ in range(n)]
    if tag == b"R":
        return Ref(r.unpack("<Q"))
    if tag == b"G":
        return NodeRef(r.unpack("<I"))
    if tag == b"V":
        return Dev(r.string())
    if tag == b"H":
        dtype = _dec(r, depth + 1)
        shape = _dec(r, depth + 1)
        data = _dec(r, depth + 1)
        if (not isinstance(dtype, torch.dtype) or not isinstance(data, bytes)
                or not isinstance(shape, list)
                or not all(isinstance(s, int) and s >= 0 for s in shape)):
            raise DecodeError("malformed host tensor")
        numel = 1
        for s in shape:
            numel *= s
        if numel * torch.empty((), dtype=dtype).element_size() != len(data):
            raise DecodeError("host tensor size does not match its data")
        return Host(dtype, shape, data)
    if tag == b"Y":
        return _lookup(_DTYPES, r.string(), "dtype")
    if tag == b"A":
        return _lookup(_LAYOUTS, r.string(), "layout")
    if tag == b"M":
        return _lookup(_FORMATS, r.string(), "memory format")
    raise DecodeError(f"unknown tag {tag!r}")


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def send_frame(sock, payload):
    sock.sendall(struct.pack("<Q", len(payload)))
    if payload:
        sock.sendall(payload)


def recv_frame(sock):
    (n,) = struct.unpack("<Q", recv_exact(sock, 8))
    if n > MAX_FRAME:
        raise DecodeError(f"frame of {n} bytes exceeds the limit")
    return recv_exact(sock, n)
```

- [ ] **Step 5: Create the environment and run the tests**

Run the three commands from **Environment** above, then:
`python/.venv/bin/pytest python/tests/test_wire.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add .gitignore python/pyproject.toml python/rgpu/__init__.py python/rgpu/server/__init__.py python/rgpu/wire.py python/tests/test_wire.py
git commit -m "Start the op-level rgpu package with its wire format

A tagged encoding that can carry only the handful of types an aten op
call needs, so decoding a frame can never produce anything else, and no
pickle. Limits on depth, list length and frame size keep a malformed
frame from making the reader allocate without bound.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 2: Server op lookup and session execution

**Files:**
- Create: `python/rgpu/server/ops.py`, `python/rgpu/server/session.py`
- Test: `python/tests/test_server_session.py`

**Interfaces:**
- Consumes: `rgpu.wire` (Task 1).
- Produces:
  - `rgpu.server.ops.resolve(name: str, overload: str) -> torch._ops.OpOverload`, raising `rgpu.server.ops.Refused`.
  - `rgpu.server.session.Session(device: str)` with `execute(message: list) -> bytes | None`. It returns an encoded reply frame for `RUN_SYNC`, `DOWNLOAD` and `SYNC`, and `None` otherwise. Attributes used by later tasks: `tensors: dict[int, Tensor | Poison]`, `last_seq: int`, `last_reply: tuple[int, bytes] | None`, `graphs: dict`, `lock: threading.Lock`, methods `_arg(v)`, `_device(name)`, `_store(out, ids)`, `_poison(ids, poison)`, `_fail(op, exc, ids)`.
  - `rgpu.server.session.Poison(op, message)`.
- Message layout: every message is `[seq, kind, *fields]`. Replies are `[seq, status, value]`, with `value = [op, message]` when `status == ERROR`.

- [ ] **Step 1: Write the failing tests**

`python/tests/test_server_session.py`:

```python
import torch

from rgpu import wire
from rgpu.server.session import Session


class Client:
    """Drives a Session directly, the way the network layer will."""

    def __init__(self):
        self.s = Session("cpu")
        self.seq = 0

    def send(self, kind, *fields):
        self.seq += 1
        return self.s.execute([self.seq, kind, *fields])

    def reply(self, kind, *fields):
        seq, status, value = wire.decode(self.send(kind, *fields))
        assert seq == self.seq
        return status, value

    def put(self, tid, t):
        self.send(wire.RUN, "aten::empty_strided", "default",
                  [list(t.shape), list(t.stride())],
                  {"dtype": t.dtype, "device": wire.Dev("rgpu")}, [tid])
        self.send(wire.UPLOAD, tid, wire.Host.of(t))

    def get(self, tid):
        status, value = self.reply(wire.DOWNLOAD, tid)
        assert status == wire.OK, value
        return value.tensor()


def test_ops_run_in_order_and_results_come_back():
    c = Client()
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    c.put(1, a)
    c.put(2, b)
    c.send(wire.RUN, "aten::add", "Tensor", [wire.Ref(1), wire.Ref(2)], {}, [3])
    assert torch.allclose(c.get(3), a + b)


def test_cpu_tensors_travel_inline():
    c = Client()
    a = torch.randn(5)
    c.put(1, a)
    c.send(wire.RUN, "aten::mul", "Tensor",
           [wire.Ref(1), wire.Host.of(torch.tensor(2.0))], {}, [2])
    assert torch.allclose(c.get(2), a * 2)


def test_views_alias_their_base_on_the_server():
    c = Client()
    c.put(1, torch.zeros(2, 3))
    c.send(wire.RUN, "aten::t", "default", [wire.Ref(1)], {}, [2])
    c.send(wire.RUN, "aten::add_", "Scalar", [wire.Ref(2), 1.0], {}, [None])
    assert torch.equal(c.get(1), torch.ones(2, 3))


def test_a_failure_poisons_what_depends_on_it_and_is_reported_once():
    c = Client()
    c.put(1, torch.randn(2, 3))
    c.put(2, torch.randn(4, 5))
    c.send(wire.RUN, "aten::mm", "default", [wire.Ref(1), wire.Ref(2)], {}, [3])
    c.send(wire.RUN, "aten::relu", "default", [wire.Ref(3)], {}, [4])
    status, (op, message) = c.reply(wire.SYNC)
    assert status == wire.ERROR and op == "aten::mm.default"
    assert c.reply(wire.SYNC)[0] == wire.OK          # reported once, not twice
    status, (op, _) = c.reply(wire.DOWNLOAD, 4)
    assert status == wire.ERROR and op == "aten::mm.default"  # but it sticks


def test_blocked_and_unknown_ops_are_refused():
    c = Client()
    c.send(wire.RUN, "aten::from_file", "default", ["/etc/passwd"], {}, [1])
    status, (op, message) = c.reply(wire.SYNC)
    assert status == wire.ERROR and "not allowed" in message
    c.send(wire.RUN, "aten::__dict__", "default", [], {}, [2])
    assert c.reply(wire.SYNC)[0] == wire.ERROR
    c.send(wire.RUN, "prims::add", "default", [], {}, [3])
    assert c.reply(wire.SYNC)[0] == wire.ERROR


def test_run_sync_reports_shapes_it_chose_and_keeps_the_tensor():
    c = Client()
    c.put(1, torch.tensor([0.0, 3.0, 0.0, 5.0]))
    status, desc = c.reply(wire.RUN_SYNC, "aten::nonzero", "default", [wire.Ref(1)], {})
    assert status == wire.OK
    tag, tid, dtype, shape, stride, offset = desc
    assert tag == "__tensor__" and tid >= wire.SERVER_ID_BASE
    assert shape == [2, 1] and dtype == torch.int64
    assert torch.equal(c.get(tid), torch.tensor([[1], [3]]))


def test_replayed_messages_are_ignored():
    c = Client()
    c.put(1, torch.zeros(3))
    c.send(wire.RUN, "aten::add_", "Scalar", [wire.Ref(1), 1.0], {}, [None])
    c.seq -= 1   # the same message again, as after a reconnect
    c.send(wire.RUN, "aten::add_", "Scalar", [wire.Ref(1), 1.0], {}, [None])
    assert torch.equal(c.get(1), torch.ones(3))


def test_an_op_returning_nothing_is_not_mistaken_for_a_failure():
    c = Client()
    c.put(1, torch.zeros(3))
    c.put(2, torch.zeros(3))
    c.send(wire.RUN, "aten::_foreach_add_", "Scalar", [[wire.Ref(1), wire.Ref(2)], 1.0], {}, [])
    assert c.reply(wire.SYNC)[0] == wire.OK
    assert torch.equal(c.get(2), torch.ones(3))


def test_free_forgets_tensors():
    c = Client()
    c.put(1, torch.zeros(3))
    c.send(wire.FREE, [1])
    assert c.reply(wire.DOWNLOAD, 1)[0] == wire.ERROR


def test_seed_makes_random_ops_repeat():
    c = Client()
    for tid in (1, 2):
        c.send(wire.SEED, 7)
        c.send(wire.RUN, "aten::randn", "default", [[8]],
               {"dtype": torch.float32, "device": wire.Dev("rgpu")}, [tid])
    assert torch.equal(c.get(1), c.get(2))
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python/.venv/bin/pytest python/tests/test_server_session.py -q`
Expected: `ModuleNotFoundError: No module named 'rgpu.server.session'`.

- [ ] **Step 3: Write `python/rgpu/server/ops.py`**

```python
"""Which ops the server will run, found by name and nothing else.

Only aten ops, looked up through torch.ops.aten by name and overload. There
is no general attribute lookup on torch and nothing a client names is ever
imported. A few aten ops reach outside the GPU; those are refused outright.
"""

import torch

# from_file reads a file from the server's disk into a tensor.
BLOCKED = frozenset({"aten::from_file"})


class Refused(Exception):
    """An op the server will not run."""


def resolve(name, overload):
    if name in BLOCKED:
        raise Refused(f"{name} is not allowed on rgpu-opserver")
    ns, sep, op = name.partition("::")
    if (ns != "aten" or not sep or not op.isidentifier() or op.startswith("__")
            or not overload.isidentifier() or overload.startswith("__")):
        raise Refused(f"{name}.{overload} is not an aten op")
    try:
        found = getattr(getattr(torch.ops.aten, op), overload)
    except (AttributeError, RuntimeError):
        raise Refused(f"no such op {name}.{overload}") from None
    if not isinstance(found, torch._ops.OpOverload):
        raise Refused(f"{name}.{overload} is not an aten op")
    return found
```

- [ ] **Step 4: Write `python/rgpu/server/session.py`**

```python
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
        torch.manual_seed(seed)

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

    _ASYNC = {wire.RUN: _run, wire.UPLOAD: _upload, wire.FREE: _free, wire.SEED: _seed}
    _WAITED = {wire.RUN_SYNC: _run_sync, wire.DOWNLOAD: _download, wire.SYNC: _sync}

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
```

- [ ] **Step 5: Run the tests**

Run: `python/.venv/bin/pytest python/tests/test_server_session.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add python/rgpu/server/ops.py python/rgpu/server/session.py python/tests/test_server_session.py
git commit -m "Run aten ops by name on the server, poisoning what depends on a failure

A session applies messages strictly in order, which is what lets the
client stream ops without waiting. A failed op poisons its outputs, any
op using a poisoned tensor is skipped and poisons its own, and the first
failure is reported once, at the next message the client waits for.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 3: The server process: listener, handshake, sessions

**Files:**
- Create: `python/rgpu/server/__main__.py`, `python/tests/conftest.py`
- Test: `python/tests/test_server_net.py`

**Interfaces:**
- Consumes: `Session` (Task 2), `rgpu.wire` (Task 1).
- Produces:
  - The command `rgpu-opserver [--device cuda|cpu|mps] [--port 9720] [--bind 127.0.0.1]`, also runnable as `python -m rgpu.server`. Environment: `RGPU_SESSION_GRACE` (seconds, default 120) and `RGPU_DROP_AFTER` (a test hook: close the connection on the Nth message, 0 = never).
  - The handshake. The client sends `MAGIC`, then a frame `[VERSION, session_id: bytes(16), last_acked: int, torch_version: str]`. The server replies `[VERSION, resumed: bool, last_seq: int, torch_version: str]`. On resume, if the session's cached reply is newer than `last_acked`, the server sends that reply frame next.
  - After the handshake the client sends frames, each an encoded list of messages. The server writes a reply frame for every message that waits.
  - `tests/conftest.py`: `free_port() -> int`, `start_server(port=None, env=None, device="cpu") -> (subprocess.Popen, "host:port")`, and an autouse session fixture `opserver` that starts a CPU server and sets `RGPU_OPSERVER`, unless `RGPU_OPSERVER` is already set. When it is already set, the tests run against that server; this is how Task 13 runs the suite against a real GPU.

- [ ] **Step 1: Write `python/tests/conftest.py`**

```python
import os
import socket
import subprocess
import sys
import time

import pytest


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port=None, env=None, device="cpu"):
    port = port or free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "rgpu.server", "--device", device, "--port", str(port)],
        env={**os.environ, **(env or {})})
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return proc, f"127.0.0.1:{port}"
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("rgpu-opserver exited while starting")
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("rgpu-opserver did not start within 30 s")


@pytest.fixture(scope="session", autouse=True)
def opserver():
    """A CPU server for the whole run, unless RGPU_OPSERVER names a real one."""
    if os.environ.get("RGPU_OPSERVER"):
        yield None
        return
    proc, address = start_server(device=os.environ.get("RGPU_TEST_DEVICE", "cpu"))
    os.environ["RGPU_OPSERVER"] = address
    yield proc
    proc.terminate()
    proc.wait(timeout=10)
```

- [ ] **Step 2: Write the failing tests**

`python/tests/test_server_net.py`:

```python
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
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `python/.venv/bin/pytest python/tests/test_server_net.py -q`
Expected: failure starting the server, `No module named rgpu.server.__main__`.

- [ ] **Step 4: Write `python/rgpu/server/__main__.py`**

```python
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
```

- [ ] **Step 5: Run the tests**

Run: `python/.venv/bin/pytest python/tests/test_server_net.py python/tests/test_server_session.py python/tests/test_wire.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add python/rgpu/server/__main__.py python/tests/conftest.py python/tests/test_server_net.py
git commit -m "Serve rgpu sessions over TCP, and keep them through a dropped connection

A session is a client process and a connection is one attempt by it to
reach us, so the server keeps a session for a grace period after its
connection goes and hands it back on reconnect. The last reply is kept
too: a request that ran but whose answer was lost is answered again
rather than run twice.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 4: The client connection

**Files:**
- Create: `python/rgpu/session.py`
- Test: `python/tests/test_connection.py`

**Interfaces:**
- Consumes: `rgpu.wire` (Task 1), the handshake and message layout (Task 3).
- Produces:
  - `rgpu.session.Connection(address: str)` with:
    - `post(kind, *fields, size=0) -> None`: queue a message, without waiting.
    - `request(kind, *fields) -> value`: queue, flush and wait. Raises `RemoteError` when the reply is an error.
    - `flush()`.
    - `stats: dict` with keys `messages`, `waits`, `bytes_out`, `bytes_in`.
  - `rgpu.session.get() -> Connection`: one per process, created on first use from `RGPU_OPSERVER` (default `127.0.0.1:9720`), and recreated after `fork`. `rgpu.session.reset()` drops it; tests use this.
  - `rgpu.session.stats() -> dict`: a copy of `get().stats`.
  - `rgpu.session.pending_frees`: a `collections.deque` of tensor ids. Finalizers append to it, and the connection sends them as one `FREE` at the next safe point, never from inside a finalizer.
  - Exceptions `RemoteError(op, message)` (a `RuntimeError`) and `SessionLost` (a `RuntimeError`).
  - Replay state used by Task 12: `unacked: deque[list]`, `unacked_bytes: int`, `replay_possible: bool`, `last_acked: int`.

- [ ] **Step 1: Write the failing tests**

`python/tests/test_connection.py`:

```python
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python/.venv/bin/pytest python/tests/test_connection.py -q`
Expected: `ImportError: cannot import name 'session' from 'rgpu'`.

- [ ] **Step 3: Write `python/rgpu/session.py`**

```python
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

    def _flush(self):
        if not self.pending:
            return
        if self.sock is None:
            self._connect()
        frame = wire.encode(self.pending)
        wire.send_frame(self.sock, frame)
        self.stats["bytes_out"] += len(frame)
        self.pending = []
        self.pending_bytes = 0

    def _await(self, seq):
        while True:
            frame = wire.recv_frame(self.sock)
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
            self.unacked_bytes -= 64
            self.unacked.popleft()
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
```

- [ ] **Step 4: Run the tests**

Run: `python/.venv/bin/pytest python/tests/test_connection.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add python/rgpu/session.py python/tests/test_connection.py
git commit -m "Connect to rgpu-opserver, streaming most messages and waiting on few

Messages go out in batches, flushed every 64 or 256 KB and at every
wait, so the GPU works while the client is still issuing. Only replies
cost a round trip. Finalizers never touch the socket: they queue ids,
and the frees go out with the next batch.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 5: The `rgpu` device: RemoteTensor, factories, uploads, downloads, streaming

**Files:**
- Create: `python/rgpu/tensor.py`, `python/rgpu/device.py`, `python/rgpu/dispatch.py`
- Modify: `python/rgpu/__init__.py`
- Test: `python/tests/test_device.py`

**Interfaces:**
- Consumes: `rgpu.session.get()`, `rgpu.session.pending_frees` (Task 4), `rgpu.wire` (Task 1).
- Produces:
  - `rgpu.tensor.RemoteTensor(meta)`: a wrapper subclass reporting `device=rgpu:0`. Its meta tensor is kept on `._rgpu_meta`. It is traceable through `__tensor_flatten__`, which returns `["_rgpu_meta"]`.
  - `rgpu.tensor.register(meta, tid=None) -> int`: stores the id on `meta._rgpu_id`, sets a finalizer that queues a free, and returns the id.
  - `rgpu.tensor.id_of(meta) -> int`.
  - `rgpu.tensor.meta_like(dtype, shape, stride, offset) -> meta Tensor`.
  - `rgpu.dispatch.handle(func, args, kwargs)`, and the helpers `wire_of(x)` and `meta_of(x)` that Task 6 and Task 11 reuse.
  - `rgpu.device.register_device()`, `rgpu.device.synchronize()`, `rgpu.device.manual_seed(seed)`.
  - `import rgpu` registers the device and exports `RemoteError`, `SessionLost` and `stats`.

- [ ] **Step 1: Write the failing tests**

`python/tests/test_device.py`:

```python
import pytest
import torch

import rgpu


def test_a_tensor_round_trips_exactly():
    x = torch.arange(1024, dtype=torch.float32)
    assert torch.equal(x.to("rgpu").cpu(), x)


def test_tensors_report_the_rgpu_device():
    t = torch.randn(3, 4, device="rgpu")
    assert t.device == torch.device("rgpu", 0) and t.shape == (3, 4)
    assert torch.rgpu.is_available() and torch.rgpu.device_count() == 1


@pytest.mark.parametrize("make, expect", [
    (lambda: torch.zeros(5, device="rgpu"), torch.zeros(5)),
    (lambda: torch.ones(2, 3, device="rgpu"), torch.ones(2, 3)),
    (lambda: torch.full((4,), 7.0, device="rgpu"), torch.full((4,), 7.0)),
    (lambda: torch.arange(10, device="rgpu"), torch.arange(10)),
    (lambda: torch.linspace(0, 1, 5, device="rgpu"), torch.linspace(0, 1, 5)),
    (lambda: torch.eye(3, device="rgpu"), torch.eye(3)),
    (lambda: torch.tensor([1, 2, 3], device="rgpu"), torch.tensor([1, 2, 3])),
])
def test_factories_make_the_right_values(make, expect):
    t = make()
    assert t.shape == expect.shape and torch.equal(t.cpu(), expect)


def test_elementwise_reduction_and_matmul_match_cpu():
    a, b = torch.randn(64, 32), torch.randn(32, 16)
    ra, rb = a.to("rgpu"), b.to("rgpu")
    assert torch.allclose((ra * 2 + 1).relu().cpu(), (a * 2 + 1).relu())
    assert torch.allclose(ra.sum().cpu(), a.sum(), atol=1e-4)
    assert torch.allclose((ra @ rb).cpu(), a @ b, atol=1e-4)


def test_item_returns_a_python_number():
    assert torch.tensor([2.5, 0.5]).to("rgpu").sum().item() == 3.0


def test_dtype_conversion_happens_on_the_device():
    x = torch.randn(8)
    half = x.to("rgpu").to(torch.float16)
    assert half.dtype == torch.float16 and half.device.type == "rgpu"
    assert torch.equal(half.cpu(), x.half())


def test_cpu_scalars_can_join_rgpu_ops():
    x = torch.randn(6)
    assert torch.allclose((x.to("rgpu") + torch.tensor(2.0)).cpu(), x + 2)


def test_a_shape_error_raises_here_without_asking_the_server():
    a = torch.randn(2, 3, device="rgpu")
    b = torch.randn(4, 5, device="rgpu")
    waits = rgpu.stats()["waits"]
    with pytest.raises(RuntimeError, match="must have same reduction dim"):
        a @ b
    assert rgpu.stats()["waits"] == waits


def test_an_upload_takes_the_bytes_at_the_call():
    x = torch.zeros(4)
    r = x.to("rgpu")
    x += 1
    assert torch.equal(r.cpu(), torch.zeros(4))


def test_streaming_ops_do_not_wait():
    x = torch.randn(128, 128, device="rgpu")
    waits = rgpu.stats()["waits"]
    y = (x @ x).relu().sum(dim=0)
    assert rgpu.stats()["waits"] == waits
    y.cpu()
    assert rgpu.stats()["waits"] == waits + 1


def test_seeding_makes_random_tensors_repeat():
    torch.manual_seed(3)
    a = torch.randn(16, device="rgpu").cpu()
    torch.manual_seed(3)
    b = torch.randn(16, device="rgpu").cpu()
    assert torch.equal(a, b)


def test_repr_shows_the_values():
    text = repr(torch.tensor([1.5, 2.5]).to("rgpu"))
    assert "1.5" in text and "rgpu" in text
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python/.venv/bin/pytest python/tests/test_device.py -q`
Expected: failures such as `RuntimeError: Expected one of cpu, cuda, ... device type at start of device string: rgpu`.

- [ ] **Step 3: Write `python/rgpu/tensor.py`**

```python
"""RemoteTensor: a tensor whose data lives on the server.

On this side a RemoteTensor is only metadata - a meta tensor holding shape,
dtype and strides - plus an id naming the real tensor on the server. The id
is kept on the meta tensor rather than on the wrapper, because compiled code
works on the unwrapped meta tensors: rewrapping one gets its id back.
"""

import itertools
import weakref

import torch

from . import session

_next_id = itertools.count(1)


def register(meta, tid=None):
    """Give a meta tensor an id, and free it on the server when it dies."""
    if tid is None:
        tid = next(_next_id)
    meta._rgpu_id = tid
    weakref.finalize(meta, session.pending_frees.append, tid)
    return tid


def id_of(meta):
    try:
        return meta._rgpu_id
    except AttributeError:
        raise RuntimeError("this tensor was never sent to the server") from None


def meta_like(dtype, shape, stride, offset=0):
    """A meta tensor with exactly this layout, offset included."""
    if offset == 0 or any(s == 0 for s in shape):
        return torch.empty_strided(shape, stride, dtype=dtype, device="meta")
    extent = offset + 1 + sum((s - 1) * st for s, st in zip(shape, stride))
    return torch.empty(extent, dtype=dtype, device="meta").as_strided(shape, stride, offset)


class RemoteTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, meta, requires_grad=False):
        r = torch.Tensor._make_wrapper_subclass(
            cls, meta.shape, strides=meta.stride(), storage_offset=meta.storage_offset(),
            dtype=meta.dtype, device=torch.device("rgpu", 0), requires_grad=requires_grad)
        r._rgpu_meta = meta
        return r

    def __init__(self, meta, requires_grad=False):
        pass

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from . import dispatch
        return dispatch.handle(func, args, kwargs or {})

    # Traceable, so torch.compile sees one graph rather than one per op: the
    # compiler traces with the meta tensor, which is all it needs.
    def __tensor_flatten__(self):
        return ["_rgpu_meta"], None

    @staticmethod
    def __tensor_unflatten__(inner, ctx, outer_size, outer_stride):
        return RemoteTensor(inner["_rgpu_meta"])

    def __repr__(self, *, tensor_contents=None):
        return "rgpu:" + repr(self.detach().cpu())
```

- [ ] **Step 4: Write `python/rgpu/dispatch.py`**

```python
"""Where every op on an rgpu tensor lands, after autograd.

Each op is one of four kinds, and only two of them wait for the server:

  stream     outputs worked out here on meta tensors, ids chosen here, the op
             queued - no wait. Almost everything.
  upload     a CPU tensor copied into an rgpu one; the bytes are taken now,
             so later changes to the CPU tensor cannot leak in - no wait.
  download   .cpu(), .item(), printing - one round trip.
  unknown    ops whose output size depends on the data - the server runs
             them and reports the shapes, one round trip. (Task 6.)

Meta kernels make the same shape and dtype checks as real ones, so most
mistakes raise here, on the line that made them, before anything is sent.
"""

import torch
from torch.utils._pytree import tree_flatten, tree_map

from . import session, wire
from .tensor import RemoteTensor, id_of, register

aten = torch.ops.aten


def meta_of(x):
    if isinstance(x, RemoteTensor):
        return x._rgpu_meta
    if isinstance(x, torch.Tensor):
        return x.to("meta")
    if isinstance(x, torch.device) and x.type == "rgpu":
        return torch.device("meta")
    return x


def wire_of(x):
    if isinstance(x, RemoteTensor):
        return wire.Ref(id_of(x._rgpu_meta))
    if isinstance(x, torch.Tensor):
        if x.device.type != "cpu":
            raise RuntimeError(f"cannot mix a {x.device} tensor with rgpu tensors")
        return wire.Host.of(x)
    if isinstance(x, torch.device):
        return wire.Dev("rgpu" if x.type == "rgpu" else str(x))
    return x


def _wire_args(args, kwargs):
    return tree_map(wire_of, list(args)), {k: tree_map(wire_of, v) for k, v in kwargs.items()}


def _download(t):
    return session.get().request(wire.DOWNLOAD, id_of(t._rgpu_meta)).tensor()


def _written_input(func, args, kwargs, ret):
    """The input an in-place or out= op writes to and returns."""
    want = ret.alias_info.before_set
    for i, a in enumerate(func._schema.arguments):
        if a.alias_info is not None and a.alias_info.is_write and a.alias_info.before_set == want:
            return args[i] if i < len(args) else kwargs[a.name]
    raise RuntimeError(f"cannot tell which input {func} writes to")


def _wrap_outputs(func, args, kwargs, out):
    """Wrap meta outputs, choosing ids, in the order the server will flatten them."""
    returns = func._schema.returns
    if not returns:
        return None, []
    single = len(returns) == 1
    outs = [out] if single else list(out)
    results, out_ids = [], []
    for ret, o in zip(returns, outs):
        if ret.alias_info is not None and ret.alias_info.is_write:
            results.append(_written_input(func, args, kwargs, ret))
            out_ids.extend([None] * len(tree_flatten(o)[0]))
            continue

        def one(m):
            if not isinstance(m, torch.Tensor):
                out_ids.append(None)
                return m
            if hasattr(m, "_rgpu_id"):
                out_ids.append(None)   # the op handed back one of its inputs
            else:
                out_ids.append(register(m))
            return RemoteTensor(m)

        results.append(tree_map(one, o))
    return (results[0] if single else tuple(results)), out_ids


def _run(func, args, kwargs):
    out = func(*tree_map(meta_of, args), **tree_map(meta_of, kwargs))
    results, out_ids = _wrap_outputs(func, args, kwargs, out)
    a, kw = _wire_args(args, kwargs)
    session.get().post(wire.RUN, func._schema.name, func._overloadname, a, kw, out_ids)
    return results


def handle(func, args, kwargs):
    if func is aten._local_scalar_dense.default:
        return _download(args[0]).item()
    if func is aten._to_copy.default:
        target = kwargs.get("device")
        if target is not None and torch.device(target).type == "cpu":
            out = _download(args[0])
            dtype = kwargs.get("dtype")
            return out if dtype is None else out.to(dtype)
    if func is aten.copy_.default:
        dst, src = args[0], args[1]
        if not isinstance(dst, RemoteTensor):
            return dst.copy_(_download(src))
        if not isinstance(src, RemoteTensor):
            session.get().post(wire.UPLOAD, id_of(dst._rgpu_meta), wire.Host.of(src),
                               size=src.numel() * src.element_size())
            return dst
    return _run(func, args, kwargs)
```

- [ ] **Step 5: Write `python/rgpu/device.py`**

```python
"""Makes "rgpu" a PyTorch device, from Python alone.

PyTorch reserves one device slot for backends outside it (PrivateUse1). Its
Python-backend hook, marked experimental, registers the device guard and
hooks a backend would otherwise have to write in C++.

Factory functions - torch.zeros(..., device="rgpu") and the like - have no
tensor argument to dispatch on, so they need kernels of their own. Every
factory op is registered, not just empty: ops such as arange build an empty
tensor and resize it through out=, and a wrapper's shape is fixed when it is
made, so going through empty would leave it describing a size-zero tensor.
"""

import types

import torch

from . import session, wire
from .tensor import RemoteTensor, register

_SKIP_FACTORIES = ("sparse", "quantized", "cudnn", "mkldnn", "nested", "_make_dep_token",
                   "from_file")


def synchronize(device=None):
    session.get().request(wire.SYNC)


def manual_seed(seed):
    session.get().post(wire.SEED, int(seed))


def _set_device(device):
    index = device.index if isinstance(device, torch.device) else device
    if index not in (0, None):
        raise ValueError("rgpu has one device, rgpu:0")


def _module():
    m = types.ModuleType("torch.rgpu")
    m.is_available = lambda: True
    m.device_count = lambda: 1
    m.current_device = lambda: 0
    m.set_device = _set_device
    m.is_initialized = lambda: True
    m._is_in_bad_fork = lambda: False
    m.synchronize = synchronize
    m.manual_seed = manual_seed
    m.manual_seed_all = manual_seed
    m.get_amp_supported_dtype = lambda: [torch.float16, torch.bfloat16]
    return m


def _factory_kernel(op):
    name, overload = op._schema.name, op._overloadname

    def kernel(*args, **kwargs):
        meta_kwargs = dict(kwargs)
        meta_kwargs["device"] = torch.device("meta")
        if "pin_memory" in meta_kwargs:
            meta_kwargs["pin_memory"] = None
        meta = op(*args, **meta_kwargs)
        wire_kwargs = {k: v for k, v in kwargs.items() if k != "pin_memory"}
        wire_kwargs["device"] = wire.Dev("rgpu")
        session.get().post(wire.RUN, name, overload, list(args), wire_kwargs, [register(meta)])
        return RemoteTensor(meta)

    return kernel


def _factories():
    """Every aten overload with a device argument and no tensor arguments."""
    for full in torch._C._dispatch_get_all_op_names():
        if not full.startswith("aten::") or any(s in full for s in _SKIP_FACTORIES):
            continue
        name, _, overload = full[len("aten::"):].partition(".")
        try:
            op = getattr(getattr(torch.ops.aten, name), overload or "default")
        except (AttributeError, RuntimeError):
            continue
        schema = op._schema
        if len(schema.returns) != 1 or "Tensor" not in str(schema.returns[0].type):
            continue
        if not any(a.name == "device" for a in schema.arguments):
            continue
        if any("Tensor" in str(a.type) for a in schema.arguments):
            continue
        yield full[len("aten::"):], op


_libs = []


def register_device():
    if _libs:
        return
    torch.utils.backend_registration._setup_privateuseone_for_python_backend(
        "rgpu", backend_module=_module())
    lib = torch.library.Library("aten", "IMPL")
    for qualified, op in _factories():
        lib.impl(qualified, _factory_kernel(op), "PrivateUse1")
    _libs.append(lib)   # registrations live only as long as the Library object
```

- [ ] **Step 6: Replace `python/rgpu/__init__.py`**

```python
"""A PyTorch device whose tensors live on a remote GPU.

    import rgpu, torch
    x = torch.randn(1024, 1024, device="rgpu")   # the data is on the server
    y = (x @ x).relu().sum().item()               # one round trip, at .item()

Set RGPU_OPSERVER=host:port to reach rgpu-opserver (default 127.0.0.1:9720),
normally through an ssh tunnel: the server has no authentication.
"""

from .device import register_device as _register_device

_register_device()

from .session import RemoteError, SessionLost, stats  # noqa: E402

__all__ = ["RemoteError", "SessionLost", "stats"]
```

- [ ] **Step 7: Run the tests**

Run: `python/.venv/bin/pytest python/tests -q`
Expected: every test so far passes. If a factory test fails, run it alone with `-x` and check which overload `_factories()` missed (`python/.venv/bin/python -c "import rgpu.device as d; print(sorted(n for n, _ in d._factories()))"`).

- [ ] **Step 8: Commit**

```bash
git add python/rgpu/__init__.py python/rgpu/tensor.py python/rgpu/device.py python/rgpu/dispatch.py python/tests/test_device.py
git commit -m "Make rgpu a PyTorch device whose tensors live on the server

A RemoteTensor holds a meta tensor and an id. Every op after autograd
has its outputs worked out here on meta, gets ids chosen here, and is
streamed without waiting; only downloads wait. Shape errors raise on the
line that made them, before anything is sent. Every factory op is
registered for the device, because arange and friends resize an empty
tensor through out= and a wrapper's shape cannot follow.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 6: Views, in-place ops, and ops whose size depends on the data

**Files:**
- Modify: `python/rgpu/dispatch.py`
- Test: `python/tests/test_aliasing.py`

**Interfaces:**
- Consumes: `_run`, `_wrap_outputs`, `_written_input`, `_wire_args` (Task 5); `rgpu.tensor.meta_like`, `register` (Task 5); `wire.RUN_SYNC` and the server's `["__tensor__", id, dtype, shape, stride, offset]` output description (Task 2).
- Produces: `rgpu.dispatch._run_sync(func, args, kwargs)`. Changes to `handle` and `_run`:
  - Metadata-mutating in-place ops are refused with `NotImplementedError`.
  - `aten.equal` always waits.
  - When meta raises `NotImplementedError`, an op that returns only its written inputs is streamed without shape inference. Any other op goes to `_run_sync`.

- [ ] **Step 1: Write the failing tests**

`python/tests/test_aliasing.py`:

```python
import pytest
import torch

import rgpu


def test_an_in_place_op_on_a_view_changes_the_base():
    a = torch.randn(3, 4)
    r = a.to("rgpu")
    r.view(-1).add_(1)
    r[1].mul_(0)
    r.t()[0].fill_(5)
    expect = a.clone()
    expect.view(-1).add_(1)
    expect[1].mul_(0)
    expect.t()[0].fill_(5)
    assert torch.equal(r.cpu(), expect)


def test_in_place_ops_return_the_same_object():
    r = torch.zeros(3, device="rgpu")
    assert r.add_(1) is r


def test_out_variants_write_to_the_given_tensor():
    x, y = torch.randn(4), torch.randn(4)
    out = torch.empty(4, device="rgpu")
    result = torch.add(x.to("rgpu"), y.to("rgpu"), out=out)
    assert result is out and torch.allclose(out.cpu(), x + y)


def test_split_chunks_are_views_of_the_base():
    r = torch.zeros(6, device="rgpu")
    first, second = r.chunk(2)
    second.fill_(1)
    assert torch.equal(r.cpu(), torch.tensor([0.0, 0, 0, 1, 1, 1]))


def test_expand_and_broadcast():
    a = torch.randn(1, 4)
    assert torch.allclose(a.to("rgpu").expand(3, 4).sum(0).cpu(), a.expand(3, 4).sum(0))


@pytest.mark.parametrize("op", [
    lambda t: torch.nonzero(t),
    lambda t: t[t > 0],
    lambda t: torch.unique(t),
    lambda t: torch.masked_select(t, t > 0),
])
def test_ops_whose_size_depends_on_the_data(op):
    x = torch.tensor([0.0, 3.0, -1.0, 3.0, 5.0])
    got = op(x.to("rgpu"))
    want = op(x)
    assert got.shape == want.shape and torch.equal(got.cpu(), want)


def test_the_output_of_a_data_dependent_op_streams_onward():
    x = torch.tensor([0.0, 3.0, 5.0]).to("rgpu")
    picked = x[x > 0]
    waits = rgpu.stats()["waits"]
    doubled = picked * 2
    assert rgpu.stats()["waits"] == waits
    assert torch.equal(doubled.cpu(), torch.tensor([6.0, 10.0]))


def test_equal_asks_the_server():
    a = torch.tensor([1.0, 2.0])
    assert torch.equal(a.to("rgpu"), a.to("rgpu"))
    assert not torch.equal(a.to("rgpu"), (a + 1).to("rgpu"))


def test_changing_shape_in_place_is_refused_clearly():
    r = torch.zeros(2, 3, device="rgpu")
    with pytest.raises(NotImplementedError, match="changes a tensor's shape"):
        r.t_()
```

- [ ] **Step 2: Run the tests to see which fail**

Run: `python/.venv/bin/pytest python/tests/test_aliasing.py -q`
Expected: the data-dependent tests fail with `NotImplementedError` from meta, `test_equal_asks_the_server` fails with `RuntimeError: Tensor.item() cannot be called on meta tensors`, and `t_` fails some other way. The view and in-place tests may already pass, since Task 5's `_wrap_outputs` handles aliasing. Keep them: they pin that behaviour down.

- [ ] **Step 3: Extend `python/rgpu/dispatch.py`**

Add these definitions below `aten = torch.ops.aten`:

```python
# In-place ops that change a tensor's shape or strides rather than its
# values. A wrapper's metadata is fixed when it is made, so these would leave
# it describing the wrong tensor. Refused, clearly, until that is handled.
_METADATA_MUTATING = frozenset({
    "resize_", "resize_as_", "set_", "as_strided_", "t_", "transpose_", "squeeze_",
    "unsqueeze_", "swapaxes_", "swapdims_",
})

# Ops whose result depends on the data, but whose meta kernel says so with a
# RuntimeError rather than NotImplementedError. They always wait.
_ALWAYS_SYNC = frozenset({aten.equal.default})
```

Add these functions above `handle`:

```python
def _returns_only_inputs(func):
    returns = func._schema.returns
    return all(r.alias_info is not None and r.alias_info.is_write for r in returns)


def _run_sync(func, args, kwargs):
    """The server runs it and says what came out. Output ids are the server's."""
    if not _returns_only_inputs(func) and any(
            r.alias_info is not None and r.alias_info.is_write for r in func._schema.returns):
        raise NotImplementedError(
            f"rgpu cannot run {func}: it writes to an input and returns a new tensor "
            "whose size only the data can tell")
    a, kw = _wire_args(args, kwargs)
    value = session.get().request(wire.RUN_SYNC, func._schema.name, func._overloadname, a, kw)

    def rebuild(v):
        if isinstance(v, list) and v and v[0] == "__tensor__":
            _, tid, dtype, shape, stride, offset = v
            meta = meta_like(dtype, shape, stride, offset)
            register(meta, tid)
            return RemoteTensor(meta)
        if isinstance(v, list):
            return [rebuild(x) for x in v]
        return v

    out = rebuild(value)
    return tuple(out) if len(func._schema.returns) > 1 else out
```

Update the import line for `rgpu.tensor`:

```python
from .tensor import RemoteTensor, id_of, meta_like, register
```

Replace `_run` with:

```python
def _run(func, args, kwargs):
    try:
        out = func(*tree_map(meta_of, args), **tree_map(meta_of, kwargs))
    except NotImplementedError:
        if not _returns_only_inputs(func):
            return _run_sync(func, args, kwargs)
        # No meta kernel, but nothing new comes out, so there is no shape to
        # infer: send it and hand back the inputs it wrote to.
        returns = func._schema.returns
        results = [_written_input(func, args, kwargs, r) for r in returns]
        a, kw = _wire_args(args, kwargs)
        out_ids = [None] * sum(len(tree_flatten(r)[0]) for r in results)
        session.get().post(wire.RUN, func._schema.name, func._overloadname, a, kw, out_ids)
        if not returns:
            return None
        return results[0] if len(results) == 1 else tuple(results)
    results, out_ids = _wrap_outputs(func, args, kwargs, out)
    a, kw = _wire_args(args, kwargs)
    session.get().post(wire.RUN, func._schema.name, func._overloadname, a, kw, out_ids)
    return results
```

At the top of `handle`, before the `_local_scalar_dense` check, add:

```python
    base = func._schema.name.split("::", 1)[1]
    if base in _METADATA_MUTATING:
        raise NotImplementedError(
            f"rgpu cannot run {func} yet: it changes a tensor's shape in place")
    if func in _ALWAYS_SYNC:
        return _run_sync(func, args, kwargs)
```

- [ ] **Step 4: Run the tests**

Run: `python/.venv/bin/pytest python/tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add python/rgpu/dispatch.py python/tests/test_aliasing.py
git commit -m "Keep views aliased, and ask the server when only the data knows the size

View ops run on the server too, so its aliasing is exactly PyTorch's and
an in-place op on a view changes the base. Ops like nonzero and boolean
masks cannot be sized on meta, so they wait while the server runs them
and reports the shapes; what comes out streams onward as usual.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 7: Server memory follows client lifetimes

**Files:**
- Test: `python/tests/test_memory.py`
- Modify (only if the test fails): `python/rgpu/tensor.py`

**Interfaces:**
- Consumes: `rgpu.tensor.register`/`id_of`, `rgpu.session.pending_frees`, `rgpu.session.get()`, `torch.rgpu.synchronize` (Tasks 4–5).
- Produces: nothing new. This task pins down that frees reach the server, which Task 5 built but did not test.

- [ ] **Step 1: Write the tests**

`python/tests/test_memory.py`:

```python
import gc

import pytest
import torch

import rgpu
from rgpu import session, wire
from rgpu.tensor import id_of


def test_a_collected_tensor_is_freed_on_the_server():
    t = torch.ones(1000, device="rgpu")
    tid = id_of(t._rgpu_meta)
    assert torch.equal(session.get().request(wire.DOWNLOAD, tid).tensor(), torch.ones(1000))
    del t
    gc.collect()
    torch.rgpu.synchronize()   # a safe point: the free goes out first
    with pytest.raises(rgpu.RemoteError, match="no tensor"):
        session.get().request(wire.DOWNLOAD, tid)


def test_a_view_keeps_the_storage_after_its_base_is_freed():
    base = torch.arange(6.0).to("rgpu")
    view = base[2:]
    del base
    gc.collect()
    torch.rgpu.synchronize()
    assert torch.equal(view.cpu(), torch.arange(2.0, 6.0))


def test_a_long_chain_of_freed_temporaries_stays_correct():
    x = torch.randn(256, device="rgpu")
    for _ in range(200):
        x = x * 1.0 + 0.0
    gc.collect()
    torch.rgpu.synchronize()
    ids = [id_of(x._rgpu_meta)]
    assert torch.allclose(session.get().request(wire.DOWNLOAD, ids[0]).tensor(), x.cpu())
```

- [ ] **Step 2: Run the tests**

Run: `python/.venv/bin/pytest python/tests/test_memory.py -q`
Expected: all pass. If `test_a_collected_tensor_is_freed_on_the_server` fails because the id is still downloadable, the finalizer is not firing. Check that nothing keeps the meta tensor alive: the wrapper holds it, and `_wrap_outputs` must not store it anywhere else. Fix that in `rgpu/tensor.py`.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_memory.py
git commit -m "Check that tensors collected here are freed on the server

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 8: The inference ladder, up to ResNet-18

**Files:**
- Test: `python/tests/test_ladder.py`
- Modify (as the tests demand): `python/rgpu/dispatch.py`

**Interfaces:**
- Consumes: the device (Tasks 5–6). `torchvision` must be installed in `python/.venv`.
- Produces: nothing new. This is where ops missing from the dispatcher show up. Fix each in `dispatch.py` with the smallest change, and add a test to `test_aliasing.py` or `test_device.py` for anything that is a new kind of op rather than just a new op.

- [ ] **Step 1: Write the tests**

`python/tests/test_ladder.py`:

```python
import torch
import torch.nn as nn


def test_convolution_matches_cpu():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    want = conv(x)
    got = conv.to("rgpu")(x.to("rgpu")).cpu()
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)


def test_pooling_and_batch_norm_in_eval_mode():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(),
                        nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    x = torch.randn(4, 3, 16, 16)
    with torch.no_grad():
        want = net(x)
        got = net.to("rgpu")(x.to("rgpu")).cpu()
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)


def test_resnet18_inference_matches_cpu():
    import torchvision.models as models
    torch.manual_seed(0)
    net = models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        want = net(x)
        got = net.to("rgpu")(x.to("rgpu")).cpu()
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4), (got - want).abs().max()


def test_resnet18_inference_waits_once():
    import rgpu
    import torchvision.models as models
    net = models.resnet18(weights=None).eval().to("rgpu")
    x = torch.randn(1, 3, 224, 224).to("rgpu")
    with torch.no_grad():
        net(x)                        # warm up
        waits = rgpu.stats()["waits"]
        net(x).cpu()
    assert rgpu.stats()["waits"] - waits == 1, "inference should wait only for .cpu()"
```

- [ ] **Step 2: Run the tests and fix what they find**

Run: `python/.venv/bin/pytest python/tests/test_ladder.py -q -x`
Expected: pass. If an op fails, run the failing test with `-x --tb=long`. Find the op in the traceback, and fix `dispatch.py` for that class of op, not for that op alone. For example, if an op returns a tuple containing a `None`, the fix belongs in `_wrap_outputs`. Re-run the whole suite after each fix.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_ladder.py python/rgpu/dispatch.py
git commit -m "Run ResNet-18 inference on rgpu, matching CPU, waiting once

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 9: Training, and the one-round-trip claim

**Files:**
- Test: `python/tests/test_training.py`
- Modify (as the tests demand): `python/rgpu/dispatch.py`

**Interfaces:**
- Consumes: the device (Tasks 5–8).
- Produces: nothing new. The last test checks the spec's headline claim, that eager training costs about one round trip per step, without needing a GPU.

- [ ] **Step 1: Write the tests**

`python/tests/test_training.py`:

```python
import torch
import torch.nn as nn

import rgpu


def paired(make, seed=0):
    torch.manual_seed(seed)
    cpu = make()
    torch.manual_seed(seed)
    return cpu, make().to("rgpu")


def grads_match(cpu, remote, rtol=1e-4, atol=1e-5):
    for (name, a), (_, b) in zip(cpu.named_parameters(), remote.named_parameters()):
        g = b.grad.cpu()
        assert g.abs().sum() > 0 or a.grad.abs().sum() == 0, f"{name} came back all zeros"
        assert torch.allclose(g, a.grad, rtol=rtol, atol=atol), f"{name} differs"


def test_backward_through_linear_and_convolution():
    for make, shape in [(lambda: nn.Linear(64, 32), (8, 64)),
                        (lambda: nn.Conv2d(3, 8, 3, padding=1), (2, 3, 16, 16))]:
        cpu, remote = paired(make)
        x = torch.randn(*shape)
        cpu(x).pow(2).mean().backward()
        remote(x.to("rgpu")).pow(2).mean().backward()
        grads_match(cpu, remote)


def test_batch_norm_in_training_mode():
    cpu, remote = paired(lambda: nn.BatchNorm2d(8))
    x = torch.randn(4, 8, 8, 8)
    cpu(x).pow(2).mean().backward()
    remote(x.to("rgpu")).pow(2).mean().backward()
    grads_match(cpu, remote)
    assert torch.allclose(remote.running_mean.cpu(), cpu.running_mean, atol=1e-6)


def test_sgd_and_adam_steps_match_cpu():
    for make_opt in [lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
                     lambda p: torch.optim.Adam(p, lr=1e-2)]:
        cpu, remote = paired(lambda: nn.Linear(32, 32))
        oc, orr = make_opt(cpu.parameters()), make_opt(remote.parameters())
        x = torch.randn(8, 32)
        for _ in range(3):
            oc.zero_grad()
            cpu(x).sum().backward()
            oc.step()
            orr.zero_grad()
            remote(x.to("rgpu")).sum().backward()
            orr.step()
        for a, b in zip(cpu.parameters(), remote.parameters()):
            assert torch.allclose(b.detach().cpu(), a.detach(), atol=1e-5)


def test_resnet18_training_step_matches_cpu():
    import torchvision.models as models
    cpu, remote = paired(lambda: models.resnet18(weights=None))
    x, y = torch.randn(2, 3, 64, 64), torch.randint(0, 1000, (2,))
    lc = nn.functional.cross_entropy(cpu(x), y)
    lr = nn.functional.cross_entropy(remote(x.to("rgpu")), y.to("rgpu"))
    lc.backward()
    lr.backward()
    assert abs(lr.item() - lc.item()) < 1e-4
    grads_match(cpu, remote, rtol=1e-3, atol=1e-4)


def test_loss_falls_over_ten_steps():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).to("rgpu")
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    x, y = torch.randn(32, 64).to("rgpu"), torch.randint(0, 10, (32,)).to("rgpu")
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_an_eager_training_step_waits_once():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).to("rgpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = torch.randn(32, 64).to("rgpu"), torch.randint(0, 10, (32,)).to("rgpu")

    def step():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss.item()

    step()   # warm up: the first step creates optimizer state
    waits = rgpu.stats()["waits"]
    step()
    assert rgpu.stats()["waits"] - waits == 1, "a step should wait only for loss.item()"
```

- [ ] **Step 2: Run the tests and fix what they find**

Run: `python/.venv/bin/pytest python/tests/test_training.py -q -x`
Expected: pass. If `test_an_eager_training_step_waits_once` reports more than one wait, count them by op. Temporarily wrap `session.Connection.request` to print `fields[:2]`, then find which op forced the wait. The usual cause is a data-dependent op inside an optimizer, reached through `_run_sync` or a download. Fix it in `dispatch.py` when an equivalent streaming form exists. If it doesn't, change the assertion to the measured count and add a comment naming the op and why it waits. Don't hide it. The count goes into the spec's table in Task 13.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_training.py python/rgpu/dispatch.py
git commit -m "Train on rgpu, matching CPU, at one round trip per step

Autograd runs on the client, above the dispatch point, so forward,
backward and the optimizer step all stream. A step waits only where the
code pulls data back, here loss.item().

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 10: Mixed precision

**Files:**
- Create: `python/rgpu/autocast.py`
- Modify: `python/rgpu/device.py` (`register_device` calls `autocast.install()`)
- Test: `python/tests/test_amp.py`

**Interfaces:**
- Consumes: `register_device` (Task 5).
- Produces: `rgpu.autocast.install()`. It registers a fallthrough for every op at `AutocastPrivateUse1`, plus cast kernels for two lists of ops:
  - **lower precision**: ops that run in the autocast dtype;
  - **fp32**: ops whose half-precision inputs are widened to float32.

  Both lists mirror PyTorch's documented CUDA autocast lists. After this, `torch.amp.autocast("rgpu", dtype=torch.float16)` and `torch.amp.GradScaler("rgpu")` work.

- [ ] **Step 1: Write the failing tests**

`python/tests/test_amp.py`:

```python
import torch
import torch.nn as nn


def test_matmul_runs_in_half_and_softmax_in_float():
    a, b = torch.randn(16, 32), torch.randn(32, 8)
    with torch.amp.autocast("rgpu", dtype=torch.float16):
        y = a.to("rgpu") @ b.to("rgpu")
        s = torch.log_softmax(y, dim=-1)
        r = torch.relu(y)
    assert y.dtype == torch.float16
    assert s.dtype == torch.float32
    assert r.dtype == torch.float16          # untouched ops keep their input's dtype
    assert torch.allclose(y.float().cpu(), a @ b, atol=5e-2, rtol=1e-2)


def test_autocast_off_leaves_float32_alone():
    a = torch.randn(4, 4).to("rgpu")
    assert (a @ a).dtype == torch.float32


def test_training_with_a_gradient_scaler():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).to("rgpu")
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    scaler = torch.amp.GradScaler("rgpu")
    x, y = torch.randn(32, 64).to("rgpu"), torch.randint(0, 10, (32,)).to("rgpu")
    losses = []
    for _ in range(5):
        opt.zero_grad()
        with torch.amp.autocast("rgpu", dtype=torch.float16):
            loss = nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python/.venv/bin/pytest python/tests/test_amp.py -q`
Expected: `NotImplementedError: Could not run 'aten::matmul' with arguments from the 'Autocastrgpu' backend`.

- [ ] **Step 3: Write `python/rgpu/autocast.py`**

```python
"""Autocast for rgpu: which ops run in half precision and which stay float32.

PyTorch ships an autocast policy only for its own devices. A new device gets
none, and under autocast every op fails until something is registered at its
Autocast key. This registers a fallthrough for everything, then cast kernels
for the same two lists PyTorch documents for CUDA, so mixed precision on
rgpu behaves as it would on a local GPU.
"""

import torch
from torch.utils._pytree import tree_map

# Ops that run in the autocast dtype (PyTorch's CUDA "lower precision" list).
LOWER = [
    "mm", "addmm", "bmm", "baddbmm", "addbmm", "addmv", "addr", "matmul", "mv",
    "linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
    "conv_transpose3d", "convolution", "_convolution", "prelu", "chain_matmul",
    "linalg_multi_dot",
]

# Ops whose half-precision inputs are widened to float32 (the CUDA fp32 list).
FP32 = [
    "acos", "asin", "cosh", "erfinv", "exp", "expm1", "log", "log10", "log2",
    "log1p", "reciprocal", "rsqrt", "sinh", "tan", "pow", "softplus", "layer_norm",
    "group_norm", "norm", "nuclear_norm", "cosine_similarity", "poisson_nll_loss",
    "cosine_embedding_loss", "nll_loss", "nll_loss2d", "hinge_embedding_loss",
    "kl_div", "l1_loss", "smooth_l1_loss", "huber_loss", "mse_loss",
    "margin_ranking_loss", "multilabel_margin_loss", "soft_margin_loss",
    "triplet_margin_loss", "multi_margin_loss", "binary_cross_entropy_with_logits",
    "dist", "pdist", "cdist", "renorm", "logsumexp", "softmax", "log_softmax",
    "cross_entropy_loss", "cumprod", "cumsum", "prod", "sum",
]

_KEY = torch._C.DispatchKey.AutocastPrivateUse1
_libs = []


def _kernel(op, target, eligible):
    def kernel(*args, **kwargs):
        dtype = target()

        def cast(x):
            if (isinstance(x, torch.Tensor) and x.device.type == "rgpu"
                    and eligible(x.dtype) and x.dtype != dtype):
                return x.to(dtype)
            return x

        with torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(_KEY)):
            return op(*tree_map(cast, args), **tree_map(cast, kwargs))

    return kernel


def install():
    if _libs:
        return
    everything = torch.library.Library("_", "IMPL")
    everything.fallback(torch.library.fallthrough_kernel, "AutocastPrivateUse1")
    lib = torch.library.Library("aten", "IMPL")

    def lower_eligible(dt):
        return dt.is_floating_point and dt != torch.float64

    def fp32_eligible(dt):
        return dt in (torch.float16, torch.bfloat16)

    for names, target, eligible in [
        (LOWER, lambda: torch.get_autocast_dtype("rgpu"), lower_eligible),
        (FP32, lambda: torch.float32, fp32_eligible),
    ]:
        for name in names:
            packet = getattr(torch.ops.aten, name, None)
            if packet is None:
                continue
            for overload in packet.overloads():
                qualified = name if overload == "default" else f"{name}.{overload}"
                lib.impl(qualified, _kernel(getattr(packet, overload), target, eligible),
                         "AutocastPrivateUse1")
    _libs.extend([everything, lib])
```

In `python/rgpu/device.py`, at the end of `register_device()`, add:

```python
    from . import autocast
    autocast.install()
```

- [ ] **Step 4: Run the tests**

Run: `python/.venv/bin/pytest python/tests -q`
Expected: all pass. If `lib.impl` raises for an overload that already has a kernel at this key, catch `RuntimeError` around that single `impl` call and skip it. Some composite overloads refuse Autocast kernels.

- [ ] **Step 5: Commit**

```bash
git add python/rgpu/autocast.py python/rgpu/device.py python/tests/test_amp.py
git commit -m "Give rgpu an autocast policy, mirroring CUDA's

A new device gets no autocast policy, so under autocast every op failed.
Registering a fallthrough for everything and cast kernels for PyTorch's
documented CUDA lists makes mixed precision behave as it would locally,
gradient scaler included.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 11: torch.compile

**Files:**
- Create: `python/rgpu/compile.py`, `python/rgpu/server/graphs.py`
- Modify: `python/rgpu/server/session.py` (handlers for `COMPILE` and `CALL`), `python/rgpu/__init__.py` (export `compile_backend`, register the name `"rgpu"`)
- Test: `python/tests/test_compile.py`

**Interfaces:**
- Consumes: `RemoteTensor` traceability and `rgpu.tensor.register`/`id_of`/`meta_like` (Task 5); `wire.COMPILE`, `wire.CALL`, `wire.NodeRef` (Task 1); `Session._arg`, `_device`, `_store`, `_poison`, `_fail` (Task 2); `rgpu.server.ops.resolve` (Task 2).
- Produces:
  - `rgpu.compile_backend(mode=None, compiler="inductor")` returns a torch.compile backend. `torch.compile(model, backend="rgpu")` uses the defaults, and `torch.compile(model, backend=rgpu.compile_backend(mode="reduce-overhead"))` selects a mode. The spec's `mode="reduce-overhead"` is passed this way, because torch.compile's own `mode=` applies only to inductor.
  - `rgpu.compile.serialize(gm) -> list`: nodes `["placeholder"]`, `["call", op_name, overload, args, kwargs]` or `["output", [refs]]`, with node references as `wire.NodeRef`. It raises `rgpu.compile.Unshippable`.
  - `rgpu.server.graphs.build(nodes, session, compiler, mode) -> callable`.
  - Server messages: `COMPILE [graph_id, nodes, compiler, mode]` and `CALL [graph_id, input_ids, output_ids]`. Neither gets a reply.

- [ ] **Step 1: Write the failing tests**

`python/tests/test_compile.py`:

```python
import pytest
import torch
import torch.nn as nn

import rgpu
from rgpu.compile import Unshippable, serialize

EAGER = rgpu.compile_backend(compiler="eager")   # fast: runs the shipped graph as is


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_an_elementwise_function():
    f = lambda x, y: (x * y + x).relu()
    a, b = torch.randn(64), torch.randn(64)
    got = torch.compile(f, backend=EAGER)(a.to("rgpu"), b.to("rgpu"))
    assert torch.allclose(got.cpu(), f(a, b))


def test_a_fused_reduction_and_a_matmul_with_an_epilogue():
    f = lambda x: (x * 2).softmax(dim=-1).sum(dim=-1)
    g = lambda a, b, c: torch.relu(a @ b + c)
    x = torch.randn(8, 32)
    a, b, c = torch.randn(16, 32), torch.randn(32, 8), torch.randn(8)
    assert torch.allclose(torch.compile(f, backend=EAGER)(x.to("rgpu")).cpu(), f(x), atol=1e-5)
    got = torch.compile(g, backend=EAGER)(a.to("rgpu"), b.to("rgpu"), c.to("rgpu"))
    assert torch.allclose(got.cpu(), g(a, b, c), atol=1e-4)


def test_resnet18_forward():
    import torchvision.models as models
    torch.manual_seed(0)
    net = models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        want = net(x)
        got = torch.compile(net.to("rgpu"), backend=EAGER)(x.to("rgpu")).cpu()
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4)


def test_a_compiled_training_step_matches_eager_and_waits_once():
    torch.manual_seed(0)
    make = lambda: nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
    torch.manual_seed(0)
    ref = make()
    torch.manual_seed(0)
    model = make().to("rgpu")
    step = torch.compile(model, backend=EAGER)
    x, y = torch.randn(16, 32), torch.randint(0, 4, (16,))
    nn.functional.cross_entropy(ref(x), y).backward()
    loss = nn.functional.cross_entropy(step(x.to("rgpu")), y.to("rgpu"))
    loss.backward()
    for a, b in zip(ref.parameters(), model.parameters()):
        assert torch.allclose(b.grad.cpu(), a.grad, atol=1e-5)
    waits = rgpu.stats()["waits"]
    loss = nn.functional.cross_entropy(step(x.to("rgpu")), y.to("rgpu"))
    loss.backward()
    loss.item()
    assert rgpu.stats()["waits"] - waits == 1


def test_one_forward_and_one_backward_graph_are_shipped():
    shipped = []
    import rgpu.compile as rc
    real = rc._ship
    rc._ship = lambda *a: shipped.append(a[0]) or real(*a)
    try:
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU()).to("rgpu")
        torch.compile(model, backend=EAGER)(torch.randn(2, 8).to("rgpu")).sum().backward()
    finally:
        rc._ship = real
    assert len(shipped) == 2


def test_inductor_compiles_on_the_server():
    f = lambda x: (x.sin() * 2).cos().sum(dim=0)
    x = torch.randn(32, 16)
    got = torch.compile(f, backend=rgpu.compile_backend())(x.to("rgpu"))
    assert torch.allclose(got.cpu(), f(x), atol=1e-5)


def test_graphs_with_things_the_wire_cannot_carry_are_refused():
    gm = torch.fx.symbolic_trace(lambda x: x + torch.ones(3))
    with pytest.raises(Unshippable):
        serialize(gm)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python/.venv/bin/pytest python/tests/test_compile.py -q`
Expected: `AttributeError: module 'rgpu' has no attribute 'compile_backend'`.

- [ ] **Step 3: Write `python/rgpu/server/graphs.py`**

```python
"""Rebuild a graph the client shipped, and compile it for this server.

A graph arrives as a list of nodes whose ops are aten ops named exactly as in
a RUN message, and which are looked up the same way: the compile path gives
a client no op it could not already run one at a time.
"""

import torch
from torch.utils._pytree import tree_map

from .. import wire
from . import ops


def build(nodes, session, compiler, mode):
    graph = torch.fx.Graph()
    env = []

    def arg(a):
        if isinstance(a, wire.NodeRef):
            return env[a.index]
        if isinstance(a, wire.Dev):
            return session._device(a.name)
        if isinstance(a, (wire.Ref, wire.Host)):
            raise ValueError("a graph cannot carry tensors")
        return a

    for node in nodes:
        kind = node[0]
        if kind == "placeholder":
            env.append(graph.placeholder(f"arg{len(env)}"))
        elif kind == "call":
            _, name, overload, args, kwargs = node
            op = ops.resolve(name, overload)
            env.append(graph.call_function(
                op, tuple(tree_map(arg, args)), {k: tree_map(arg, v) for k, v in kwargs.items()}))
        elif kind == "output":
            graph.output(tuple(tree_map(arg, node[1])))
            env.append(None)
        else:
            raise ValueError(f"unknown node kind {kind!r}")

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    if compiler == "eager":
        return gm
    if compiler == "inductor":
        return torch.compile(gm, mode=mode, dynamic=False)
    raise ValueError(f"unknown compiler {compiler!r}")
```

- [ ] **Step 4: Add `COMPILE` and `CALL` to `python/rgpu/server/session.py`**

Add these methods to `Session`, next to `_run`:

```python
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
        if fn is None or isinstance(fn, Poison):
            self._poison(out_ids, fn or Poison(f"graph {gid}", "was never compiled"))
            return
        try:
            inputs = [self._arg(wire.Ref(i)) for i in in_ids]
            outs = fn(*inputs)
            self._store(list(outs), out_ids)
        except _Skip as s:
            self._poison(out_ids, s.poison)
        except Exception as e:  # noqa: BLE001
            self._fail(f"graph {gid}", e, out_ids)
```

Replace the `_ASYNC` table:

```python
    _ASYNC = {wire.RUN: _run, wire.UPLOAD: _upload, wire.FREE: _free, wire.SEED: _seed,
              wire.COMPILE: _compile, wire.CALL: _call}
```

- [ ] **Step 5: Write `python/rgpu/compile.py`**

```python
"""torch.compile for rgpu: ship the graph once, then one message per call.

The backend hooks aot_autograd, which hands over forward and backward graphs
of aten ops. Each is serialized with the same op names and argument encoding
as eager ops, sent once, and compiled on the server. After that a call is a
single CALL message, and it streams: output shapes come from the traced
graph, so nothing has to wait for them.

Compiled code works on the unwrapped meta tensors, never on RemoteTensor
itself, which is why each tensor's id lives on its meta tensor: inputs are
looked up by it and outputs are given one.
"""

import itertools
import logging

import torch
from torch._dynamo.backends.common import aot_autograd
from torch.utils._pytree import tree_map

from . import session, wire
from .tensor import RemoteTensor, id_of, meta_like, register

log = logging.getLogger("rgpu")
_graph_ids = itertools.count(1)


class Unshippable(Exception):
    """A graph containing something the wire cannot carry."""


def serialize(gm):
    nodes, index = [], {}

    def ref(a):
        if isinstance(a, torch.fx.Node):
            return wire.NodeRef(index[a])
        if isinstance(a, torch.Tensor):
            raise Unshippable("a constant tensor inside the graph")
        if isinstance(a, torch.device):
            return wire.Dev("rgpu" if a.type == "rgpu" else str(a))
        if isinstance(a, (torch.SymInt, torch.SymFloat)):
            raise Unshippable("dynamic shapes; compile with dynamic=False")
        try:
            wire.encode(a)
        except wire.EncodeError as e:
            raise Unshippable(str(e)) from None
        return a

    for n in gm.graph.nodes:
        if n.op == "placeholder":
            nodes.append(["placeholder"])
        elif n.op == "call_function":
            t = n.target
            if not isinstance(t, torch._ops.OpOverload) or t.namespace != "aten":
                raise Unshippable(f"{t} is not an aten op")
            nodes.append(["call", t._schema.name, t._overloadname,
                          tree_map(ref, list(n.args)),
                          {k: tree_map(ref, v) for k, v in n.kwargs.items()}])
        elif n.op == "output":
            outs = n.args[0] if isinstance(n.args[0], (list, tuple)) else [n.args[0]]
            nodes.append(["output", [tree_map(ref, o) for o in outs]])
        else:
            raise Unshippable(f"a {n.op} node")
        index[n] = len(nodes) - 1
    return nodes


def _output_values(gm):
    out = next(n for n in gm.graph.nodes if n.op == "output")
    outs = out.args[0] if isinstance(out.args[0], (list, tuple)) else [out.args[0]]
    return [o.meta.get("val") if isinstance(o, torch.fx.Node) else o for o in outs]


def _boxed(fn):
    def run(args):
        return fn(*args)
    run._boxed_call = True
    return run


def _ship(gid, nodes, compiler, mode):
    session.get().post(wire.COMPILE, gid, nodes, compiler, mode)


def _eager(gm):
    """Run a graph that cannot be shipped op by op, through the normal path."""
    def run(*inner):
        wrapped = [RemoteTensor(x) if isinstance(x, torch.Tensor) else x for x in inner]
        outs = gm(*wrapped)
        return [o._rgpu_meta if isinstance(o, RemoteTensor) else o for o in outs]
    return _boxed(run)


def _compiler(mode, compiler):
    def compile_graph(gm, example_inputs):
        try:
            nodes = serialize(gm)
            values = _output_values(gm)
            if any(isinstance(s, torch.SymInt) for v in values
                   if isinstance(v, torch.Tensor) for s in v.shape):
                raise Unshippable("dynamic shapes; compile with dynamic=False")
        except Unshippable as e:
            log.warning("rgpu: running a graph eagerly because it has %s", e)
            return _eager(gm)
        gid = next(_graph_ids)
        _ship(gid, nodes, compiler, mode)

        def run(*inputs):
            in_ids = [id_of(x) for x in inputs]
            outs, out_ids = [], []
            for v in values:
                if isinstance(v, torch.Tensor):
                    m = meta_like(v.dtype, list(v.shape), list(v.stride()), v.storage_offset())
                    out_ids.append(register(m))
                    outs.append(m)
                else:
                    out_ids.append(None)
                    outs.append(v)
            session.get().post(wire.CALL, gid, in_ids, out_ids)
            return outs

        return _boxed(run)

    return compile_graph


def compile_backend(mode=None, compiler="inductor"):
    """A torch.compile backend that compiles on the rgpu server."""
    comp = _compiler(mode, compiler)
    return aot_autograd(fw_compiler=comp, bw_compiler=comp)


torch._dynamo.register_backend(name="rgpu")(compile_backend())
```

- [ ] **Step 6: Export it from `python/rgpu/__init__.py`**

Replace the last two lines:

```python
from .compile import compile_backend  # noqa: E402
from .session import RemoteError, SessionLost, stats  # noqa: E402

__all__ = ["RemoteError", "SessionLost", "compile_backend", "stats"]
```

- [ ] **Step 7: Run the tests**

Run: `python/.venv/bin/pytest python/tests -q`
Expected: all pass. `test_inductor_compiles_on_the_server` needs a C++ compiler on the Mac, which the Xcode command line tools provide. If inductor cannot build there, mark only that test `@pytest.mark.skipif(sys.platform == "darwin" and not shutil.which("clang++"), reason=...)`. Task 13 runs inductor on CUDA regardless.

- [ ] **Step 8: Commit**

```bash
git add python/rgpu/compile.py python/rgpu/server/graphs.py python/rgpu/server/session.py python/rgpu/__init__.py python/tests/test_compile.py
git commit -m "Compile on the rgpu server: ship a graph once, then one message per call

aot_autograd hands over forward and backward graphs of aten ops, which
travel with the same op names and encoding as eager ops and compile on
the server. A call is then a single CALL message that streams, because
output shapes come from the traced graph. A graph the wire cannot carry
runs eagerly instead, with a warning naming why.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 12: Surviving a dropped connection

**Files:**
- Modify: `python/rgpu/session.py` (reconnect and replay in `_flush` and `_await`)
- Test: `python/tests/test_reconnect.py`

**Interfaces:**
- Consumes: the server's resume and reply cache and `RGPU_DROP_AFTER` (Task 3); `Connection.unacked`, `replay_possible`, `last_acked`, `had_session` (Task 4); `conftest.start_server` (Task 3).
- Produces: `Connection._reconnect()`. It retries until `RGPU_RECONNECT_SECONDS` runs out (default 60), with backoff capped at 3 s. After resuming it drops acknowledged messages and resends the rest in one frame. It raises `SessionLost` when the server no longer has the session, or when messages the server lacks were not kept because of the 64 MB replay limit.

- [ ] **Step 1: Write the failing tests**

`python/tests/test_reconnect.py`:

```python
import os

import pytest
import torch
import torch.nn as nn

import rgpu
from rgpu import session
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
    start_server(port=port)[0]   # a new server knows nothing of us
    with pytest.raises(rgpu.SessionLost):
        (t + 1).cpu()
```

`RGPU_REAL_SERVER=1` marks a run against a remote GPU (Task 13), where there's no local drop hook.

- [ ] **Step 2: Run the tests to see them fail**

Run: `python/.venv/bin/pytest python/tests/test_reconnect.py -q`
Expected: the first test fails with `ConnectionError: connection closed`. The second raises `ConnectionError` instead of `SessionLost`.

- [ ] **Step 3: Add reconnection to `python/rgpu/session.py`**

Add `import time` at the top. Add these methods to `Connection`:

```python
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
```

Replace `_flush` and `_await` with versions that reconnect once:

```python
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
                continue
            self._ack(rseq)
            if status == wire.ERROR:
                raise RemoteError(*value)
            return value
```

`ConnectionError` is a subclass of `OSError`, so a closed connection takes the same path as a reset.

- [ ] **Step 4: Run the tests**

Run: `python/.venv/bin/pytest python/tests -q`
Expected: all pass. If the drop test's losses differ in the last digit, the replay has sent something twice. Check that `_reconnect` drops messages up to and including `server_seq`, and that `pending` is cleared: everything in `pending` is already in `unacked`.

- [ ] **Step 5: Commit**

```bash
git add python/rgpu/session.py python/tests/test_reconnect.py
git commit -m "Pick an rgpu session back up after a dropped connection

The client resends whatever the server had not acknowledged and
continues; the server resends a reply that was lost rather than running
the request twice. A training run through a deliberate drop produces
the same losses, to the last digit, as one without. A server that
restarted has no session to return, and says so.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 13: Real hardware, from the Mac

**Files:**
- Create: `python/benchmarks/round_trips.py`, `python/benchmarks/survive_drop.py`, `scripts/opserver_pod.sh`
- Modify: `docs/superpowers/specs/2026-09-11-rgpu-op-backend-design.md` (fill in the results table), `docs/PRODUCT_SPEC.md` (the macOS path), `README.md` (a short "macOS" section)

**Interfaces:**
- Consumes: everything above, and the `rcuda-runpod` skill in `.claude/skills/rcuda-runpod/SKILL.md`. Read it first: it covers renting the GPU, confirming nothing is billing, the stale host key, and the 1Password account.
- Produces: measured round trips and milliseconds per step, from native macOS Python against a RunPod GPU.

- [ ] **Step 1: Write `python/benchmarks/round_trips.py`**

```python
"""Round trips and milliseconds per step, eager and compiled, on ResNet-18.

    RGPU_OPSERVER=127.0.0.1:9720 python benchmarks/round_trips.py [steps]
"""

import sys
import time

import torch
import torch.nn as nn
import torchvision.models as models

import rgpu

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def measure(label, step):
    step()
    step()   # warm up, and compile if compiling
    torch.rgpu.synchronize()
    waits, start = rgpu.stats()["waits"], time.time()
    for _ in range(steps):
        step()
    torch.rgpu.synchronize()
    ms = (time.time() - start) / steps * 1000
    per = (rgpu.stats()["waits"] - waits - 1) / steps   # minus the final synchronize
    print(f"{label:28} {ms:9.1f} ms/step {per:6.2f} round trips/step")


torch.manual_seed(0)
net = models.resnet18(weights=None).to("rgpu")
x = torch.randn(1, 3, 224, 224).to("rgpu")
y = torch.randint(0, 1000, (1,)).to("rgpu")
opt = torch.optim.SGD(net.parameters(), lr=0.01)


def infer(model):
    def step():
        with torch.no_grad():
            model.eval()
            model(x).cpu()
    return step


def train(model):
    def step():
        model.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        loss.item()
    return step


compiled = torch.compile(net, backend=rgpu.compile_backend(), dynamic=False)
measure("eager inference", infer(net))
measure("eager training step", train(net))
measure("compiled training step", train(compiled))
```

- [ ] **Step 2: Write `python/benchmarks/survive_drop.py`**

```python
"""Per-step losses with a fixed seed, to run while the link is broken and restored.

A run that survived the break prints exactly the same losses as one that
never saw it; a reconnect that had quietly started a new session could not.
"""

import sys
import time

import torch
import torch.nn as nn

import rgpu  # noqa: F401 - registers the device

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10)).to("rgpu")
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
x, y = torch.randn(128, 256).to("rgpu"), torch.randint(0, 10, (128,)).to("rgpu")
start = time.time()
for step in range(steps):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    print(f"step {step:2d}  t={time.time() - start:5.1f}s  loss {loss.item():.6f}", flush=True)
```

- [ ] **Step 3: Write `scripts/opserver_pod.sh`**

This runs **on the pod**. It installs torch with the Mac's `major.minor`, installs the package, and starts the server detached, then verifies that it's listening. It prints `LISTENING` or `DEAD`. Checking that the port answers catches a server that died, which a `pgrep` check can mistake for a live one.

```bash
#!/usr/bin/env bash
# Start rgpu-opserver on a GPU pod, detached, and prove it is listening.
#
#   bash scripts/opserver_pod.sh 2.14        # the Mac's torch major.minor
set -euo pipefail
cd "$(dirname "$0")/.."
want=${1:?give the client torch version, e.g. 2.14}
if [[ ! -x /root/opvenv/bin/python ]]; then
  python3 -m venv /root/opvenv
fi
if ! /root/opvenv/bin/python -c "import torch,sys; sys.exit(not torch.__version__.startswith('$want'))" 2>/dev/null; then
  for cu in cu128 cu126 cu130; do
    /root/opvenv/bin/pip install -q "torch==$want.*" --index-url "https://download.pytorch.org/whl/$cu" && break
  done
fi
/root/opvenv/bin/pip install -q -e python --no-deps
pkill -f "rgpu.server" 2>/dev/null || true
sleep 1
setsid nohup /root/opvenv/bin/python -m rgpu.server --device cuda > /root/opserver.log 2>&1 < /dev/null &
sleep 5
if (exec 3<>/dev/tcp/127.0.0.1/9720) 2>/dev/null; then echo LISTENING; else echo DEAD; tail -20 /root/opserver.log; fi
```

- [ ] **Step 4: Rent a GPU and start the server**

Follow the `rcuda-runpod` skill. `export OP_ACCOUNT=my.1password.com`, then run `./scripts/runpod.sh status` (it must say nothing is billing) and `./scripts/runpod.sh create`. Refresh the host key as the skill describes. Copy the package:

```bash
tar czf - python scripts/opserver_pod.sh | ssh -i ~/.ssh/rgpu_runpod -p PORT root@HOST "mkdir -p /root/rgpu && tar xzf - -C /root/rgpu"
ssh -i ~/.ssh/rgpu_runpod -p PORT root@HOST "bash /root/rgpu/scripts/opserver_pod.sh $(python/.venv/bin/python -c 'import torch; print(".".join(torch.__version__.split(".")[:2]))')"
```

Expected: `LISTENING`. Then open the tunnel: `ssh -f -N -i ~/.ssh/rgpu_runpod -p PORT -L 9720:localhost:9720 root@HOST`.

- [ ] **Step 5: Run the suites from the Mac against the GPU**

```bash
RGPU_OPSERVER=127.0.0.1:9720 RGPU_REAL_SERVER=1 python/.venv/bin/pytest python/tests -q
```

Expected: everything passes except `test_reconnect.py`, which is skipped. The CPU-reference tolerances in `test_ladder.py`, `test_training.py` and `test_compile.py` were set for a CPU server. On CUDA, TF32 and cuDNN algorithm choice move results by about 1e-3. If an `allclose` fails only by that amount, add `torch.backends.cuda.matmul.allow_tf32 = False` to the server's startup in `rgpu/server/__main__.py`, rather than loosening the test. A real mismatch is a bug.

- [ ] **Step 6: Measure**

```bash
RGPU_OPSERVER=127.0.0.1:9720 python/.venv/bin/python python/benchmarks/round_trips.py 10
```

Expected: three lines. Copy them into the table under **Done when** in the spec, next to the CUDA-level figures (49, and 2 for graph replay).

- [ ] **Step 7: Kill the tunnel mid-run**

Run `survive_drop.py` in the background, writing to a file. After step 4 prints, run `pkill -f "ssh -f -N.*9720"`, wait 10 s, and reopen the tunnel. Then compare the losses with an uninterrupted run:

```bash
diff <(grep -oE "step +[0-9]+ .*loss [0-9.]+" clean.txt | sed -E 's/t= *[0-9.]+s//') \
     <(grep -oE "step +[0-9]+ .*loss [0-9.]+" dropped.txt | sed -E 's/t= *[0-9.]+s//') && echo identical
```

Expected: `identical`.

- [ ] **Step 8: Release the GPU, and confirm it**

Close the tunnel, then run `./scripts/runpod.sh delete --yes` and `./scripts/runpod.sh status`. Expected: `no pods: nothing is billing`. If `delete` printed "Malformed Bearer token", the 1Password session expired. Sign in again (`op signin --account my.1password.com`) and repeat: the pod is still billing until the list is empty.

- [ ] **Step 9: Write it down and commit**

- Fill in the spec's results table.
- Add a **The macOS path** section to `docs/PRODUCT_SPEC.md` giving the measured numbers and the one-round-trip result, or the measured count and the op responsible.
- Add three lines to `README.md` on how to use it from a Mac: `pip install -e python`, the tunnel, `import rgpu`.

```bash
git add python/benchmarks scripts/opserver_pod.sh docs/superpowers/specs/2026-09-11-rgpu-op-backend-design.md docs/PRODUCT_SPEC.md README.md
git commit -m "Measure the rgpu device from a Mac against a rented GPU

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
git push origin main
```
