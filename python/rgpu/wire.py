"""The wire format shared by the rgpu client and rgpu-opserver.

Frames are length-prefixed. Inside a frame, values use a small tagged
encoding that can carry only these types:

  None, bool, int, float, str, bytes, list (tuples arrive as lists),
  dict with str keys (op kwargs),
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
import socket
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
MAX_FRAME = int(os.environ.get("RGPU_MAX_FRAME", 1 << 30))

# Keepalive: how long a connection may be idle before the kernel starts
# probing, how far apart the probes are, and how many go unanswered before it
# gives up. Roughly a minute to notice a peer that has gone away without
# closing - a network partition, a suspended laptop - which no amount of
# waiting in recv() would ever reveal on its own.
KEEPALIVE_IDLE = 30
KEEPALIVE_INTERVAL = 10
KEEPALIVE_COUNT = 3


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
    elif isinstance(v, dict):
        if not all(isinstance(k, str) for k in v.keys()):
            raise EncodeError("dict keys must be strings")
        out += b"K" + struct.pack("<I", len(v))
        for k, val in v.items():
            b = k.encode()
            out += struct.pack("<I", len(b)) + b
            _enc(val, out, depth + 1)
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
    if tag == b"K":
        n = r.unpack("<I")
        if n > MAX_ITEMS:
            raise DecodeError("dict too large")
        return {r.string(): _dec(r, depth + 1) for _ in range(n)}
    raise DecodeError(f"unknown tag {tag!r}")


def set_keepalive(sock):
    """Ask the kernel to notice a peer that has stopped answering.

    The three knobs are spelled differently per platform - macOS calls the
    idle time TCP_KEEPALIVE, Linux calls it TCP_KEEPIDLE, and the interval
    and count are missing entirely on some - so each is probed rather than
    assumed. SO_KEEPALIVE alone still works everywhere; only the timing
    falls back to the system default, which is two hours.
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (("TCP_KEEPIDLE", KEEPALIVE_IDLE),
                        ("TCP_KEEPALIVE", KEEPALIVE_IDLE),
                        ("TCP_KEEPINTVL", KEEPALIVE_INTERVAL),
                        ("TCP_KEEPCNT", KEEPALIVE_COUNT)):
        option = getattr(socket, name, None)
        if option is None:
            continue
        try:
            sock.setsockopt(socket.IPPROTO_TCP, option, value)
        except OSError:
            pass   # the name exists but this kernel will not take it


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
