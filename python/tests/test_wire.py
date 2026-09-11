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
