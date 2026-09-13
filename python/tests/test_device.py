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


def test_tolist_matches_cpu():
    x, y = torch.arange(5.0), torch.randn(2, 3)
    assert x.to("rgpu").tolist() == x.tolist()
    assert y.to("rgpu").tolist() == y.tolist()


def test_cpu_keeps_the_strides_a_transposed_view_has():
    """What comes back over the wire is contiguous, but the tensor is not:
    CPU and CUDA both keep the strides of a view that is dense, and code that
    reads .stride() or .is_contiguous() after .cpu() must see the same."""
    x = torch.randn(2, 3)
    want = x.t()
    got = x.to("rgpu").t().cpu()
    assert got.stride() == want.stride()
    assert torch.equal(got, want)


def test_cpu_keeps_channels_last():
    x = torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last)
    got = x.to("rgpu").cpu()
    assert got.stride() == x.stride()
    assert got.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(got, x)


def test_a_memory_format_asked_for_on_the_way_back_is_honoured():
    x = torch.randn(2, 3, 4, 5)
    want = x.to("cpu", copy=True, memory_format=torch.channels_last)
    got = x.to("rgpu").to("cpu", memory_format=torch.channels_last)
    assert got.stride() == want.stride()
    assert torch.equal(got, want)


def test_a_view_that_is_not_dense_comes_back_contiguous():
    """A copy cannot keep the strides of a view with gaps in it - PyTorch's
    preserve_format makes such a tensor contiguous, and so does rgpu."""
    x = torch.randn(4, 4)
    got = x.to("rgpu")[:, ::2].cpu()
    assert got.is_contiguous()
    assert torch.equal(got, x[:, ::2])


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


def test_a_wrong_sized_out_raises_and_leaves_it_consistent():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="rgpu")
    r = torch.empty(0, device="rgpu")
    with pytest.raises(NotImplementedError):
        torch.add(a, a, out=r)
    assert r.shape == r._rgpu_meta.shape == (0,)


def test_a_correctly_sized_out_still_works():
    x, y = torch.randn(4), torch.randn(4)
    rx, ry = x.to("rgpu"), y.to("rgpu")
    out = torch.empty(4, device="rgpu")
    torch.add(rx, ry, out=out)
    assert torch.allclose(out.cpu(), x + y)


def test_an_inplace_reshape_raises_and_leaves_the_shape_unchanged():
    x = torch.randn(4, device="rgpu")
    with pytest.raises(NotImplementedError):
        x.unsqueeze_(0)
    assert x.shape == (4,)


def test_an_unsupported_argument_raises_at_the_call_and_spares_the_queue():
    """The wire carries a fixed set of types; anything else raises on the
    client, at the call that passed it. It must not be queued first, or the
    batch it is sitting in poisons every later flush in the process."""
    r = torch.tensor([1.0, 2.0]).to("rgpu")
    with pytest.raises(TypeError, match="Generator"):
        torch.normal(0.0, 1.0, size=(3,), device="rgpu", generator=torch.Generator())
    assert torch.equal(r.cpu(), torch.tensor([1.0, 2.0]))
    assert torch.equal(r.cpu(), torch.tensor([1.0, 2.0]))
