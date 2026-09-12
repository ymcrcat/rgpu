import pytest
import torch

import rgpu
from rgpu import dispatch


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
    r = x.to("rgpu")
    waits = rgpu.stats()["waits"]
    got = op(r)
    assert rgpu.stats()["waits"] == waits + 1
    want = op(x)
    assert got.shape == want.shape and torch.equal(got.cpu(), want)


def test_a_data_dependent_op_refuses_an_out_tensor():
    x = torch.tensor([0.0, 3.0, -1.0]).to("rgpu")
    out = torch.empty(0, dtype=torch.long, device="rgpu")
    with pytest.raises(NotImplementedError, match="out="):
        torch.nonzero(x, out=out)


def test_run_sync_refuses_an_op_that_writes_an_input_and_returns_a_new_tensor():
    lib = torch.library.Library("rgputest", "DEF")
    lib.define("mixed(Tensor(a!) x) -> (Tensor(a!), Tensor)")
    op = torch.ops.rgputest.mixed.default
    r = torch.zeros(3, device="rgpu")
    waits = rgpu.stats()["waits"]
    with pytest.raises(NotImplementedError, match="whose size only the data can tell"):
        dispatch._run_sync(op, (r,), {})
    assert rgpu.stats()["waits"] == waits


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
