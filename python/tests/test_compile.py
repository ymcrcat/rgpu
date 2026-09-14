import logging

import pytest
import torch
import torch.nn as nn

import rgpu
from rgpu.compile import Unshippable, UnsupportedOp, serialize

EAGER = rgpu.compile_backend(compiler="eager")   # fast: runs the shipped graph as is


@torch.library.custom_op("rgpu_probe::twice", mutates_args=())
def _twice(x: torch.Tensor) -> torch.Tensor:
    return x + x


@_twice.register_fake
def _(x):
    return torch.empty_like(x)


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


def test_an_op_with_several_outputs(caplog):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.MaxPool2d(2))   # max pool: two outputs
    x = torch.randn(2, 3, 16, 16)
    with caplog.at_level(logging.WARNING, logger="rgpu"), torch.no_grad():
        want = model(x)
        got = torch.compile(model.to("rgpu"), backend=EAGER)(x.to("rgpu")).cpu()
    assert "eagerly" not in caplog.text
    assert torch.allclose(got, want, atol=1e-5)


def test_resnet18_ships_its_graph_rather_than_falling_back(caplog):
    import torchvision.models as models
    torch.manual_seed(0)
    net = models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    with caplog.at_level(logging.WARNING, logger="rgpu"), torch.no_grad():
        want = net(x)
        got = torch.compile(net.to("rgpu"), backend=EAGER)(x.to("rgpu")).cpu()
    assert "eagerly" not in caplog.text
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


def test_a_backward_graph_over_an_op_with_several_outputs_is_shipped_too(caplog):
    make = lambda: nn.Sequential(nn.Conv2d(3, 4, 3), nn.MaxPool2d(2), nn.Flatten(),
                                 nn.Linear(4 * 7 * 7, 3))
    torch.manual_seed(0)
    ref = make()
    torch.manual_seed(0)
    model = make().to("rgpu")
    x = torch.randn(2, 3, 16, 16)
    ref(x).sum().backward()
    shipped = []
    import rgpu.compile as rc
    real = rc._ship
    rc._ship = lambda *a: shipped.append(a[0]) or real(*a)
    try:
        with caplog.at_level(logging.WARNING, logger="rgpu"):
            torch.compile(model, backend=EAGER)(x.to("rgpu")).sum().backward()
    finally:
        rc._ship = real
    assert "eagerly" not in caplog.text
    assert len(shipped) == 2
    for a, b in zip(ref.parameters(), model.parameters()):
        assert torch.allclose(b.grad.cpu(), a.grad, atol=1e-5)


def test_inductor_compiles_on_the_server():
    f = lambda x: (x.sin() * 2).cos().sum(dim=0)
    x = torch.randn(32, 16)
    got = torch.compile(f, backend=rgpu.compile_backend())(x.to("rgpu"))
    assert torch.allclose(got.cpu(), f(x), atol=1e-5)


def test_a_graph_that_makes_a_tensor_of_its_own_makes_it_on_the_server():
    f = lambda x: x + torch.arange(4, device=x.device, dtype=x.dtype)
    x = torch.randn(4)
    got = torch.compile(f, backend=EAGER)(x.to("rgpu"))
    assert torch.allclose(got.cpu(), f(x))


def test_graphs_with_things_the_wire_cannot_carry_are_refused():
    gm = torch.fx.symbolic_trace(lambda x: x + torch.ones(3))
    with pytest.raises(Unshippable):
        serialize(gm)


def test_a_custom_op_is_refused_at_compile_time_naming_the_op():
    """There is no eager fallback for this one and pretending otherwise only
    delays the failure: running the graph op by op posts the same custom op
    as a RUN, and the server looks ops up by name and runs only aten ops."""
    f = lambda x: torch.ops.rgpu_probe.twice(x) + 1
    with pytest.raises(Exception) as e:
        torch.compile(f, backend=EAGER)(torch.randn(4).to("rgpu"))
    # dynamo wraps whatever a backend raises, so the report is what matters:
    # the op is named, the reason is given, and it did not reach the server.
    assert "rgpu_probe::twice" in str(e.value) and "aten ops only" in str(e.value)
    assert not isinstance(e.value, rgpu.RemoteError)
    assert UnsupportedOp.__name__ in str(e.value)


def test_a_graph_that_is_only_unshippable_still_runs_eagerly_and_correctly(caplog,
                                                                           monkeypatch):
    """The other half of the fence: a graph refused for what it is rather
    than for an op it contains - a constant tensor, dynamic shapes - still
    falls back to the ordinary dispatch path and gets the right answer."""
    import rgpu.compile as rc

    def refuse(gm):
        raise rc.Unshippable("a constant tensor inside the graph")
    monkeypatch.setattr(rc, "serialize", refuse)

    f = lambda x: (x * 2).relu().sum(dim=0)
    x = torch.randn(8, 4)
    with caplog.at_level(logging.WARNING, logger="rgpu"):
        got = torch.compile(f, backend=EAGER)(x.to("rgpu"))
    assert "eagerly" in caplog.text and "a constant tensor" in caplog.text
    assert torch.allclose(got.cpu(), f(x), atol=1e-5)


def test_the_default_backend_says_which_backend_to_use():
    """torch.compile(model) picks inductor, which cannot work here: it generates
    code that allocates tensors rgpu never sees, and the first one to reach the
    wire used to raise "this tensor was never sent to the server" from deep in
    dispatch - true, but a symptom several layers below the cause. The error has
    to name the backend and the fix, because that is what the user has to
    change."""
    net = nn.Linear(8, 8).to("rgpu")
    x = torch.randn(4, 8, device="rgpu")
    with pytest.raises(RuntimeError, match=r'backend="rgpu"'):
        torch.compile(net)(x).sum().item()


def test_the_rgpu_backend_still_works_after_that():
    """The advice the message gives has to be advice that works."""
    net = nn.Linear(8, 8).to("rgpu")
    x = torch.randn(4, 8, device="rgpu")
    got = torch.compile(net, backend="rgpu", dynamic=False)(x)
    assert got.shape == (4, 8)
    assert torch.isfinite(got.cpu()).all()
