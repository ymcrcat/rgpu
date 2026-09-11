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


def test_a_graph_that_makes_a_tensor_of_its_own_makes_it_on_the_server():
    f = lambda x: x + torch.arange(4, device=x.device, dtype=x.dtype)
    x = torch.randn(4)
    got = torch.compile(f, backend=EAGER)(x.to("rgpu"))
    assert torch.allclose(got.cpu(), f(x))


def test_graphs_with_things_the_wire_cannot_carry_are_refused():
    gm = torch.fx.symbolic_trace(lambda x: x + torch.ones(3))
    with pytest.raises(Unshippable):
        serialize(gm)
