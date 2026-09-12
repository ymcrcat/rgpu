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
