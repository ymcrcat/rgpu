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


def test_out_variant_is_not_cast_away_from_the_callers_tensor():
    a = torch.randn(4, 4).to("rgpu").half()
    out = torch.empty(4, dtype=torch.float16, device="rgpu")
    with torch.amp.autocast("rgpu", dtype=torch.float16):
        r = torch.sum(a, dim=0, out=out)
    assert r is out
    assert torch.allclose(out.cpu(), a.float().cpu().sum(dim=0).half().cpu(),
                           atol=1e-2, rtol=1e-2)


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
