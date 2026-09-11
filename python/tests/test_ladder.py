import torch
import torch.nn as nn


def test_convolution_matches_cpu():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    # Under no_grad because moving a module to rgpu goes through
    # torch.utils.swap_tensors - chosen because RemoteTensor is a traceable
    # wrapper subclass - and swap refuses a parameter that the autograd graph of
    # an earlier forward still holds a reference to.
    with torch.no_grad():
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
