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
