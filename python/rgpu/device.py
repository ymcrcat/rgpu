"""Makes "rgpu" a PyTorch device, from Python alone.

PyTorch reserves one device slot for backends outside it (PrivateUse1). Its
Python-backend hook, marked experimental, registers the device guard and
hooks a backend would otherwise have to write in C++.

Factory functions - torch.zeros(..., device="rgpu") and the like - have no
tensor argument to dispatch on, so they need kernels of their own. Every
factory op is registered, not just empty: ops such as arange build an empty
tensor and resize it through out=, and a wrapper's shape is fixed when it is
made, so going through empty would leave it describing a size-zero tensor.
"""

import types

import torch

from . import session, wire
from .tensor import RemoteTensor, register

_SKIP_FACTORIES = ("sparse", "quantized", "cudnn", "mkldnn", "nested", "_make_dep_token",
                   "from_file")


def synchronize(device=None):
    session.get().request(wire.SYNC)


def manual_seed(seed):
    session.get().post(wire.SEED, int(seed))


def _set_device(device):
    index = device.index if isinstance(device, torch.device) else device
    if index not in (0, None):
        raise ValueError("rgpu has one device, rgpu:0")


def _module():
    m = types.ModuleType("torch.rgpu")
    m.is_available = lambda: True
    m.device_count = lambda: 1
    m.current_device = lambda: 0
    m.set_device = _set_device
    m.is_initialized = lambda: True
    m._is_in_bad_fork = lambda: False
    m.synchronize = synchronize
    m.manual_seed = manual_seed
    m.manual_seed_all = manual_seed
    m.get_amp_supported_dtype = lambda: [torch.float16, torch.bfloat16]
    return m


def _factory_kernel(op):
    name, overload = op._schema.name, op._overloadname

    def kernel(*args, **kwargs):
        meta_kwargs = dict(kwargs)
        meta_kwargs["device"] = torch.device("meta")
        if "pin_memory" in meta_kwargs:
            meta_kwargs["pin_memory"] = None
        meta = op(*args, **meta_kwargs)
        wire_kwargs = {k: v for k, v in kwargs.items() if k != "pin_memory"}
        wire_kwargs["device"] = wire.Dev("rgpu")
        session.get().post(wire.RUN, name, overload, list(args), wire_kwargs, [register(meta)])
        return RemoteTensor(meta)

    return kernel


def _copy_from_kernel(self, dst, non_blocking=False):
    """Fills a tensor this backend just made, in place of copy_.

    torch.tensor(data, device="rgpu") builds the destination through a
    factory kernel above, then reaches PrivateUse1's _copy_from directly
    rather than going through __torch_dispatch__ - the same aten op that
    the ordinary CPU->rgpu path already runs as copy_.default.
    """
    from . import dispatch
    return dispatch.handle(torch.ops.aten.copy_.default, (dst, self, non_blocking), {})


def _factories():
    """Every aten overload with a device argument and no tensor arguments."""
    for full in torch._C._dispatch_get_all_op_names():
        if not full.startswith("aten::") or any(s in full for s in _SKIP_FACTORIES):
            continue
        name, _, overload = full[len("aten::"):].partition(".")
        try:
            op = getattr(getattr(torch.ops.aten, name), overload or "default")
        except (AttributeError, RuntimeError):
            continue
        schema = op._schema
        if len(schema.returns) != 1 or "Tensor" not in str(schema.returns[0].type):
            continue
        if not any(a.name == "device" for a in schema.arguments):
            continue
        if any("Tensor" in str(a.type) for a in schema.arguments):
            continue
        yield full[len("aten::"):], op


_libs = []


def register_device():
    if _libs:
        return
    torch.utils.backend_registration._setup_privateuseone_for_python_backend(
        "rgpu", backend_module=_module())
    lib = torch.library.Library("aten", "IMPL")
    for qualified, op in _factories():
        lib.impl(qualified, _factory_kernel(op), "PrivateUse1")
    # torch.tensor(data, device="rgpu") calls this directly, bypassing
    # __torch_dispatch__: see _copy_from_kernel.
    lib.impl("_copy_from", _copy_from_kernel, "PrivateUse1")
    _libs.append(lib)   # registrations live only as long as the Library object
