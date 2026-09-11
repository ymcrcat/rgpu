"""RemoteTensor: a tensor whose data lives on the server.

On this side a RemoteTensor is only metadata - a meta tensor holding shape,
dtype and strides - plus an id naming the real tensor on the server. The id
is kept on the meta tensor rather than on the wrapper, because compiled code
works on the unwrapped meta tensors: rewrapping one gets its id back.

Move a model to rgpu before any grad-tracking forward pass. Because this is a
wrapper subclass, nn.Module.to("rgpu") installs parameters with
torch.utils.swap_tensors, which refuses a parameter that something else still
holds - and the autograd graph of an earlier forward holds exactly that. Moving
afterwards raises "_apply(): Couldn't swap <Module>.<param>". A real CUDA device
has no such rule; there the move happens in place. Under torch.no_grad(), or
before the first forward, rgpu behaves the same as CUDA.
"""

import itertools
import weakref

import torch

from . import session

_next_id = itertools.count(1)


def register(meta, tid=None):
    """Give a meta tensor an id, and free it on the server when it dies."""
    if tid is None:
        tid = next(_next_id)
    meta._rgpu_id = tid
    weakref.finalize(meta, session.pending_frees.append, tid)
    return tid


def id_of(meta):
    try:
        return meta._rgpu_id
    except AttributeError:
        raise RuntimeError("this tensor was never sent to the server") from None


def meta_like(dtype, shape, stride, offset=0):
    """A meta tensor with exactly this layout, offset included."""
    if offset == 0 or any(s == 0 for s in shape):
        return torch.empty_strided(shape, stride, dtype=dtype, device="meta")
    extent = offset + 1 + sum((s - 1) * st for s, st in zip(shape, stride))
    return torch.empty(extent, dtype=dtype, device="meta").as_strided(shape, stride, offset)


class RemoteTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, meta, requires_grad=False):
        r = torch.Tensor._make_wrapper_subclass(
            cls, meta.shape, strides=meta.stride(), storage_offset=meta.storage_offset(),
            dtype=meta.dtype, device=torch.device("rgpu", 0), requires_grad=requires_grad)
        r._rgpu_meta = meta
        return r

    def __init__(self, meta, requires_grad=False):
        pass

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from . import dispatch
        return dispatch.handle(func, args, kwargs or {})

    # Traceable, so torch.compile sees one graph rather than one per op: the
    # compiler traces with the meta tensor, which is all it needs.
    def __tensor_flatten__(self):
        return ["_rgpu_meta"], None

    @staticmethod
    def __tensor_unflatten__(inner, ctx, outer_size, outer_stride):
        return RemoteTensor(inner["_rgpu_meta"])

    def __repr__(self, *, tensor_contents=None):
        return "rgpu:" + repr(self.detach().cpu())
