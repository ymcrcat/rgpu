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
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor

from . import session

_next_id = itertools.count(1)

# What a tracer puts in place of the meta tensor: dynamo and aot_autograd trace
# with fake tensors, and aot_autograd functionalizes on top of them. A tensor
# holding one of these has no data anywhere yet, and no id.
_TRACED = (FakeTensor, FunctionalTensor)


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
        # Overwhelmingly this is torch.compile with the wrong backend. The
        # default is inductor, which cannot work here by construction: it
        # generates code that allocates its own tensors, and rgpu never sees
        # them, so the first one to reach the wire has no id. Saying only that
        # a tensor was never sent describes a symptom several layers below the
        # cause, and leaves the user with nothing to change.
        raise RuntimeError(
            "this tensor was never sent to the server. If you called "
            "torch.compile, pass backend=\"rgpu\", dynamic=False: other "
            "backends (inductor is the default) compile to code that "
            "allocates tensors rgpu never sees, so they cannot run on an "
            "rgpu tensor") from None


def is_traced(t):
    """True while torch.compile traces this tensor: its meta is the tracer's."""
    return isinstance(t, RemoteTensor) and isinstance(t._rgpu_meta, _TRACED)


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
        # The conjugate and negative bits are metadata, like the strides, and
        # _make_wrapper_subclass has no argument for them. They have to be on
        # the wrapper and not only on the meta inside it: torch reads them
        # above __torch_dispatch__ - conj() is a bit flip, and .real, .imag
        # and resolve_conj() each branch on the bit before any op is
        # dispatched - so a wrapper without them describes a different tensor
        # from the one the server holds.
        if meta.is_conj():
            torch._C._set_conj(r, True)
        if meta.is_neg():
            torch._C._set_neg(r, True)
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

    def tolist(self):
        # torch.Tensor.tolist refuses a subclass outright, so this download
        # has to be spelled out here the way .cpu() and .item() are.
        return self.detach().cpu().tolist()

    def __repr__(self, *, tensor_contents=None):
        if is_traced(self):
            return f"rgpu:<traced {self.dtype} {tuple(self.shape)}>"   # no data to fetch
        return "rgpu:" + repr(self.detach().cpu())
