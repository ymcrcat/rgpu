"""Where every op on an rgpu tensor lands, after autograd.

Each op is one of four kinds, and only two of them wait for the server:

  stream     outputs worked out here on meta tensors, ids chosen here, the op
             queued - no wait. Almost everything.
  upload     a CPU tensor copied into an rgpu one; the bytes are taken now,
             so later changes to the CPU tensor cannot leak in - no wait.
  download   .cpu(), .item(), printing - one round trip.
  unknown    ops whose output size depends on the data - the server runs
             them and reports the shapes, one round trip. (Task 6.)

Meta kernels make the same shape and dtype checks as real ones, so most
mistakes raise here, on the line that made them, before anything is sent.
"""

import torch
from torch.utils._pytree import tree_flatten, tree_map

from . import session, wire
from .tensor import RemoteTensor, id_of, register

aten = torch.ops.aten


def meta_of(x):
    if isinstance(x, RemoteTensor):
        return x._rgpu_meta
    if isinstance(x, torch.Tensor):
        return x.to("meta")
    if isinstance(x, torch.device) and x.type == "rgpu":
        return torch.device("meta")
    return x


def wire_of(x):
    if isinstance(x, RemoteTensor):
        return wire.Ref(id_of(x._rgpu_meta))
    if isinstance(x, torch.Tensor):
        if x.device.type != "cpu":
            raise RuntimeError(f"cannot mix a {x.device} tensor with rgpu tensors")
        return wire.Host.of(x)
    if isinstance(x, torch.device):
        return wire.Dev("rgpu" if x.type == "rgpu" else str(x))
    return x


def _wire_args(args, kwargs):
    return tree_map(wire_of, list(args)), {k: tree_map(wire_of, v) for k, v in kwargs.items()}


def _download(t):
    return session.get().request(wire.DOWNLOAD, id_of(t._rgpu_meta)).tensor()


def _written_input(func, args, kwargs, ret):
    """The input an in-place or out= op writes to and returns."""
    want = ret.alias_info.before_set
    for i, a in enumerate(func._schema.arguments):
        if a.alias_info is not None and a.alias_info.is_write and a.alias_info.before_set == want:
            return args[i] if i < len(args) else kwargs[a.name]
    raise RuntimeError(f"cannot tell which input {func} writes to")


def _wrap_outputs(func, args, kwargs, out):
    """Wrap meta outputs, choosing ids, in the order the server will flatten them."""
    returns = func._schema.returns
    if not returns:
        return None, []
    single = len(returns) == 1
    outs = [out] if single else list(out)
    results, out_ids = [], []
    for ret, o in zip(returns, outs):
        if ret.alias_info is not None and ret.alias_info.is_write:
            results.append(_written_input(func, args, kwargs, ret))
            out_ids.extend([None] * len(tree_flatten(o)[0]))
            continue

        def one(m):
            if not isinstance(m, torch.Tensor):
                out_ids.append(None)
                return m
            if hasattr(m, "_rgpu_id"):
                out_ids.append(None)   # the op handed back one of its inputs
            else:
                out_ids.append(register(m))
            return RemoteTensor(m)

        results.append(tree_map(one, o))
    return (results[0] if single else tuple(results)), out_ids


def _writable_metas(func, args, kwargs):
    """Metas of plain-Tensor inputs the op writes to (an out= or in-place self).

    A Tensor(a!)[] written through a list is not covered - none of aten's
    resizing ops (out=, in-place) take a list there, only a lone Tensor(a!).
    """
    metas = []
    for i, a in enumerate(func._schema.arguments):
        if a.alias_info is None or not a.alias_info.is_write:
            continue
        v = args[i] if i < len(args) else kwargs.get(a.name)
        if isinstance(v, RemoteTensor):
            metas.append(v._rgpu_meta)
    return metas


def _run(func, args, kwargs):
    metas = _writable_metas(func, args, kwargs)
    before = [(m.shape, m.stride(), m.storage_offset()) for m in metas]
    out = func(*tree_map(meta_of, args), **tree_map(meta_of, kwargs))
    changed = False
    for m, (shape, stride, offset) in zip(metas, before):
        if m.shape != shape or m.stride() != stride or m.storage_offset() != offset:
            m.as_strided_(shape, stride, offset)
            changed = True
    if changed:
        raise NotImplementedError(
            f"rgpu cannot run {func}: it changes the shape of a tensor it writes to "
            "(resize an out= tensor to the right size first)")
    results, out_ids = _wrap_outputs(func, args, kwargs, out)
    a, kw = _wire_args(args, kwargs)
    session.get().post(wire.RUN, func._schema.name, func._overloadname, a, kw, out_ids)
    return results


def handle(func, args, kwargs):
    if func is aten._local_scalar_dense.default:
        return _download(args[0]).item()
    if func is aten._to_copy.default:
        target = kwargs.get("device")
        if target is not None and torch.device(target).type == "cpu":
            out = _download(args[0])
            dtype = kwargs.get("dtype")
            return out if dtype is None else out.to(dtype)
    if func is aten.copy_.default:
        dst, src = args[0], args[1]
        if not isinstance(dst, RemoteTensor):
            return dst.copy_(_download(src))
        if not isinstance(src, RemoteTensor):
            session.get().post(wire.UPLOAD, id_of(dst._rgpu_meta), wire.Host.of(src),
                               size=src.numel() * src.element_size())
            return dst
    return _run(func, args, kwargs)
