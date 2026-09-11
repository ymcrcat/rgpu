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
from .tensor import RemoteTensor, id_of, meta_like, register

aten = torch.ops.aten

# In-place ops that change a tensor's shape or strides rather than its
# values. A wrapper's metadata is fixed when it is made, so these would leave
# it describing the wrong tensor. Refused, clearly, until that is handled.
_METADATA_MUTATING = frozenset({
    "resize_", "resize_as_", "set_", "as_strided_", "t_", "transpose_", "squeeze_",
    "unsqueeze_", "swapaxes_", "swapdims_",
})

# Ops whose result always depends on the data. Their meta kernel does not fail
# with NotImplementedError (aten.equal's raises plain RuntimeError on torch
# 2.14), so routing them here just skips a meta attempt that can only fail.
_ALWAYS_SYNC = frozenset({aten.equal.default})


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


def _returns_only_inputs(func):
    returns = func._schema.returns
    return all(r.alias_info is not None and r.alias_info.is_write for r in returns)


def _run_sync(func, args, kwargs):
    """The server runs it and says what came out. Output ids are the server's."""
    if not _returns_only_inputs(func) and any(
            r.alias_info is not None and r.alias_info.is_write for r in func._schema.returns):
        raise NotImplementedError(
            f"rgpu cannot run {func}: it writes to an input and returns a new tensor "
            "whose size only the data can tell")
    a, kw = _wire_args(args, kwargs)
    value = session.get().request(wire.RUN_SYNC, func._schema.name, func._overloadname, a, kw)

    def rebuild(v):
        if isinstance(v, list) and v and v[0] == "__tensor__":
            _, tid, dtype, shape, stride, offset = v
            meta = meta_like(dtype, shape, stride, offset)
            register(meta, tid)
            return RemoteTensor(meta)
        if isinstance(v, list):
            return [rebuild(x) for x in v]
        return v

    out = rebuild(value)
    return tuple(out) if len(func._schema.returns) > 1 else out


def _run(func, args, kwargs):
    metas = _writable_metas(func, args, kwargs)
    before = [(m.shape, m.stride(), m.storage_offset()) for m in metas]
    try:
        out = func(*tree_map(meta_of, args), **tree_map(meta_of, kwargs))
    except NotImplementedError:
        if not _returns_only_inputs(func):
            return _run_sync(func, args, kwargs)
        # Nothing new comes out, so there would be no shape to infer even with
        # a meta kernel - but an out= tensor still needs to be the right size
        # before it is written, and only the data can say what that is.
        if any(a.alias_info is not None and a.alias_info.is_write and a.is_out
               for a in func._schema.arguments):
            raise NotImplementedError(
                f"rgpu cannot run {func}: its output size depends on the data, so it "
                "cannot write into an out= tensor; call it without out=")
        # An in-place (self-writing, non-out=) op with no meta kernel. No aten
        # op currently falls here - a survey of every in-place overload found
        # none missing a meta kernel - so this is refused rather than shipping
        # a streaming path that has never run against a real op.
        raise NotImplementedError(
            f"rgpu cannot run {func}: it has no meta kernel to infer its output from, "
            "and rgpu has no other way to run an in-place op without one")
    changed = False
    for m, (shape, stride, offset) in zip(metas, before):
        if m.shape != shape or m.stride() != stride or m.storage_offset() != offset:
            m.as_strided_(shape, stride, offset)
            changed = True
    if changed:
        raise NotImplementedError(
            f"rgpu cannot run {func}: it changes the shape of a tensor it writes to "
            "(allocate the out= tensor at the right size)")
    results, out_ids = _wrap_outputs(func, args, kwargs, out)
    a, kw = _wire_args(args, kwargs)
    session.get().post(wire.RUN, func._schema.name, func._overloadname, a, kw, out_ids)
    return results


def handle(func, args, kwargs):
    base = func._schema.name.split("::", 1)[1]
    if base in _METADATA_MUTATING:
        raise NotImplementedError(
            f"rgpu cannot run {func} yet: it changes a tensor's shape in place")
    if func in _ALWAYS_SYNC:
        return _run_sync(func, args, kwargs)
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
