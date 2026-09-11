"""torch.compile for rgpu: ship the graph once, then one message per call.

The backend hooks aot_autograd, which hands over forward and backward graphs
of aten ops. Each is serialized with the same op names and argument encoding
as eager ops, sent once, and compiled on the server. After that a call is a
single CALL message, and it streams: output shapes come from the traced
graph, so nothing has to wait for them.

Compiled code works on the unwrapped meta tensors, never on RemoteTensor
itself, which is why each tensor's id lives on its meta tensor: inputs are
looked up by it and outputs are given one.
"""

import itertools
import logging
import operator

import torch
from torch._dynamo.backends.common import aot_autograd
from torch.utils._pytree import tree_map

from . import session, wire
from .tensor import RemoteTensor, id_of, meta_like, register

log = logging.getLogger("rgpu")
_graph_ids = itertools.count(1)


class Unshippable(Exception):
    """A graph containing something the wire cannot carry."""


def serialize(gm):
    nodes, index = [], {}

    def ref(a):
        if isinstance(a, torch.fx.Node):
            return wire.NodeRef(index[a])
        if isinstance(a, torch.Tensor):
            raise Unshippable("a constant tensor inside the graph")
        if isinstance(a, torch.device):
            # An rgpu tensor is a meta tensor on this side, so a graph traced
            # through one allocates on "meta". On the server that means rgpu.
            return wire.Dev("rgpu" if a.type in ("rgpu", "meta") else str(a))
        if isinstance(a, (torch.SymInt, torch.SymFloat)):
            raise Unshippable("dynamic shapes; compile with dynamic=False")
        try:
            wire.encode(a)
        except wire.EncodeError as e:
            raise Unshippable(str(e)) from None
        return a

    for n in gm.graph.nodes:
        if n.op == "placeholder":
            nodes.append(["placeholder"])
        elif n.op == "call_function":
            t = n.target
            if t is operator.getitem:
                # Not an op: how a graph picks one output of an op that has
                # several, which aot_autograd writes for max_pool2d_with_indices,
                # native_batch_norm and the rest. It travels as its own kind of
                # node, so the server still runs nothing but aten ops.
                src, i = n.args
                if not isinstance(src, torch.fx.Node) or not isinstance(i, int):
                    raise Unshippable(f"getitem with a {type(i).__name__} index")
                nodes.append(["getitem", wire.NodeRef(index[src]), i])
                index[n] = len(nodes) - 1
                continue
            if not isinstance(t, torch._ops.OpOverload) or t.namespace != "aten":
                raise Unshippable(f"{t} is not an aten op")
            nodes.append(["call", t._schema.name, t._overloadname,
                          tree_map(ref, list(n.args)),
                          {k: tree_map(ref, v) for k, v in n.kwargs.items()}])
        elif n.op == "output":
            outs = n.args[0] if isinstance(n.args[0], (list, tuple)) else [n.args[0]]
            nodes.append(["output", [tree_map(ref, o) for o in outs]])
        else:
            raise Unshippable(f"a {n.op} node")
        index[n] = len(nodes) - 1
    return nodes


def _output_values(gm):
    out = next(n for n in gm.graph.nodes if n.op == "output")
    outs = out.args[0] if isinstance(out.args[0], (list, tuple)) else [out.args[0]]
    return [o.meta.get("val") if isinstance(o, torch.fx.Node) else o for o in outs]


def _boxed(fn):
    def run(args):
        return fn(*args)
    run._boxed_call = True
    return run


def _ship(gid, nodes, compiler, mode):
    session.get().post(wire.COMPILE, gid, nodes, compiler, mode)


def _eager(gm):
    """Run a graph that cannot be shipped op by op, through the normal path."""
    def run(*inner):
        wrapped = [RemoteTensor(x) if isinstance(x, torch.Tensor) else x for x in inner]
        outs = gm(*wrapped)
        return [o._rgpu_meta if isinstance(o, RemoteTensor) else o for o in outs]
    return _boxed(run)


def _compiler(mode, compiler):
    def compile_graph(gm, example_inputs):
        try:
            nodes = serialize(gm)
            values = _output_values(gm)
            if any(isinstance(s, torch.SymInt) for v in values
                   if isinstance(v, torch.Tensor) for s in v.shape):
                raise Unshippable("dynamic shapes; compile with dynamic=False")
        except Unshippable as e:
            log.warning("rgpu: running a graph eagerly because it has %s", e)
            return _eager(gm)
        gid = next(_graph_ids)
        _ship(gid, nodes, compiler, mode)

        def run(*inputs):
            in_ids = [id_of(x) for x in inputs]
            outs, out_ids = [], []
            for v in values:
                if isinstance(v, torch.Tensor):
                    m = meta_like(v.dtype, list(v.shape), list(v.stride()), v.storage_offset())
                    out_ids.append(register(m))
                    outs.append(m)
                else:
                    out_ids.append(None)
                    outs.append(v)
            session.get().post(wire.CALL, gid, in_ids, out_ids)
            return outs

        return _boxed(run)

    return compile_graph


def compile_backend(mode=None, compiler="inductor"):
    """A torch.compile backend that compiles on the rgpu server."""
    comp = _compiler(mode, compiler)
    return aot_autograd(fw_compiler=comp, bw_compiler=comp)


torch._dynamo.register_backend(name="rgpu")(compile_backend())
