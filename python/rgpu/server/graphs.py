"""Rebuild a graph the client shipped, and compile it for this server.

A graph arrives as a list of nodes whose ops are aten ops named exactly as in
a RUN message, and which are looked up the same way: the compile path gives
a client no op it could not already run one at a time.
"""

import torch
from torch.utils._pytree import tree_map

from .. import wire
from . import ops


def build(nodes, session, compiler, mode):
    graph = torch.fx.Graph()
    env = []

    def arg(a):
        if isinstance(a, wire.NodeRef):
            return env[a.index]
        if isinstance(a, wire.Dev):
            return session._device(a.name)
        if isinstance(a, (wire.Ref, wire.Host)):
            raise ValueError("a graph cannot carry tensors")
        return a

    for node in nodes:
        kind = node[0]
        if kind == "placeholder":
            env.append(graph.placeholder(f"arg{len(env)}"))
        elif kind == "call":
            _, name, overload, args, kwargs = node
            op = ops.resolve(name, overload)
            env.append(graph.call_function(
                op, tuple(tree_map(arg, args)), {k: tree_map(arg, v) for k, v in kwargs.items()}))
        elif kind == "output":
            graph.output(tuple(tree_map(arg, node[1])))
            env.append(None)
        else:
            raise ValueError(f"unknown node kind {kind!r}")

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    if compiler == "eager":
        return gm
    if compiler == "inductor":
        return torch.compile(gm, mode=mode, dynamic=False)
    raise ValueError(f"unknown compiler {compiler!r}")
