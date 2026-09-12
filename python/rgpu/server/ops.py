"""Which ops the server will run, found by name and nothing else.

Only aten ops, looked up through torch.ops.aten by name and overload. There
is no general attribute lookup on torch and nothing a client names is ever
imported. A few aten ops reach outside the GPU; those are refused outright.
"""

import torch

# from_file reads a file from the server's disk into a tensor.
BLOCKED = frozenset({"aten::from_file"})


class Refused(Exception):
    """An op the server will not run."""


def resolve(name, overload):
    if name in BLOCKED:
        raise Refused(f"{name} is not allowed on rgpu-opserver")
    ns, sep, op = name.partition("::")
    if (ns != "aten" or not sep or not op.isidentifier() or op.startswith("__")
            or not overload.isidentifier() or overload.startswith("__")):
        raise Refused(f"{name}.{overload} is not an aten op")
    try:
        found = getattr(getattr(torch.ops.aten, op), overload)
    except (AttributeError, RuntimeError):
        raise Refused(f"no such op {name}.{overload}") from None
    if not isinstance(found, torch._ops.OpOverload):
        raise Refused(f"{name}.{overload} is not an aten op")
    return found
