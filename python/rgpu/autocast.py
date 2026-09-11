"""Autocast for rgpu: which ops run in half precision and which stay float32.

PyTorch ships an autocast policy only for its own devices. A new device gets
none, and under autocast every op fails until something is registered at its
Autocast key. This registers a fallthrough for everything, then cast kernels
for the same two lists PyTorch documents for CUDA, so mixed precision on
rgpu behaves as it would on a local GPU.
"""

import torch
from torch.utils._pytree import tree_map

# Ops that run in the autocast dtype (PyTorch's CUDA "lower precision" list).
LOWER = [
    "mm", "addmm", "bmm", "baddbmm", "addbmm", "addmv", "addr", "matmul", "mv",
    "linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
    "conv_transpose3d", "convolution", "_convolution", "prelu", "chain_matmul",
    "linalg_multi_dot",
]

# Ops whose half-precision inputs are widened to float32 (the CUDA fp32 list).
FP32 = [
    "acos", "asin", "cosh", "erfinv", "exp", "expm1", "log", "log10", "log2",
    "log1p", "reciprocal", "rsqrt", "sinh", "tan", "pow", "softplus", "layer_norm",
    "group_norm", "norm", "nuclear_norm", "cosine_similarity", "poisson_nll_loss",
    "cosine_embedding_loss", "nll_loss", "nll_loss2d", "hinge_embedding_loss",
    "kl_div", "l1_loss", "smooth_l1_loss", "huber_loss", "mse_loss",
    "margin_ranking_loss", "multilabel_margin_loss", "soft_margin_loss",
    "triplet_margin_loss", "multi_margin_loss", "binary_cross_entropy_with_logits",
    "dist", "pdist", "cdist", "renorm", "logsumexp", "softmax", "log_softmax",
    "cross_entropy_loss", "cumprod", "cumsum", "prod", "sum",
]

_KEY = torch._C.DispatchKey.AutocastPrivateUse1
_libs = []


def _kernel(op, target, eligible):
    def kernel(*args, **kwargs):
        dtype = target()

        def cast(x):
            if (isinstance(x, torch.Tensor) and x.device.type == "rgpu"
                    and eligible(x.dtype) and x.dtype != dtype):
                return x.to(dtype)
            return x

        with torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(_KEY)):
            return op(*tree_map(cast, args), **tree_map(cast, kwargs))

    return kernel


def _has_out_argument(schema):
    """True if any argument writes into a tensor the caller passed in.

    An out= kernel's cast() would allocate a fresh cast tensor, compute into
    that, and leave the caller's out tensor untouched - silently wrong
    values, no exception. PyTorch's own CUDA autocast policy does not cover
    out= variants either, so - like CUDA - they fall through to the uncast
    path instead of getting a cast kernel here.
    """
    return any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments)


def install():
    if _libs:
        return
    everything = torch.library.Library("_", "IMPL")
    everything.fallback(torch.library.fallthrough_kernel, "AutocastPrivateUse1")
    lib = torch.library.Library("aten", "IMPL")

    def lower_eligible(dt):
        return dt.is_floating_point and dt != torch.float64

    def fp32_eligible(dt):
        return dt in (torch.float16, torch.bfloat16)

    for names, target, eligible in [
        (LOWER, lambda: torch.get_autocast_dtype("rgpu"), lower_eligible),
        (FP32, lambda: torch.float32, fp32_eligible),
    ]:
        for name in names:
            packet = getattr(torch.ops.aten, name, None)
            if packet is None:
                continue
            for overload in packet.overloads():
                op = getattr(packet, overload)
                if _has_out_argument(op._schema):
                    continue
                qualified = name if overload == "default" else f"{name}.{overload}"
                lib.impl(qualified, _kernel(op, target, eligible), "AutocastPrivateUse1")
    _libs.extend([everything, lib])
