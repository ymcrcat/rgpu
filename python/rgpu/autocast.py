"""Autocast for rgpu: which ops run in half precision and which stay float32.

PyTorch ships an autocast policy only for its own devices. A new device gets
none, and under autocast every op fails until something is registered at its
Autocast key. This registers a fallthrough for everything, then kernels for a
documented subset of what PyTorch registers for AutocastCUDA: LOWER runs in
the autocast dtype, FP32 widens half inputs back to float32, and BANNED
raises the way CUDA does. Everything else falls through uncast.

Falling through is right for most of what is missing, because much of CUDA's
list is CompositeImplicitAutograd and decomposes into ops that are covered -
einsum and layer_norm were checked and come out as they do on CUDA. These are
the differences that remain, where rgpu is not CUDA:

  - grid_sampler and the RNN cells (lstm_cell, gru_cell, rnn_tanh_cell,
    rnn_relu_cell, and the fused _thnn_* forms), which CUDA runs in the
    autocast dtype and rgpu leaves alone.
  - dot, vdot, bilinear and conv_tbc, likewise on CUDA's lower-precision list
    and not here.
  - CUDA's whole "promote" category, which runs an op at the widest dtype
    among its inputs: addcmul, addcdiv, atan2, cross, index_put, scatter_add
    and the upsample_* family. On rgpu a mix of float16 and float32 inputs
    reaches them unpromoted.
"""

import torch
from torch.utils._pytree import tree_map

# Ops that run in the autocast dtype (PyTorch's CUDA "lower precision" list).
LOWER = [
    "mm", "addmm", "bmm", "baddbmm", "addbmm", "addmv", "addr", "matmul", "mv",
    "linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
    "conv_transpose3d", "convolution", "_convolution", "prelu", "chain_matmul",
    "linalg_multi_dot", "scaled_dot_product_attention",
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

# Ops CUDA refuses under autocast rather than casting, because they are
# unsafe in half precision. Running one anyway would diverge silently from a
# local GPU, so rgpu refuses them with the same advice.
BANNED = {
    "binary_cross_entropy": (
        "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to "
        "autocast. Many models use a sigmoid layer right before the binary cross "
        "entropy layer. In this case, combine the two layers using "
        "torch.nn.functional.binary_cross_entropy_with_logits or "
        "torch.nn.BCEWithLogitsLoss. binary_cross_entropy_with_logits and "
        "BCEWithLogits are safe to autocast."),
}

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


def _refusing_kernel(message):
    def kernel(*args, **kwargs):
        raise RuntimeError(message)

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

    for name, message in BANNED.items():
        packet = getattr(torch.ops.aten, name, None)
        if packet is None:
            continue
        # Every overload, out= included: this kernel computes nothing, so the
        # reason out= variants are skipped above does not apply.
        for overload in packet.overloads():
            qualified = name if overload == "default" else f"{name}.{overload}"
            lib.impl(qualified, _refusing_kernel(message), "AutocastPrivateUse1")

    _libs.extend([everything, lib])
