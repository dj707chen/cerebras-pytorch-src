# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Iterable

import torch
import torch._prims_common as utils
from torch._decomp.decompositions import (
    _reflection_pad,
    hardswish,
    index_add,
    index_add_,
    softplus_backward,
)
from torch._prims_common import TensorLikeType
from torch._refs import dot, roll, std, unbind
from torch._refs.nn.functional import glu
from torch.jit._decompositions import var_decomposition as var

from cerebras.pytorch.decomp.registry import register_decomposition

# to be decorated on decompositions for ops that have overloads with extra parameters
from torch._prims_common.wrappers import out_wrapper  # noqa

aten = torch._ops.ops.aten

register_decomposition(aten.dot)(dot)
register_decomposition(aten.glu)(glu)
register_decomposition(aten.hardswish)(hardswish)
register_decomposition(aten.index_add)(index_add)
register_decomposition(aten.index_add_)(index_add_)
register_decomposition(aten.reflection_pad1d)(_reflection_pad)
register_decomposition(aten.reflection_pad2d)(_reflection_pad)
register_decomposition(aten.roll)(roll)
register_decomposition(aten.softplus_backward)(softplus_backward)
register_decomposition(aten.std.correction)(std)
register_decomposition(aten.var.correction)(var)
register_decomposition(aten.unbind)(unbind)


@register_decomposition(aten.acos)
def acos(x: torch.Tensor) -> torch.Tensor:
    """
    Reference - https://developer.download.nvidia.com/cg/acos.html
    """
    dtype = x.dtype
    x = x.to(torch.float32)

    negate = (x < 0).float()
    x = x.abs()

    ret = -0.0187293
    ret = ret * x
    ret = ret + 0.0742610
    ret = ret * x
    ret = ret - 0.2121144
    ret = ret * x
    ret = ret + 1.5707288
    ret = ret * torch.sqrt(1.0 - x)
    ret = ret - 2 * negate * ret

    return (negate * 3.14159265358979 + ret).to(dtype)


@register_decomposition(aten.asin)
def asin(x: torch.Tensor) -> torch.Tensor:
    """
    Reference - https://developer.download.nvidia.com/cg/asin.html
    """
    dtype = x.dtype
    x = x.to(torch.float32)

    negate = (x < 0).float()
    x = x.abs()

    ret = -0.0187293
    ret = ret * x
    ret = ret + 0.0742610
    ret = ret * x
    ret = ret - 0.2121144
    ret = ret * x
    ret = ret + 1.5707288
    ret = 1.57079632679 - torch.sqrt(1.0 - x) * ret
    ret = ret - 2.0 * negate * ret

    return ret.to(dtype)


@register_decomposition(aten._weight_norm_interface)
def _weight_norm_interface(v, g, dim=0):
    """
    Decomposition for torch.aten._weight_norm_interface copied from PyTorch:
        https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L4993-L4999

    This implementation removes the line that does dtype checking:
        norm_dtype = torch.float if g.dtype == torch.bfloat16 else None
    as it is CUDA specific.
    """
    keep_dim = tuple(i for i in range(len(v.shape)) if i != dim)
    norm = v.norm(2, keep_dim, keepdim=True)
    return v * (g / norm.to(g.dtype)), norm


# NOTE: this decomposition is not registered by default and requires manual registration whenever needed:
#     register_decomposition(torch._ops.ops.aten.native_layer_norm)(native_layer_norm)
def native_layer_norm(input, normalized_shape, weight, bias, eps):
    """
    Decomposition for torch.aten.native_layer_norm mostly copied from PyTorch:
    https://github.com/pytorch/pytorch/blob/main/torch/_refs/__init__.py#L3209

    1. PyTorch implementation always casts output to input type, but in this implementation,
    the type of output and weight, bias computations is based on POL setting.

    2. In PyTorch implementation, when computing mean of input tensor, it decomposes aten.mean
    further and calls torch._prims.sum, but we don't support this. We call torch.mean instead.

    3. In PyTorch implementation, when computing variance of input tensor, it decomposes aten.var
    further and calls torch._prims.var, but we don't support this. we compute the variance based on
    PyTorch torch.var definition with correction=0
    https://pytorch.org/docs/stable/generated/torch.var.html.
    """
    normalized_ndim = len(normalized_shape)
    torch._check(
        normalized_ndim >= 1,
        lambda: "Expected normalized_shape to be at least 1-dimensional, i.e., "
        + "containing at least one element, but got normalized_shape = "
        + str(normalized_shape),
    )
    torch._check(
        weight is None or weight.shape == tuple(normalized_shape),
        lambda: "Expected weight to be of same shape as normalized_shape, but got "
        + "weight of shape "
        + str(weight.shape)
        + " and normalized_shape = "
        + str(normalized_shape),
    )
    torch._check(
        bias is None or bias.shape == tuple(normalized_shape),
        lambda: "Expected bias to be of same shape as normalized_shape, but got "
        + "bias of shape "
        + str(bias.shape)
        + " and normalized_shape = "
        + str(normalized_shape),
    )
    torch._check(
        input.ndim >= normalized_ndim
        and input.shape[(input.ndim - normalized_ndim) :]
        == tuple(normalized_shape),
        lambda: "Given normalized_shape="
        + str(normalized_shape)
        + ", expected input with shape "
        + str(normalized_shape)
        + ", but got input of size "
        + str(input.shape),
    )
    input = input.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    axis = input.ndim - normalized_ndim
    reduction_dims = list(range(axis, input.ndim))

    a = input.to(torch.float32)
    mean = torch.mean(a, dim=reduction_dims, keepdim=True)
    var = torch.mean((a - mean) * (a - mean), dim=reduction_dims, keepdim=True)

    rstd = torch.rsqrt(var + eps)
    out = (a - mean) * rstd

    # Query the per-layer POL of the model.
    import cerebras.pytorch as cstorch

    pol = cstorch.current_pol()
    # If pol is 1 or 2, we always set the output_type to half type regardless the
    # input, to align with our layernorm kernel.
    if pol == 0:
        output_type = input.dtype
    else:
        output_type = cstorch.amp.get_half_dtype()
    out = out.to(output_type)
    if weight is None and bias is not None:
        out = out + bias.to(output_type)
    elif weight is not None and bias is None:
        out = out * weight.to(output_type)
    elif weight is not None and bias is not None:
        out = out * weight.to(output_type) + bias.to(output_type)

    return (out, mean, rstd)


@register_decomposition(aten.atan)
def atan(input_tensor):
    one_tensor = torch.ones(
        1, dtype=input_tensor.dtype, device=input_tensor.device
    )
    return torch.atan2(input_tensor, one_tensor)


@register_decomposition(aten.cosh)
def cosh(input_tensor):
    exp_x = torch.exp(input_tensor)
    exp_neg_x = torch.exp(-input_tensor)
    return (exp_x + exp_neg_x) / 2


@register_decomposition(aten.acosh)
def acosh(input_tensor):
    sqrt_sub_one = torch.sqrt(input_tensor**2 - 1)
    return torch.log(input_tensor + sqrt_sub_one)


@register_decomposition(aten.log1p)
def log1p(input_tensor):
    input_tensor = input_tensor + 1
    return torch.log(input_tensor)


@register_decomposition(aten.log10)
def log10(input_tensor):
    multiplication_constant = 0.4343  # equivalent to (1 / log(10))
    return torch.log(input_tensor) * multiplication_constant


@register_decomposition(aten.log2)
def log2(input_tensor):
    multiplication_constant = 1.4427  # equivalent to (1 / log(2))
    return torch.log(input_tensor) * multiplication_constant


@register_decomposition(aten.squeeze.dims)
def squeeze(input_tensor, dim):
    """
    Decomposition for torch.aten.squeeze.dims adapted from PyTorch:
    https://github.com/pytorch/pytorch/blob/baba7beed2b58a976f5ffab3db5eef2a269296c4/torch/_inductor/lowering.py#L1002-L1024
    """
    dims = utils.canonicalize_dims(input_tensor.dim(), dim)
    input_shape = list(input_tensor.shape)
    new_shape = []
    for d, s in enumerate(input_shape):
        if not (d in dims and s == 1):
            new_shape.append(s)

    # squeeze does nothing if the size isn't 1
    return (
        aten.view(input_tensor, new_shape)
        if new_shape != input_shape
        else input_tensor
    )


@register_decomposition(aten.tan)
def tan(input_tensor):
    return torch.sin(input_tensor) / torch.cos(input_tensor)


@register_decomposition(aten.triu)
def triu(a: TensorLikeType, diagonal: int = 0) -> TensorLikeType:
    torch._check(
        a.ndim >= 2,
        lambda: "triu: input tensor must have at least 2 dimensions",
    )
    h, w = a.shape[-2:]

    # Issue: Retaining two CIRH::Arange operations that generate row and column tensors
    # in CIRH passes, where both outputs have identical sizes and values.
    #
    # Solution:
    # - When the height & width are same and input is 2D, we manually increment the 1D row tensor by 1.
    #   After reshaping it into 2D, we subtract 1 to introduce differentiation
    #   and ensure both CIRH::Arange operations are retained.

    if h == w and a.ndim == 2:
        # Create index tensors
        row_idx = torch.arange(h, device=a.device, dtype=a.dtype) + 1
        col_idx = torch.arange(w, device=a.device, dtype=a.dtype)

        # Create mask using broadcasting
        mask = (row_idx.view(-1, 1) - 1) <= col_idx - diagonal
    else:
        mask = (
            torch.arange(w, device=a.device).unsqueeze(-2).to(a.dtype)
            - torch.arange(h, device=a.device).unsqueeze(-1).to(a.dtype)
        ) >= diagonal

    # aten.triu always returns a new contiguous tensor
    # contiguous() is needed to correctly model the output stride
    return utils.mask_tensor(mask, a).contiguous()


@register_decomposition(aten.sinh)
def sinh(input_tensor):
    exp_x = torch.exp(input_tensor)
    exp_neg_x = torch.exp(-input_tensor)
    return (exp_x - exp_neg_x) / 2


@register_decomposition(aten.cdist)
def cdist(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p: float = 2.0,
    compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
):
    """
    Decomposition for torch.aten.cdist.
    Args:
        x1: Tensor [..., P, M]
        x2: Tensor [..., R, M]
        p:  p in p-norm distance
        compute_mode:
            use_mm_for_euclid_dist[1]:
                If True and p=2.0, use the formula:
                    ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1·x2
                which avoids the full pairwise broadcast.
            use_mm_for_euclid_dist_if_necessary:
                If True and p=2.0, use use_mm_for_euclid_dist
            donot_use_mm_for_euclid_dist[2]:
                Doesn't use matrix mulitplication for euclidean distance calculation
    Returns:
        A tensor of shape [..., P, R] with the pairwise distances.
    """
    # Basic checks
    torch._check(
        len(x1.shape) >= 2,
        lambda: "cdist only supports at least 2D tensors, "
        + "X1 got: "
        + str(len(x1.shape))
        + "D",
    )
    torch._check(
        len(x2.shape) >= 2,
        lambda: "cdist only supports at least 2D tensors, "
        + "X2 got: "
        + str(len(x2.shape))
        + "D",
    )
    torch._check(
        x1.shape[-1] == x2.shape[-1],
        lambda: "X1 and X2 must have the same number of columns. "
        + "X1 = "
        + str(x1.shape[-1])
        + ", X2 = "
        + str(x2.shape[-1]),
    )
    use_mm_for_euclid_dist = compute_mode in ("use_mm_for_euclid_dist", 1) or (
        compute_mode == "use_mm_for_euclid_dist_if_necessary"
        and x1.shape[-2] > 25
        and x2.shape[-2] > 25
    )
    if (abs(p - 2.0)) < 1e-7 and use_mm_for_euclid_dist:
        # --------------------------------------------------
        # 1) Precompute squared norms of x1, x2
        #    shape(x1): [..., P, M], shape(x2): [..., R, M]
        # --------------------------------------------------
        # sum along the last dimension => shape([..., P]) or ([..., R])
        x1_sq = x1.pow(2).sum(dim=-1, keepdim=True)  # [..., P, 1]
        x2_sq = x2.pow(2).sum(dim=-1, keepdim=True)  # [..., R, 1]

        # --------------------------------------------------
        # 2) Compute x1 dot x2 using matmul
        #    x1: [..., P, M]
        #    x2.transpose(-1, -2): [..., M, R]
        #    => matmul => shape([..., P, R])
        # --------------------------------------------------
        # NOTE: This automatically handles any number of batch dims in front.
        dot = torch.matmul(x1, x2.transpose(-1, -2).contiguous())  # [..., P, R]

        # --------------------------------------------------
        # 3) Combine to get dist^2, then sqrt
        # dist_sq = x1_sq + x2_sq - 2 * dot
        # x1_sq: [..., P, 1]
        # x2_sq: [..., R, 1] => for broadcasting, we do x2_sq.transpose(-2, -1)
        x2_sq_bcast = x2_sq.transpose(-2, -1)  # => [..., 1, R]
        dist_sq = x1_sq + x2_sq_bcast - 2 * dot

        # Numerical safety: clamp any tiny negative values to 0
        dist_sq = dist_sq.clamp_min(0.0)
        dist = dist_sq.sqrt()  # Euclidean
        return dist

    else:
        x1_expanded = x1.unsqueeze(-2)  # => [..., P, 1, M]
        x2_expanded = x2.unsqueeze(-3)  # => [..., 1, R, M]
        if p == 0:
            diff = x1_expanded != x2_expanded  # Element-wise comparison
            return diff.to(torch.float32).sum(
                dim=-1
            )  # Count of non-zero differences
        diff = x1_expanded - x2_expanded
        abs_diff = diff.abs()
        dist_p = abs_diff.pow(p).sum(dim=-1)  # => [..., P, R]

        if abs(p - 2.0) < 1e-7:
            return dist_p.sqrt()  # Euclidean norm
        else:
            return dist_p.pow(1.0 / p)


@register_decomposition(aten.angle)
def angle(input_tensor):

    assert input_tensor.dtype not in (
        torch.complex64,
        torch.complex32,
    ), "Complex datatypes are not yet supported."
    real_part = input_tensor
    imag_part = torch.zeros(
        input_tensor.size(),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    return torch.atan2(imag_part, real_part)


@register_decomposition(aten.flip)
def flip(a, dims):
    if not isinstance(dims, tuple) and not isinstance(dims, list):
        raise ValueError("dims has to be a sequence of ints")
    dims = utils.canonicalize_dims(a.ndim, dims)  # type: ignore[assignment]
    utils.validate_no_repeating_dims(dims)

    for dim in dims:
        # Create reversed indices for the current dim
        indices = torch.arange(a.size(dim) - 1, -1, -1, device=a.device)
        a = torch.index_select(a, dim, indices)
    return a


@register_decomposition(aten.logsumexp)
def logsumexp(self, dim, keepdim=False):
    """
    Reference:
    https://github.com/pytorch/pytorch/blob/c7515da7b00de40942c83dc5856b6daec727e280/torch/_refs/__init__.py#L824
    """
    if not isinstance(dim, Iterable):
        dim = (dim,)
    if self.numel() == 0:
        return torch.sum(torch.exp(self), dim, keepdim).log()
    maxes = torch.amax(torch.real(self), dim, keepdim=True)
    maxes = torch.masked_fill(maxes, maxes.abs() == float("inf"), 0)
    # Note: use torch.amax instead of torch.squeeze to work around missing
    # support for aten.squeeze_copy.dims
    maxes_squeezed = maxes if keepdim else torch.amax(maxes, dim)
    result = torch.sum(torch.exp(self - maxes), dim, keepdim)
    return result.log().add(maxes_squeezed)


@register_decomposition(aten.expm1)
def expm1(input_tensor):
    return torch.exp(input_tensor) - 1
