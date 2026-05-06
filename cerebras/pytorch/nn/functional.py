"""Cerebras specific functional op implementations."""

# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

import torch

import cerebras.pytorch as cstorch
from cerebras.appliance import logger

# Pass in value for optional attributes in cirh ops
SKIP_OPTIONAL_ATTR = "SKIP_OPTIONAL_ATTR"


def one_hot(array, num_classes):
    """Cerebras specific implementation of one_hot"""
    if num_classes == -1:
        logger.error("num_class argument to one_hot cannot be -1")
    init = torch.zeros(
        array.shape + (num_classes,), device=array.device, dtype=torch.int
    )
    res = init.scatter_(-1, array.unsqueeze(-1), 1)
    return res


# CIRH ScopeBoundary op boundary_type enum
BEGIN_FORWARD = 'BEGIN_FORWARD'
BEGIN_BACKWARD = 'BEGIN_BACKWARD'
END_FORWARD = 'END_FORWARD'
END_BACKWARD = 'END_BACKWARD'


def scope_boundary(input, boundary_type, scope_name):
    """
    This function is used to set a boundary after input, or place the cirh.ScopeBoundary op
    after `input` in the CIRH graph.

    Args:
        boundary_type (str): The type of the boundary. One of `BEGIN_FORWARD`, 'BEGIN_BACKWARD',
            'END_FORWARD`, or `END_BACKWARD`.
        scope_name (str): The name of the scope.
    """

    if cstorch.use_cs():
        from cerebras.pytorch import cirh

        return cirh.ScopeBoundary(
            [input],
            boundary_type=boundary_type,
            scope_name=scope_name,
        )[0]
    return input


def enter_scope(input, scope_name):
    """
    This module is used as a wrapper function of 'EnterFunction' autograd functions,
    which can set the "BEGIN" boundaries in CIRH graph.
    """
    return EnterFunction.apply(input, scope_name)


def exit_scope(input, scope_name):
    """
    This module is used as a wrapper function of 'ExitFunction' autograd functions,
    which can set the "END" boundaries in CIRH graph.
    """
    return ExitFunction.apply(input, scope_name)


class EnterFunction(torch.autograd.Function):
    """
    This module is used to set a boundary after 'input'. In the foward pass, the type of
    boundary is BEGIN_FORWARD. In the backward, the type of boundary is END_BACKWARD.

    `scope_boundary()` is used to invoke the custom call to generate cirh.ScopeBoundary.
    """

    @staticmethod
    def forward(ctx, input, scope_name):
        ctx.scope_name = scope_name
        return scope_boundary(input, BEGIN_FORWARD, scope_name)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            scope_boundary(grad_output, END_BACKWARD, ctx.scope_name),
            None,
            None,
        )


class ExitFunction(torch.autograd.Function):
    """
    This module is used to set a boundary after 'input'. In the foward pass, the type of
    boundary is END_FORWARD. In the backward, the type of boundary is BEGIN_BACKWARD.

    `scope_boundary()` is used to invoke the custom call to generate cirh.ScopeBoundary.
    """

    @staticmethod
    def forward(ctx, input, scope_name):
        ctx.scope_name = scope_name
        return scope_boundary(input, END_FORWARD, scope_name)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            scope_boundary(grad_output, BEGIN_BACKWARD, ctx.scope_name),
            None,
            None,
        )


class CSXSparseMatMul(torch.autograd.Function):
    """CSX SparseMatMul Op."""

    @staticmethod
    def forward(ctx, input_values, input_indices, weight):
        ctx.save_for_backward(input_values, input_indices, weight)
        return cstorch.cirh.SparseMatMul(input_values, input_indices, weight)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_values, grad_weight = cstorch.cirh.SparseMatMulGrad(
            grad_output, *ctx.saved_tensors
        )
        grad_input_indices = None

        return grad_input_values, grad_input_indices, grad_weight


def sparse_matmul(
    input_values: torch.Tensor,
    input_indices: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return sparse output from matmul of sparse input and dense weight.

    Consider the batch matmul `einsum('...BMN, BNK -> ...BMK', in0, in1)`. For
    `in0`, if the dim `B` is sparse with only `b` values present, we can
    accelerate it with this function. The sparse representation of `in0` would
    be `in0_values` of shape (..., b, M, N) and `in0_indices` of shape (..., b,
    N, K). The relation between them is,
        ```
        in0 = torch.scatter(zeros, -3, in0_broadcasted_indices, in0_values))
        ```
    where in0_broadcasted_indices is in0_indices broadcasted to in0_values
    shape. Similarly the full dense output can be got from,
        ```
        out = torch.scatter(zeros, -3, out_broadcasted_indices, out_values))
        ```
    where out_broadcasted_indices is in0_indices broadcasted to out_values shape.

    Note the expected layout of `input` and `weight` are different from `in0`
    and `in1` used for explanation.

    Requires `input_indices` to fit in uint16 tensor.

    Args:
        input_values: Sparse input values.
        input_indices: Sparse input indices.
        weight: Dense weight.

    Returns:
        output_values: Sparse output values.

    Shapes:
        input_values: (..., M, compressed_sparse_dim, N)
        input_indices: (..., M, compressed_sparse_dim)
        weight: (K, full_sparse_dim, N)
        output_values: (..., M, compressed_sparse_dim, K)
    """
    if weight.size(1) >= 2**16:
        raise NotImplemented("Requires `input_indices` to fit in uint16 tensor")

    if cstorch.use_cs():
        return CSXSparseMatMul.apply(input_values, input_indices, weight)

    sparse_dim = -2
    compressed_sparse_dim_size = input_values.shape[sparse_dim]
    full_sparse_dim_size = weight.shape[sparse_dim]
    dense_input_shape = (
        *input_values.shape[:sparse_dim],
        full_sparse_dim_size,
        *input_values.shape[sparse_dim + 1 :],
    )
    indices = input_indices[..., None]

    # Scatter into dense input.
    dense_input = torch.zeros(
        dense_input_shape, dtype=input_values.dtype, device=input_values.device
    )
    dense_input.scatter_(
        sparse_dim, indices.broadcast_to(input_values.shape), input_values
    )

    output = torch.einsum("...MBN, KBN -> ...MBK", dense_input, weight)

    # Gather into sparse output.
    output_values_shape = (*input_values.shape[:-1], weight.shape[0])
    output_values = torch.gather(
        output, sparse_dim, indices.broadcast_to(output_values_shape)
    )
    return output_values


class CSXSparseActMatMul(torch.autograd.Function):
    """CSX SparseActMatMul Op."""

    @staticmethod
    def forward(ctx, input_value, input_mask, weight, dense_capacity):
        ctx.save_for_backward(input_value, input_mask, weight)
        ctx.dense_capacity = dense_capacity
        return cstorch.cirh.SparseActMatMul(
            input_value, input_mask, weight, denseCapacity=dense_capacity
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_value, grad_weight = cstorch.cirh.SparseActMatMulGrad(
            grad_output, *ctx.saved_tensors, denseCapacity=ctx.dense_capacity
        )
        grad_input_mask = None
        grad_dense_capacity = None

        return (
            grad_input_value,
            grad_input_mask,
            grad_weight,
            grad_dense_capacity,
        )


def sparse_act_matmul(
    input_value: torch.Tensor,
    input_mask: torch.Tensor,
    weight: torch.Tensor,
    dense_capacity: Optional[int] = None,
) -> torch.Tensor:
    """Applies "row-level" sparse mask to input. Then performs matmul on
    sparse input and dense weight. In the "row-level" input_mask a 1 at pos "i"
    indicates that the ith (of M) rows should be used in the computation,
    while a 0 indicates that it should be ignored. An optional `dense_capacity`
    integer parameter can also be provided to specify the maximum number of rows
    guaranteed to be used in the computation.

    On CSX: Performance speedup is proportional to sparsity applied to input.

    In order to use this kernel the user must call CSXSparseActMatMul.apply(...).
    PyTorch primitives (matmul, einsum, etc.) will not map to the CSX optimized
    implementation.

    Args:
        input_value: Dense input value.
        input_mask: Token level sparse mask. 1 to use token in matmul, 0 otherwise
        weight: Dense weight.
        dense_capacity: Optional integer parameter that specifies the maximum number of dense rows

    Returns:
        output_values: Dense output value.

    Shapes:
        input_value: (..., M, N)
        input_mask: (..., M)
        weight: (K, N)
        output_values: (..., M, K)
    """

    assert (
        input_value.shape[:-1] == input_mask.shape
    ), f"Expect input_value and input_mask to have the same shape (except for last dim of input_value).\nInstead got input_value.shape = {input_value.shape} and input_mask.shape = {input_mask.shape}"

    """Optimized CSX implementation"""
    if cstorch.use_cs():
        return CSXSparseActMatMul.apply(
            input_value, input_mask, weight, dense_capacity
        )

    """Non-CSX functional implementation"""
    output_value = torch.einsum("...MN, KN -> ...MK", input_value, weight)
    output_value = output_value * input_mask.unsqueeze(-1).broadcast_to(
        output_value.shape
    ).to(output_value.dtype)

    return output_value


class CSXBiasAdd(torch.autograd.Function):
    """CSX BiasAdd Op."""

    @staticmethod
    def forward(ctx, input_value, bias, dim):
        ctx.save_for_backward(bias)
        ctx.dim = dim
        return cstorch.cirh.BiasAdd(input_value, bias, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        (bias,) = ctx.saved_tensors

        grad_input_value = grad_output
        grad_bias = cstorch.cirh.ReduceBias(grad_output, bias, dim=ctx.dim)

        return grad_input_value, grad_bias, None


def bias_add(
    input_value: torch.Tensor, bias: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    Broadcasts a 1D bias weight along `dim` and adds to the `input_value`.
    Maps directly to an optimized CSX implementation on the forwards and backwards pass.

    Args:
        bias: Dense weight.
        input_value: Dense input value.
        dim: integer.

    Returns:
        output_values: Dense output value.

    Shapes:
        input_value: (..., H, ...)
        bias: (H)
        dim: ()
        output_value: (..., H, ...)
    """

    assert (
        len(bias.shape) == 1
    ), f"Expected 1D bias weight. Instead got bias.shape = {bias.shape}"

    assert (
        len(input_value.shape) >= 2
    ), f"Expected >= 2D input_value. Instead got input_value.shape = {input_value.shape}"

    assert dim >= 0 and dim <= len(
        input_value.shape
    ), f"`dim` must in [0, len(input_value.shape) - 1]. Instead got: 0 <= dim={dim} <= {len(input_value.shape)}"

    assert (
        bias.shape[0] == input_value.shape[dim]
    ), f"Expect bias.shape[0] == input_value.shape[dim={dim}]. Instead got bias.shape = {bias.shape} and input_value.shape = {input_value.shape}"

    """Optimized CSX implementation"""
    if cstorch.use_cs():
        return CSXBiasAdd.apply(input_value, bias, dim)

    """Non-CSX functional implementation"""
    view_shape = [1] * input_value.dim()
    view_shape[dim] = bias.shape[0]
    bias_expanded = bias.view(view_shape).expand(input_value.shape)
    return input_value + bias_expanded


class CSXMatMul(torch.autograd.Function):
    """CSX MatMul Op."""

    @staticmethod
    def forward(ctx, weight, input_value, weight_transposed):
        ctx.save_for_backward(weight, input_value, weight_transposed)
        return torch.matmul(weight, input_value)

    @staticmethod
    def backward(ctx, grad_output):
        weight, input_value, weight_transposed = ctx.saved_tensors

        grad_input_value = torch.matmul(weight_transposed, grad_output)
        grad_weight = cstorch.cirh.DMatMul(grad_output, input_value, weight)

        return grad_weight, grad_input_value, None


def matmul(
    weight: torch.Tensor,
    input_value: torch.Tensor,
    weight_transposed: torch.Tensor = None,
) -> torch.Tensor:
    """
    Performs [Ho][Hi] x [Hi][S] = [Ho][S] matmul.

    Maps directly to an optimized CSX implementation on the forwards and backwards pass.

    Args:
        weight: Dense weight.
        input_value: Dense input value.
        weight_transposed: Transpose of weight for backwards pass (optional)

    Returns:
        output_values: Dense output value.

    Shapes:
        weight: (Ho, Hi)
        input_value: (Hi, S)
        weight_transposed: (Hi, Ho)
        output_value: (Ho, S)
    """

    assert (
        len(weight.shape) == 2
    ), f"Expected 2D weight. Instead got weight.shape = {weight.shape}"

    assert (
        len(input_value.shape) == 2
    ), f"Expected 2D input_value. Instead got input_value.shape = {input_value.shape}"

    assert (
        weight.shape[1] == input_value.shape[0]
    ), f"Expect weight.shape(1) == input_value.shape[0]. Instead got weight.shape = {weight.shape} and input_value.shape = {input_value.shape}"

    if weight_transposed is not None:
        assert (
            weight_transposed.shape == torch.transpose(weight, 1, 0).shape
        ), f"Expected weight_transposed.shape to be the transpose of weight.shape. {weight_transposed.shape} != transpose({weight.shape}, 1, 0)."
    else:
        weight_transposed = torch.transpose(weight, 1, 0)

    """Optimized CSX implementation"""
    if cstorch.use_cs():
        return CSXMatMul.apply(weight, input_value, weight_transposed)

    """Non-CSX functional implementation"""
    return torch.matmul(weight, input_value)


class CSXRangedMatMul(torch.autograd.Function):
    """CSX RangedMatMul Op."""

    @staticmethod
    def forward(ctx, weight, input_value, num_cols, weight_transposed):
        ctx.save_for_backward(weight, input_value, weight_transposed)
        ctx.num_cols = num_cols
        return cstorch.cirh.RangedMatMul(weight, input_value, num_cols)

    @staticmethod
    def backward(ctx, grad_output):
        weight, input_value, weight_transposed = ctx.saved_tensors
        num_cols = ctx.num_cols

        grad_input_value = cstorch.cirh.RangedMatMul(
            weight_transposed, grad_output, num_cols
        )
        grad_weight = cstorch.cirh.RangedDMatMul(
            grad_output, input_value, weight, num_cols
        )

        return grad_weight, grad_input_value, None, None


def ranged_matmul(
    weight: torch.Tensor,
    input_value: torch.Tensor,
    num_cols: torch.Tensor,
    weight_transposed: torch.Tensor = None,
) -> torch.Tensor:
    """
    Performs [Ho][Hi] x [B][S][Hi] = [B][S][Ho] matmul.
    For each B, only reads/writes to the first "num_cols" columns on each
    "input_value" matrix.

    Constraint: 0 <= num_cols[i] <= S

    Behaviour is undefined for columns [num_cols:-1].

    On CSX: Performance speedup is proportional to num_cols.

    In order to use this kernel the user must call CSXRangedMatMul.apply(...).
    PyTorch primitives (matmul, einsum, etc.) will not map to the CSX optimized
    implementation.

    Args:
        weight: Dense weight.
        input_value: Dense input value.
        num_cols: Scalar value
        weight_transposed: Transpose of weight for backwards pass (optional)

    Returns:
        output_values: Dense output value.

    Shapes:
        weight: (Ho, Hi)
        input_value: (B, S, Hi)
        num_cols: (B)
        weight_transposed: (Hi, Ho)
        output_values: (B, S, Ho)

    TODO: Explain scrambled layout once layouts are exposed to torch.
    (There is no speedup if matmul is assigned a normal layout). Until then
    stack will alway apply a scrambled layout to this kernel.
    """

    assert (
        len(weight.shape) == 2
    ), f"Expected 2D weight. Instead got weight.shape = {weight.shape}"

    assert (
        len(input_value.shape) == 3
    ), f"Expected 3D input_value. Instead got input_value.shape = {input_value.shape}"

    assert (
        weight.shape[1] == input_value.shape[2]
    ), f"Expect weight.shape(1) == input_value.shape[2]. Instead got weight.shape = {weight.shape} and input_value.shape = {input_value.shape}"

    assert (
        len(num_cols.shape) == 1
    ), f"Expected len(num_cols.shape) == 1. Instead got len(num_cols.shape) = {len(num_cols.shape)}"

    if weight_transposed is not None:
        assert (
            weight_transposed.shape == torch.transpose(weight, 1, 0).shape
        ), f"Expected weight_transposed.shape to be the transpose of weight.shape. {weight_transposed.shape} != transpose({weight.shape}, 1, 0)."
    else:
        weight_transposed = torch.transpose(weight, 1, 0)

    """Optimized CSX implementation"""
    if cstorch.use_cs():
        return CSXRangedMatMul.apply(
            weight, input_value, num_cols, weight_transposed
        )

    """Non-CSX functional implementation"""
    return torch.einsum("...MN, KN -> ...MK", input_value, weight)


class CSXConditionalArange(torch.autograd.Function):
    """CSX ConditionalArange Op."""

    @staticmethod
    def forward(ctx, mask, start, step, dtype):
        return cstorch.cirh.ConditionalArange(mask, start, step, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        assert False, "CSXConditionalArange is not a reversible function."


def conditional_arange(
    mask: torch.Tensor,
    start: torch.Tensor,
    step: torch.Tensor,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Returns evenly spaced values where `mask` indices are 1. Indices where
    the `mask` is 0 are ignored.

    Constraints:
        - The "dim" along which arange is performed is the last dimension of the `mask`'s shape.
        - start and step must be compile-time constants
        - step must be non-zero

    Args:
        mask: binary tensor
        start: Scalar integer value
        step: Scalar integer value. Must be non-zero.

    Returns:
        output_values: Dense output value.

    Shapes:
        mask: Any >= 1D
        start: N/A
        step: N/A

    Example 1:
        start = 0, step = 1
        mask   = [0, 1, 1, 1, 0, 0, 1]
        result = [X, 0, 1, 2, X, X, 3]

    Example 2:
        start = 2, step = 7
        mask   = [0, 1, 1, 1, 0, 0, 1]
        result = [X, 2, 9, 16, X, X, 23]

    """

    assert (
        len(start.shape) == 0
    ), f"Expected 0D tensor for start. Instead got {start.shape}"

    assert (
        len(step.shape) == 0
    ), f"Expected 0D tensor for step. Instead got {step.shape}"

    if cstorch.use_cs():
        return CSXConditionalArange.apply(mask, start, step, dtype)

    mask = mask.bool()
    cumsum = (
        torch.cumsum(mask, dim=-1) - 1
    )  # Subtract 1 so first masked element is 0
    scaled = start + step * cumsum
    result = torch.where(mask, scaled, torch.full_like(scaled, -1))
    if dtype is not None:
        result = result.to(dtype)
    return result


class CSXNthOrderStatisic(torch.autograd.Function):
    """CSX NthOrderStatistic Op."""

    @staticmethod
    def forward(ctx, input_value, k, dimension):
        return cstorch.cirh.NthOrderStatistic(
            input_value, k=k, dimension=dimension
        )

    @staticmethod
    def backward(ctx, grad_output):
        assert False, "CSXNthOrderStatisic is not a reversible function."


def nth_order_statistic(
    input: torch.Tensor,
    k: int,
    dimension: int,
) -> torch.Tensor:
    """
    Performs a topk operation along the specified dimension on the input, and
    returns a mask tensor with the same shape as the input, with `1` indicating
    that the corresponding element on the input tensor is part of the top k
    elements.

    Constraints:
        - shape(input) == shape(result)
        - 0 <= dimension < rank(input)
        - 0 < k <= shape(input)[dimension]

    Example:
        dim = 0, k = 3
        input  = [0, 2, 3, 5, 4, 1, 6]
        result = [0, 0, 0, 1, 1, 0, 1]
    """

    if cstorch.use_cs():
        return CSXNthOrderStatisic.apply(input, k, dimension)

    topk_values, topk_indices = input.topk(k, dim=dimension)
    mask = torch.scatter(
        torch.zeros_like(input, dtype=torch.int16),
        dim=dimension,
        index=topk_indices,
        src=torch.ones_like(topk_indices, dtype=torch.int16),
    )
    return mask


def unbind(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Unbinds the input tensor along the first dimension.

    Args:
        tensor: Input tensor to unbind.

    Returns:
        A tuple of tensors obtained by unbinding the input tensor along the first dimension.
    """
    return UnbindFunction.apply(tensor)


class CSXD2DTransfer(torch.autograd.Function):
    """CSX DeviceToDeviceTransfer Op."""

    @staticmethod
    def forward(ctx, input, dst_device):
        return cstorch.cirh.DeviceToDeviceTransfer(
            input, dst_device=dst_device, dst_constraints=SKIP_OPTIONAL_ATTR
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class CSXM2MTransfer(torch.autograd.Function):
    """CSX MeshToMeshTransfer Op."""

    @staticmethod
    def forward(ctx, input, src_mesh_id, dst_mesh_id, src_root_id=None):
        if src_root_id is None:
            src_root_id = SKIP_OPTIONAL_ATTR
        return cstorch.cirh.MeshToMeshTransfer(
            input,
            srcMeshId=src_mesh_id,
            dstMeshId=dst_mesh_id,
            srcRootId=src_root_id,
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class CSXMeshAllSlice(torch.autograd.Function):
    """CSX MeshAllSlice Op."""

    @staticmethod
    def forward(ctx, input, split_axes, names):
        return cstorch.cirh.MeshAllSlice(
            input, split_axes=split_axes, names=names
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class CSXMeshAllGather(torch.autograd.Function):
    """CSX MeshAllGather Op."""

    @staticmethod
    def forward(ctx, input, gather_axes):
        return cstorch.cirh.MeshAllGather(input, gather_axes=gather_axes)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class CSXMeshShard(torch.autograd.Function):
    """CSX MeshShard Op."""

    @staticmethod
    def forward(
        ctx,
        input,
        mesh_id,
        split_axes,
        names,
        partial_axes,
        partial_type,
        devices,
    ):
        if partial_axes is None:
            partial_axes = []
        if partial_type is None:
            partial_type = SKIP_OPTIONAL_ATTR

        if devices is None or devices == []:
            device_string = SKIP_OPTIONAL_ATTR
        elif len(devices) == 3:
            device_string = SKIP_OPTIONAL_ATTR
        else:
            device_string = "_".join(sorted(devices, reverse=True))

        return cstorch.cirh.MeshShard(
            input,
            mesh_id=mesh_id,
            split_axes=split_axes,
            names=names,
            partial_axes=partial_axes,
            partial_type=partial_type,
            devices=device_string,
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class UnbindFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.input_shape = input.shape
        return tuple(torch.unbind(input, dim=0))

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        return (torch.stack(grad_outputs, dim=0),)
