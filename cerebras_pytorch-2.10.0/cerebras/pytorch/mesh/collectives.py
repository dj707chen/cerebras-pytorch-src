# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Union, get_args

import torch

import cerebras.pytorch as cstorch
import cerebras.pytorch.nn.functional as F

from .constants import BoxDeviceType
from .mesh import Mesh
from .placement import Replicate, Shard


def d2d_transfer(
    tensor: torch.Tensor, dst_device: BoxDeviceType
) -> torch.Tensor:
    """
    Transfers an unsharded (ie replicated) tensor to a specific dst_device type within the same mesh.

    Args:
        input: the input tensor to be transferred. Must be unsharded (replicated)
        dst_device: WSE, LocalHost or GlobalHost string specifying where the input tensor should be moved.

    Expects:
        - dst_device is one of the BoxDeviceType enum values

    Returns:
        A tensor that has been transferred to the specified device
    """

    if dst_device not in get_args(BoxDeviceType):
        raise TypeError(
            f"Expected `dst_device` to be one of {get_args(BoxDeviceType)}."
        )

    if cstorch.use_cs():
        return F.CSXD2DTransfer.apply(tensor, dst_device)
    return tensor


def m2m_transfer(
    tensor: torch.Tensor, src_mesh: Mesh, dst_mesh: Mesh, src_root_id=0
) -> torch.Tensor:
    """
    Transfers a tensor from one mesh to another mesh.

    Args:
        input: the input tensor to be transferred. Must be unsharded (replicated)
        src_mesh: the source mesh from which the tensor is to be transferred
        dst_mesh: the destination mesh to which the tensor is to be transferred
        src_root_id (Optional): the root id of the source mesh

    Returns:
        A tensor that has been transferred to the specified mesh
    """

    src_mesh_id: int = src_mesh.mesh_id
    dst_mesh_id: int = dst_mesh.mesh_id

    if cstorch.use_cs():
        return F.CSXM2MTransfer.apply(
            tensor,
            src_mesh_id,
            dst_mesh_id,
            src_root_id,
        )

    return tensor


def all_slice(
    tensor: torch.Tensor,
    sharding_array: List[Union[Replicate, Shard]],
    names: List[str],
) -> torch.Tensor:
    """
    Returns a copy of a replicated input tensor, sharded along the same mesh as specified by the sharded array.

    Args:
        tensor: the input tensor to be sliced
        sharding_array: a list of `Replicate` and `Shard` instances.
        names: list of dimension names, e.g. ["N", ...]

    Expects:
        - length of sharding_array <= tensor rank
        - sharding_array contains `Replicate` and `Shard` instances only
        - names should match the number of split axes

    Returns:
        A tensor that has been sliced according to the sharding array
    """

    tensor_rank: int = len(tensor.shape)
    if len(sharding_array) > tensor_rank:
        raise ValueError(
            f"Expected length of `sharding_array` {len(sharding_array)} to be less than or equal to the tensor rank {tensor_rank}."
        )

    if not all(
        isinstance(placement, (Replicate, Shard))
        for placement in sharding_array
    ):
        raise ValueError(
            f"Expected all placements in `sharding_array` to be instances of `Replicate` or `Shard`."
        )

    split_axes: List[int] = [
        (placement.dim if isinstance(placement, Shard) else -1)
        for placement in sharding_array
    ]
    sharded_axes_only: List[int] = [
        placement for placement in split_axes if placement != -1
    ]
    if len(names) != len(sharded_axes_only):
        raise ValueError(
            f"Expected `names` length {len(names)} to match the number of sharded axes {len(sharded_axes_only)}."
        )

    if cstorch.use_cs():
        names_str = ",".join(names)

        return F.CSXMeshAllSlice.apply(tensor, split_axes, names_str)
    return tensor


def all_gather(
    tensor: torch.Tensor,
    gather_axes: List[int],
) -> torch.Tensor:
    """
    Returns a copy of a sharded input tensor, gathered along the specified axis.

    Args:
        tensor: the input tensor to be gathered
        gather_axis: the axis along which to gather the tensor

    Expects:
        - gather_axes is a list of non-negative integers
        - gather_axes is a list of integers less than the tensor rank

    Returns:
        A tensor that has been gathered along the specified axis
    """
    tensor_rank = len(tensor.shape)

    for i, gather_axis in enumerate(gather_axes):
        if gather_axis < 0:
            raise ValueError(
                f"All gather axes must be non-negative, got {gather_axes} at index gather_axes[{i}]."
            )
        if gather_axis >= tensor_rank:
            raise ValueError(
                f"All axes must be less than the tensor rank. Gather axis {gather_axis} exceeds tensor rank {tensor_rank} at index gather_axes[{i}]."
            )

    if cstorch.use_cs():
        return F.CSXMeshAllGather.apply(tensor, gather_axes)
    return tensor
