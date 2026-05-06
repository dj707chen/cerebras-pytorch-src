# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Union, get_args

import torch

import cerebras.pytorch as cstorch
import cerebras.pytorch.nn.functional as F

from .constants import DEVICE_TYPE_GLOBALHOST, BoxDeviceType
from .mesh import Mesh
from .placement import Replicate, Shard
from .spec import ShardingSpec


def shard(dim: int) -> Shard:
    """
    Create a `Shard` instance for a given dimension.

    Args:
        dim: The mesh dimension to shard along.

    Returns:
        A `Shard` instance.
    """
    return Shard(dim)


def replicate() -> Replicate:
    """
    Create a `Replicate` instance.

    Returns:
        A `Replicate` instance.
    """
    return Replicate()


def sharding_spec(
    sharding_array: List[Union[Replicate, Shard]],
    mesh: Mesh,
    device: Optional['BoxDeviceType'] = None,
) -> ShardingSpec:
    """
    Create a `ShardingSpec` instance from a sharding array that can
    be passed to `distribute_tensor`. The `mesh` must be specified,
    and tensor storange can be further constrained to the optional `device`.

    Args:
        sharding_array: A list of `Replicate` and `Shard` instances.
        mesh: The `Mesh` instance to shard along.
        device: Optional device type to constrain tensor storage.

    Expects:
        - device is a valid `BoxDeviceType` device or None
        - sharding_array contains `Replicate` and `Shard` instances only
        - if device is "GlobalHost", all sharding_array items must be `Replicate`

    Returns:
        A `ShardingSpec` instance.
    """
    if device and device not in get_args(BoxDeviceType):
        raise ValueError(
            f"Expected `device` to be one of: {get_args(BoxDeviceType)}."
        )

    if not all(
        (isinstance(element, (Replicate, Shard))) for element in sharding_array
    ):
        raise TypeError(
            "Expected `sharding_array` to contain `Replicate` and `Shard` instances only."
        )

    if device == DEVICE_TYPE_GLOBALHOST and not all(
        isinstance(element, Replicate) for element in sharding_array
    ):
        raise TypeError(
            "Expected all sharding_array items to be `Replicate` for `GlobalHost` device type."
        )

    mesh_id = mesh.mesh_id
    split_axes = [
        i if element.is_shard() else -1
        for i, element in enumerate(sharding_array)
    ]
    partial_axes = None
    partial_types = None
    device = [device] if device else None

    return ShardingSpec(
        mesh_id, split_axes, partial_axes, partial_types, device
    )


def distribute_tensor(
    tensor: torch.Tensor,
    sharding_spec: ShardingSpec,
    names: List[str],
) -> torch.Tensor:
    """
    Returns a copy of the input tensor, annotated with the specified sharding_spec

    Args:
        tensor: the input tensor to shard (or replicate) over a mesh
        sharding_spec: a ShardingSpec object whose spec should be annotated on the input tensor
        names: list of dimension names, e.g. ["N", ...]

    Expects:
        - tensor is unsharded (replicated)
        - sharding_spec is a valid ShardingSpec object
        - sharding_spec.split_axes match the rank of the tensor
        - if sharding_spec.partial_axes is specified, it must match the rank of the tensor
        - names should match the number of sharded axes

    Returns:
        A tensor that has been sharded or replicated according to the sharding_spec
    """
    tensor_rank: int = len(tensor.shape)

    if (
        sharding_spec.split_axes
        and len(sharding_spec.split_axes) != tensor_rank
    ):
        raise ValueError(
            f"Split axes {len(sharding_spec.split_axes)} exceed tensor rank {tensor_rank}."
        )

    num_sharded_axes = len(
        [placement for placement in sharding_spec.split_axes if placement != -1]
    )
    if len(names) != num_sharded_axes:
        raise ValueError(
            f"Expected `names` length {len(names)} to match the number of sharded axes {num_sharded_axes}."
        )

    if sharding_spec.partial_axes:
        if len(sharding_spec.partial_axes) > tensor_rank:
            raise ValueError(
                f"Partial axes {len(sharding_spec.partial_axes)} exceed tensor rank {tensor_rank}."
            )
        if sharding_spec.partial_axes == sharding_spec.split_axes:
            raise ValueError(
                f"Partial axes: {sharding_spec.partial_axes} and split axes: {sharding_spec.split_axes} cannot be the same."
            )

    if cstorch.use_cs():
        names_str = ",".join(names)

        return F.CSXMeshShard.apply(
            tensor,
            sharding_spec.mesh_id,
            sharding_spec.split_axes,
            names_str,
            sharding_spec.partial_axes,
            sharding_spec.partial_type,
            sharding_spec.devices,
        )
    return tensor
