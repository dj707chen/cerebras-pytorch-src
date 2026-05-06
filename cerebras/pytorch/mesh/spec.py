# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional

from cerebras.pytorch.mesh.constants import BoxDeviceType, ShardingPartialType


@dataclass
class ShardingSpec:
    """
    Sharding spec for a tensor.

    Attributes:
        mesh_id (int): The mesh to which the sharding spec will apply.
        split_axes (List[int]): The axes to split the tensor along.
        partial_axes (Optional[List[int]]): The axes to partially replicate the tensor along.
        partial_types (Optional[ShardingPartialType]): The type of partial replication to use.
        device (Optional[BoxDeviceType]): Specifies the device to constrain tensor storage to.
            Can be “WSE”, “LocalHost” or “GlobalHost”.
    """

    mesh_id: int
    split_axes: List[int]
    partial_axes: Optional[List[int]]
    partial_type: Optional[ShardingPartialType]
    devices: Optional[List[BoxDeviceType]]
