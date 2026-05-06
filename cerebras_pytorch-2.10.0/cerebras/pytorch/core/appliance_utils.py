# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from collections import deque
from heapq import heappop, heappush
from typing import Dict, List, Tuple, Union

import numpy as np

from cerebras.appliance.data.conversions import (
    np_dtype_from_rtfx_dtype,
    rtfx_dtype_from_np_dtype,
)
from cerebras.appliance.pb.ws.rtfx_pb2 import RtFxProto


def assign_tensor_ids_to_shards(
    tensor_costs: Dict[Union[int, Tuple[int]], Dict[str, float]],
    num_shards: int,
) -> List[Tuple[int, int]]:
    """
    Assigns the jit graphs to the shards based on their cost.

    Args:
        tensor_costs: Dictionary of tensor id to costs dictionary.
        num_shards: Number of shards to assign the jit graphs to.

    Returns:
        List of pairs of tensor id and the shard id.
    """

    # Heap item is cost, num_tensors, shard_id
    shard_costs = [(0, 0, i) for i in range(num_shards)]
    shard_map = [deque() for i in range(num_shards)]

    min_cpu_time = min(cost["total_cpu_time"] for cost in tensor_costs.values())
    max_cpu_time = max(cost["total_cpu_time"] for cost in tensor_costs.values())
    if max_cpu_time == min_cpu_time:
        max_cpu_time = 1
        min_cpu_time = 0

    min_total_memory = min(
        cost["total_memory"] for cost in tensor_costs.values()
    )
    max_total_memory = max(
        cost["total_memory"] for cost in tensor_costs.values()
    )
    if max_total_memory == min_total_memory:
        max_total_memory = 1
        min_total_memory = 0

    # Normalize the costs so that we can sum them up
    for tensor_id, costs in tensor_costs.items():
        normalized_cpu_time = (costs["total_cpu_time"] - min_cpu_time) / (
            max_cpu_time - min_cpu_time
        )
        normalized_memory = (costs["total_memory"] - min_total_memory) / (
            max_total_memory - min_total_memory
        )

        # Cost is the sum of normalized cpu time and memory
        costs["normalized_cost"] = normalized_cpu_time + normalized_memory

    for tensor_id, costs in sorted(
        ((tensor_id, costs) for tensor_id, costs in tensor_costs.items()),
        key=lambda x: x[1]["normalized_cost"],
    ):
        # Get the shard with the least cost. If there are multiple
        # with the same cost, get the one with the least number
        # of tensors assigned to it
        cost, num_tensors, shard_id = heappop(shard_costs)
        if isinstance(tensor_id, (list, tuple)):
            shard_map[shard_id].extend(tensor_id)
            num_tensors += len(tensor_id)
        else:
            shard_map[shard_id].append(tensor_id)
            num_tensors += 1

        cost += costs["normalized_cost"]
        heappush(shard_costs, (cost, num_tensors, shard_id))

    # order the tensor ids in round robin fashion so that
    # we can optimize the parallized sharding and broadcasting
    def get_tensor_id_shard_id():
        while any(map(len, shard_map)):
            for shard_id, tensor_ids in enumerate(shard_map):
                if len(tensor_ids) > 0:
                    tensor_id = tensor_ids.popleft()
                    yield tensor_id, shard_id

    return list(get_tensor_id_shard_id())


def np_array_to_rtfx_proto(np_array):
    rtfx_dtype = rtfx_dtype_from_np_dtype(np_array.dtype)
    if rtfx_dtype == RtFxProto.T_I1:
        # I1 needs to be encoded as int16
        np_array = np_array.astype(np.int16)

    rtfx = RtFxProto()
    rtfx.dtype = rtfx_dtype
    rtfx.tensor.data = np_array.data.tobytes()
    rtfx.tensor.shape.extend(np_array.shape)
    return rtfx


def np_array_to_rtfx_scalar(np_array):
    if np_array.ndim != 0:
        raise ValueError("Array must be a scalar")

    rtfx_dtype = rtfx_dtype_from_np_dtype(np_array.dtype)
    if rtfx_dtype == RtFxProto.T_I1:
        # I1 needs to be encoded as int16
        np_array = np_array.astype(np.int16)

    rtfx = RtFxProto()
    rtfx.dtype = rtfx_dtype
    rtfx.scalar.data = np_array.data.tobytes()
    return rtfx


def rtfx_to_np_array(data, shape, dtype):
    if dtype == RtFxProto.T_I1:
        # I1 is encoded as int16
        np_dtype = np.int16
    else:
        np_dtype = np_dtype_from_rtfx_dtype(dtype)

    if not shape:
        shape = []
    np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)

    # I1 comes through as int16, but it _should_ be bool...
    if dtype == RtFxProto.T_I1 and np_array.dtype != bool:
        np_array = np_array.astype(bool)

    return np_array
