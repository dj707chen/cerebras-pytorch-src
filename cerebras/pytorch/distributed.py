# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Get information about the current cluster setup."""

import os
from pathlib import Path
from typing import List, Optional

from cerebras.appliance.cluster_config import ClusterConfig
from cerebras.appliance.utils._contexts import ValueContext
from cerebras.appliance.utils.units import bytes_to_human
from cerebras.pytorch.utils.utils import get_dir_size

from .cluster_resolver import TaskRole
from .service_resolver import BaseServiceResolver
from .worker_state import WorkerState

_STREAMING_BATCH_SIZES = ValueContext(None)


def get_worker_state():
    return WorkerState.get_worker_state()


def service_resolver():
    resolver = BaseServiceResolver.get_resolver()
    return resolver


def num_tasks():
    """Returns total number of tasks in the cluster."""
    return service_resolver().cluster_resolver.num_tasks


def num_streamers():
    """Returns total number of tasks responsible for streaming inputs."""
    return len(service_resolver().streamer_ordinals())


def num_receivers():
    """Returns total number of tasks responsible for receiving outputs."""
    return len(service_resolver().receiver_ordinals())


def get_ordinal():
    """Returns the ordinal number of the current task."""
    return service_resolver().cluster_resolver.rank


def get_streaming_rank():
    """Returns the rank of the current task among streamers."""
    streamers = sorted(service_resolver().streamer_ordinals())
    ordinal = get_ordinal()
    assert ordinal in streamers, f"Ordinal {ordinal} is not a streamer."
    return streamers.index(ordinal)


def get_streaming_batch_size(
    effective_batch_size: int, global_rank: Optional[int] = None
) -> int:
    if is_streamer():
        global _STREAMING_BATCH_SIZES
        return _STREAMING_BATCH_SIZES.value[
            service_resolver().cluster_spec.task(global_rank).wse_id
        ]

    if not isinstance(effective_batch_size, int):
        raise TypeError(
            f"Expected effective batch size to be an integer, but got type "
            f"{type(effective_batch_size)} with value {effective_batch_size}."
        )
    if effective_batch_size <= 0:
        raise ValueError(
            f"Expected effective batch size to be a positive integer, but got "
            f"value {effective_batch_size}."
        )

    return effective_batch_size


def _set_streaming_batch_sizes(subbatch_sizes: List[List[int]]) -> None:
    assert is_streamer(), "This method must only be called in the streamer."

    cluster_spec = service_resolver().cluster_spec
    if len(subbatch_sizes) != cluster_spec.num_csx:
        raise ValueError(
            f"`subbatch_sizes` must be a list of subbatch sizes per CSX. But "
            f"num_csx is {cluster_spec.num_csx} and subbatch_sizes are {subbatch_sizes}."
        )

    per_box_batch_sizes = tuple(sum(x) for x in subbatch_sizes)
    if any(x <= 0 for x in per_box_batch_sizes):
        raise ValueError(
            f"Per-box batch sizes must all be greater than zero, but got "
            f"{per_box_batch_sizes}."
        )

    global _STREAMING_BATCH_SIZES
    _STREAMING_BATCH_SIZES.value = per_box_batch_sizes


def is_master_ordinal(local=False):
    """Returns True if the current task is the master task."""
    return service_resolver().cluster_resolver.assumes_role(TaskRole.MASTER)


def is_streamer():
    """Returns True if the current task is a streamer task."""
    return get_ordinal() in service_resolver().streamer_ordinals()


def is_receiver():
    """Returns True if the current task is a receiver task."""
    return get_ordinal() in service_resolver().receiver_ordinals()


SSD_LIMIT = 0.8
WORKER_CACHE_ROOT = "/n0/cache"


def hit_worker_cache_limit(src_dir: str, dest_dir: str):
    Path(dest_dir).resolve().relative_to(Path(WORKER_CACHE_ROOT).resolve())

    ssd_mount = WORKER_CACHE_ROOT
    statvfs = os.statvfs(ssd_mount)
    max_size = statvfs.f_frsize * statvfs.f_blocks
    dir_size = get_dir_size(src_dir)
    ssd_available = statvfs.f_frsize * statvfs.f_bavail
    ssd_occupied = max_size - ssd_available
    removal_size = get_dir_size(dest_dir)
    cap = SSD_LIMIT * max_size
    new_size = dir_size + ssd_occupied - removal_size
    is_limit_hit = new_size > cap
    available_space_for_copy = (
        cap - ssd_occupied + removal_size
        if cap > (ssd_occupied - removal_size)
        else 0
    )

    return (
        is_limit_hit,
        bytes_to_human(dir_size),
        bytes_to_human(available_space_for_copy),
    )
