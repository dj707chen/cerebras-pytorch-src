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

# The current streaming batch sizes per box. The value is only available in the
# workers and must be queried using `get_streaming_batch_size()` API below.
_STREAMING_BATCH_SIZES = ValueContext(None)


def get_worker_state():
    """API exposing internal state info captured by each CSX Worker
    for the current run at a checkpoint step. This state info is
    represented in the :py:class:`DataLoaderCheckpoint` dataclass format:

    Returns:
        :py:class:`DataLoaderCheckpoint` instance holding worker state information
        at the checkpoint step

    .. note::
        - This method may only be called inside of a custom implementation of `state_dict` for
        dataloaders conforming to the :py:class:`RestartableDataLoader` protocol, since
        `state_dict` is well-defined only at a checkpoint step.
        - Use this method to save any of the aforementioned state info recorded by each worker
        when defining `state_dict` for custom implementations of restartable dataloaders.
        - This state info captured by each worker is for the current run only, i.e. if you pause and
        restart a run, the counters gathering information returned by this function will be reset.
    """
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
    """Returns the streaming batch size of the given task.

    In a Wafer-Scaler Cluster setup with more than 1 CS-X node, the batch size
    used in compile and specified by user is the effective batch size at
    which gradient updates are done. However, each worker node streams a local
    batch of data to a given CS-X node to consitute data parallel training.

    This helper method returns the local batch size that the current task should
    use given the desired effective batch size. Note that when the effective batch
    size is not divisible by number of CS-X nodes, the streaming batch size of
    workers may be different depending on the CS-X node that they are streaming to.

    Args:
        effective_batch_size: The effective batch size of the model.
        global_rank: The global rank of the task to return the streaming batch size for. If None,
            it returns the streaming batch size of the current task.
    Returns:
        The local batch size to be streamed by the given task. If queried on the
        user node (used when compiling the model), this returns the original
        effective batch size as passed in the argument.
    """
    # If queried on the worker, return the current streaming batch size value that's been set
    # by the streamer for this worker process.
    if is_streamer():
        global _STREAMING_BATCH_SIZES
        return _STREAMING_BATCH_SIZES.value[
            service_resolver().cluster_spec.task(global_rank).wse_id
        ]

    # If not queried on the worker node, return the effective batch size as is
    # so the compile can automatically handle data parallel and gradient
    # accumulation.
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
    """Set the current streaming batch sizes.

    This method is internal because it's the streamer that sets this value and is not meant to
    be used externally.
    """
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
    # Note: keeping `local` argument for compatibility with XLA API.
    return service_resolver().cluster_resolver.assumes_role(TaskRole.MASTER)


def is_streamer():
    """Returns True if the current task is a streamer task."""
    return get_ordinal() in service_resolver().streamer_ordinals()


def is_receiver():
    """Returns True if the current task is a receiver task."""
    return get_ordinal() in service_resolver().receiver_ordinals()


# constants
SSD_LIMIT = 0.8
WORKER_CACHE_ROOT = "/n0/cache"


def hit_worker_cache_limit(src_dir: str, dest_dir: str):
    """
    Identifies whether copying the src_dir to a dest_dir (within
    worker_cache), will lead to a cache overflow
    Args:
        src_dir (str, required): directory path of the source
        dest_dir (str, required): directory path of the destination within
        the worker cache
    Returns:
        A tuple of (``is_limit_hit``, ``dir_size``, ``available_space_for_copy``)
        where ``is_limit_hit`` is a bool indicating whether cache limit
        will be hit with the copy,
        ``dir_size`` is the size of the src_dir to be copied to the cache,
        ``available_space_for_copy`` is the space available for src_dir copy,
        including the space occupied by the currently cached_dir
        corresponding to src_dir.
    """
    # Raises if dest_dir path is not a descendant of WORKER_CACHE_ROOT
    Path(dest_dir).resolve().relative_to(Path(WORKER_CACHE_ROOT).resolve())

    # Only add things to cache if < SSD_LIMIT occupied
    ssd_mount = WORKER_CACHE_ROOT
    # Get size of SSD mount
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
