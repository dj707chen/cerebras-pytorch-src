# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for saving and loading checkpoints."""
import os
from typing import IO, Any, Callable, Union

import torch

import cerebras.pytorch as cstorch
from cerebras.appliance import logger
from cerebras.appliance.utils.file import StrPath, get_path_size, is_pathlike
from cerebras.appliance.utils.memory import (
    get_available_memory,
    with_memory_info_logged,
)
from cerebras.pytorch.backend import current_backend_impl

from .checkpoint_reader import CheckpointReader
from .pt_h5_saver import PyTorchH5Saver

# A file-like object, which has to implement `read`, `readline`, `tell`, and
# `seek` methods.
_CkptFileT = Union[StrPath, IO]
_MapLocT = Union[str, torch.device, Callable, dict, None]
_StateDictT = Any


@with_memory_info_logged(
    "loading checkpoint",
    info=["available", "used"],
    logger=logger,
)
def legacy_load(
    checkpoint_file: _CkptFileT,
    map_location: _MapLocT = None,
    **kwargs,
) -> _StateDictT:
    """Load a legacy PyTorch checkpoint from a file.

    Args:
        checkpoint_file: The path to the checkpoint to load.
        map_location: A mapping of where to load the checkpoint content to.
            If the map_location is `None`, then the tensors will be lazily loaded
            from the checkpoint file every single time the tensor is accessed.
            If the map_location is "cache", then the tensors will be cached
            once they are lazily loaded from the checkpoint file.
            If the map location is "cpu", then the tensors will be eagerly loaded
            into memory from the checkpoint file.
        **kwargs: Additional keyword arguments to pass to the vanilla torch
            checkpoint loader. These are ignored if the checkpoint is a
            Cerebras HDF5 checkpoint.
    Returns:
        The loaded checkpoint file.
    Raises:
        RuntimeError: If the checkpoint file does not exist or checkpoint is not
            a valid HDF5 or vanilla torch checkpoint.
    """
    if not is_pathlike(
        checkpoint_file
    ) or not PyTorchH5Saver.is_valid_checkpoint(checkpoint_file):
        raise RuntimeError(f"Checkpoint is not a valid legacy HDF5 checkpoint.")

    logger.debug(
        "Checkpoint is a legacy HDF5 checkpoint. "
        "Using the legacy HDF5 checkpoint loader."
    )

    res = _cstorch_load(checkpoint_file, map_location, **kwargs)
    logger.debug(f"Loaded HDF5 checkpoint {checkpoint_file}.")
    return res


def _cstorch_load(
    checkpoint_file: _CkptFileT,
    map_location: _MapLocT = None,
    **kwargs,
) -> _StateDictT:
    cache_tensors = False
    if map_location == "cache":
        cache_tensors = True
        map_location = None

    if map_location is not None:
        if isinstance(map_location, (str, torch.device)):
            map_location = torch.device(map_location)
        else:
            raise TypeError(
                f"Unsupported `map_location` provided for loading HDF5 "
                f"checkpoint. Expected `None` or a torch device, but got "
                f"`{map_location}`"
            )

    CheckpointReader.saver_cls = PyTorchH5Saver
    reader = CheckpointReader(checkpoint_file)
    tensor_names = reader.tensor_names
    spec = reader.spec

    if not spec:
        raise RuntimeError(
            f"Checkpoint `{checkpoint_file}` is an HDF5 file but does not "
            f"conform to the Cerebras HDF5 checkpoint specification. Please "
            f"ensure that the checkpoint was saved using `cstorch.save()`."
        )

    from cerebras.pytorch.utils.nest import recurse_spec

    spec_keys = list(map(".".join, recurse_spec(spec)))
    unique_spec_keys = set(spec_keys)

    missing = unique_spec_keys - set(tensor_names)
    present = unique_spec_keys - missing
    if missing:
        logger.warning(
            f"The checkpoint is missing the following keys that are "
            f"found in the spec: {sorted(missing)}"
        )

    backend = current_backend_impl(raise_exception=False)
    if backend is None:
        logger.debug(
            "No backend has been initialized. Loading tensors onto CPU."
        )
        map_location = torch.device("cpu")

    saver = PyTorchH5Saver()

    with cstorch.storage.serializers.cache_deferred_tensors(cache_tensors):
        values = []
        # Load all (present) values in one file read, as flocking the h5 file
        # can be expensive on some filesystems.
        # Because we're using PyTorchH5Saver(), they aren't actually in memory,
        # each torch.Tensor is backed by a DeferredH5Tensor.
        vals = saver.load(checkpoint_file, present)
        for key in spec_keys:
            if key in vals:
                val = vals[key]
                if map_location is not None and isinstance(val, torch.Tensor):
                    val = val.to(map_location)
            else:
                val = None
            values.append(val)

    # pylint: disable=protected-access
    state_dict = torch.utils._pytree.tree_unflatten(values, spec)
    logger.debug(f"Loaded HDF5 checkpoint {checkpoint_file}.")

    return state_dict


__all__ = ["load"]
