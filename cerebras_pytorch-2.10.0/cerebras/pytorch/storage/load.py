# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from typing import IO, Any, Callable, Union

import torch

import cerebras.pytorch as cstorch
from cerebras.appliance import logger
from cerebras.appliance.storage import StorageReader
from cerebras.appliance.storage.context import (
    DeserializationContext,
    NestedObject,
    SerializedObject,
)
from cerebras.appliance.storage.serializers import (
    DeferredNumpyArray,
    DeferredObject,
    LinkedObject,
)
from cerebras.appliance.utils.file import StrPath, get_path_size, is_pathlike
from cerebras.appliance.utils.memory import (
    get_available_memory,
    with_memory_info_logged,
)
from cerebras.pytorch.backend import current_backend_impl
from cerebras.pytorch.storage.constants import (
    __METADATA__,
    __OBJECTS__,
    __SPEC__,
    __VERSION__,
)
from cerebras.pytorch.storage.serializers import DeferredTorchTensor

_CkptFileT = Union[StrPath, IO]
_MapLocT = Union[str, torch.device, Callable, dict, None]
_StateDictT = Any


def _load_spec(value):
    if NestedObject not in torch.utils._pytree.SUPPORTED_NODES:
        torch.utils._pytree._register_namedtuple(
            NestedObject, serialized_type_name="NestedObject"
        )

    _, spec = value
    return torch.utils._pytree.treespec_loads(spec)


@with_memory_info_logged(
    "loading checkpoint",
    info=["available", "used"],
    logger=logger,
)
def load(
    ckpt_path: StrPath, map_location: _MapLocT = None, **kwargs
) -> _StateDictT:
    if not is_pathlike(ckpt_path):
        return _torch_load(ckpt_path, map_location, **kwargs)

    try:
        reader = StorageReader.get(ckpt_path)
    except ValueError as e:
        try:
            return _torch_load(ckpt_path, map_location, **kwargs)
        except Exception as e2:
            raise e2 from e

    global_metadata = reader.global_metadata

    version = global_metadata[__VERSION__]
    if version < 0.5:
        from cerebras.pytorch.storage._legacy import legacy_load

        return legacy_load(ckpt_path, map_location, **kwargs)

    if not global_metadata.get("__SUCCESS__", False):
        raise RuntimeError(
            f"Cannot load partially written checkpoint at {ckpt_path}."
        )

    spec = _load_spec(json.loads(global_metadata[__SPEC__]))
    top_level_keys = list(map(".".join, cstorch.utils.nest.recurse_spec(spec)))

    def deserialize(value):
        if isinstance(value, SerializedObject):
            key = value.object
            context = DeserializationContext(key, value.metadata)

            if context.metadata.get("__NONE__"):
                obj = context.deserialize(None)
            elif context.metadata.get("__NUMPY__"):
                obj = DeferredNumpyArray(reader, key, metadata=context.metadata)
            else:
                obj = DeferredObject(reader, key, metadata=context.metadata)

            if context.metadata.get("__TORCH__"):
                shape = context.metadata["shapes"][0]
                _, dtype = context.metadata["dtypes"][0].split(".")
                dtype = getattr(torch, dtype)
                obj = DeferredTorchTensor(obj, shape, dtype)

            return obj

        if isinstance(value, NestedObject):
            obj = deserialize(value.object)
            key = None

            context = DeserializationContext(key, value.metadata)
            obj = context.deserialize(obj)
            return obj

        if isinstance(value, dict):
            return type(value)((k, deserialize(v)) for k, v in value.items())

        if isinstance(value, (list, tuple)):
            return type(value)(map(deserialize, value))

        raise TypeError(f"Unexpected type {type(value)}")

    cache_tensors = False
    if map_location == "cache":
        cache_tensors = True
        map_location = None

    with cstorch.storage.serializers.cache_deferred_tensors(cache_tensors):
        values = []
        for top_level_key in top_level_keys:
            m = json.loads(global_metadata[f"{top_level_key}.{__METADATA__}"])

            serialized_spec = _load_spec(m[__SPEC__])
            serialized_keys = map(
                ".".join, cstorch.utils.nest.recurse_spec(serialized_spec)
            )

            # List of non nested objects
            objects = m[__OBJECTS__]

            serialized_object = serialized_spec.unflatten(
                # Must be a concrete serialized object
                (
                    SerializedObject(k, m[f"{k}.{__METADATA__}"])
                    if k in objects
                    # Otherwise must be metadata for a nested object
                    else m[f"{k}.{__METADATA__}"]
                )
                for k in serialized_keys
            )

            deserialized = deserialize(serialized_object[top_level_key])

            if map_location is not None and isinstance(
                deserialized, torch.Tensor
            ):
                deserialized = deserialized.to(map_location)

            values.append(deserialized)

    values_map = dict(zip(top_level_keys, values))

    return torch.utils._pytree.tree_unflatten(
        (
            # Re-tie objects in state dict
            values_map[v.key] if isinstance(v, LinkedObject) else v
            for k, v in values_map.items()
        ),
        spec,
    )


def _torch_load(
    checkpoint_file: _CkptFileT,
    map_location: _MapLocT = None,
    **kwargs,
) -> _StateDictT:
    """Load a PyTorch checkpoint using vanilla torch.load.

    Args:
        checkpoint_file: The path to the checkpoint to load.
        map_location: A mapping of where to load the checkpoint content to.
        **kwargs: Additional keyword arguments to pass to torch.load.
    """
    # Attempt to load the checkpoint using safetensors if available
    # Otherwise, fall back to vanilla torch.load
    try:
        from cerebras.pytorch.storage.serializers import DeferredSafeTensor

        cache_tensors = False
        if map_location == "cache":
            cache_tensors = True
            map_location = None

        with cstorch.storage.serializers.cache_deferred_tensors(cache_tensors):
            state_dict = DeferredSafeTensor.load_file(checkpoint_file)

            if map_location is not None:

                def map_tensor(t):
                    if isinstance(t, torch.Tensor):
                        return t.to(map_location)
                    return t

                state_dict = torch.utils._pytree.tree_map(
                    map_tensor, state_dict
                )

            return state_dict
    except:
        pass

    if is_pathlike(checkpoint_file) and os.path.exists(checkpoint_file):
        unit = "GB"
        file_size = get_path_size(checkpoint_file, unit=unit)
        free_mem = get_available_memory(unit=unit)

        if file_size > 10:
            backend = current_backend_impl(raise_exception=False)
            if backend is not None and backend.backend_type.is_csx:
                extra_msg = ", could significantly slow down weight transfer,"
            else:
                extra_msg = ""
            logger.warning(
                f"Checkpoint file is a vanilla torch checkpoint and has "
                f"size {file_size} {unit}. This may take a while to load"
                f"{extra_msg} and may occupy a large amount of memory."
            )

        if file_size > free_mem:
            logger.warning(
                f"Checkpoint file is a vanilla torch checkpoint and has "
                f"size {file_size} {unit}, which is larger than the "
                f"currently available memory {free_mem} {unit}. Since "
                f"torch checkpoints are loaded in their entirety into "
                f"memory, this may cause out-of-memory errors."
            )

    try:
        state_dict = torch.load(
            checkpoint_file, map_location=map_location, **kwargs
        )
    except FileNotFoundError as e:
        # Error message is already descriptive enough
        raise e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint file `{checkpoint_file}`."
        ) from e

    logger.debug(f"Loaded checkpoint {checkpoint_file} into memory.")

    return state_dict
