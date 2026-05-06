# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import ExitStack

import torch
from tqdm import tqdm

import cerebras.pytorch as cstorch
from cerebras.appliance.storage import DeferredStorageReader, StorageWriter
from cerebras.appliance.storage.context import (
    HoistedObject,
    NestedObject,
    SerializationContext,
    SerializedMetadata,
    SerializedObject,
)
from cerebras.appliance.storage.serializers import DeferredObject, LinkedObject
from cerebras.appliance.utils.signal import on_sigint
from cerebras.pytorch.backend import current_backend_impl
from cerebras.pytorch.lib import cerebras_pytorch_lib
from cerebras.pytorch.storage.constants import (
    __CSTORCH_VERSION__,
    __METADATA__,
    __OBJECTS__,
    __SPEC__,
    __VERSION__,
    CURRENT_CKPT_FORMAT_VERSION,
)
from cerebras.pytorch.storage.serializers import DeferredTorchTensor


def _serialize_spec(spec):
    unflattened = torch.utils._pytree.tree_unflatten(
        ("*" for _ in range(spec.num_leaves)), spec
    )
    try:
        json.dumps(unflattened)
    except TypeError:
        unflattened = None

    if NestedObject not in torch.utils._pytree.SUPPORTED_NODES:
        torch.utils._pytree._register_namedtuple(
            NestedObject, serialized_type_name="NestedObject"
        )

    return unflattened, torch.utils._pytree.treespec_dumps(spec)


def save(state_dict: dict, ckpt_path: str):
    backend = current_backend_impl(raise_exception=False)
    use_cs = backend is not None and backend.is_csx

    with ExitStack() as exit_stack:
        writer = exit_stack.enter_context(StorageWriter.get(ckpt_path))

        if use_cs:
            # Create a deferred reader to the checkpoint we're about to write to.
            # Must be deferred, as it hasn't been written to yet
            deferred_reader = DeferredStorageReader(ckpt_path)
            # We don't want any deferred tensors created here to be cached
            exit_stack.enter_context(
                cstorch.storage.serializers.cache_deferred_tensors(False)
            )

        flattened, state_dict_spec = torch.utils._pytree.tree_flatten(
            state_dict
        )
        flattened_keys = list(
            map(".".join, cstorch.utils.nest.recurse_spec(state_dict_spec))
        )

        global_metadata = {
            __SPEC__: json.dumps(_serialize_spec(state_dict_spec), indent=4),
            __VERSION__: CURRENT_CKPT_FORMAT_VERSION,
            __CSTORCH_VERSION__: cstorch.__version__,
        }

        # Write global metadata first in case of errors
        writer.write(None, None, global_metadata)

        seen = {}
        duplicates = defaultdict(list)
        unique_values = {}

        for key, val in zip(flattened_keys, flattened):
            if id(val) in seen:
                duplicates[seen[id(val)]].append(key)
            else:
                seen[id(val)] = key
                unique_values[key] = val

        progress = None
        # Enable progress bar only on TTY
        if sys.stdout.isatty():
            progress = exit_stack.enter_context(
                tqdm(
                    total=len(flattened),
                    desc="Saving checkpoint",
                    dynamic_ncols=True,  # Match console width
                    unit=" tensors",
                    file=sys.stdout,
                )
            )

        metadata_map = {}

        def write_val(key, val):
            context = SerializationContext(
                ckpt_path, key, cstorch.storage.serializers.use_external_link
            )
            serialized = context.serialize(val)

            vals, spec = torch.utils._pytree.tree_flatten({key: serialized})
            keys = map(".".join, cstorch.utils.nest.recurse_spec(spec))

            metadata_map[key] = {
                __SPEC__: _serialize_spec(spec),
                # Objects for which we may write data for
                __OBJECTS__: [],
            }

            for k, v in zip(keys, vals):
                if isinstance(v, HoistedObject):
                    v = v.object

                if isinstance(v, SerializedMetadata):
                    metadata_map[key][f"{k}.{__METADATA__}"] = v.metadata
                elif isinstance(v, SerializedObject):
                    if v.object is not None:
                        v.metadata.setdefault(
                            "compressed",
                            cstorch.backends.csx.performance.compress_weights,
                        )
                        writer.write(k, v.object, v.metadata)

                    metadata_map[key][f"{k}.{__METADATA__}"] = v.metadata
                    metadata_map[key][__OBJECTS__].append(k)
                else:
                    raise RuntimeError(f"Unexpected val: {type(val)}")

            if progress is not None:
                progress.update(1)

        futures = {}

        def cancel_futures(*args, **kwargs):
            for future in futures.values():
                future.cancel()

        exit_stack.enter_context(on_sigint(cancel_futures))
        # Even if no sigint encountered, cancel all futures on exit
        exit_stack.callback(cancel_futures)

        executor = exit_stack.enter_context(
            ThreadPoolExecutor(
                max_workers=cstorch.backends.csx.performance.transfer_processes
            )
        )

        def submit_job(key, val):
            futures[key] = executor.submit(write_val, key, val)

            # Handle tied weights
            for alias in duplicates.get(key, []):
                write_val(alias, LinkedObject(key))

        for key, val in unique_values.items():
            if not isinstance(val, torch.Tensor):
                submit_job(key, val)

        tensors = [
            (k, t)
            for k, t in unique_values.items()
            if isinstance(t, torch.Tensor)
        ]

        if use_cs:
            tensors = backend.save_tensors(tensors, ckpt_path)

        for key, val in tensors:
            submit_job(key, val)

            if use_cs:
                if unique_values[key] is val:
                    val = DeferredTorchTensor(
                        DeferredObject(deferred_reader, key),
                        val.shape,
                        val.dtype,
                    )

                _share_storage(unique_values[key], val)

        # Wait for all futures to complete
        completed, pending = wait(futures.values(), return_when=FIRST_EXCEPTION)

        # Iterate over completed futures and handle exceptions
        for f in completed:
            if (e := f.exception()) is not None:
                # Find the corresponding key for logging
                k = next(k for k, future in futures.items() if future == f)

                logging.error(f"Ran into error while saving '{k}'. {str(e)}")

                # Cancel all pending futures
                for p in pending:
                    p.cancel()

                # raise on the first error found, this causes an early abort
                raise e

        global_metadata.update(
            **{
                f"{k}.{__METADATA__}": json.dumps(v, indent=4)
                for k, v in metadata_map.items()
            }
        )
        global_metadata["__SUCCESS__"] = True

        # Update the global metadata with the metadata map
        writer.write(None, None, global_metadata)

    if use_cs:
        # Materialize the cached reader now that the checkpoint has been written
        _ = deferred_reader.reader


def _share_storage(val, deferred):
    # To avoid excess fetching from the appliance, we update appliance info
    # tensors with deferred tensors, so they won't be fetched from the appliance.

    if isinstance(val, cerebras_pytorch_lib.ApplianceDataInfo):
        appliance_data = val
    elif isinstance(val, torch.Tensor):
        if val.device.type != "lazy":
            return
        appliance_data = cerebras_pytorch_lib.get_appliance_data(val)
    else:
        return

    # If tensor is already available there is no need
    # to update the storage with deferred tensor.
    if appliance_data.is_tensor_available:
        return

    deferred = deferred.to(val.device)

    deferred_appliance_data = cerebras_pytorch_lib.get_appliance_data(deferred)
    deferred_appliance_data.tensor_descriptor = appliance_data.tensor_descriptor

    # In case we have appliance info underneath, we update its
    # storage with the deferred tensor, so we can later differentiate
    # appliance info tensors (that can be carried over between sessions)
    # from the materialized tensors that needs to be send as a part of
    # initial checkpoint.
    # Note: if the tensor with appliance info was modified, so the appliance
    # info will be replaced with graph/file/memory info which means that this
    # tensor will be sent as a part of initial checkpoint and the original
    # tensor inside PTR will be dropped.
    if appliance_info := appliance_data.get_appliance_info():
        # need to perform shape check here since "shape" is implemented
        # differently for ApplianceInfo versus other storage types
        if deferred_appliance_data.shape != appliance_info.shape:
            raise RuntimeError(
                f"The shape of the tensor from the appliance {deferred_appliance_data.shape} "
                f"does not match the shape of the checkpointed tensor {appliance_info.shape}. "
                f"This indicates an internal bug. Please contact Cerebras Support for help."
            )
        appliance_info.share_storage(deferred_appliance_data)
    else:
        if deferred_appliance_data.shape != appliance_data.shape:
            raise RuntimeError(
                f"The shape of the tensor from the appliance {deferred_appliance_data.shape} "
                f"does not match the shape of the checkpointed tensor {appliance_data.shape}. "
                f"This indicates an internal bug. Please contact Cerebras Support for help."
            )
        appliance_data.share_storage(deferred_appliance_data)
