# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import logging
from contextlib import contextmanager
from threading import Event, Thread
from typing import Union

import numpy
import torch

import cerebras.pytorch as cstorch
from cerebras.appliance.data.conversions import (
    np_dtype_from_rtfx_dtype,
    rtfx_dtype_from_np_dtype,
)
from cerebras.pytorch.lib import cerebras_pytorch_lib


def lazy_tensor_data_wrapper(
    tensor: Union[torch.Tensor, "cerebras_pytorch_lib.ApplianceDataInfo"]
) -> torch.Tensor:
    """A wrapper for tensors that returns the underlying CPU view.

    Args:
        tensor: The tensor to return the CPU view of. If tensor is on cpu, it
            is returned as is. If tensor is on a lazy device, the underlying
            ApplianceDataInfo object is queried first, then the CPU view of it
            is returned.
    Returns:
        The CPU view of the tensor. Modifying this tensor will modify the
        lazy tensor's device data.
    """
    from cerebras.pytorch import cerebras_pytorch_lib

    from .serializers import DeferredFullTensor, DeferredGraphTensor

    if isinstance(tensor, torch.Tensor):
        if tensor.device.type == "lazy":
            app_data = cerebras_pytorch_lib.get_appliance_data(tensor)
        else:
            return tensor
    elif isinstance(tensor, cerebras_pytorch_lib.ApplianceDataInfo):
        app_data = tensor
    else:
        raise ValueError(
            f"Attempting to create a lazy tensor wrapper for a value of type "
            f"{type(tensor)}, but one of torch.Tensor and ApplianceDataInfo "
            f"was expected."
        )

    if full_tensor := app_data.full_tensor:
        # If a tensor can be represented as a full tensor, return a deferred
        # full tensor so that we can transfer the data without much overhead.
        return DeferredFullTensor(
            full_tensor.size,
            getattr(torch, full_tensor.dtype),
            full_tensor.value,
        )
    elif jit_graph := app_data.jit_graph:
        # If a tensor has a JIT graph, we can use it to create a tensor.
        return DeferredGraphTensor(
            jit_graph.graph,
            jit_graph.arguments,
            jit_graph.size,
            getattr(torch, jit_graph.dtype),
        )
    else:
        # This tensor may be one of the DeferredTensor's below
        # Or raise an exception if tensor data does not exist
        return app_data.tensor


def has_lazy_tensor_data_impl(tensor: torch.Tensor) -> bool:
    """Returns True if the lazy tensor has data it can use to create a CPU view."""
    if isinstance(tensor, torch.Tensor) and tensor.device.type == "lazy":
        from cerebras.pytorch import cerebras_pytorch_lib

        if cerebras_pytorch_lib.has_backend_data(tensor):
            app_data = cerebras_pytorch_lib.get_appliance_data(tensor)
            return app_data.filename is not None or app_data.has_tensor_storage

    return False


def to_deferred(tensors):
    from cerebras.pytorch.backend import current_backend_impl

    backend = current_backend_impl(raise_exception=False)
    if backend is not None and backend.is_csx:
        backend.initial_mark_step()

    def _to_deferred(tensor):
        if isinstance(tensor, torch.Tensor):
            return lazy_tensor_data_wrapper(tensor)
        return tensor

    return torch.utils._pytree.tree_map(_to_deferred, tensors)


def torch_to_np_dtype(dtype: torch.dtype) -> numpy.dtype:
    """Converts a torch dtype to a numpy dtype."""
    return cstorch.to_numpy(torch.empty(0, dtype=dtype, device="cpu")).dtype


def np_to_torch_dtype(dtype: numpy.dtype) -> torch.dtype:
    """Converts a numpy dtype to a torch dtype."""
    return cstorch.from_numpy(numpy.empty(0).astype(dtype)).dtype


def torch_to_rtfx_dtype(dtype: torch.dtype):
    """Converts a torch dtype to a numpy dtype."""
    np_dtype = torch_to_np_dtype(dtype)
    return rtfx_dtype_from_np_dtype(np_dtype)


def rtfx_to_torch_dtype(rtfx_dtype: int):
    """Converts an RtFx dtype to a torch dtype."""
    np_dtype = np_dtype_from_rtfx_dtype(rtfx_dtype)
    return np_to_torch_dtype(np_dtype)


@contextmanager
def monitor_execute_jit_graph(jit_graph: str, timeout: int | None = None):
    """
    Context manager to monitor the execution of a JIT graph.
    This will log a warning if the JIT graph execution takes longer than
    the configured timeout.

    Args:
        jit_graph: The JIT graph to monitor.
        timeout: The timeout in seconds. If None, no monitoring is done.
    """
    if timeout is None:
        yield
        return

    execute_finished = Event()

    def monitor_function():
        wait_timeout = timeout // 10
        for i in range(10):
            if execute_finished.wait(timeout=wait_timeout):
                return
            logging.warning(
                f"Still waiting for JIT graph to execute ({i * wait_timeout} secs)"
            )

        logging.error(
            f"JIT graph execution is taking longer than {timeout} seconds. "
            f"This may indicate a problem.\n"
            f"JIT Graph:\n{jit_graph}"
        )

    thread = Thread(target=monitor_function)
    thread.start()

    try:
        yield
    finally:
        execute_finished.set()
