# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Cerebras subclass of module parameter and buffers."""
import warnings
from typing import Optional

import torch
from typing_extensions import Self

from cerebras.pytorch.backend import current_backend_impl
from cerebras.pytorch.storage.utils import lazy_tensor_data_wrapper


class _WrapperTensorMixin:
    """Mixin for tensors where the underlying storage is a lazy/cpu tensor.

    This class overrides certain methods to handle cases for moving/copying
    tensors between devices. This mixin is used to implement our custom version
    of `torch.nn.Parameter` and module buffers.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __new__(cls, data=None):  # pylint: disable=signature-differs
        assert isinstance(data, torch.Tensor)

        if isinstance(data, _WrapperTensorMixin):
            return data

        # pylint: disable=protected-access,attribute-defined-outside-init
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._lazy_param: Optional[Self] = None
        self._cpu_param: Optional[Self] = None

    def copy_(self, tensor: torch.Tensor) -> Self:
        """Copy the tensor into this parameter.

        This overrides the default copy_() method to handle copying tensors with
        storage onto this parameter. This is needed because the default copy_()
        method only traces the "copy_()" op. However, for cases where we're
        loading a checkpoint tensor into this parameter, we want to copy actual
        storage instead of tracing.
        """
        if self.device.type == "lazy" and (
            not current_backend_impl().is_tracing
        ):
            _lazy_copy_storage(self, tensor)
            # Clear the CPU param since we've replaced the underlying storage
            self._cpu_param = None
        else:
            super().copy_(tensor)

        return self

    def clone(self, *args, **kwargs) -> Self:
        if self.device.type == "cpu":  # pylint: disable=no-member
            assert self._lazy_param is not None
            cloned = self._lazy_param.clone()
            cloned.requires_grad = self.requires_grad
            return cloned.to("cpu")
        elif self.device.type == "lazy":
            from cerebras.pytorch.lib import cerebras_pytorch_lib

            if not current_backend_impl().is_tracing:
                cloned = type(self)(
                    cerebras_pytorch_lib.clone_tensor(self._data)
                )
                cloned.requires_grad = self.requires_grad
                return cloned
            else:
                return super().clone(*args, **kwargs)
        else:
            raise RuntimeError(
                f"Unsupported device type: {self.device.type}. Expected one of "
                f"\"cpu\" or \"lazy\"."
            )

    def __deepcopy__(self, memo: dict) -> Self:
        return memo.setdefault(id(self), self.clone())

    def detach(self) -> Self:
        return self

    def __getattribute__(self, name):
        if name == "data" and self.device.type == "lazy":
            backend = current_backend_impl()
            if backend.is_tracing:
                raise RuntimeError(
                    "Cannot access data attribute of a lazy tensor while tracing. "
                    "Please modify the tensor directly so that all operations are traced."
                )

            from cerebras.pytorch.lib import cerebras_pytorch_lib

            # Accessing the data attribute has the semantics of
            # Getting a new tensor object that shares the same storage
            # as the original tensor (but with requires_grad=false)
            # This is done in the implementation of get_parameter_data
            # by creating a new lazy tensor with the same data pointer.
            # Now, any inplace ops that happen to the returned tensor
            # are reflected back on the original tensor (i.e. it is traced)
            return cerebras_pytorch_lib.get_parameter_data(self)

        # For some cases in Parameter methods we still need access to the original
        # data. So as a workaround  we call self._data which bypass self.data
        # access logic.
        if name == "_data":
            name = "data"

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "data" and self.device.type == "lazy":
            from cerebras.pytorch.lib import cerebras_pytorch_lib

            # Setting the data attribute has the semantics of wanting to
            # share the storage of another tensor, namely `value`.
            # This is done by changing the underlying lazy tensor
            # that the LTCTensorImpl uses, thereby getting this
            # parameter to use the same data pointer as the value tensor
            cerebras_pytorch_lib.set_parameter_data(src=value, dst=self)
            return

        # Symmetric method to set data using self._data=value to original tensor
        # data field.
        if name == "_data":
            name = "data"

        return super().__setattr__(name, value)

    def to(self, *args, **kwargs) -> Self:
        # pylint: disable=protected-access
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        # TODO: Error out if dtype, non_blocking, or memory_format are specified

        if device is None:
            pass
        elif device.type != self.device.type:  # pylint: disable=no-member
            if device.type != "cpu":  # pylint: disable=no-member
                assert self._lazy_param is not None
                # User may have modified the parameter since we captured
                # _lazy_param, so update the replacement to keep it consistent
                self._lazy_param.requires_grad = self.requires_grad
                # Clear the CPU param to release virtual memory in case of
                # file-backed appliance data. This is needed to avoid OOM errors
                # during python subprocess forking.
                self._lazy_param._cpu_param = None

                return self._lazy_param
            else:
                if self._cpu_param is not None:
                    # User may have modified the parameter since we captured
                    # _lazy_param, so update the replacement to keep it
                    # consistent
                    self._cpu_param.requires_grad = self.requires_grad
                    return self._cpu_param

                if self.device.type == "lazy":
                    from cerebras.pytorch.lib import cerebras_pytorch_lib

                    try:
                        data = cerebras_pytorch_lib.get_appliance_data(
                            self._data
                        )
                        tensor = data.tensor
                    except RuntimeError:
                        backend = current_backend_impl()
                        if backend.device.config.drop_data:
                            # In compile only, we actually discard app data to
                            # save memory. However, we may need the metadata of
                            # the tensor, so return an empty tensor like it.
                            tensor = torch.empty_like(self, device=device)
                        else:
                            # Otherwise, that was a valid error...
                            raise
                    cpu_param = type(self)(tensor)
                    cpu_param.requires_grad = self.requires_grad
                else:
                    cpu_param = super().to(*args, **kwargs)

                # pylint: disable=protected-access
                cpu_param._lazy_param = self

                self._cpu_param = cpu_param
                return cpu_param
        else:
            return self

        return super().to(*args, **kwargs)

    def __repr__(self) -> str:
        if self.device.type != "cpu":
            warnings.warn(
                "Parameter repr may not contain actual stored values. "
                "Move the tensor to cpu via .to(\"cpu\") to view real values."
            )
        return super().__repr__()

    def __str__(self) -> str:
        if self.device.type != "cpu":
            warnings.warn(
                "Parameter str may not contain actual stored values. "
                "Move the tensor to cpu via .to(\"cpu\") to view real values."
            )
        return super().__str__()


class Parameter(_WrapperTensorMixin, torch.nn.Parameter):
    """Custom wrapper for module parameters."""


class Buffer(_WrapperTensorMixin, torch.Tensor):
    """Custom wrapper for module buffers."""


def _lazy_copy_storage(
    self: torch.Tensor, tensor: torch.Tensor
) -> torch.Tensor:
    """Copies the data from `tensor` into `self` without tracing.

    This is to handle 2 cases where we want to copy storage instead of tracing:
    1. Copying a CPU tensor into a lazy parameter. This can happen
       when we load a vanilla torch checkpoint onto CPU and then
       load it into a lazy parameter.
    2. Copying a DeferredTensor into a lazy parameter. This can happen
       when we load an H5 checkpoint onto the lazy device and then
       load it into a lazy parameter.

    Args:
        self: The lazy tensor to copy into.
        tensor: The tensor to copy from.
    Returns:
        self.
    """
    from cerebras.pytorch.backend import current_backend
    from cerebras.pytorch.lib import cerebras_pytorch_lib

    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"Attempting to copy a non-tensor type {type(tensor)} into a "
            f"tensor."
        )

    if self.shape != tensor.shape:
        raise RuntimeError(
            f"Cannot copy tensor of different shape ({tensor.shape}) "
            f"into a lazy buffer with shape {self.shape}."
        )

    if self.dtype != tensor.dtype:
        # If the dtype is different we need to get the data wrapper
        # and convert the tensor to the correct dtype before copying.
        # Note, for deferred tensors, this will involve an extra
        # read and write to disk, but this should be rare as most
        # cases should be handled by the checkpoint converter

        tensor = lazy_tensor_data_wrapper(tensor)
        tensor = tensor.type(self.dtype)

    # Currently a vanilla torch checkpoint is loaded onto CPU, so we need
    # to handle copying from a CPU tensor to a lazy tensor.
    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
        with current_backend(raise_warning=False).device:
            lazy_tensor = tensor.to("lazy")
    else:
        lazy_tensor = tensor

    # lazy tensor `copy_()` traces the op and doesn't actually copy the
    # underlying storage.
    # We want to copy the ApplianceData storage through sharing data with the `lazy_tensor`.
    # Or copy the IR value if we're currently tracing the initialization.
    with current_backend(raise_warning=False).device:
        cerebras_pytorch_lib.tensor_copy_(lazy_tensor, self)

    return self
