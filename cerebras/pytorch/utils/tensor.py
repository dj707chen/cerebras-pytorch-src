# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch

from cerebras.pytorch.backend import current_torch_device, use_cs
from cerebras.pytorch.storage.serializers import DeferredFullTensor


def full(shape, value: float, dtype=None):
    """
    Returns an lazily initialized tensor filled with the provided value.

    Args:
        shape: The shape of the tensor.
        value: The value to fill the tensor with.
        dtype: The dtype of the tensor.
    """
    if not use_cs():
        return torch.full(shape, value, dtype=dtype)

    return DeferredFullTensor(shape, dtype=dtype, value=value).to(
        current_torch_device()
    )


def full_like(other: torch.Tensor, value: float, dtype=None):
    """
    Returns an lazily initialized full tensor with the same properties as the
    provided tensor.

    Args:
        other: The tensor to copy the properties from
        value: The value to fill the tensor with
        dtype: The dtype of the tensor. If not provided, the dtype of the other
            tensor is used
    """
    if not dtype:
        dtype = other.dtype
    if not use_cs():
        return torch.full_like(other, value, dtype=dtype)

    return DeferredFullTensor(other.shape, dtype=dtype, value=value).to(
        current_torch_device()
    )


def ones(shape, dtype=None):
    """
    Returns an lazily initialized tensor filled with ones.

    Args:
        shape: The shape of the tensor
        dtype: The dtype of the tensor
    """
    if not use_cs():
        return torch.ones(shape, dtype=dtype)

    return DeferredFullTensor(shape, dtype=dtype, value=1).to(
        current_torch_device()
    )


def ones_like(other: torch.Tensor, dtype=None):
    """
    Returns an lazily initialized tensor full of ones with the same properties
    as the provided tensor.

    Args:
        other: The tensor to copy the properties from
        dtype: The dtype of the tensor. If not provided, the dtype of the other
            tensor is used
    """
    if not dtype:
        dtype = other.dtype
    if not use_cs():
        return torch.ones_like(other, dtype=dtype)

    return DeferredFullTensor(other.shape, dtype=dtype, value=1).to(
        current_torch_device()
    )


def zeros(shape, dtype=None):
    """
    Returns an lazily initialized tensor filled with zeros.

    Args:
        shape: The shape of the tensor
        dtype: The dtype of the tensor
    """
    if not use_cs():
        return torch.zeros(shape, dtype=dtype)

    return DeferredFullTensor(shape, dtype=dtype, value=0).to(
        current_torch_device()
    )


def zeros_like(other: torch.Tensor, dtype=None):
    """
    Returns an lazily initialized tensor full of zeros with the same properties
    as the provided tensor.

    Args:
        other: The tensor to copy the properties from
        dtype: The dtype of the tensor. If not provided, the dtype of the other
            tensor is used
    """
    if not dtype:
        dtype = other.dtype
    if not use_cs():
        return torch.zeros_like(other, dtype=dtype)

    return DeferredFullTensor(other.shape, dtype=dtype, value=0).to(
        current_torch_device()
    )
