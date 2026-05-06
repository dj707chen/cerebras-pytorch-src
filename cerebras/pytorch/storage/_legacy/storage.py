# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Implementations of the containers for appliance data tensors.
"""

import dill
import h5py as h5
import torch

from cerebras.appliance.data.dtypes import bf16
from cerebras.appliance.saver.h5_saver import H5Saver, register_h5_type
from cerebras.pytorch.storage.utils import np_to_torch_dtype


@register_h5_type()
class StoredTensorH5Type:
    """Class for loading custom torch.Tensor's from previous releases."""

    @staticmethod
    def save(tensor, f: h5.File, key: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load(f: h5.File, key: str):
        return DeferredH5Tensor.load(f.filename, key)


@register_h5_type()
class StoredApplianceTensorH5Type:
    """Class for loading custom torch.Tensor's from previous releases."""

    @staticmethod
    def save(tensor, f: h5.File, key: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load(f: h5.File, key: str) -> torch.Tensor:
        return DeferredFileTensor.load(f, key)


@register_h5_type()
class FullTensorH5Type:
    """Class for loading custom torch.Tensor's from previous releases."""

    @staticmethod
    def save(tensor, f: h5.File, key: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load(f: h5.File, key: str):
        return DeferredFullTensor.load(f, key)


@register_h5_type()
class DeferredFileTensor:
    """Class for loading deferred file tensors from previous releases."""

    def save(self, f: h5.File, name: str, **kwargs) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(f: h5.File, key: str) -> "DeferredFileTensor":
        from cerebras.pytorch.storage.serializers import (
            DeferredFileTensor as NewDeferredFileTensor,
        )

        dataset = f[key]

        return NewDeferredFileTensor(
            filepath=dataset.attrs["filepath"],
            size=torch.Size(dataset.attrs["shape"]),
            dtype=dill.loads(bytes.fromhex(dataset.attrs["dtype"])),
        )


@register_h5_type()
class DeferredFullTensor:
    """Class for loading deferred full tensors from previous releases."""

    def save(self, f: h5.File, name: str, **kwargs) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(f: h5.File, key: str) -> "DeferredFullTensor":
        from cerebras.pytorch.storage.serializers import (
            DeferredFullTensor as NewDeferredFullTensor,
        )

        dset = f[key]

        size = torch.Size(dset.attrs["shape"])
        value = dset.attrs["fill_value"].item()
        np_dtype = dset.dtype
        if dset.attrs["is_bfloat16"]:
            np_dtype = bf16
        dtype = np_to_torch_dtype(np_dtype)

        return NewDeferredFullTensor(size, dtype=dtype, value=value)


@register_h5_type()
class DeferredGraphTensor:
    """Class for loading deferred graph tensors from previous releases."""

    def save(self, f: h5.File, name: str, **kwargs) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(f: h5.File, key: str) -> "DeferredGraphTensor":
        from cerebras.pytorch.storage.serializers import (
            DeferredGraphTensor as NewDeferredGraphTensor,
        )

        dset = f[key]

        jit_graph = dset.attrs["jit_graph"]

        saver = H5Saver()
        args = [
            saver._load_tensor_from_checkpoint(f, arg_name)
            for arg_name in dset.attrs["args"]
        ]

        size = torch.Size(dset.attrs["shape"])
        np_dtype = dset.dtype
        if dset.attrs.get("is_bfloat16"):
            np_dtype = bf16
        dtype = np_to_torch_dtype(np_dtype)

        return NewDeferredGraphTensor(jit_graph, args, size, dtype)


@register_h5_type()
class DeferredH5Tensor:
    """A deferred tensor whose data is stored in an H5 file."""

    def save(self, f: h5.File, name: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load(f: h5.File, key: str) -> "DeferredH5Tensor":
        dset = f[key]
        key = dset.attrs["key"]
        link_to_self = dset.attrs.get("link_to_self", False)
        filepath = f.filename if link_to_self else dset.attrs["filepath"]

        from cerebras.appliance.storage.serializers import DeferredObject
        from cerebras.pytorch.storage.serializers import DeferredTorchTensor
        from cerebras.pytorch.storage.utils import np_to_torch_dtype

        with h5.File(filepath, "r") as _f:
            _dset = _f[key]
            shape = _dset.shape
            dtype = _dset.dtype
            is_bfloat16 = _dset.attrs.get("is_bfloat16")

        dtype = torch.bfloat16 if is_bfloat16 else np_to_torch_dtype(dtype)

        return DeferredTorchTensor(
            DeferredObject(
                filepath,
                key,
                metadata={
                    "__TYPE__": "TorchTensorSerializer",
                    "shapes": [shape],
                    "dtypes": [str(dtype)],
                },
            ),
            shape,
            dtype,
        )
