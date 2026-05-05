# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Implementations of the containers for appliance data tensors
"""
import copy
import os
import weakref
from numbers import Number
from typing import List, Optional, TextIO, Type, Union

import dill
import h5py as h5
import numpy
import torch
from torch.utils._pytree import tree_map

import cerebras.pytorch as cstorch
from cerebras.appliance.appliance_manager import ApplianceInfo
from cerebras.appliance.data.dtypes import bf16, is_bf16
from cerebras.appliance.saver.h5_saver import (
    NumpyArrayH5Type,
    ObjectH5Type,
    hdf5_locking,
    register_h5_type,
)
from cerebras.appliance.utils._contexts import BooleanContext
from cerebras.pytorch.backend import current_torch_device, use_cs
from cerebras.pytorch.lib import cerebras_pytorch_lib

from .pt_h5_saver import TorchTensorH5Type

# Flag for controlling whether to store tensors to H5 via external links.
use_external_link = BooleanContext(default=False)
# Flag for controlling whether to pickle cstorch tensors as torch tensors.
use_cstorch_types = BooleanContext(default=False)
# Flag for controlling whether to cache deferred tensors.
cache_deferred_tensors = BooleanContext(default=True)
# Flag for controlling whether to validate deferrerd tensor "hash" when materializing.
# Deferred tensors validate that at point of materialization, the timestamp of
# the backing file (if any) has not changed (to prevent indirect modification
# of the tensor data). On some filesystems, timestamps are not always accurate
# and we could get false positives. By default, we are strict and do the check.
# This context provides a way to opt-out of this check to workaround the limitations
# of timestamps.
check_deferred_backing_storage = BooleanContext(default=True)
# Initial checkpointing context that enables custom h5 types serialization support.
saving_initial_state = BooleanContext(default=False)


@register_h5_type(cerebras_pytorch_lib.ApplianceInfo)
class ApplianceInfoH5Type(ObjectH5Type):
    """
    Class for saving cerebras_pytorch_lib.ApplianceInfo storage to the H5 checkpoint
    """

    @staticmethod
    def save(
        appliance_info: cerebras_pytorch_lib.ApplianceInfo,
        f: h5.File,
        key: str,
        **kwargs,
    ) -> Type["ApplianceInfoH5Type"]:
        """Saves the ApplianceInfo storage to the provided H5 File"""
        if not saving_initial_state:
            raise RuntimeError(
                f"Saving ApplianceInfo objects to checkpoints is not allowed. "
                f"Please ensure that \"{key}\" is a `ApplianceDataInfo` instance instead."
            )
        ObjectH5Type.save(
            ApplianceInfo(
                tensor_id=appliance_info.uid,
            ),
            f,
            key,
            **kwargs,
        )

        return ApplianceInfoH5Type


@register_h5_type()
class StoredTensorH5Type:
    """Class for loading custom torch.Tensor's from previous releases."""

    @staticmethod
    def save(tensor, f: h5.File, key: str, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load(f: h5.File, key: str):
        return DeferredH5Tensor(f.filename, key)


@register_h5_type()
class StoredApplianceTensorH5Type:
    """Class for loading custom torch.Tensor's from previous releases."""

    @staticmethod
    def save(tensor, f: h5.File, key: str, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load(f: h5.File, key: str) -> torch.Tensor:
        return DeferredFileTensor.load(f, key)


@register_h5_type()
class FullTensorH5Type:
    """Class for loading custom torch.Tensor's from previous releases."""

    @staticmethod
    def save(tensor, f: h5.File, key: str, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load(f: h5.File, key: str):
        return DeferredFullTensor.load(f, key)


def full(shape, value: float, dtype=None):
    """
    Returns an lazily initialized tensor filled with the provided value

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
    provided tensor

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
    Returns an lazily initialized tensor filled with ones

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
    as the provided tensor

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
    Returns an lazily initialized tensor filled with zeros

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
    as the provided tensor

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

    if app_data.filename:
        # Currently, LTC creates a file-backed tensor internally with a normal
        # torch.Tensor type, so we need to wrap it in a DeferredFileTensor to
        # avoid copying when creating initial state.
        return DeferredFileTensor(
            app_data.filename, app_data.tensor.size(), app_data.tensor.dtype
        )
    else:
        # This tensor may be one of the DeferredTensor's below
        # Or raise an exception if tensor data does not exist
        return app_data.tensor


def has_lazy_tensor_data_impl(tensor: torch.Tensor) -> bool:
    """Returns True if the lazy tensor has data it can use to create a CPU view"""
    if isinstance(tensor, torch.Tensor) and tensor.device.type == "lazy":
        from cerebras.pytorch import cerebras_pytorch_lib

        if cerebras_pytorch_lib.has_backend_data(tensor):
            app_data = cerebras_pytorch_lib.get_appliance_data(tensor)
            return app_data.filename is not None or app_data.has_tensor_storage

    return False


class DeferredTensor(torch.Tensor):
    """A deferred tensor that is lazily materialized on the CPU.

    This is a base class for a tensor that provides a recipe for getting its
    value. The tensor is not materialized until some torch operation is called
    on it, at which point it's materialized to CPU and all subsequent accesses
    are applied to to the materialized CPU tensor.

    Deferred tensors are especially useful when moving to lazy tensors. Instead
    of incurring copies, the tensor handle is stored in the lazy tensor. If the
    tensor is materialized and modified, moving the tensor to lazy incurs a full
    copy because at that point the recipe is already out of data.

    NOTE: that all subclass names must start with "Deferred" and end with
    "Tensor" to be recognized by appliance data to avoid copying when moving to
    lazy device.
    """

    # If __torch_dispatch__ is defined, the default torch function
    # implementation (which preserves subclasses) typically must be disabled.
    __torch_function__ = torch._C._disabled_torch_function_impl

    def __init__(self):
        super().__init__()

        # The cpu tensor that is materialized when the tensor is accessed.
        self._tensor: Optional[torch.Tensor] = None
        # Keep track of any changes to the CPU tensor. If there are changes,
        # when moving to lazy, we need to use the CPU tensor. Otherwise, we
        # use the original data.
        self._is_dirty = False
        # If True, we cache the CPU tensor. This is useful for tensors that
        # are accessed multiple times, or are modified inplace.
        # However, it can cause memory issues if the tensor is large, or if
        # many larger tensors are cached.
        self._cache_tensor = bool(cache_deferred_tensors)

    @property
    def is_modified(self) -> bool:
        """Returns True if the tensor has been materialized and modified."""
        return self._tensor is not None and self._is_dirty

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        tensor_handles: List[cls] = []

        def unwrap(t):
            if isinstance(t, cls):
                cpu_handle = t._materialize()
                tensor_handles.append((t, cpu_handle))
                return cpu_handle
            return t

        res = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        for t, c in tensor_handles:
            if (
                # In-place modification
                (
                    hasattr(func, "_schema")
                    and func._schema.name[-1] == "_"
                    and t is args[0]
                )
                or (  # built-in inplace ops don't have schemas. Check name attribute
                    hasattr(func, "__name__")
                    and func.__name__[-1] == "_"
                    and t is args[0]
                )
                # Modified through being an output of an operation
                or ("out" in kwargs and t is kwargs["out"])
            ):
                # Explicitly cache a tensor if it's modified inplace
                t._cache_tensor = True
                t._tensor = c
                t._is_dirty = True

        return res

    def save(
        self, f: h5.File, name: str, **kwargs
    ) -> Union[TorchTensorH5Type, None]:
        """Saves the tensor to an H5 file.

        If tensor is materialized on CPU and modified, this uses a normal torch
        tensor H5 type to save and returns the type used for saving. Otherwise,
        it uses the deferred tensor type and returns None, so that subsequent
        loads use the same type.

        Args:
            f: The H5 file to save to.
            name: The name of the dataset to save to.
            **kwargs: Additional arguments to pass to the H5 save function for
                compression.
        Returns:
            The H5 type used to save the tensor, or None if type(self) was used
            to save.
        """

        if self._is_dirty:
            return TorchTensorH5Type.save(self._tensor, f, name, **kwargs)
        return self._save(f, name, **kwargs)

    def _materialize(
        self, cache_override: Optional[bool] = None
    ) -> torch.Tensor:
        """Returns the materialized CPU tensor.

        If tensor was already materialized, this returns the already cached
        tensor. Otherwise, it materializes the tensor, (conditionally) caches
        it, and returns it.

        Args:
            cache: Whether to override the default cache settings when materializing.
        """
        tensor = self._to_cpu() if self._tensor is None else self._tensor
        tensor.requires_grad = self.requires_grad

        should_cache = (
            cache_override if cache_override is not None else self._cache_tensor
        )
        if should_cache:
            self._tensor = tensor

        return tensor

    ############################################################################
    # torch.Tensor overrides                                                   #
    ############################################################################

    def to(self, *args, **kwargs):
        """Overrides the default to() implementation to handle lazy tensors."""
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None and device.type == "lazy":
            from cerebras.pytorch.backend import current_backend
            from cerebras.pytorch.lib import cerebras_pytorch_lib

            with current_backend(raise_warning=False).device:
                if not self.is_modified:
                    # This custom implementation creates a new lazy tensor whose
                    # underlying device data is set to this tensor handle. This
                    # avoids copying any data. Note that moving to lazy deletes
                    # the storage of "self". This is fine because the storage
                    # is never directly used as any CPU operation is done on
                    # the materialized tensor.
                    return cerebras_pytorch_lib.eager_to_lazy(self)
                else:
                    # If materialized tensor has been modified, we need to use
                    # the default implementation which copies the data.
                    return super().to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def numpy(self) -> numpy.ndarray:
        """Implements numpy() for deferred tensors."""
        # Set dirty to True as it is possible that the numpy array
        # will be modified inplace
        self._is_dirty = True
        return self._materialize().numpy()

    def tolist(self) -> list:
        """Implements tolist() for deferred tensors."""
        return self._materialize().tolist()

    def clone(self) -> "DeferredTensor":
        """Implements clone() for deferred tensors."""
        if not self.is_modified:
            cloned = self._clone()
            cloned.requires_grad = self.requires_grad
            return cloned
        return super().clone()

    def detach(self) -> torch.Tensor:
        """Implements detach() for deferred tensors.

        Note that this currently falls back to the original implementation,
        which materializes the tensor. The contract of detach is that the
        returned tensor shares the same storage with the original one. However,
        imagine the following case:
            1. A is a deferred tensor not materialized yet.
            2. B = A.detach() is called
            3. A += 1 is called, which materialies A
        In this sequence, B does not see the modification to A. To avoid this
        issue, we currently materialize the tensor when detach() is called.
        """
        return super().detach()

    def __deepcopy__(self, memo: dict) -> "DeferredTensor":
        """Implements deepcopy() for deferred tensors."""
        if not self.is_modified:
            return memo.setdefault(id(self), self._clone())
        new_tensor = copy.deepcopy(self._materialize(), memo)
        new_tensor.requires_grad = self.requires_grad
        return new_tensor

    def __reduce_ex__(self, protocol):
        """Implements __reduce_ex__() for deferred tensors.

        This add special pickling support for deferred tensors (e.g., used in
        torch.save()). If saving cstorch types is allowed, the tensor subclass
        is pickled as is. Otherwise, the tensor is materialized and the class
        is pickled as a normal torch tensor. This is to avoid strict dependency
        on cstorch types in checkpoints when needed.
        """
        if use_cstorch_types:
            return super().__reduce_ex__(protocol)
        return self._materialize().__reduce_ex__(protocol)

    ############################################################################
    # Abstract methods to override                                             #
    ############################################################################

    def _save(self, f: h5.File, name: str, **kwargs) -> None:
        """Saves the tensor to an H5 file.

        This is called when the tensor has not been previously not materialized
        on CPU, which means the deferred type can be saved to H5 for further
        retrieval.
        """
        raise NotImplementedError

    @staticmethod
    def load(f: h5.File, key: str) -> Type["DeferredTensor"]:
        """Loads a tensor from an H5 file.

        Args:
            f: The H5 file to load from.
            key: The dataset name that holds the tensor value.
        """
        raise NotImplementedError

    def _to_cpu(self) -> torch.Tensor:
        """Materializes the tensor to CPU and returns it."""
        raise NotImplementedError

    def _clone(self) -> "DeferredTensor":
        """Clones the non-materialized tensor and returns it."""
        raise NotImplementedError


@register_h5_type()
class DeferredFileTensor(DeferredTensor):
    """A deferred tensor whose data is stored in a binary file."""

    def __new__(cls, filepath: str, size: torch.Size, dtype: torch.dtype):
        data = torch.empty(size, dtype=dtype, device="cpu")
        return cls._make_subclass(cls, data, require_grad=False)

    def __init__(self, filepath: str, size: torch.Size, dtype: torch.dtype):
        """Constructs a `DeferredFileTensor` instance.

        Args:
            filepath: The path to the binary file that holds the tensor data.
            size: The size of the tensor.
            dtype: The data type of the tensor.
        """
        super().__init__()

        self._filepath = os.path.abspath(filepath)

        # Store the last stat of the file so we can check if the file
        # has been modified since the tensor was created before materializing it
        self._last_stats = os.stat(filepath)

    def _save(self, f: h5.File, name: str, **kwargs) -> None:
        if not use_external_link or not self.shape:
            # When external links are disabled, we need to materialize the
            # tensor and save it to file. But note that we don't cache the
            # materialized tensor to avoid OOM.
            return TorchTensorH5Type.save(
                self._materialize(cache_override=False), f, name, **kwargs
            )

        dset = f.create_dataset(name, data=h5.Empty("f"))
        dset.attrs["filepath"] = self._filepath
        dset.attrs["shape"] = tuple(self.shape)
        dset.attrs["dtype"] = dill.dumps(self.dtype).hex()

    @staticmethod
    def load(f: h5.File, key: str) -> "DeferredFileTensor":
        dataset = f[key]
        return DeferredFileTensor(
            filepath=dataset.attrs["filepath"],
            size=torch.Size(dataset.attrs["shape"]),
            dtype=dill.loads(bytes.fromhex(dataset.attrs["dtype"])),
        )

    def _to_cpu(self) -> torch.Tensor:
        _check_file_modification(
            self._filepath,
            self._last_stats,
            f"materialize deferred tensor from file {self._filepath}",
        )

        # Return a read-only file-backed tensor. Upon write, the tensor will
        # be converted to an in-memory tensor.
        return torch.from_file(
            self._filepath,
            shared=False,  # Opens in read-only mode
            size=self.shape.numel(),
            dtype=self.dtype,
        ).reshape(self.shape)

    def _clone(self) -> "DeferredFileTensor":
        cloned = DeferredFileTensor(self._filepath, self.shape, self.dtype)
        cloned.requires_grad = self.requires_grad
        return cloned


@register_h5_type()
class DeferredFullTensor(DeferredTensor):
    """A deferred torch.full() tensor."""

    def __new__(
        cls,
        size: torch.Size,
        dtype: Optional[torch.dtype] = None,
        value: Optional[Number] = None,
    ):
        data = torch.empty(size, dtype=dtype, device="cpu")
        return cls._make_subclass(cls, data, require_grad=False)

    def __init__(
        self,
        size: torch.Size,
        dtype: Optional[torch.dtype] = None,
        value: Optional[Number] = None,
    ):
        """Constructs a `DeferredFullTensor` instance.

        Args:
            size: The size of the tensor.
            dtype: The data type of the tensor. If not specified, defaults to
                the default torch dtype.
            value: The value to fill the tensor with. If not specified, defaults
                to uninitialized data.
        """
        super().__init__()

        self._value = value

    @property
    def fill_value(self) -> Number:
        """Returns the fill value."""
        return self._value

    def _save(self, f: h5.File, name: str, **kwargs) -> None:
        np_dtype = torch_to_np_dtype(self.dtype)

        dset = f.create_dataset(name, dtype=np_dtype)
        dset.attrs["shape"] = tuple(self.shape)
        dset.attrs["fill_value"] = self._value
        dset.attrs["is_bfloat16"] = is_bf16(np_dtype)

    @staticmethod
    def load(f: h5.File, key: str) -> "DeferredFullTensor":
        dset = f[key]

        size = torch.Size(dset.attrs["shape"])
        value = dset.attrs["fill_value"]
        np_dtype = dset.dtype
        if dset.attrs["is_bfloat16"]:
            np_dtype = bf16
        dtype = _np_to_torch_dtype(np_dtype)

        return DeferredFullTensor(size, dtype=dtype, value=value)

    def _to_cpu(self) -> torch.Tensor:
        if self._value is None:
            return torch.empty(self.shape, dtype=self.dtype)
        elif self._value == 0:
            return torch.zeros(self.shape, dtype=self.dtype)
        elif self._value == 1:
            return torch.ones(self.shape, dtype=self.dtype)
        else:
            return torch.full(self.shape, self._value, dtype=self.dtype)

    def _clone(self) -> "DeferredFullTensor":
        cloned = DeferredFullTensor(self.shape, self.dtype, self._value)
        cloned.requires_grad = self.requires_grad
        return cloned


class _CachingFileOpener:
    """File opener that reuses file descriptors from previous opened files."""

    def __init__(self):
        # Keep a weak reference to the file descriptors
        self._open_files = weakref.WeakValueDictionary()

    def __call__(self, path, *args, **kwargs):
        """Opens the file (or reuses previously opened file) and returns the file descriptor."""
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns)

        if key in self._open_files:
            return self._open_files[key]

        fp = open(path, *args, **kwargs)
        self._open_files[key] = fp
        return fp


@register_h5_type()
class DeferredH5Tensor(DeferredTensor):
    """A deferred tensor whose data is stored in an H5 file."""

    # Class property for opening files using cached descriptors. This is to avoid opening the same
    # file each time and instead reusing the descriptor for it. The cache helps with memory usages.
    _FILE_OPENER = _CachingFileOpener()

    def __new__(cls, filepath: str, key: str, fp: Optional[TextIO] = None):
        if fp is not None:
            ctx = h5.File(fp, "r", locking=hdf5_locking.value)
        elif h5.is_hdf5(filepath):
            ctx = h5.File(filepath, "r", locking=hdf5_locking.value)
        else:
            raise ValueError(f"{filepath} is not a valid HDF5 file.")

        with ctx as f:
            size = f[key].shape
            np_dtype = f[key].dtype
            if f[key].attrs.get("is_bfloat16"):
                np_dtype = bf16
            dtype = _np_to_torch_dtype(np_dtype)
            data = torch.empty(size, dtype=dtype, device="cpu")

        return cls._make_subclass(cls, data, require_grad=False)

    def __init__(self, filepath: str, key: str, fp: Optional[TextIO] = None):
        """Constructs a `DeferredH5Tensor` instance.

        Args:
            filepath: The path to the H5 file that holds the tensor data.
            key: The dataset name with which to retrieve the tensor value from the H5 file.
            fp: An optional file pointer to the opened filepath. If provided, `filepath` is not
                opened and the file pointer is used instead.
        """
        super().__init__()

        self._filepath = os.path.abspath(filepath)
        self._key = key

        # We keep the file open to avoid it being deleted from under us. DeferredH5Tensor's are
        # generally used for lazily loading values from checkpoints. By keeping this reference,
        # we're avoiding the case where during training someone deletes the original checkpoint
        # then we go and save this tensor to a new checkpoint but fail because the original was
        # deleted.
        # Note: H5 doesn't allow keeping a file open in different modes. So to keep the file open,
        # we use a regular `open()` instead of `h5.File()`.
        self._fp = self._FILE_OPENER(filepath, "rb") if fp is None else fp

        # Store the last modified time of the file so we can check if the file
        # has been modified since the tensor was created before materializing it
        self._stat = os.fstat(self._fileno)

    @property
    def _fileno(self) -> int:
        """Returns the file number of the unerlyding file descriptor."""
        return self._fp.fileno()

    def __getstate__(self):
        clear_cache = False
        if self._tensor is None and (
            not os.path.exists(self._filepath)
            or _timestamp_changed(os.stat(self._filepath), self._stat)
        ):
            self._materialize(cache_override=True)
            clear_cache = True

        state = self.__dict__.copy()
        # Delete file descriptors since we need to reopen when unpickling. The attribute might not
        # exist if this is an unpickled instance whose `self._tensor` existed. See `__setstate__`.
        state.pop("_fp", None)

        if clear_cache:
            self._tensor = None

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # If tensor is non-materialized, we need to ensure the backing file still exists and its
        # timestamp hasn't changed from before.
        if self._tensor is None:
            if not os.path.exists(self._filepath):
                raise RuntimeError(
                    f"Attempting to unpickle a deferred tensor whose backing file {self._filepath} "
                    f"no longer exists."
                )
            self._fp = self._FILE_OPENER(self._filepath, "rb")
            # Here we're checking against the file stats now vs when the original tensor that was
            # pickled and we loaded into `self._stat` now.
            self._check_file_modification("unpickle")

    def _save(
        self, f: h5.File, name: str, **kwargs
    ) -> Union[None, Type[TorchTensorH5Type]]:
        same_file = os.path.realpath(f.filename) == os.path.realpath(
            self._filepath
        )

        if (not same_file and not use_external_link) or (
            same_file and name == self._key
        ):
            # When external links are disabled, we need to materialize the
            # tensor and save it to file. But note that we don't cache the
            # materialized tensor to avoid OOM. This mainly happens when we load
            # from an initial H5 checkpoint and save the initial weights to
            # another H5 checkpoint.
            return TorchTensorH5Type.save(
                self._materialize(cache_override=False), f, name, **kwargs
            )

        dset = f.create_dataset(name, data=h5.Empty("f"))
        dset.attrs["key"] = self._key
        if same_file:
            dset.attrs["link_to_self"] = same_file
        else:
            dset.attrs["filepath"] = self._filepath
        # TODO: Need to save fstat and compare upon load to avoid loading tampered file

        return None

    @staticmethod
    def load(f: h5.File, key: str) -> "DeferredH5Tensor":
        dset = f[key]
        link_to_self = dset.attrs.get("link_to_self", False)
        filepath = f.filename if link_to_self else dset.attrs["filepath"]
        return DeferredH5Tensor(filepath, dset.attrs["key"])

    def _to_cpu(self) -> torch.Tensor:
        self._check_file_modification("materialize")
        with h5.File(self._fp, "r", locking=hdf5_locking.value) as f:
            return cstorch.from_numpy(NumpyArrayH5Type.load(f, self._key))

    def _check_file_modification(self, action: str):
        """Check whether the backing file has been modified since the tensors was created."""
        _check_file_modification(
            self._fileno,
            self._stat,
            f"{action} deferred tensor with key \"{self._key}\" from file {self._filepath}",
        )

    def _clone(self) -> "DeferredH5Tensor":
        # Use the file descriptor, since the filepath may have been unlinked. But since we hold the
        # descriptor open, the file itself hasn't been deleted.
        self._check_file_modification("clone")
        cloned = DeferredH5Tensor(self._filepath, self._key, fp=self._fp)
        cloned.requires_grad = self.requires_grad
        return cloned


def torch_to_np_dtype(dtype: torch.dtype) -> numpy.dtype:
    """Converts a torch dtype to a numpy dtype."""
    return cstorch.to_numpy(torch.empty(0, dtype=dtype)).dtype


def _np_to_torch_dtype(dtype: numpy.dtype) -> torch.dtype:
    """Converts a numpy dtype to a torch dtype."""
    return cstorch.from_numpy(numpy.empty(0).astype(dtype)).dtype


def _check_file_modification(
    fd: Union[str, int], expected_stats: os.stat_result, msg: str
):
    """Check whether the backing file has been modified since the tensors was created."""
    if check_deferred_backing_storage and _timestamp_changed(
        os.stat(fd), expected_stats
    ):
        raise RuntimeError(
            f"Attempting to {msg}, but the file has "
            f"since been modified. The loaded tensor value may be "
            f"different from originally loaded tensor. Please refrain "
            f"from modifying the file while the run is in progress. "
            f"If this is a false positive, you can disable this check "
            f"by using the following context:"
            f"\n\tfrom cerebras.pytorch.saver.storage import check_deferred_backing_storage"
            f"\n\twith check_deferred_backing_storage(False):"
            f"\n\t\t... # Code that materializes the tensor\n"
        )


def _timestamp_changed(lhs: os.stat_result, rhs: os.stat_result) -> bool:
    """Returns whether timestamp difference is above a certain threshold."""
    return abs(rhs.st_mtime - lhs.st_mtime) > 1
