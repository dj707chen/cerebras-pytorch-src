# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
PyTorch specific overrides for saving a state dict to a
Cerebras H5 based checkpoint.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Union

import dill
import h5py as h5
import torch
from typing_extensions import Self

from cerebras.appliance._version import __version__
from cerebras.appliance.saver.h5_saver import (
    H5Saver,
    NumpyArrayH5Type,
    UnknownH5TypeError,
    hdf5_locking,
    register_h5_type,
)
from cerebras.pytorch.utils.nest import recurse_spec

from cerebras.pytorch.storage._legacy.storage import *  # noqa


@dataclass
class CkptVersion:
    """Dataclass to hold the checkpoint version information.

    Args:
        ckpt_version: The checkpoint format version.
        cstorch_version: cstorch release version.
    """

    __ckpt_version_key__ = "__version__"
    __cstorch_version_key___ = "__cstorch_version__"

    ckpt_version: float
    cstorch_version: str

    def __post_init__(self):
        if not isinstance(self.ckpt_version, float):
            raise TypeError(
                f"ckpt_version must be a float, not {type(self.ckpt_version).__name__}"
            )
        if not isinstance(self.cstorch_version, (str, type(__version__))):
            raise TypeError(
                f"cstorch_version must be a str, not {type(self.cstorch_version).__name__}"
            )

    @classmethod
    def from_file(cls, fp: Union[str, h5.File]) -> Self:
        """Load checkpoint version from file."""
        ctx = (
            nullcontext(fp)
            if isinstance(fp, h5.File)
            else h5.File(fp, "r", locking=hdf5_locking.value)
        )
        with ctx as f:
            ckpt_version = f.attrs.get(cls.__ckpt_version_key__, 0.1)
            cstorch_version = f.attrs.get(cls.__cstorch_version_key___, None)

        if cstorch_version is None:
            # Checkpoints from versions 2.0 and below didn't have a cstorch version embedded.
            # For those, we can infer the cstorch version from the ckpt_version.
            _version_map = {
                0.1: "1.7",
                0.2: "1.8",
                0.3: "1.9",
                0.4: "2.0",
            }

            if ckpt_version in _version_map:
                cstorch_version = _version_map[ckpt_version]
            else:
                raise RuntimeError(
                    "Cannot determine the release version of the checkpoint. "
                    "Is this a malformed checkpoint?"
                )

        return cls(ckpt_version, cstorch_version)

    def to_file(self, fp: Union[str, h5.File]) -> None:
        """Save checkpoint version to file."""
        ctx = (
            nullcontext(fp)
            if isinstance(fp, h5.File)
            else h5.File(fp, "a", locking=hdf5_locking.value)
        )
        with ctx as f:
            f.attrs[self.__ckpt_version_key__] = self.ckpt_version
            f.attrs[self.__cstorch_version_key___] = self.cstorch_version

    def __lt__(self, other: Self) -> bool:
        return self.ckpt_version < other.ckpt_version

    def __eq__(self, other: Self) -> bool:
        return self.ckpt_version == other.ckpt_version

    def __gt__(self, other: Self) -> bool:
        return self.ckpt_version > other.ckpt_version


# The current checkpoint version. Update when checkpoint format changes.
_CURR_CKPT_VERSION = CkptVersion(0.4, __version__)


class PyTorchH5Saver(H5Saver):
    """
    A PyTorch specific H5Saver subclass.
    """

    # The key at which the state dict's spec is saved
    __spec__ = "__spec__"

    @classmethod
    def extract_version(
        cls, checkpoint_file: Union[str, h5.File]
    ) -> Optional[CkptVersion]:
        """Extract the checkpoint version from the checkpoint file.

        Args:
            checkpoint_file: The path to the checkpoint to extract the version from.
        """
        if isinstance(checkpoint_file, h5.File) or cls.is_valid_checkpoint(
            checkpoint_file
        ):
            return CkptVersion.from_file(checkpoint_file)
        return None

    @classmethod
    def check_version(cls, checkpoint_file: str):
        """Check compatibility of the checkpoint file with the current release.

        Args:
            checkpoint_file: The path to the checkpoint to check
        """
        version = cls.extract_version(checkpoint_file)
        if version is None:
            raise TypeError(f"Invalid checkpoint file: {checkpoint_file}")

        if version < _CURR_CKPT_VERSION:
            cls.logger.debug(
                f"Loading checkpoint from an older release {version.cstorch_version}. "
                f"The current release is {_CURR_CKPT_VERSION.cstorch_version}."
            )
        elif version > _CURR_CKPT_VERSION:
            raise RuntimeError(
                f"Checkpoint was saved using release {version.cstorch_version} which is newer than "
                f"the current release {_CURR_CKPT_VERSION.cstorch_version}. Please update to a "
                f"newer release."
            )

    @staticmethod
    def flatten_state_dict(state_dict: dict):
        """
        Returns a flattened dictionary where the keys are the compressed scope.
        By definition, the dictionary has depth 1.

        Args:
            state_dict: The nested dictionary to flatten
        """
        # pylint: disable=protected-access
        flattened, spec = torch.utils._pytree.tree_flatten(state_dict)
        state_dict = {
            ".".join(scope): v
            for scope, v in zip(recurse_spec(spec), flattened)
        }
        return state_dict, spec

    # pylint: disable=arguments-renamed
    def save(self, ckpt_file: str, state_dict: dict, save_spec: bool = True):
        flattened, spec = self.flatten_state_dict(state_dict)

        if save_spec:
            # Save the spec before saving the rest in case something goes wrong
            self.save_spec(ckpt_file, spec)

        super().save(ckpt_file, flattened)

    def save_tensor(
        self,
        ckpt_file: str,
        tensor_name: str,
        tensor_value: torch.Tensor,
    ):
        # This override is not useless, it skips calling self.save() which would
        # otherwise call save_spec and destroy it
        super().save(ckpt_file, {tensor_name: tensor_value})

    def save_spec(self, ckpt_file, spec):
        """
        Saves the tensor spec to the checkpoint file.

        Args:
            ckpt_file: The path to the checkpoint to save the spec to
            spec: The spec object produced by the call to flatten_state_dict
        """
        with h5.File(ckpt_file, "a", locking=hdf5_locking.value) as f:
            f.attrs[self.__spec__] = dill.dumps(spec).hex()
            _CURR_CKPT_VERSION.to_file(f)

    def _load_tensor_from_checkpoint(self, f: h5.File, key: str):
        try:
            return super()._load_tensor_from_checkpoint(f, key)
        except UnknownH5TypeError:
            pass

        # If the key does not have an H5Type, then it must be from a
        # checkpoint saved in a release 1.8 or lower.
        # The following loading code is only needed for backwards compatibility

        from .storage import (
            DeferredFileTensor,
            DeferredFullTensor,
            DeferredH5Tensor,
            _np_to_torch_dtype,
        )

        dset = f[key]

        if dset.attrs.get("filepath") is not None:
            filepath = dset.attrs["filepath"]
            shape = torch.Size(dset.attrs["shape"])
            dtype = dill.loads(bytes.fromhex(dset.attrs["dtype"]))

            return DeferredFileTensor(filepath, shape, dtype)

        # Keep the following loading code for backwards compatibility
        elif dset.attrs.get("is_constructible", False):
            shape = torch.Size(dset.attrs.get("shape"))
            dtype = _np_to_torch_dtype(dset.dtype)
            fill_value = dset.attrs.get("fill_value")

            return DeferredFullTensor(shape, dtype, fill_value)

        elif dset.attrs.get("is_none", False):
            return None
        elif dset.attrs.get("is_string", False):
            return dset.attrs.get("string_val")
        else:
            tensor = DeferredH5Tensor(f.filename, key)

            if dset.attrs.get("is_scalar", False):
                return tensor.to("cpu").item()
            return tensor

    def _load_by_typename(self, h5_type_name: str, f: h5.File, key: str):
        """Load the value using the given H5Type class."""
        # If checkpoint version is <= 0.3, then we assume that all numpy arrays
        # are torch tensors. This is because in previous releases, we saved
        # torch tensors as numpy arrays.
        if h5_type_name == NumpyArrayH5Type.__name__ and (
            self.extract_version(f).ckpt_version <= 0.3
        ):
            return TorchTensorH5Type.load(f, key)
        return super()._load_by_typename(h5_type_name, f, key)


@register_h5_type(torch.Tensor)
class TorchTensorH5Type:
    """
    Class for saving and loading PyTorch tensors to and from the H5 checkpoint.
    """

    @staticmethod
    def save(tensor: torch.Tensor, f: h5.File, key: str, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load(f: h5.File, key: str) -> torch.Tensor:
        """Loads the PyTorch tensor from the provided H5 File."""
        from cerebras.appliance.storage.serializers import DeferredObject
        from cerebras.pytorch.storage.serializers import DeferredTorchTensor
        from cerebras.pytorch.storage.utils import np_to_torch_dtype

        shape = f[key].shape
        dtype = (
            torch.bfloat16
            if f[key].attrs.get("is_bfloat16")
            else np_to_torch_dtype(f[key].dtype)
        )

        return DeferredTorchTensor(
            DeferredObject(
                f.filename,
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
