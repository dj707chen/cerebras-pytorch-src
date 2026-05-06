# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Class for easier reading of different Pytorch Checkpoint Formats"""
import sys
from contextlib import contextmanager
from types import ModuleType

import dill
import h5py as h5
import torch

import cerebras.pytorch as cstorch
from cerebras.appliance.saver.h5_saver import H5Saver, hdf5_locking
from cerebras.pytorch.storage._legacy.pt_h5_saver import PyTorchH5Saver


class CheckpointReader:
    """Reader class to parse different checkpoint formats"""

    def __init__(
        self, checkpoint_file: str, saver_cls: H5Saver = PyTorchH5Saver
    ):
        self._checkpoint_file = checkpoint_file
        assert H5Saver.is_valid_checkpoint(
            checkpoint_file
        ), f"Must provide a valid checkpoint to read. Received {checkpoint_file}"

        self._reader = H5Saver()
        self._spec = None
        self._version = None
        self._tensor_names = None

        self.saver_cls = saver_cls
        self.saver_cls.check_version(checkpoint_file)

    @property
    def tensor_names(self):
        """Tensor names in checkpoint"""
        if self._tensor_names is None:
            self._tensor_names = self._reader.tensor_names(
                self._checkpoint_file
            )
        return self._tensor_names

    def _fix_spec(self, spec):
        # Checkpoints saved with older versions of PyTorch don't have this
        # attribute and thus need to manually calculate

        if not hasattr(spec, "num_nodes"):
            for child in spec.children_specs:
                self._fix_spec(child)

            spec.__post_init__()

    @property
    def spec(self):
        """Specification for checkpoint"""
        if self._spec is None:
            with h5.File(
                self._checkpoint_file, "r", locking=hdf5_locking.value
            ) as f:
                if self.saver_cls.__spec__ in f.attrs.keys():
                    with _patch_missing_imports():
                        self._spec = dill.loads(
                            bytes.fromhex(f.attrs[self.saver_cls.__spec__])
                        )
                        self._fix_spec(self._spec)
                else:
                    self._spec = []

        return self._spec

    @property
    def state_dict(self):
        """State dict defined in checkpoint"""
        fill_value = None
        spec = self.spec
        # pylint: disable=protected-access
        return torch.utils._pytree.tree_unflatten(
            [fill_value for i in range(spec.num_leaves)], spec
        )

    def __getitem__(self, tensor_name: str):
        return self._reader.load_tensor(self._checkpoint_file, tensor_name)

    def __iter__(self):
        for var_name in self.tensor_names:
            yield var_name, self[var_name]

    def __contains__(self, tensor_name: str):
        return tensor_name in self.tensor_names


@contextmanager
def _patch_missing_imports():
    # Dynamically patch this import so that dill can load the spec
    cerebras_pytorch = ModuleType("cerebras.pytorch")  # noqa
    old_cerebras_pytorch = ModuleType("cerebras_pytorch")  # noqa
    tensor_scalar_dict = ModuleType("tensor_scalar_dict")
    tensor_scalar_dict.TensorScalarDict = dict
    try:
        # Keep for backward compatibility with the old checkpoints.
        sys.modules["cerebras_pytorch.utils.tensor_scalar_dict"] = (
            tensor_scalar_dict
        )

        sys.modules["cerebras_pytorch"] = cstorch

        yield
    finally:
        sys.modules.pop("cerebras_pytorch.utils.tensor_scalar_dict")
        sys.modules.pop("cerebras_pytorch")
