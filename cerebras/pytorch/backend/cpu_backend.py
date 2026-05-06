# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Contains the CPU backend subclass. """
import torch

from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.pytorch.backend.base_backend import BaseBackend
from cerebras.pytorch.core.device import CPUDevice

UNSPECIFIED = object


@named_class_logger("CpuBackend")
class CpuBackendImpl(BaseBackend, ClassLogger):
    """The CPU backend subclass."""

    def __init__(
        self,
        backend_type,
        artifact_dir: str = None,
        mixed_precision: bool = UNSPECIFIED,
    ):
        super().__init__(backend_type, CPUDevice(), artifact_dir)

        if mixed_precision is not UNSPECIFIED:
            raise ValueError(
                f"Passing mixed_precision to the CPU backend is no longer allowed. "
                f"Please use the cstorch.amp.autocast() context manager instead."
            )

    def on_run_start(self):
        """Runs once at the beginning of the run.

        Used by cstorch.utils.data.DataLoader
        """
        super().on_run_start()
        self.run_step_closures()

    def to_cpu(self, tensor, *args, **kwargs):
        return torch.utils._pytree.tree_map(lambda t: t.to("cpu"), tensor)
