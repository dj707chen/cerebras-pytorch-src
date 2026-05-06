# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Contains the GPU backend subclass. """
import os

import torch
import torch.distributed as dist

from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.pytorch.amp.autocast import autocast
from cerebras.pytorch.backend.base_backend import BaseBackend
from cerebras.pytorch.core.device import GPUDevice


@named_class_logger("GpuBackend")
class GpuBackendImpl(BaseBackend, ClassLogger):
    """The GPU backend subclass."""

    def __init__(
        self,
        backend_type: str,
        artifact_dir: str = None,
        enable_distributed: bool = False,
        main_process_id: int = 0,
        dist_backend: str = "nccl",
        init_method: str = None,
        sync_batchnorm: bool = False,
    ):
        super().__init__(
            backend_type,
            GPUDevice(enable_distributed, dist_backend, init_method),
            artifact_dir,
        )

        self.enable_distributed = enable_distributed
        self.main_process_id = main_process_id
        self.sync_batchnorm = sync_batchnorm

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if enable_distributed else 1

    def setup_model(self, model):
        model.to(self.torch_device)

        if self.enable_distributed:
            if self.sync_batchnorm:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        return super().setup_model(model)

    def setup_optimizer(self, optimizer):
        super().setup_optimizer(optimizer)

        # Move the optimizer state to the current GPU
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                for key in state:
                    state[key] = state[key].to(self.torch_device)

    @property
    def is_main_process(self) -> bool:
        return not self.enable_distributed or (
            torch.distributed.is_initialized()
            and torch.distributed.get_rank() == self.main_process_id
        )

    def reduce(self, tensor):
        if self.enable_distributed:
            # not using AVG since it's only available with NCCL
            dist.reduce(tensor, self.main_process_id, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
            dist.barrier()
        return tensor

    def on_run_start(self):
        """Runs once at the beginning of the run.

        Used by cstorch.utils.data.DataLoader
        """
        super().on_run_start()
        self.run_step_closures()

    def forward(self, *args, **kwargs):
        with autocast():  # only required for GPU runs
            return super().forward(*args, **kwargs)

    def run_step_closures(self):  # pylint: disable=no-self-use
        """Run all the queued closures."""
        step_closures = self.step_closures
        self.step_closures = []

        for closure, args, kwargs, repeat in step_closures:
            reduced_args, reduced_kwargs = torch.utils._pytree.tree_map(
                lambda arg: (
                    self.reduce(arg).detach().to("cpu")
                    if isinstance(arg, torch.Tensor)
                    and arg.device.type == self.torch_device.type
                    else arg
                ),
                (args, kwargs),
            )

            if self.is_main_process:
                closure(*reduced_args, **reduced_kwargs)

            if repeat:
                self.step_closures.append((closure, args, kwargs, repeat))

    def to_cpu(self, tensor, *args, **kwargs):
        return torch.utils._pytree.tree_map(lambda t: t.to("cpu"), tensor)
