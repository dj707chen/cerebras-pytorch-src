# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Base class for all dynamic sparsity optimizer, plus dynamic schedule helpers.
"""
import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional, Union

import torch

import cerebras.pytorch as cstorch
from cerebras.pytorch.utils.weak import DefaultWeakIdKeyDictionary

from .base import SparseParameter, SparsityAlgorithm
from .utils import UpdateScheduleType, make_update_schedule


class DynamicSparsityAlgorithm(SparsityAlgorithm, ABC):
    def __init__(
        self,
        sparsity: Union[float, dict, None] = None,
        update: Optional[UpdateScheduleType] = None,
        **kwargs,
    ):
        """Constructs a `DynamicSparsityAlgorithm` instance.

        Args:
            sparsity: A float specifying the level of sparsity to apply to each
                parameter or a dictionary specifying the schedule to use for
                sparsity. The dictionary must have a "type" key, which specifies
                the type of schedule to use. The remaining keys are
                schedule-specific. The following schedule types
                are supported:
                    - ":py:class:`constant <cerebras.pytorch.sparse.utils.Constant>`"
                    - ":py:class:`linear <cerebras.pytorch.sparse.utils.Linear>`"
                    - ":py:class:`exp <cerebras.pytorch.sparse.utils.Exp>`"
                    - ":py:class:`power <cerebras.pytorch.sparse.utils.Power>`"
                    - ":py:class:`cosine <cerebras.pytorch.sparse.utils.Cosine>`"
                    - ":py:class:`cycling <cerebras.pytorch.sparse.utils.Cycling>`"

            update: A dictionary specifying the schedule to use for updating the sparsity pattern.
                The dictionary must contain keys that can be used to construct either a
                :py:class:`~cerebras.pytorch.sparse.utils.FreqSchedule` or a
                :py:class:`~cerebras.pytorch.sparse.utils.ListSchedule`.
                If not provided, the sparsity pattern will be updated every step.

        """
        super().__init__(sparsity=sparsity, **kwargs)

        self.update_schedule = make_update_schedule(update)
        self.starts_sparse = self.update_schedule(
            torch.tensor(0, dtype=torch.int64)
        )
        if not self.starts_sparse:
            self.init_method = lambda p, sparsity, **kwargs: cstorch.ones_like(
                p, dtype=torch.bool
            )

        self.step = torch.tensor(0, dtype=torch.int64)

    def csx_annotate_sparsity(self, param: SparseParameter) -> None:
        begin_step = getattr(self.update_schedule, "start", None) or 0
        # If the schedule has a `stop` step use that, otherwise pick
        # 100,000 arbitrarily.
        end_step = getattr(self.update_schedule, "stop", None) or 100000

        # This simple scalar computation does not need to be traced
        with torch.device("cpu"):
            min_max_end = self.sparsity[param.data].get_min_max_end(
                begin_step, end_step
            )
            if min_max_end and not self.starts_sparse:
                # If we we don't start sparse, there is a period of dense
                # training, or 0% sparsity.
                _, max_v, end_v = min_max_end
                min_max_end = (0.0, max_v, end_v)

        min_v, max_v, ending_v = min_max_end
        param.annotate("min_sparsity", min_v)
        param.annotate("max_sparsity", max_v)
        param.annotate("sparsity", ending_v)

    def sparsify_parameter(
        self, module: torch.nn.Module, name: str, param: torch.Tensor
    ) -> None:
        super().sparsify_parameter(module, name, param)
        self.sparsity[param].update(self.starts_sparse)

    @cached_property
    def is_update_step(self) -> torch.BoolTensor:
        """
        Returns a boolean tensor indificating whether the current step is an
        update step according to the update schedule.
        """
        return self.update_schedule(self.step)

    @torch.no_grad()
    def update(self, optimizer: Optional[torch.optim.Optimizer] = None):
        cstorch.amp.update_if_finite(optimizer, self.step)
        self.step += 1

        # Collect all sparse params that a given sparsity schedule is assigned to
        unique_schedules = DefaultWeakIdKeyDictionary(list)
        for sparse_param in self.get_sparse_params(optimizer):
            unique_schedules[self.sparsity[sparse_param.param]].append(
                sparse_param.name
            )

        if optimizer:
            if not isinstance(optimizer, cstorch.optim.Optimizer):
                raise TypeError(
                    f"Expected a Cerebras Optimizer. Got: {type(optimizer)}"
                )
            # Only should update if no optimizer gradients are NaN/inf
            isfinite = cstorch.amp.isfinite(optimizer)
            if isinstance(isfinite, torch.Tensor):
                self.is_update_step &= isfinite

        for sparse_param in self.get_sparse_params(optimizer):
            p = sparse_param.param
            mask = sparse_param.mask

            if p.grad is None:
                # If the gradient is None, then the parameter was not updated
                # so there is no need to update the mask
                continue

            schedule = self.sparsity[p]
            # Compute sparsity level for the parameter at the current step
            sparsity = schedule(self.step).to(p.device)
            # Ensure dynamic sparsity stays between [0, 1)
            sparsity = torch.clamp(sparsity, min=0.0, max=1.0)
            # update the sparsity schedule if it is an update step
            # so that we get the latest sparsity value
            # This can technically cause issues if there are multiple optimizers
            # sparsified by the same sparsity algorithm
            schedule.update(self.is_update_step)

            if schedule in unique_schedules:
                # We only want to call the hooks once per unique schedule
                names = unique_schedules.pop(schedule)
                # Create a "glob" using the common prefix, e.g.
                # [fc1.weight, fc2.weight] would yield "fc*"
                name_glob = os.path.commonprefix(names) + "*"
                for hook in self._target_sparsity_hooks.values():
                    hook(self, name_glob, sparsity)

            new_mask = self.update_mask(p, mask, sparsity)
            # Rewrite into the existing mask tensor for state tracking
            new_mask = torch.where(self.is_update_step, new_mask, mask)
            sparse_param.mask = new_mask

            if self._computed_sparsity_hooks:
                actual_sparsity_level = 1 - new_mask.sum() / new_mask.numel()
                for hook in self._computed_sparsity_hooks.values():
                    hook(
                        self,
                        sparse_param.name,
                        actual_sparsity_level,
                    )

        # Clear the scheduler's cache. We don't want to carry this over
        # to the next iteration.
        # Technically this can be handled by not caching the values
        # and relying on common subexpression elimination, but for now
        # keep as is
        for sparsity in self.sparsity.values():
            sparsity.cache_clear()

        # Clear update step cache. We don't want to carry this over
        # to the next iteration.
        self.__dict__.pop("is_update_step", None)

    @abstractmethod
    @torch.no_grad()
    def update_mask(self, p, mask, sparsity) -> torch.Tensor:
        """
        Compute an updated sparsity pattern.

        Args:
            p (torch.Tensor): the parameter to sparsify
            mask (torch.tensor(dtype=torch.bool)): the current mask
                of param p
            sparsity (torch.tensor(dtype=torch.float32)): the desired
                sparsity level
        Returns:
            The updated sparsity pattern on parameter p
        """

    def visit_state(self, f):
        super().visit_state(f)

        out = f(self.step)
        if out is not None:
            self.step = out

        # Iterate a unique list of sparsity hyperparam objects
        for sparsity in torch.utils.weak.WeakIdKeyDictionary(
            {
                self.sparsity[sparse_param.param]: None
                for sparse_param in self.sparse_params.values()
            }
        ):
            sparsity.visit_state(f)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["step"] = self.step

        state_dict["sparsity"] = {
            name: s
            # Only need to save unique sparsity schedules
            for sparsity, name in torch.utils.weak.WeakIdKeyDictionary(
                {
                    self.sparsity[sparse_param.param]: sparse_param.name
                    for sparse_param in self.sparse_params.values()
                }
            ).items()
            if (s := sparsity.state_dict())
        }

        return state_dict

    def load_state_dict(self, state_dict):
        self.step = state_dict.pop("step")

        super().load_state_dict(state_dict)

        state_dict["sparsity"] = {}
        for sparse_param in self.sparse_params.values():
            sparsity = self.sparsity[sparse_param.param]
            if s := state_dict["sparsity"].get(sparse_param.name):
                sparsity.load_state_dict(s)

        with self._backend.device:
            self.visit_state(lambda x: x.to(self._backend.torch_device))
