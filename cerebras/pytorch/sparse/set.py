# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Provide an optimizer implementing SET for use with the WSE.
"""

import torch

from .dynamic import DynamicSparsityAlgorithm
from .utils import (
    HyperParameterScheduleType,
    make_hyperparam_schedule,
    make_mask_drop_minimum,
    make_mask_grow_maximum,
)


class SET(DynamicSparsityAlgorithm):
    r"""Implements Sparse Evolutionary Training (SET)

    Sparsity levels stay constant throughout training, but the lowest
    magnitude weights are pruned and then regrown randomly.

    See: https://arxiv.org/abs/1707.04780
    """

    def __init__(
        self, drop_fraction: HyperParameterScheduleType = 0.3, **kwargs
    ):
        """
        Args:
            drop_fraction: Fraction of non-pruned weights to drop each update step.
                Either a constant or a step-aware hyperparamter.
            **kwargs: Any additional arguments are passed to the
                :py:class:`~cerebras.pytorch.sparse.DynamicSparsityAlgorithm`'s constructor.

        Example:

        .. code-block:: python

            sparsity_opt = cstorch.sparse.SET(
                sparsity=0.9,
                update={"freq": 100, "stop": 1000},
                drop_fraction={"type": "cosine", "init": 0.3, "half_period": 1000},
            )
        """
        super().__init__(**kwargs)

        # drop_fraction is a required value for SET though it has a default
        # value. Pass it as dynamic optimizer kwarg. It will be configured
        # on each param_group.
        self.drop_fraction = make_hyperparam_schedule(drop_fraction)

    @torch.no_grad()
    def update_mask(self, p, mask, sparsity):
        drop_fraction = self.drop_fraction(self.step)

        # Update the drop fraction schedule if it is an update step
        self.drop_fraction.update(self.is_update_step)

        # Keep the connections of highest magnitude weights but drop some.
        p_score = p.abs()
        mask, k = make_mask_drop_minimum(p_score, mask, drop_fraction)

        # Regrow randomly.
        regrow_score = torch.rand_like(p)
        return make_mask_grow_maximum(regrow_score, mask, sparsity, k)
