# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Provide an optimizer implementing RigL for use with the WSE.
"""
from functools import partial

import torch

from .dynamic import DynamicSparsityAlgorithm
from .utils import (
    HyperParameterScheduleType,
    InputGroupScoreShaper,
    OutputGroupScoreShaper,
    make_hyperparam_schedule,
    make_mask_drop_minimum,
    make_mask_grow_maximum,
)


class RigL(DynamicSparsityAlgorithm):
    r"""Implements Rigging the Lottery (RigL)

    Sparsity levels stay constant throughout training, but the lowest magnitude
    weights are pruned and then regrown using a proxy measure of where a pruned
    connection  would have had the most impact by finding the highest magnitude
    (dense) gradients of pruned weights.

    See: https://arxiv.org/abs/1911.11134
    """

    def __init__(
        self,
        drop_fraction: HyperParameterScheduleType = 0.3,
        balance_in_groups: int = None,
        balance_out_groups: int = None,
        **kwargs,
    ):
        """
        Args:
            drop_fraction: Fraction of non-pruned weights to drop each update step.
                Either a constant or a step-aware hyperparamter.
            balance_in_groups: The number of groups used by
                :py:func:`~cerebras.pytorch.sparse.utils.InputGroupScoreShaper`
            balance_out_groups: The number of groups used by
                :py:func:`~cerebras.pytorch.sparse.utils.OutputGroupScoreShaper`
            **kwargs: Any additional arguments are passed to the
                :py:func:`~cerebras.pytorch.sparse.DynamicSparsityAlgorithm`'s constructor.
            super().__init__(**kwargs)

        Example:

        .. code-block:: python

            sparsity = cstorch.sparse.RiGL(
                sparsity=0.9,
                update={"freq": 100, "stop": 1000},
                drop_fraction={"type": "cosine", "init": 0.3, "half_period": 1000},
            )
        """

        super().__init__(**kwargs)

        # drop_fraction is a required value for RigL though it has a default
        # value. Pass it as dynamic optimizer kwarg. It will be configured
        # on each param_group.
        self.drop_fraction = make_hyperparam_schedule(drop_fraction)

        self.balance_in_groups = balance_in_groups
        self.balance_out_groups = balance_out_groups
        self.score_shaper = None

        self._dense_grads = {}

    def sparsify_parameter(
        self, module: torch.nn.Module, name: str, param: torch.Tensor
    ) -> None:
        if self.score_shaper is None:

            def validate_balance(groups, err_key):
                for dim in param.shape:
                    if dim % groups == 0:
                        break
                else:
                    raise ValueError(
                        f"Sparsity group configured with `{err_key}`={groups} "
                        f"but parameter {name} does not have a dimension with "
                        f"a multiple of {groups}: {param.shape}"
                    )

            if self.balance_out_groups:
                if self.balance_in_groups:
                    raise ValueError(
                        "Only one of `balance_in_groups` and `balance_out_groups` "
                        "can be specified at a time."
                    )
                validate_balance(self.balance_out_groups, "balance_out_groups")
                self.score_shaper = OutputGroupScoreShaper(
                    self.balance_out_groups
                )
            elif self.balance_in_groups:
                validate_balance(self.balance_in_groups, "balance_in_groups")
                self.score_shaper = InputGroupScoreShaper(
                    self.balance_in_groups
                )
            else:
                self.score_shaper = None

            self.init_method = partial(
                self.init_method, score_shaper=self.score_shaper
            )

        return super().sparsify_parameter(module, name, param)

    def _grad_hook(self, p, grad):
        # Save a copy of the dense gradients before masking.
        if p in self._dense_grads:
            # GPU gradient accumulation mode.
            self._dense_grads[p] += grad
        else:
            self._dense_grads[p] = grad.clone()

        return super()._grad_hook(p, grad)

    def sparsify_optimizer(self, optimizer):
        super().sparsify_optimizer(optimizer)

        def clear_accumulated_dense_grads(set_to_none: bool = True):
            if set_to_none:
                self._dense_grads = {}
            else:
                for g in self._dense_grads.values():
                    g.zero_()

        self.zero_grad_post_hook = optimizer.register_zero_grad_post_hook(
            lambda optimizer, args, kwargs: clear_accumulated_dense_grads(
                *args, **kwargs
            )
        )

    @torch.no_grad()
    def update_mask(self, p, mask, sparsity):
        if p not in self._dense_grads:
            raise RuntimeError(
                "RigL requires dense gradients, ensure you have called "
                f"sparsity.prune_weights(), {len(self._dense_grads)}"
            )

        # RigL may need per-head balancing of attention projection weights
        drop_fraction = self.drop_fraction(self.step)

        # update the drop fraction schedule if it is an update step
        self.drop_fraction.update(self.is_update_step)

        # Keep the connections of highest magnitude weights but drop some.
        p_score = p.abs()
        mask, k = make_mask_drop_minimum(
            p_score, mask, drop_fraction, score_shaper=self.score_shaper
        )

        # Regrow where the gradient magnitude is the largest.
        regrow_score = self._dense_grads[p].abs()
        return make_mask_grow_maximum(
            regrow_score,
            mask,
            sparsity,
            k,
            score_shaper=self.score_shaper,
        )
