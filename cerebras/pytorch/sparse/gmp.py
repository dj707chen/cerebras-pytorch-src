# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Provide an optimizer implementing GMP for use with the WSE.
"""
import torch

from .dynamic import DynamicSparsityAlgorithm
from .utils import Constant, make_mask_topk_sparsity


class GMP(DynamicSparsityAlgorithm):
    r"""Implements Gradual Magnitude Pruning

    Sparsity increases monotonically based on weight magnitude.

    See: https://arxiv.org/abs/1710.01878
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: All arguments are passed to the
                :py:func:`~cerebras.pytorch.sparse.DynamicSparsityAlgorithm`'s constructor.

        Example:
            sparsity_opt = cstorch.sparse.GMP(
                schedule={"type": "exp", "init": 0, "gamma": 1000*math.log(0.3)
                update={"freq": 1000},
            )
        """

        super().__init__(**kwargs)

        if isinstance(self.sparsity.default, Constant):
            raise ValueError(
                f"Configured with constant sparsity {self.sparsity.default.value}. "
                f"This is not valid, because the sparsity pattern would not change "
                f"during training. For a static sparsity pattern, use `algorithm=\"static\".`"
            )

    @torch.no_grad()
    def update_mask(self, p, mask, sparsity):
        score = p.abs()
        return make_mask_topk_sparsity(score, sparsity)
