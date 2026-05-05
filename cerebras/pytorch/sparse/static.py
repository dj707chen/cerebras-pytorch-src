# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Provide an "optimizer" implementing static sparsity.
"""

from typing import Optional

import torch

from .base import SparseParameter, SparsityAlgorithm


class Static(SparsityAlgorithm):
    def __init__(self, sparsity: Optional[float] = None, **kwargs):
        """Constructs a `Static` sparsity instance.

        Args:
            sparsity: A float specifying the level of sparsity to apply to each parameter
        """
        if sparsity is not None:
            if not isinstance(sparsity, float):
                raise ValueError(
                    "Static sparsity algorithm only supports constant sparsity"
                )
            if not (0.0 <= sparsity < 1.0):
                raise ValueError(
                    f"Invalid sparsity level {sparsity}. Must be 0.0 <= s < 1.0"
                )

        self.sparsity_level = sparsity

        super().__init__(sparsity, **kwargs)

    def csx_annotate_sparsity(self, param: SparseParameter):
        # This simple scalar computation does not need to be traced
        with torch.device("cpu"):
            # We can just take the sparsity value at step 0
            # as the sparsity value is constant
            sparsity = self.sparsity[param.data](step=0).item()
            self.sparsity_level = sparsity

        param.annotate("min_sparsity", sparsity)
        param.annotate("max_sparsity", sparsity)
        param.annotate("sparsity", sparsity)

    def update(self, optimizer):
        # Nothing to update for static sparsity

        for hook in self._target_sparsity_hooks.values():
            hook(self, "*", self.sparsity_level)

        for hook in self._computed_sparsity_hooks.values():
            hook(self, "*", self.sparsity_level)
