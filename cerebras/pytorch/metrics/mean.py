# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Mean Metric for PyTorch """

import torch
from torch import Tensor

from cerebras.pytorch.metrics.metric import Metric


class MeanMetric(Metric):
    """Aggregate a stream of value into their mean value

    Args:
        name: Name of the metric
    """

    def reset(self):
        self.register_state("mean_value", torch.tensor(0, dtype=torch.float32))
        self.register_state("weight", torch.tensor(0, dtype=torch.float32))
        self._dtype = None

    def update(self, value, weight=1.0, dtype=None):
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = torch.as_tensor(
                value, device=self.mean_value.device, dtype=torch.float32
            )
        if not isinstance(weight, Tensor):
            weight = torch.as_tensor(
                weight, device=self.weight.device, dtype=torch.float32
            )
        weight = torch.broadcast_to(weight, value.shape)

        self.mean_value.add_((value * weight).sum())
        self.weight.add_(weight.sum())

        self._dtype = dtype

    def compute(self) -> torch.Tensor:
        result = self.mean_value / self.weight
        if self._dtype is not None:
            result = result.to(self._dtype)
        return result
