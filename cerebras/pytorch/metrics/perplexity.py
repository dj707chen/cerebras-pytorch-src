# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Perplexity metric for PyTorch """

import torch

from cerebras.pytorch.metrics.metric import Metric


class PerplexityMetric(Metric):
    """Computes the perplexity of the model's predictions

    Args:
        name: Name of the metric
    """

    def reset(self):
        self.register_state("total_loss", torch.tensor(0, dtype=torch.float32))
        self.register_state(
            "total_num_tokens", torch.tensor(0, dtype=torch.float32)
        )
        self._dtype = None

    def update(self, labels, loss, weights=None, dtype=None):
        if weights is None:
            num_tokens = torch.tensor(
                labels.numel(), dtype=torch.float32, device=labels.device
            )
        else:
            num_tokens = (weights > 0).float().sum()

        self.total_loss.add_(loss)
        self.total_num_tokens.add_(num_tokens)

        self._dtype = dtype

    def compute(self) -> torch.Tensor:
        result = torch.exp(self.total_loss / self.total_num_tokens)
        if self._dtype is not None:
            result = result.to(self._dtype)
        return result
