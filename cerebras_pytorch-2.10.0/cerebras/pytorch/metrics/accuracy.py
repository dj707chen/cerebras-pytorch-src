# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Accuracy metric for PyTorch """
import warnings

import torch

from cerebras.pytorch.metrics.metric import Metric


class AccuracyMetric(Metric):
    """Computes the accuracy of the model's predictions

    Args:
        name: Name of the metric
    """

    def reset(self):
        self.register_state(
            "total_correct_predictions", torch.tensor(0, dtype=torch.float32)
        )
        self.register_state(
            "total_num_tokens", torch.tensor(0, dtype=torch.float32)
        )
        self._dtype = None

    def update(
        self, labels, predictions, weights=None, dtype=None
    ):  # pylint: disable=arguments-differ
        if labels.shape != predictions.shape:
            warnings.warn(
                "Shapes mismatch in accuracy metric"
                f"\n    labels: {labels.shape}"
                f"\n    predictions {predictions.shape}"
            )
            predictions = predictions.reshape(labels.shape)
        correct_predictions = (labels == predictions).float()
        if weights is None:
            num_correct_predictions = correct_predictions.sum()
            num_tokens = torch.tensor(
                correct_predictions.numel(),
                dtype=torch.float32,
                device=predictions.device,
            )
        else:
            correct_predictions = correct_predictions * weights
            num_correct_predictions = correct_predictions.sum()
            num_tokens = (weights > 0).float().sum()
        self.total_correct_predictions.add_(num_correct_predictions)
        self.total_num_tokens.add_(num_tokens)

        self._dtype = dtype

    def compute(self) -> torch.Tensor:
        result = self.total_correct_predictions / self.total_num_tokens
        if self._dtype is not None:
            result = result.to(self._dtype)
        return result
