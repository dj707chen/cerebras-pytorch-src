# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Mean per class Accuracy metric for PyTorch.
Calculates the accuracy for each class, then takes the mean of that.

"""
from typing import Optional

import torch

from cerebras.pytorch.metrics.metric import Metric
from cerebras.pytorch.metrics.utils import divide_no_nan


def compute_helper(total_per_class_correct_predictions, total_per_class_tokens):
    """Computes the per class accuracy"""
    per_class_accuracy = divide_no_nan(
        total_per_class_correct_predictions,
        total_per_class_tokens,
    )
    return torch.mean(per_class_accuracy)


class MeanPerClassAccuracyMetric(Metric):
    """
    Calculates the accuracy for each class, then takes the mean of that.

    Args:
        num_classes: The possible number of labels the prediction task can
            have. This value must be provided, since two variables with
            shape=[num_classes] will be allocated.
        name: Optional `string` which indicates name of the metric.
            If None or empty string, it defaults to the name of the class.
    """

    def __init__(self, num_classes, name: Optional[str] = None):
        self.num_classes = num_classes
        super().__init__(name=name)

    def reset(self):
        self.register_state(
            "total_per_class_correct_predictions",
            torch.zeros(self.num_classes, dtype=torch.float32),
        )
        self.register_state(
            "total_per_class_tokens",
            torch.zeros(self.num_classes, dtype=torch.float32),
        )
        self._dtype = None

    def update(
        self,
        labels,
        predictions,
        weights=None,
        dtype=None,
    ):  # pylint: disable=arguments-differ
        """
        Updates the mean per class accuracy metric.

        Args:
            labels: A `Tensor` of ground truth labels of type `int32` or `int64`.
                The tensor will be flattened if its rank > 1.
            predictions: A `Tensor` of prediction results for semantic labels,
                of type `int32` or `int64`. The tensor will be
                flattened if its rank > 1.
            weights: Optional `Tensor` whose rank is either 0, or the same rank as
                `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
                be either `1`, or the same as the corresponding `labels` dimension).
                If `weights` is `None`, weights default to 1.
                Use weights of 0 to mask values.

        Raises:
            ValueError: If `predictions` and `labels` have mismatched shapes, or if
                `weights` is not `None` and its shape doesn't match `predictions`.
        """
        if labels.shape != predictions.shape:
            raise ValueError(
                f"`labels` and `predictions` have mismatched shapes of "
                f"{labels.shape} and {predictions.shape} respectively."
            )
        if weights is not None:
            if weights.shape != labels.shape:
                raise ValueError(
                    f"`labels`={labels.shape} and ",
                    f"`weights`={weights.shape} have mismatched shapes",
                )
            weights = weights.flatten()

        labels = labels.to(torch.long)

        if len(labels.shape) > 1:
            labels = labels.flatten()
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()

        correct_predictions = labels == predictions
        num_tokens = torch.ones_like(predictions)

        if weights is not None:
            correct_predictions = correct_predictions * weights
            num_tokens = num_tokens * weights

        per_class_correct_predictions = torch.zeros(
            self.num_classes,
            dtype=torch.float32,
            device=predictions.device,
        )
        per_class_tokens = torch.zeros(
            self.num_classes,
            dtype=torch.float32,
            device=predictions.device,
        )

        per_class_correct_predictions.scatter_add_(
            dim=0, index=labels, src=correct_predictions.to(torch.float32)
        )
        per_class_tokens.scatter_add_(
            dim=0, index=labels, src=num_tokens.to(torch.float32)
        )

        self.total_per_class_correct_predictions.add_(
            per_class_correct_predictions
        )
        self.total_per_class_tokens.add_(per_class_tokens)

        self._dtype = dtype

    def compute(self) -> torch.Tensor:
        mean_per_class_accuracy = compute_helper(
            self.total_per_class_correct_predictions,
            self.total_per_class_tokens,
        )
        # WS Stack limitation: Need to cast to fp16 before store output
        if self._dtype is not None:
            mean_per_class_accuracy = mean_per_class_accuracy.to(self._dtype)
        return mean_per_class_accuracy
