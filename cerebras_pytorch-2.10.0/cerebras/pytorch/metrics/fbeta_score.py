# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
F Beta Score metric for PyTorch.
Confusion matrix calculation in Pytorch referenced from:
https://github.com/pytorch/ignite/blob/master/ignite/metrics/confusion_matrix.py
"""
from typing import List, Optional

import torch

import cerebras.pytorch as cstorch
from cerebras.pytorch.metrics.metric import Metric
from cerebras.pytorch.metrics.utils import (
    compute_confusion_matrix,
    compute_mask,
    divide_no_nan,
)


def compute_helper(
    confusion_matrix,
    mask,
    beta: Optional[float] = 1.0,
    average_type: Optional[str] = "micro",
) -> float:
    """Helper function to compute fbeta score"""

    # true_pos = torch.diagonal(confusion_matrix).to(dtype=torch.float)
    # TODO: Use diagonal op when support is added (SW-91547)
    wgth_id = torch.eye(
        confusion_matrix.shape[0], device=confusion_matrix.device
    )
    true_pos = (wgth_id * confusion_matrix).sum(axis=-1, dtype=torch.float)
    predicted_per_class = confusion_matrix.sum(dim=0).type(torch.float)
    actual_per_class = confusion_matrix.sum(dim=1).type(torch.float)

    mask = cstorch.make_constant(mask)

    num_labels_to_consider = mask.sum()
    beta = torch.tensor(beta).to(torch.float)

    if average_type == "micro":
        precision = divide_no_nan(
            (true_pos * mask).sum(), (predicted_per_class * mask).sum()
        )
        recall = divide_no_nan(
            (true_pos * mask).sum(), (actual_per_class * mask).sum()
        )
        fbeta = divide_no_nan(
            (1.0 + beta**2) * precision * recall,
            (beta**2) * precision + recall,
        )
    else:  # "macro"
        precision_per_class = divide_no_nan(true_pos, predicted_per_class)
        recall_per_class = divide_no_nan(true_pos, actual_per_class)
        fbeta_per_class = divide_no_nan(
            (1.0 + beta**2) * precision_per_class * recall_per_class,
            (beta**2) * precision_per_class + recall_per_class,
        )
        precision = (precision_per_class * mask).sum() / num_labels_to_consider
        recall = (recall_per_class * mask).sum() / num_labels_to_consider
        fbeta = (fbeta_per_class * mask).sum() / num_labels_to_consider

    return fbeta


class FBetaScoreMetric(Metric):
    """Calculates F Score from labels and predictions.

    fbeta = (1 + beta^2) * (precision*recall) / ((beta^2 * precision) + recall)

    Where beta is some positive real factor.
    Args:
        num_classes: Number of classes.
        beta: Beta coefficient in the F measure.
        average_type: Defines the reduction that is applied. Should be one
            of the following:
            - 'micro' [default]: Calculate the metric globally, across all
                samples and classes.
            - 'macro': Calculate the metric for each class separately, and
                average the metrics across classes (with equal weights for
                each class). This does not take label imbalance into account.
        ignore_labels: Integer specifying a target classes to ignore.
        name: Name of the metric
    """

    def __init__(
        self,
        num_classes,
        beta: float = 1.0,
        average_type: str = "micro",
        ignore_labels: Optional[List] = None,
        name: Optional[str] = None,
    ):
        if num_classes <= 1:
            raise ValueError(
                f"'num_classes' should be at least 2, got {num_classes}"
            )
        self.num_classes = num_classes
        with torch.device("cpu"):
            # We want the mask to be computed on the CPU so that it can be
            # encoded into the graph as a constant
            self.mask = compute_mask(num_classes, ignore_labels)

        super().__init__(name=name)

        if beta <= 0:
            raise ValueError(f"'beta' should be a positive number, got {beta}")
        self.beta = beta

        allowed_average = ["micro", "macro"]
        if average_type not in allowed_average:
            raise ValueError(
                f"The average_type has to be one of {allowed_average}, "
                f"got {average_type}."
            )
        self.average_type = average_type

    def reset(self):
        self.register_state(
            "confusion_matrix",
            torch.zeros(
                (self.num_classes, self.num_classes), dtype=torch.float
            ),
        )
        self._dtype = None

    def update(self, labels, predictions, dtype=None):
        if labels.shape != predictions.shape:
            raise ValueError(
                f"`labels` and `predictions` have mismatched shapes. "
                f"Their shapes were {labels.shape} and {predictions.shape} respectively."
            )

        confusion_matrix = compute_confusion_matrix(
            labels=labels,
            predictions=predictions,
            num_classes=self.num_classes,
            on_device=cstorch.use_cs(),
        )
        self.confusion_matrix.add_(confusion_matrix)

        self._dtype = dtype

    def compute(self) -> torch.Tensor:
        fbeta = compute_helper(
            self.confusion_matrix,
            self.mask,
            beta=self.beta,
            average_type=self.average_type,
        )
        if self._dtype is not None:
            fbeta = fbeta.to(self._dtype)
        return fbeta
