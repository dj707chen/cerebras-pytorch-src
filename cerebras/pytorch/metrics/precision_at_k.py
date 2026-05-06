# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Precision@K metric for PyTorch.
"""
from typing import Optional

import torch

import cerebras.pytorch as cstorch
from cerebras.pytorch.metrics.metric import Metric


class PrecisionAtKMetric(Metric):
    """
    Precision@K takes the top K predictions and computes the true positive at K
    and false positive at K. For K = 1, it is the same as Precision.

    Precision@K is defined as follows:
    Precision@K = true_positive_at_k / (true_positive_at_k + false_positive_at_k).

    Internally, we keep track of true_positive_at_k and false_positive_at_k,
    weighted by `weights`.

    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

    Args:
        :param Tensor labels: A `Tensor` of ground truth labels of type `int32`
            or `int64` and shape (batch_size, num_labels).
        :param Tensor predictions: A `Tensor` of predicted logit values for
            each class in the last dimention. It is of type `float` and shape
            (batch_size, num_classes).
        :param int k: The number of predictions for @k metric.
        :param Tensor weights: Optional `Tensor` whose rank is either 0, or n-1,
            where n is the rank of `labels`. If the latter, it must be
            broadcastable to `labels` (i.e., all dimensions must be either `1`,
            or the same as the corresponding `labels` dimension).
        :param name: Optional `string` which indicates name of the metric.
                If None or empty string, it defaults to the name of the class.

    Returns:
        precision_at_k: A float representing Precision@K.

    Raises:
        ValueError: If `weights` is not `None` and its shape doesn't match `predictions`
    """

    def __init__(self, k, name: Optional[str] = None):
        if cstorch.use_cs():
            raise NotImplementedError(
                "PrecisionAtKMetric not yet supported on CSX."
            )

        self.k = k
        super(PrecisionAtKMetric, self).__init__(name=name)

    def reset(self):
        self.register_state(
            "true_positive_at_k", torch.tensor(0, dtype=torch.float32)
        )
        self.register_state(
            "false_positive_at_k", torch.tensor(0, dtype=torch.float32)
        )

    def update(self, labels, predictions, weights=None):
        if weights is not None:
            if len(weights.shape) != 0 and weights.numel() != labels.shape[0]:
                raise ValueError(
                    f"`labels`={labels.shape} and `weights`={weights.shape} so"
                    f"`weights` must be a scalar or a vector of size "
                    f"{labels.shape[0]}"
                )

        _, topk_pred_idx = torch.topk(predictions, self.k, dim=-1)

        # Compute the number of true positives per row
        lbl = labels.repeat_interleave(self.k, dim=1)
        pred_idx = topk_pred_idx.repeat(1, labels.shape[-1])
        intersection_per_row = torch.sum(lbl == pred_idx, dim=-1).float()

        if weights is not None:
            tp = intersection_per_row * weights
            fp = (self.k - intersection_per_row) * weights
        else:
            tp = intersection_per_row
            fp = self.k - intersection_per_row

        self.true_positive_at_k += torch.sum(tp)
        self.false_positive_at_k += torch.sum(fp)

    def compute(self):
        """Returns the Precision@K as a float."""
        return float(
            self.true_positive_at_k
            / (self.true_positive_at_k + self.false_positive_at_k)
        )
