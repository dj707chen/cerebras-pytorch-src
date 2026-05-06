# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
mean Intersection-Over-Union (mIOU) metric for PyTorch.
Calculate per-step mean Intersection-Over-Union (mIOU).
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


def compute_helper(confusion_matrix, mask):
    """Returns the meanIOU"""
    mask = cstorch.make_constant(mask)

    sum_over_row = torch.sum(confusion_matrix, 0, dtype=torch.float)
    sum_over_col = torch.sum(confusion_matrix, 1, dtype=torch.float)

    # TODO: workaround for SW-76827
    # cm_diag = torch.diagonal(confusion_matrix).to(dtype=torch.float)
    wgth_id = torch.eye(
        confusion_matrix.shape[0], device=confusion_matrix.device
    )
    cm_diag = (wgth_id * confusion_matrix).sum(
        axis=-1, dtype=torch.float
    ) * mask
    denominator = (sum_over_row + sum_over_col - cm_diag) * mask

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = torch.sum(torch.ne(denominator, 0).to(torch.float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    iou = divide_no_nan(cm_diag, denominator)

    # If the number of valid entries is 0 (no classes) we return 0.
    mean_iou = torch.where(
        num_valid_entries > 0.0,
        torch.sum(iou) / num_valid_entries,
        torch.tensor(0, dtype=torch.float, device=iou.device),
    )
    return mean_iou


class MeanIOUMetric(Metric):
    """
    Mean Intersection-Over-Union is a common evaluation metric for
    semantic image segmentation, which first computes the IOU for each
    semantic class and then computes the average over classes.
    iou is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by `weights`,
    and mIOU is then calculated from it.

    For estimation of the metric over a stream of data, the function creates an
    `update_op` operation that updates these variables and returns the `mean_iou`.

    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

    Args:
        num_classes: The possible number of labels the prediction task can
            have. This value must be provided, since a confusion matrix of
            dimension = [num_classes, num_classes] will be allocated.
        name: Optional `string` which indicates name of the metric.
            If None or empty string, it defaults to the name of the class.
    """

    def __init__(
        self,
        num_classes,
        ignore_classes: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        self.num_classes = num_classes
        with torch.device("cpu"):
            # We want the mask to be computed on the CPU so that it can be
            # encoded into the graph as a constant
            self.mask = compute_mask(num_classes, ignore_classes)
        super().__init__(name=name)

    def reset(self):
        self.register_state(
            "confusion_matrix",
            torch.zeros(
                (self.num_classes, self.num_classes), dtype=torch.float32
            ),
        )
        self._dtype = None

    def update(
        self, labels, predictions, weights=None, dtype=None
    ):  # pylint: disable=arguments-differ
        """
        Updates the mean IOU metric.

        Args:
            labels: A `Tensor` of ground truth labels of type `int32` or `int64`.
            predictions: A `Tensor` of prediction results for semantic labels,
                of type `int32` or `int64`.
            weights: Optional `Tensor` whose rank is either 0, or the same rank as
                `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
                be either `1`, or the same as the corresponding `labels` dimension).

        Raises:
            ValueError: If `predictions` and `labels` have mismatched shapes, or if
                `weights` is not `None` and its shape doesn't match `predictions`
        """
        if labels.shape != predictions.shape:
            raise ValueError(
                f"`labels` and `predictions` have mismatched shapes. "
                f"Their shapes were {labels.shape} and {predictions.shape} respectively."
            )
        if weights is not None:
            if weights.shape != labels.shape:
                raise ValueError(
                    f"`labels`={labels.shape} and ",
                    f"`weights`={weights.shape} have mismatched shapes",
                )

        confusion_matrix = compute_confusion_matrix(
            labels=labels,
            predictions=predictions,
            num_classes=self.num_classes,
            weights=weights,
            on_device=cstorch.use_cs(),
        )
        self.confusion_matrix.add_(confusion_matrix)

        self._dtype = dtype

    def compute(self) -> torch.Tensor:
        mean_iou = compute_helper(self.confusion_matrix, self.mask)
        if self._dtype is not None:
            mean_iou = mean_iou.to(self._dtype)
        return mean_iou
