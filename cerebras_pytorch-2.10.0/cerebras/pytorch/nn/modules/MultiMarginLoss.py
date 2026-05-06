# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from .utils import apply_loss_reduction, autogen_loss


@autogen_loss
class MultiMarginLoss(nn.Module):
    def __init__(
        self,
        p=1,
        margin=1.0,
        weight=None,
        reduction='mean',
    ):
        super(MultiMarginLoss, self).__init__()
        assert p in [1, 2], "MultiMarginLoss only supports p being 1 or 2."
        assert (
            reduction != 'none'
        ), "MultiMarginLoss does not support 'none' reduction."
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, y):
        # Obtain loss terms including the i == y term in the sum.
        # The i == y term will be subtracted out at the end.
        loss = (
            self.margin
            # - x[y]
            - torch.gather(x, -1, y.unsqueeze(-1).type(torch.int64))
            + x
        )

        loss = loss.clamp(min=0)

        # Compute the i == y term as "correction"
        # I.e., (self.margin ** p) * sum(w[y]) / x.shape[-1]
        # possibly further scaled by 1/batch depending on whether
        # the batch is present or it is a sum reduction.
        if self.p == 2:
            # pow(loss, p)
            loss = loss * loss
            correction = self.margin * self.margin
        else:
            correction = self.margin

        if self.weight is not None:
            # * w[y]
            w = torch.gather(self.weight, 0, y).unsqueeze(-1)
            loss = w * loss
            correction = correction * torch.sum(w)
            if len(x.shape) > 1:
                correction = torch.div(correction, x.shape[0])

        correction = torch.div(correction, x.shape[-1])

        if len(x.shape) > 1 and self.reduction != 'mean':
            correction = correction * x.shape[0]

        if self.reduction == "sum":
            loss = torch.div(loss, x.shape[-1])
        return apply_loss_reduction(loss, self.reduction) - correction
