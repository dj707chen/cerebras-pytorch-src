#
# This code is adapted from
# https://github.com/pytorch/pytorch/blob/473b733bae7009945cc5712699d346678e8a40ff/torch/_decomp/decompositions.py#L349
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import torch.nn as nn

from .utils import apply_loss_reduction, autogen_loss


@autogen_loss
class HuberLoss(nn.Module):
    def __init__(self, reduction='mean', delta=1.0):
        super(HuberLoss, self).__init__()
        self.reduction = reduction
        assert delta > 0, "HuberLoss only supports positive values for delta."
        self.delta = delta

    def forward(self, input, target):
        z = (input - target).abs()
        loss = torch.where(
            z < self.delta, 0.5 * z * z, self.delta * (z - 0.5 * self.delta)
        )
        return apply_loss_reduction(loss, self.reduction)
