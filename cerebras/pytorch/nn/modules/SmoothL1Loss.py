#
# This code is adapted from
# https://github.com/pytorch/pytorch/blob/f96d96a7fcaa5bb06829d2c7de1992d6ab6e9235/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L608
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import torch.nn as nn

from .utils import apply_loss_reduction, autogen_loss


@autogen_loss
class SmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean', beta=1.0):
        super(SmoothL1Loss, self).__init__()
        assert (
            beta >= 0
        ), "SmoothL1Loss only supports non-negative values for beta."
        self.reduction = reduction
        self.beta = beta

    def forward(self, input, target):
        z = torch.abs(input - target)
        loss = torch.where(
            z < self.beta, 0.5 * z * z / self.beta, z - 0.5 * self.beta
        )
        return apply_loss_reduction(loss, self.reduction)
