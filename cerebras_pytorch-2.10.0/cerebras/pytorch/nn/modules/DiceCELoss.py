# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from .DiceLoss import DiceLoss


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes,
        include_background,
        wc=0.5,
        wd=0.5,
    ):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss(
            num_classes=num_classes,
            include_background=include_background,
        )
        self.cross_entropy = nn.CrossEntropyLoss()
        self.wc = wc
        self.wd = wd
        if not include_background:
            self.mean_correction = torch.tensor(
                num_classes / (num_classes - 1),
                dtype=torch.float32,
            )
        else:
            self.mean_correction = torch.tensor(
                1.0,
                dtype=torch.float32,
            )
        self.one_const = torch.tensor(
            1.0,
            dtype=torch.float32,
        )

    def forward(self, outputs, labels, input_shape):
        ce = self.cross_entropy(outputs, labels)
        dc = self.mean_correction * torch.mean(
            self.dice(outputs, labels, input_shape)
        )
        loss = self.wc * ce + self.wd * (self.one_const - dc)
        return loss
