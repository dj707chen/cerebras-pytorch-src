# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch

import cerebras.pytorch as cstorch

from .. import functional as F


class DiceLoss:
    def __init__(
        self,
        num_classes: int,
        to_onehot_y: bool = True,
        to_onehot_x: bool = False,
        use_softmax: bool = True,
        use_argmax: bool = False,
        include_background: bool = False,
    ):
        self.num_classes = num_classes
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.to_onehot_x = to_onehot_x
        self.use_softmax = use_softmax
        self.use_argmax = use_argmax
        self.smooth_nr = 0.0
        self.smooth_dr = 1e-6
        self.include_background = include_background
        self.bg_mask = None

    def _create_background_mask(self, device, dtype, ish, chanx):
        z_shape = ish[0:chanx] + [1] + ish[chanx + 1 :]  # [N,1,D,H,W]
        o_shape = (
            ish[0:chanx] + [ish[chanx] - 1] + ish[chanx + 1 :]
        )  # [N,C-1,D,H,W]
        zeros = torch.zeros(z_shape, dtype=dtype)
        ones = torch.ones(o_shape, dtype=dtype)
        weights = torch.cat(
            (zeros, ones), chanx
        )  # [N,C,D,H,W] w/ first ch 0'ed
        if cstorch.use_cs():
            bg_mask = cstorch.make_constant(weights)
        else:
            bg_mask = weights.to(device)
        return bg_mask

    def __call__(self, prediction, target, input_shape):

        target = torch.unsqueeze(target, 1)
        channel_axis = 1
        reduce_axis = list(range(2, len(prediction.shape)))
        num_pred_ch = prediction.shape[channel_axis]

        if self.use_softmax:
            prediction = torch.softmax(prediction, dim=channel_axis)
        elif self.use_argmax:
            prediction = torch.argmax(prediction, dim=channel_axis)

        if self.to_onehot_y:
            target = to_one_hot(target, channel_axis, self.num_classes)
        if self.to_onehot_x:
            prediction = to_one_hot(prediction, channel_axis, self.num_classes)

        if not self.include_background:
            if self.bg_mask is None:
                self.bg_mask = self._create_background_mask(
                    target.device,
                    prediction.dtype,
                    input_shape,
                    channel_axis,
                )
            assert (
                num_pred_ch > 1
            ), f"To exclude background the prediction needs more than one channel. Got {num_pred_ch}."
            target = target * self.bg_mask
            prediction = prediction * self.bg_mask

        assert (
            target.shape == prediction.shape
        ), f"Target and prediction shape do not match. Target: ({target.shape}), prediction: ({prediction.shape})."

        intersection = torch.sum(target * prediction, dim=reduce_axis)
        target_sum = torch.sum(target, dim=reduce_axis)
        prediction_sum = torch.sum(prediction, dim=reduce_axis)

        res = (2.0 * intersection + self.smooth_nr) / (
            target_sum + prediction_sum + self.smooth_dr
        )
        return res


def to_one_hot(array, channel_axis, num_classes):
    if len(array.shape) >= 5:
        array = torch.squeeze(array, dim=channel_axis)

    array = F.one_hot(array.long(), num_classes).float()
    array = array.permute(0, 4, 1, 2, 3)
    return array
