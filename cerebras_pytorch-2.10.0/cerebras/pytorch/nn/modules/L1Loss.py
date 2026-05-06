# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch.nn as nn

from .utils import autogen_loss


# A dummy class to wrap nn.L1Loss loss with Autogen support. To
# enable the Autogen support, add a keyword argument `use_autogen=True`
# when initializing the loss function.
# For example:
# loss = L1Loss(..., use_autogen=True)
@autogen_loss
class L1Loss(nn.L1Loss):
    pass
