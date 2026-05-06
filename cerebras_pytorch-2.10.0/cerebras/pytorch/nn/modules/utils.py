# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch

import cerebras.pytorch as cstorch

LOSS_SCOPE = "loss"


def apply_loss_reduction(loss, reduction):
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss


# Since we do not have automatic loss scope detection yet, some changes in user model would
# be required to tag the ops belong to loss.
# A wrapper to wrap loss function for Autogen support, so that loss function
# will be handled by Autogen.
# To use this Autogen loss function wrapper, when initializing the loss function,
# using our loss in Layer API, and adding a keyword argument 'use_autogen=True' (the default is False).
# For example:
#   loss = MSELoss(..., use_autogen=True)
#
# To apply the Autogen loss wrapper to any custom loss function, just add this wrapper as the decorator
# of the custom loss class.
# For example:
#   @autogen_loss
#   class CustomLoss(nn.Module):
#       def __init__(...):
#
# In the future, we may remove this temporary wrapper after we developed better technics to support Autogen.
def autogen_loss(loss_cls):
    if cstorch._generating_docs:
        # When generating docs, there's no need to actually wrap the loss function
        return loss_cls

    loss_cls._old_init = loss_cls.__init__
    loss_cls._old_forward = loss_cls.forward

    def autogen_init(self, *args, **kwargs):
        self.autogen_enabled = kwargs.pop("use_autogen", False)
        self._old_init(*args, **kwargs)
        if self.autogen_enabled and cstorch.use_cs():
            self.mark_with_autogen = cstorch.nn.Scope(scope_name=LOSS_SCOPE)

    def autogen_forward(self, *args, **kwargs):
        if self.autogen_enabled and cstorch.use_cs():
            args = [self.mark_with_autogen(arg) for arg in args]
            kwargs = {k: self.mark_with_autogen(v) for k, v in kwargs.items()}
            loss = self._old_forward(*args, **kwargs)
            return self.mark_with_autogen.exit(loss)
        else:
            return self._old_forward(*args, **kwargs)

    loss_cls.__init__ = autogen_init
    loss_cls.forward = autogen_forward
    return loss_cls
