# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""contains the Cerebras SGD implementation"""
from typing import Iterable

import torch

import cerebras.pytorch as cstorch

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    SGD optimizer implemented to conform to execution within the constraints
    of the Cerebras WSE, including pre-initializing optimizer state
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        maximize: bool = False,
    ):
        """
        Args:
            params: Model parameters
            lr: The learning rate to use
            momentum: momentum factor
            dampening: dampening for momentum
            weight_decay: weight decay (L2 penalty)
            nesterov: enables Nesterov momentum
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                f"Nesterov momentum requires a `momentum` and zero `dampening`. "
                f"`momentum` was {momentum} and `dampening` was {dampening}."
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )

        super().__init__(params, defaults)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group['params']:
                if group['momentum'] != 0:
                    self.state[p]["momentum_buffer"] = cstorch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group['momentum']
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("SGD does not support sparse gradients.")

                grad = grad if not maximize else -grad

                grad = grad + p * weight_decay

                if momentum != 0:
                    buf = self.state[p]["momentum_buffer"]

                    buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)

                    if nesterov:
                        grad.add_(buf, alpha=momentum)
                    else:
                        grad = buf

                p.add_(-lr * grad)

        return loss
