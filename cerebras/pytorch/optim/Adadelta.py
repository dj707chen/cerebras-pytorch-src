# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""contains the Cerebras Adadelta implementation"""
from typing import Callable

import torch

import cerebras.pytorch as cstorch

from .optimizer import Optimizer


class Adadelta(Optimizer):
    """
    Adadelta optimizer implemented to perform the required
    pre-initialization of the optimizer state.
    """

    def __init__(
        self,
        params,
        lr=1.0,
        rho=0.9,
        eps=1e-6,
        weight_decay=0,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
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
                self.state[p]["square_avg"] = cstorch.zeros_like(p)
                self.state[p]["acc_delta"] = cstorch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure : A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            rho = group['rho']
            eps = group["eps"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adadelta does not support sparse gradients."
                    )

                state = self.state[p]
                square_avg = state["square_avg"]
                acc_delta = state["acc_delta"]

                grad = grad if not maximize else -grad

                grad = grad + p * weight_decay

                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
                p.add_(-lr * delta)

        return loss
