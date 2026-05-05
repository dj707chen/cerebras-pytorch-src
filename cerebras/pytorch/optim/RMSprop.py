#
# Cerebras implementation of RAdam optimizer. Adapted from the `torch.optim.RMSProp` implementation.
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

import cerebras.pytorch as cstorch

from .optimizer import Optimizer


class RMSprop(Optimizer):
    """
    RMSprop optimizer implemented to perform the required
    pre-initialization of the optimizer state.
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
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
                if group['momentum'] > 0:
                    self.state[p]["momentum_buffer"] = cstorch.zeros_like(p)
                if group['centered']:
                    self.state[p]["grad_avg"] = cstorch.zeros_like(p)

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
            alpha = group["alpha"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["eps"]
            centered = group["centered"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "RMSprop does not support sparse gradients."
                    )

                state = self.state[p]
                square_avg = state["square_avg"]

                grad = grad + p * weight_decay

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1.0 - alpha)

                if centered:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = (
                        square_avg.addcmul(grad_avg, grad_avg, value=-1.0)
                        .sqrt_()
                        .add_(eps)
                    )
                else:
                    avg = square_avg.sqrt().add_(eps)

                if momentum > 0.0:
                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.mul_(momentum).addcdiv_(grad, avg)
                    p.add_(-lr * momentum_buffer)
                else:
                    p.addcdiv_(-lr * grad, avg)

        return loss
