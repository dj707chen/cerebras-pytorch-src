#
# Cerebras implementation of Rprop optimizer. Adapted from the `torch.optim.Rprop` implementation.
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Iterable, Tuple

import torch

import cerebras.pytorch as cstorch

from .optimizer import Optimizer


class Rprop(Optimizer):
    r"""Rprop optimizer implemented to conform to execution within the constraints
    of the Cerebras WSE, including pre-initializing optimizer state

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        etas (Tuple[float, float], optional): step size multipliers
        step_size (Tuple[float, float], optional): Tuple of min, max step size
            values. Step size is clamped to be between these values.

    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-6, 50.0),
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
        )

        super().__init__(params, defaults)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state["prev"] = cstorch.zeros_like(p)
                state["step_size"] = cstorch.full_like(p, group['lr'])

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
            step_size_min, step_size_max = [
                ss if isinstance(ss, torch.Tensor) else torch.tensor(ss)
                for ss in group['step_sizes']
            ]
            etaminus, etaplus = [
                eta if isinstance(eta, torch.Tensor) else torch.tensor(eta)
                for eta in group['etas']
            ]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "Rprop does not support sparse gradients."
                    )

                state = self.state[p]

                prev = state["prev"]
                prev_step_size = state["step_size"]

                zero = torch.tensor(0.0).to(p.device)
                one = torch.tensor(1.0).to(p.device)
                neg_one = torch.tensor(-1.0).to(p.device)

                # check gradient for sign change - set step multiplier to etaplus, etaminus, or 1
                step_multiplier = grad.mul(prev)
                step_multiplier = torch.where(
                    torch.gt(step_multiplier, zero),
                    etaplus.to(p.device),
                    step_multiplier,
                )
                step_multiplier = torch.where(
                    torch.lt(step_multiplier, zero),
                    etaminus.to(p.device),
                    step_multiplier,
                )
                step_multiplier = torch.where(
                    torch.eq(step_multiplier, zero),
                    one,
                    step_multiplier,
                )

                # calculate new step size and clamp between step_size_min and step_size_max
                step_size = prev_step_size.mul_(step_multiplier).to(p.device)
                step_size = torch.clamp(
                    step_size,
                    min=step_size_min,
                    max=step_size_max,
                )

                # Zero out the gradient wherever there was a sign change
                mask = torch.where(
                    torch.eq(step_multiplier, etaminus.to(p.device)),
                    zero,
                    one,
                )

                grad.mul_(mask.to(p.device))

                # Get the sign of the gradient
                sign_grad = torch.zeros_like(grad)
                sign_grad = torch.where(torch.gt(grad, zero), one, sign_grad)
                sign_grad = torch.where(
                    torch.lt(grad, zero), neg_one, sign_grad
                )

                update = torch.mul(sign_grad, step_size).to(p.device)

                # update parameters
                p.sub_(update)

                prev.copy_(grad)
                prev_step_size.copy_(step_size)

        return loss
