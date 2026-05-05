#
# Cerebras implementation of RAdam optimizer. Adapted from the `torch.optim.RAdam` implementation.
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Iterable, Tuple

import torch

import cerebras.pytorch as cstorch

from .optimizer import Optimizer


class RAdam(Optimizer):
    r"""RAdam optimizer implemented to conform to execution within the
    constraints of the Cerebras WSE.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0]"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0]"
            )
        if weight_decay < 0.0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay} - should be >= 0.0"
            )
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults, enable_global_step=True)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                # State initialization

                # Exponential moving average of gradient values
                state["exp_avg"] = cstorch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = cstorch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            if not isinstance(beta1, torch.Tensor):
                beta1 = torch.tensor(beta1)
            if not isinstance(beta2, torch.Tensor):
                beta2 = torch.tensor(beta2)
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    state = self.state[p]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    global_step = self.increment_global_step(p)

                    grad = grad + p * weight_decay

                    beta1t = torch.pow(beta1.to(p.device), global_step)
                    beta2t = torch.pow(beta2.to(p.device), global_step)

                    bias_correction1 = 1 - beta1t
                    bias_correction2 = 1 - beta2t

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time.
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2
                    )

                    # correcting bias for the first moving moment
                    update = exp_avg / bias_correction1

                    # maximum length of the approximated SMA
                    rho_inf = 2 / (1 - beta2) - 1
                    # compute the length of the approximated SMA
                    rho_t = (
                        rho_inf - 2 * global_step * beta2t / bias_correction2
                    )

                    one = torch.tensor(1.0).to(p.device)
                    five = torch.tensor(5.0).to(p.device)

                    # Compute the variance rectification term and update parameters accordingly
                    rect = torch.where(
                        torch.gt(rho_t, five),
                        torch.sqrt(
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * rho_inf
                            / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                        ),
                        one,
                    )
                    adaptive_lr = torch.where(
                        torch.gt(rho_t, five),
                        torch.sqrt(bias_correction2)
                        / exp_avg_sq.sqrt().add_(eps),
                        one,
                    )

                    update *= rect
                    update *= adaptive_lr

                    update *= group["lr"]

                    p.sub_(update)

        return loss
