from typing import Tuple

import torch

import cerebras.pytorch as cstorch

from .optimizer import Optimizer, ParamsT


class Lion(Optimizer):
    r"""Implements Lion algorithm.
    As proposed in `Symbolic Discovery of Optimization Algorithms`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
        weight_decay (float, optional): weight decay coefficient (default: 0)

    .. _Symbolic Discovery of Optimization Algorithms: https://arxiv.org/pdf/2302.06675.pdf

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["exp_avg"] = cstorch.zeros_like(p)

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
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is not None:
                    grad = p.grad
                    state = self.state[p]
                    exp_avg = state['exp_avg']

                    p.mul_(1 - lr * weight_decay)

                    update = (
                        exp_avg.clone()
                        .mul_(beta1)
                        .add(grad, alpha=1 - beta1)
                        .sign_()
                    )
                    p.add_(-lr * update)
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
