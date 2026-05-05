#
# Cerebras implementation of ASGD optimizer. Adapted from the `torch.optim.ASGD` implementation.
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

import cerebras.pytorch as cstorch

from .optimizer import Optimizer


class ASGD(Optimizer):
    r"""ASGD optimizer implemented to conform to execution within the constraints
    of the Cerebras WSE, including pre-initializing optimizer state.

    For more details, see https://dl.acm.org/citation.cfm?id=131098
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        lambd=1e-4,
        alpha=0.75,
        t0=1e6,
        weight_decay=0,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults, enable_global_step=True)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["eta"] = torch.tensor(group["lr"]).to(p.device)
                self.state[p]["mu"] = torch.tensor(1.0).to(p.device)
                self.state[p]["ax"] = cstorch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        r"""
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the
                model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lambd = group["lambd"]
            lr = group["lr"]
            t0 = group["t0"]

            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "ASGD does not support sparse gradients"
                        )
                    alpha = group["alpha"]
                    if not isinstance(alpha, torch.Tensor):
                        alpha = torch.tensor(alpha)
                    alpha = alpha.to(p.device)
                    state = self.state[p]
                    grad = p.grad
                    grad = grad if not group["maximize"] else -grad
                    mu = state["mu"]
                    ax = state["ax"]
                    eta = state["eta"]
                    step = self.increment_global_step(p)

                    grad = grad + p * group["weight_decay"]

                    # decay term
                    p.mul_(1 - lambd * eta)

                    # update parameter
                    p.add_(grad * eta.neg())

                    # averaging
                    new_ax = torch.where(mu == 1, p, ax.add(p.sub(ax).mul(mu)))
                    ax.copy_(new_ax)

                    new_eta = lr / torch.pow(1 + lambd * lr * step, alpha)
                    eta.copy_(new_eta)

                    new_mu = 1 / torch.maximum(
                        torch.ones(size=[], dtype=mu.dtype),
                        step - t0,
                    )
                    mu.copy_(new_mu)

        return loss
