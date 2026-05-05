# This code is adapted from
# https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2019 cybertronai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

import cerebras.pytorch as cstorch

from .optimizer import Optimizer


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    .. _Large Batch Optimization for Deep Learning\: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962

    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        adam=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam=adam
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
                # Exponential moving average of gradient values
                state['exp_avg'] = cstorch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = cstorch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        r"""Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider SparseAdam instad.'
                    )

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group[
                    'lr'
                ]  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.pow(2).sum().sqrt().clamp(0, 10).to(torch.float)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])

                adam_step = adam_step + p * group["weight_decay"]

                adam_norm = adam_step.pow(2).sum().sqrt().to(torch.float)
                # pytorch version for future reference (we don't support
                #     weight_norm == 0 or adam_norm == 0)
                # if weight_norm == 0 or adam_norm == 0:
                #     trust_ratio = 1
                # else:
                #     trust_ratio = weight_norm / adam_norm
                zero = torch.tensor(
                    0.0, dtype=torch.float32, device=weight_norm.device
                )
                trust_ratio = torch.where(
                    torch.gt(weight_norm, zero),
                    torch.where(
                        torch.gt(adam_norm, zero),
                        weight_norm / adam_norm,
                        torch.tensor(
                            1.0, dtype=torch.float32, device=weight_norm.device
                        ),
                    ),
                    torch.tensor(
                        1.0, dtype=torch.float32, device=weight_norm.device
                    ),
                )
                if group['adam']:
                    trust_ratio = 1

                update_step = adam_step.mul(trust_ratio)
                p.sub_(update_step * step_size)

        return loss
