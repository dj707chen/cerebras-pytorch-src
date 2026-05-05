"""contains the Cerebras Adafactor implementation"""

# coding=utf-8
#
# This code is adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Modifications Copyright 2023 Cerebras.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch.optim import Optimizer

import cerebras.pytorch as cstorch

from .optimizer import Optimizer


class Adafactor(Optimizer):
    """
    Adafactor optimizer implemented to conform to execution within the
    constraints of the Cerebras WSE.
    """

    def __init__(
        self,
        params,
        lr,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=False,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError(
                "Cannot combine manual `lr` and `relative_step=True` options"
            )
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` is not supported yet")
        if clip_threshold != 1.0:
            raise ValueError(
                f"Only `clip_threshold=1.0` is supported now. "
                f"It was set to {clip_threshold}."
            )
        if beta1 is not None:
            raise ValueError(
                f"Only `beta1=None` is supported now. It was set to {beta1}."
            )
        if relative_step:
            raise ValueError("`relative_step=True` is not supported yet")
        if (
            not isinstance(eps, (tuple, list))
            or len(eps) != 2
            or not all([isinstance(x, float) for x in eps])
        ):
            raise ValueError(
                f"Expected `eps` to be a tuple/list of two floats, but got {eps}."
            )

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
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
                grad_shape = p.shape

                factored = len(grad_shape) >= 2
                use_first_moment = group["beta1"] is not None

                if use_first_moment:
                    state["exp_avg"] = cstorch.zeros_like(p)
                if factored:
                    state["exp_avg_sq_row"] = cstorch.zeros(grad_shape[:-1])
                    state["exp_avg_sq_col"] = cstorch.zeros(
                        grad_shape[:-2] + grad_shape[-1:]
                    )
                else:
                    state["exp_avg_sq"] = cstorch.zeros_like(p)

    @staticmethod
    def _get_lr(param_group, rms):
        rel_step_sz = param_group["lr"]
        param_scale = 1.0
        if param_group["scale_parameter"]:
            eps = param_group["eps"][1]
            if not isinstance(eps, torch.Tensor):
                eps = torch.tensor(eps)
            param_scale = torch.maximum(rms, eps)
        return param_scale * rel_step_sz

    @staticmethod
    def _rms(tensor):
        return tensor.square().mean().sqrt()

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates
            the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adafactor does not support sparse gradients."
                    )

                state = self.state[p]

                factored = "exp_avg_sq_row" in state
                use_first_moment = "exp_avg" in state

                global_step_fp32 = self.increment_global_step(p)

                lr = self._get_lr(group, self._rms(p))

                decay_rate = group["decay_rate"]
                if not isinstance(decay_rate, torch.Tensor):
                    decay_rate = torch.tensor(decay_rate)

                beta2t = 1.0 - torch.pow(
                    global_step_fp32, decay_rate.to(p.device)
                )
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1).mul(1.0 - beta2t)
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2).mul(1.0 - beta2t)
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col
                    )
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update.mul(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    torch.maximum(
                        self._rms(update) / group["clip_threshold"],
                        torch.tensor(1.0, dtype=torch.float32, device=p.device),
                    )
                )
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(
                        update.mul(1 - group["beta1"])
                    )
                    update = exp_avg

                p.sub_(p.mul(group["weight_decay"] * lr))

                p.sub_(update)

        return loss
