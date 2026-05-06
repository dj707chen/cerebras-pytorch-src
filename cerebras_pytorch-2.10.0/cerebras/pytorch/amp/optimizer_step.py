# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Contains a helper that takes care of using the GradScaler"""
import torch

import cerebras.pytorch as cstorch


def optimizer_step(
    loss: torch.Tensor,
    optimizer: "cstorch.optim.Optimizer",
    grad_scaler: "cstorch.amp.GradScaler",
    max_gradient_norm: float = None,
    max_gradient_value: float = None,
):
    """
    Performs loss scaling, gradient scaling and optimizer step

    Args:
        loss: The loss value to scale. loss.backward should be called before
            this function
        optimizer: The optimizer to step
        grad_scaler: The gradient scaler to use to scale the parameter gradients
        max_gradient_norm: the max gradient norm to use for gradient clipping
        max_gradient_value: the max gradient value to use for gradient clipping
    """
    if not isinstance(loss, torch.Tensor):
        raise ValueError(
            "Expected the wrapped function to return a single loss tensor. "
            f"Got: {type(loss)}"
        )

    if isinstance(optimizer, cstorch.optim.Optimizer):
        optimizers = [optimizer]
    elif isinstance(optimizer, (list, tuple)):
        optimizers = optimizer
        for i, optim in enumerate(optimizers):
            if not isinstance(optim, cstorch.optim.Optimizer):
                raise TypeError(
                    f"Expected optimizer {i} to be a `cstorch.optim.Optimizer`. "
                    f"Got: `{type(optim)}`"
                )
    else:
        raise TypeError(
            f"Expected optimizer {i} to be a `cstorch.optim.Optimizer`. "
            f"Got: `{type(optimizer)}`"
        )

    if not isinstance(grad_scaler, cstorch.amp.GradScaler):
        raise TypeError(
            "Expected grad_scaler to be a `cstorch.amp.GradScaler`. "
            f"Got: `{type(grad_scaler)}`"
        )

    grad_scaler.scale(loss).backward()

    for optim in optimizers:
        grad_scaler.unscale_(optim)

    # gradient clipping
    if max_gradient_norm is not None and max_gradient_norm < 0.0:
        raise ValueError(
            f"max_gradient_norm has to be a non-negative float. Got "
            f"{max_gradient_norm}"
        )
    if max_gradient_value is not None and max_gradient_value < 0.0:
        raise ValueError(
            f"max_gradient_value has to be a non-negative float. Got "
            f"{max_gradient_value}"
        )
    if max_gradient_norm is not None and max_gradient_value is not None:
        raise ValueError(
            f"Gradients can be clipped by norm(={max_gradient_norm}) or by "
            f"value(={max_gradient_value}), but not both. "
            f"Do not set both `max_gradient_norm` and `max_gradient_value`."
        )

    # TODO: add check for if max_gradient_norm is set in grad scaler
    params = (
        p
        for param_group in optimizer.param_groups
        for p in param_group["params"]
    )
    if max_gradient_norm is not None:
        torch.nn.utils.clip_grad_norm_(list(params), max_gradient_norm)
    elif max_gradient_value is not None:
        torch.nn.utils.clip_grad_value_(list(params), max_gradient_value)

    for optim in optimizers:
        grad_scaler.step(optim)

    # compute new loss scale
    grad_scaler.update()

    for optim in optimizers:
        optim.zero_grad()
