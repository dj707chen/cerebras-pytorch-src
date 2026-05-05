# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Cerebras Gradient Scaler implementation"""
import warnings
from enum import Enum, auto
from typing import Union

import torch

import cerebras.pytorch.amp as amp
from cerebras.pytorch.backend import current_backend_impl

from ._amp_state import _amp_state, maybe_print
from .conditional_update import ConditionalUpdateManager


class OptState(Enum):
    """
    An enum to specify the optimizer's current state regarding if its been
    scaled or not
    """

    READY = auto()
    SCALED = auto()
    UNSCALED = auto()
    STEPPED = auto()

    def is_unscaled(self):
        """Returns true if the state is unscaled"""
        return self == OptState.UNSCALED


class GradScaler:
    """
    Faciliates mixed precision training and DLS, DLS + GCC

    For more details please see docs for amp.initialize.

    Args:
        loss_scale:
            If loss_scale == "dynamic", then configure dynamic loss
            scaling. Otherwise, it is the loss scale value used in static
            loss scaling.
        init_scale:
            The initial loss scale value if loss_scale == "dynamic"
        steps_per_increase:
            The number of steps after which to increase the loss
            scaling condition
        min_loss_scale:
            The minimum loss scale value that can be chosen by dynamic
            loss scaling
        max_loss_scale:
            The maximum loss scale value that can be chosen by dynamic
            loss scaling
        overflow_tolerance:
            The maximum fraction of steps involving infinite or undefined
            values in the gradient we allow. We reduce the loss scale if
            the tolerance is exceeded
        max_gradient_norm:
            The maximum gradient norm to use for global gradient clipping
            Only applies in the DLS + GCC case. If GCC is not enabled,
            then this parameter has no effect
    """

    warned_unscaling_non_fp32_grad = False

    def __init__(
        self,
        loss_scale: Union[str, float] = None,
        init_scale: float = None,
        steps_per_increase: int = None,
        min_loss_scale: float = None,
        max_loss_scale: float = None,
        overflow_tolerance: float = 0.0,
        max_gradient_norm: float = None,
    ):
        fp16_type = _amp_state.half_dtype_str
        default_max_loss_scale_value = (
            2.0**31 if fp16_type == "cbfloat16" else 2.0**15
        )
        loss_scale = loss_scale if loss_scale else 1.0
        init_scale = init_scale if init_scale else default_max_loss_scale_value
        steps_per_increase = steps_per_increase if steps_per_increase else 2000
        min_loss_scale = min_loss_scale if min_loss_scale else 2.0**-14
        max_loss_scale = (
            max_loss_scale if max_loss_scale else default_max_loss_scale_value
        )

        self.loss_is_scaled = loss_scale != 1.0

        self.backend = current_backend_impl()

        if loss_scale == "dynamic":
            if min_loss_scale < 2.0**-14:
                raise ValueError("min_loss_scale too small")
            if overflow_tolerance < 0:
                raise ValueError(
                    "loss scaling counter threshold must be set >= 0"
                )

            self.dynamic = True

            self._loss_scale = torch.tensor(
                min(max_loss_scale, init_scale),
                dtype=torch.float32,
            )
            self._steps_since_rescale = torch.tensor(0, dtype=torch.int64)
            self._overflows_since_rescale = torch.tensor(0, dtype=torch.int64)

            self._overflow_tolerance = overflow_tolerance
            self._max_gradient_norm = max_gradient_norm
            if max_gradient_norm:
                warnings.warn(
                    "Using global gradient clipping built into GradScaler "
                    "is deprecated. Use torch.nn.utils.clip_grad_norm_"
                )
            # Will be set in `_unscale_helper`
            self._squared_local_norms = []
            # Will bet set in `update_scale`
            self.isfinite = None
        else:
            self.dynamic = False
            self._loss_scale = loss_scale
            self.isfinite = True
            max_gradient_norm = None

        self._max_loss_scale = max_loss_scale
        self._min_loss_scale = min_loss_scale
        self._steps_per_increase = steps_per_increase
        self.global_norm = None

        self.backend.setup_grad_scaler(self)

        for optimizer in self.backend.optimizer_registry:
            if hasattr(optimizer, "_amp_stash"):
                continue

            amp.setup_optimizer(optimizer)
            optimizer._amp_stash.state = OptState.READY

    def state_dict(self, destination=None):
        """
        Returns a dictionary containing the state to be saved to a checkpoint
        """
        if not self.backend.backend_type.is_csx:
            return {}

        if self.dynamic:
            return {
                "loss_scale": self._loss_scale,
                "steps_since_rescale": self._steps_since_rescale,
                "overflows_since_rescale": self._overflows_since_rescale,
            }
        else:
            return {"loss_scale": self._loss_scale}

    def load_state_dict(self, state_dict):
        """Loads the state dictionary into the current params"""

        def load_param(param, param_name):
            # Only load if the key exists in the state_dict
            if param_name in state_dict:
                value = state_dict[param_name]
                if isinstance(param, torch.Tensor):
                    if isinstance(value, torch.Tensor):
                        # Only move to device is the param device is not CPU
                        # Otherwise keep the original value's device
                        if (
                            value.device.type != param.device.type
                            and param.device.type != "cpu"
                        ):
                            return value.to(param.device)
                        return value
                    else:
                        return torch.tensor(value, dtype=param.dtype).to(
                            param.device
                        )
                else:
                    return value
            else:
                return param

        self._loss_scale = load_param(self._loss_scale, "loss_scale")
        if self.dynamic:
            self._steps_since_rescale = load_param(
                self._steps_since_rescale, "steps_since_rescale"
            )
            self._overflows_since_rescale = load_param(
                self._overflows_since_rescale, "overflows_since_rescale"
            )

    def scale(self, loss: torch.Tensor):
        """Scales the loss in preparation of the backwards pass"""
        # TODO: handle the case of outputs being iterable
        # which is supported by the torch interface
        if not self.backend.backend_type.is_csx:
            return loss

        with self.backend.name_scope("grad_scaler.scale"):
            self.backend.mark_output({"grad_scalar": self.state_dict()})

            if (not self.dynamic) and self._loss_scale == 1.0:
                # Mark optimizers has having been unscaled since there is
                # no scaling to be done
                for optimizer in self.backend.optimizer_registry:
                    # pylint: disable=protected-access
                    optimizer._amp_stash.state = OptState.UNSCALED

                return loss.float()

            for optimizer in self.backend.optimizer_registry:
                # pylint: disable=protected-access
                if optimizer._amp_stash.state == OptState.READY:
                    optimizer._prepare_amp_backward()
                    optimizer._amp_stash.state = OptState.SCALED
                    continue
                if optimizer._amp_stash.state != OptState.SCALED:
                    raise RuntimeError(
                        "Optimizer parameter gradients already scaled"
                    )

            return (loss.float()) * self._loss_scale

    def get_scale(self):
        """Return the loss scale"""
        return self._loss_scale

    def _unscale_helper(self, model_grads, master_grads, scale):
        for model, master in zip(model_grads, master_grads):
            if model is not None:
                if (
                    master is not model
                ):  # copy_ probably internally short-circuits this
                    master.copy_(model)

        if not self.dynamic and scale == 1.0:
            return

        if not GradScaler.warned_unscaling_non_fp32_grad:
            for master in master_grads:
                if master.dtype != torch.float32:
                    maybe_print(
                        f"Attempting to unscale a grad with type {master.type()} "
                        f"Unscaling non-fp32 grads may indicate an error. "
                        f"When using Amp, you don't need to call .half() on your model."
                    )
                    GradScaler.warned_unscaling_non_fp32_grad = True

        if self.dynamic:
            inv_scale = torch.tensor(1.0, dtype=torch.float32) / scale
        else:
            inv_scale = torch.tensor(1.0 / scale, dtype=torch.float32)

        for master in master_grads:
            master.mul_(inv_scale)

        if self.dynamic:
            # Use CS1 compatible algorithm for detcting NaN/inf by using global
            # L2 norm of all gradients
            norms_squared = [torch.sum(g * g) for g in master_grads]
            self._squared_local_norms.extend(norms_squared)

    def _unscale(
        self,
        model_grads,
        master_grads,
        unused_scale,
        models_are_masters=False,
        scale_override=None,
    ):
        # implementation
        scale = self._loss_scale
        if scale_override is not None:
            scale = scale_override

        if self.dynamic or not models_are_masters or scale != 1.0:
            self._unscale_helper(model_grads, master_grads, scale)

    def _unscale_with_stashed_python(
        self, model_grads, stashed_master_grads, master_grads, a, b
    ):  # pylint: disable=missing-function-docstring
        raise NotImplementedError("stashed grads not supported")

    def _unscale_with_stashed(
        self,
        model_grads,
        stashed_master_grads,
        master_grads,
        scale_override=None,
    ):  # pylint: disable=missing-function-docstring
        raise NotImplementedError("stashed grads not supported")

    def unscale_(self, optimizer):
        """Unscales the optimizer's params gradients inplace"""
        # Go unscale all the gradients
        # pylint: disable=protected-access
        if optimizer._amp_stash.state == OptState.UNSCALED:
            return  # no-op
        elif optimizer._amp_stash.state == OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # if not dynamic, short circuit to match the implicit context manager case
        if (not self.dynamic) and self._loss_scale == 1.0:
            optimizer._amp_stash.state = OptState.UNSCALED
            return

        with self.backend.name_scope("grad_scaler.unscale_"):
            optimizer._post_amp_backward(self)
        optimizer._amp_stash.params_have_scaled_gradients = False
        optimizer._amp_stash.state = OptState.UNSCALED

    def step_if_finite(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
        """
        Directly conditionalize the call to optimizer.step(*args, **kwargs) but
        only if this GradScaler detected finite grads.

        Args:
            optimizer: Optimizer that applies the gradients.
            args: Any arguments passed to the optimizer.step() call.
            kwargs: Any keyword arguments passed to the optimizer.step() call.

        Returns:
            The result of optimizer.step()
        """
        if self.dynamic:
            dls_update_manager: ConditionalUpdateManager = (
                optimizer._amp_stash.dls_update_manager
            )
            dls_update_manager.set_condition(self.isfinite)
            with dls_update_manager, amp.disable_casts():
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        dls_update_manager.mark_tensor(p)
                        for state in optimizer.state[p].values():
                            dls_update_manager.mark_tensor(state)
                return optimizer.step(*args, **kwargs)
        else:
            with amp.disable_casts():
                return optimizer.step(*args, **kwargs)

    def clip_gradients_and_return_isfinite(self, optimizers):
        """
        Clip the optimizer's params's gradients and return whether or not the
        norm is finite
        """
        # Compute gloal norm from all squared local norms
        # if not self.global_norm:
        self.global_norm = torch.sqrt(
            torch.sum(torch.stack(self._squared_local_norms))
        )

        def float32(value):
            return torch.tensor(
                value, dtype=torch.float32, device=self.global_norm.device
            )

        # self.isfinite = torch.isfinite(self.global_norm)
        # TODO: torch.isfinite^ hits a lowering error! so use:
        self.isfinite = self.global_norm < float32(float("inf"))

        if self._max_gradient_norm:
            # Then we're doing combo GGC + DLS
            # https://github.com/pytorch/pytorch/blob/release/1.9/torch/nn/utils/clip_grad.py#L56-L59
            clip_coef = float32(self._max_gradient_norm) / (
                self.global_norm + 1e-6
            )
            clip_coef = torch.where(
                clip_coef < 1,
                clip_coef,
                1.0,
            )
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        p.grad.detach().mul_(clip_coef)

        return self.isfinite

    def step(self, optimizer, *args, **kwargs):
        """
        `Step` carries out the following two operations:
        1.  Internally invokes ``unscale_(optimizer)`` (unless `unscale_` was
            explicitly called for ``optimizer`` earlier in the iteration).  As
            part of the `unscale_`, gradients are checked for infs/NaNs.
        2.  Invokes ``optimizer.step()`` using the unscaled gradients. Ensure
            that previous optimizer state or params carry over if we encounter
            NaNs in the gradients.
        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.
        Returns the return value of ``optimizer.step(*args, **kwargs)``.
        Args:
            optimizer (cerebras.pytorch.optim.Optimizer):
                Optimizer that applies the gradients.
            args:
                Any arguments.
            kwargs:
                Any keyword arguments.
        """
        # pylint: disable=protected-access
        if optimizer._amp_stash.state == OptState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )

        # must unscale all optimizers prior to step and update
        # so global grad norm can be computed correctly
        # guaranteed to also unscale the given optimizer if needed
        for _optimizer in self.backend.optimizer_registry:
            if _optimizer._amp_stash.state == OptState.READY:
                self.unscale_(_optimizer)

        with self.backend.name_scope("grad_scaler.step"):
            if self.dynamic and self.isfinite is None:
                self.isfinite = self.clip_gradients_and_return_isfinite(
                    self.backend.optimizer_registry
                )

                optimizer._amp_stash.isfinite = self.isfinite

            # Run optimizer's base step if self.isfinite is true
            return_val = self.step_if_finite(optimizer, *args, **kwargs)
            optimizer._amp_stash.state = OptState.STEPPED

        return return_val

    def update_scale(self, optimizers):
        """Update the scales of the optimizers"""
        if not self.dynamic:
            return

        # Compute gloal norm from all squared local norms
        # if not self.global_norm:
        self.global_norm = torch.sqrt(
            torch.sum(torch.stack(self._squared_local_norms))
        )

        def float32(value):
            return torch.tensor(
                value, dtype=torch.float32, device=self.global_norm.device
            )

        def int64(value):
            return torch.tensor(
                value, dtype=torch.int64, device=self.global_norm.device
            )

        # Reset local norms for next iteration
        self._squared_local_norms = []

        # integer representation of isfinite
        isfinite_int = self.isfinite.long()

        # Increment the step counter
        self._steps_since_rescale.add_(1)

        # If overflow, increment the overflow counter
        self._overflows_since_rescale.add_(1 - isfinite_int)

        ratio = (
            self._overflows_since_rescale.float()
            / self._steps_since_rescale.float()
        )

        # Decrease loss scale

        # decrease loss scaling condition
        # 1 if we've exceeded our overflow tolerance
        # 0 if we haven't hit too many overflows
        overflow_tolerance_exceeded = (
            float32(self._overflow_tolerance) < ratio
        ).long()
        # decrease loss scale 2x if we're decreasing, otherwise unchanged
        loss_scale_divisor = (1 + overflow_tolerance_exceeded).float()
        self._loss_scale.div_(loss_scale_divisor)
        # reset counters
        reset_because_decreasing = 1 - overflow_tolerance_exceeded
        self._overflows_since_rescale.mul_(reset_because_decreasing)
        self._steps_since_rescale.mul_(reset_because_decreasing)

        # Increasing loss scale
        # (done purposefully after decrease logic in case counter reset)

        # increase loss scaling condition
        # 1 if we've exceeded our steps per increase counter
        # 0 if we haven't yet.
        increase_counter_exceeded = (
            int64(self._steps_per_increase) < self._steps_since_rescale
        ).long()
        # increase loss scale 2x if we're increasing, otherwise unchanged
        loss_scale_multipler = (1 + increase_counter_exceeded).float()
        self._loss_scale.mul_(loss_scale_multipler)
        # reset counters
        reset_because_increasing = 1 - increase_counter_exceeded
        self._overflows_since_rescale.mul_(reset_because_increasing)
        self._steps_since_rescale.mul_(reset_because_increasing)

        # clamp loss scale to within min/max
        max_ls = float32(self._max_loss_scale)
        self._loss_scale.copy_(
            torch.where(
                self._loss_scale < max_ls,
                self._loss_scale,
                max_ls,
            )
        )
        min_ls = float32(self._min_loss_scale)
        self._loss_scale.copy_(
            torch.where(
                min_ls < self._loss_scale,
                self._loss_scale,
                min_ls,
            )
        )

    def update(self, new_scale=None):
        """Update the gradient scalar after all optimizers have been stepped"""
        if new_scale:
            raise ValueError(
                "cstorch.amp.GradScaler does not support providing a `new_scale`"
            )

        # Update scale
        if self.dynamic or self._loss_scale != 1.0:
            with self.backend.name_scope("grad_scaler.update"):
                self.update_scale(self.backend.optimizer_registry)

        # pylint: disable=protected-access,no-member
        _amp_state.handle._clear_cache()

        # clear all data from this iteration for the next
        self.isfinite = None
        for optimizer in self.backend.optimizer_registry:
            optimizer._amp_stash.state = OptState.READY
