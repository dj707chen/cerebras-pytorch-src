"""Automatic mixed precision helpers."""

from __future__ import annotations

from typing import Any, Optional

from ._compat import optional_torch, require_torch

_HALF_DTYPE = "float16"


def set_half_dtype(value):
    """Set the preferred 16-bit floating dtype and return its torch proxy dtype."""

    global _HALF_DTYPE
    torch = optional_torch()
    if isinstance(value, str):
        key = value.lower()
        if key not in {"float16", "bfloat16", "cbfloat16"}:
            raise ValueError("half dtype must be 'float16', 'bfloat16', or 'cbfloat16'")
        _HALF_DTYPE = key
        if torch is None:
            return key
        return torch.bfloat16 if key in {"bfloat16", "cbfloat16"} else torch.float16
    _HALF_DTYPE = str(value)
    return value


class GradScaler:
    """CPU/GPU-friendly gradient scaler facade.

    When torch AMP is available this delegates to ``torch.cuda.amp.GradScaler``.
    On CPU-only runs it behaves as an enabled no-op scaler.
    """

    def __init__(
        self,
        loss_scale=None,
        init_scale=None,
        steps_per_increase=None,
        min_loss_scale=None,
        max_loss_scale=None,
        overflow_tolerance=0.0,
        max_gradient_norm=None,
    ):
        self.loss_scale = loss_scale
        self.init_scale = init_scale if init_scale is not None else 1.0
        self.steps_per_increase = steps_per_increase
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        self.overflow_tolerance = overflow_tolerance
        self.max_gradient_norm = max_gradient_norm
        self._scale = float(self.init_scale)
        torch = optional_torch()
        self._impl = None
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                self._impl = torch.cuda.amp.GradScaler(init_scale=self._scale)
            except Exception:
                self._impl = None

    def state_dict(self, destination=None):
        state = self._impl.state_dict() if self._impl is not None else {"scale": self._scale}
        if destination is not None:
            destination.update(state)
            return destination
        return state

    def load_state_dict(self, state_dict):
        if self._impl is not None:
            return self._impl.load_state_dict(state_dict)
        self._scale = float(state_dict.get("scale", self._scale))
        return None

    def scale(self, loss):
        if self._impl is not None:
            return self._impl.scale(loss)
        return loss * self._scale if self.loss_scale not in (None, 1, 1.0) else loss

    def get_scale(self):
        if self._impl is not None:
            return self._impl.get_scale()
        return self._scale

    def unscale_(self, optimizer):
        if self._impl is not None:
            return self._impl.unscale_(optimizer)
        return optimizer

    def step_if_finite(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def clip_gradients_and_return_isfinite(self, optimizers):
        torch = require_torch()
        opts = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        params = [p for opt in opts for group in opt.param_groups for p in group.get("params", []) if getattr(p, "grad", None) is not None]
        if self.max_gradient_norm is not None and params:
            torch.nn.utils.clip_grad_norm_(params, self.max_gradient_norm)
        return torch.tensor(True)

    def step(self, optimizer, *args, **kwargs):
        if self._impl is not None:
            return self._impl.step(optimizer, *args, **kwargs)
        return optimizer.step(*args, **kwargs)

    def update_scale(self, optimizers):
        return self.update()

    def update(self, new_scale: Optional[Any] = None):
        if self._impl is not None:
            return self._impl.update(new_scale)
        if new_scale is not None:
            self._scale = float(new_scale)
        return None


def optimizer_step(loss, optimizer, grad_scaler, max_gradient_norm=None, max_gradient_value=None):
    scaled = grad_scaler.scale(loss) if grad_scaler is not None else loss
    if hasattr(scaled, "backward"):
        scaled.backward()
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)

    if max_gradient_norm is not None or max_gradient_value is not None:
        torch = require_torch()
        params = [p for group in optimizer.param_groups for p in group.get("params", []) if getattr(p, "grad", None) is not None]
        if max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, max_gradient_norm)
        if max_gradient_value is not None:
            torch.nn.utils.clip_grad_value_(params, max_gradient_value)

    result = grad_scaler.step(optimizer) if grad_scaler is not None else optimizer.step()
    if grad_scaler is not None:
        grad_scaler.update()
    return result
