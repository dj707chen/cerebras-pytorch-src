"""Schedules, score shapers, and mask helpers for sparsity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .._compat import require_torch


class HyperParameterSchedule:
    def __init__(self, init=0.0, *args, **kwargs):
        self.init = init
        self.step = 0
        self.kwargs = kwargs

    def compute(self):
        return self.init

    def update(self):
        self.step += 1
        return self.compute()

    def visit_state(self, fn):
        self.init = fn(self.init)
        return self

    def get_min_max_end(self):
        value = self.compute()
        return value, value, value


class Constant(HyperParameterSchedule):
    pass


class Linear(HyperParameterSchedule):
    def __init__(self, init=0.0, end=1.0, steps=1, **kwargs):
        super().__init__(init, **kwargs)
        self.end = end
        self.steps = max(1, int(steps))

    def compute(self):
        pct = min(1.0, self.step / self.steps)
        return self.init + (self.end - self.init) * pct


class Exp(HyperParameterSchedule):
    def __init__(self, init=1.0, gamma=0.99, **kwargs):
        super().__init__(init, **kwargs)
        self.gamma = gamma

    def compute(self):
        return self.init * (self.gamma ** self.step)


class Power(HyperParameterSchedule):
    def __init__(self, init=1.0, power=1.0, **kwargs):
        super().__init__(init, **kwargs)
        self.power = power

    def compute(self):
        return self.init * ((self.step + 1) ** self.power)


class Cosine(HyperParameterSchedule):
    def __init__(self, init=1.0, end=0.0, half_period=1, **kwargs):
        super().__init__(init, **kwargs)
        self.end = end
        self.half_period = max(1, int(half_period))

    def compute(self):
        pct = min(1.0, self.step / self.half_period)
        return self.end + 0.5 * (self.init - self.end) * (1 + math.cos(math.pi * pct))


class Cycling(Cosine):
    def compute(self):
        self.step %= self.half_period
        return super().compute()


class Lambda(HyperParameterSchedule):
    def __init__(self, fn, init=1.0):
        super().__init__(init)
        self.fn = fn

    def compute(self):
        return self.fn(self.step)


class FreqSchedule(HyperParameterSchedule):
    def __init__(self, freq=1, stop=None, **kwargs):
        super().__init__(False, **kwargs)
        self.freq = max(1, int(freq))
        self.stop = stop

    def compute(self):
        if self.stop is not None and self.step > self.stop:
            return False
        return self.step % self.freq == 0


class ListSchedule(HyperParameterSchedule):
    def __init__(self, values):
        super().__init__(values[0] if values else 0.0)
        self.values = list(values)

    def compute(self):
        if not self.values:
            return 0.0
        return self.values[min(self.step, len(self.values) - 1)]


def make_hyperparam_schedule(value):
    if isinstance(value, HyperParameterSchedule):
        return value
    if callable(value):
        return Lambda(value)
    if isinstance(value, (list, tuple)):
        return ListSchedule(value)
    if isinstance(value, dict):
        schedule_type = value.get("type", "constant").lower()
        args = {k: v for k, v in value.items() if k != "type"}
        return {
            "constant": Constant,
            "linear": Linear,
            "exp": Exp,
            "power": Power,
            "cosine": Cosine,
            "cycling": Cycling,
        }.get(schedule_type, Constant)(**args)
    return Constant(value)


def make_update_schedule(value):
    if isinstance(value, HyperParameterSchedule):
        return value
    if isinstance(value, dict):
        return FreqSchedule(**value)
    if value is None:
        return Constant(True)
    return make_hyperparam_schedule(value)


class ScoreFlattener:
    def __call__(self, scores):
        return scores.flatten()


class InputGroupScoreShaper:
    def __init__(self, groups):
        self.groups = groups

    def __call__(self, scores):
        return scores


class OutputGroupScoreShaper(InputGroupScoreShaper):
    pass


def make_mask_topk_sparsity(scores, sparsity):
    torch = require_torch()
    flat = scores.flatten()
    keep = max(0, int(round(flat.numel() * (1.0 - float(sparsity)))))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if keep:
        mask[flat.topk(keep).indices] = True
    return mask.reshape(scores.shape)


def make_mask_drop_minimum(scores, mask, drop_fraction):
    active_scores = scores.masked_fill(~mask.bool(), float("inf"))
    return make_mask_topk_sparsity(-active_scores, 1.0 - float(drop_fraction))


def make_mask_grow_maximum(scores, mask, grow_fraction):
    inactive_scores = scores.masked_fill(mask.bool(), float("-inf"))
    return mask.bool() | make_mask_topk_sparsity(inactive_scores, 1.0 - float(grow_fraction))
