"""Generic parameter-group scheduler implementation."""

from __future__ import annotations

import math
from bisect import bisect_right


class Scheduler:
    param_group_key = "value"

    def __init__(self, optimizer=None, initial_val=0.0, total_iters=1, last_epoch=-1, **kwargs):
        self.optimizer = optimizer
        self.initial_val = initial_val
        self.last_epoch = last_epoch
        self.total_iters = max(1, int(total_iters or kwargs.pop("total_steps", 1) or 1))
        self.last_value = initial_val
        self.kwargs = kwargs

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "last_value": self.last_value,
            "initial_val": self.initial_val,
            "total_iters": self.total_iters,
            "kwargs": dict(self.kwargs),
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict.get("last_epoch", self.last_epoch)
        self.last_value = state_dict.get("last_value", self.last_value)
        self.initial_val = state_dict.get("initial_val", self.initial_val)
        self.total_iters = state_dict.get("total_iters", self.total_iters)
        self.kwargs.update(state_dict.get("kwargs", {}))

    def increment_last_epoch(self):
        self.last_epoch += 1

    def _get_closed_form(self):
        return self.get()

    def get(self):
        return self.initial_val

    def step(self, *args, **kwargs):
        self.increment_last_epoch()
        self.update_last_value()
        self.update_groups(self.last_value)
        return self.last_value

    def update_last_value(self):
        self.last_value = self.get()

    def update_groups(self, values):
        if self.optimizer is None or not hasattr(self.optimizer, "param_groups"):
            return
        vals = values if isinstance(values, (list, tuple)) else [values] * len(self.optimizer.param_groups)
        for group, value in zip(self.optimizer.param_groups, vals):
            group[self.param_group_key] = value

    def get_last_value(self):
        return self.last_value


class _LinearSchedule(Scheduler):
    def __init__(self, optimizer=None, initial_val=0.0, end_val=0.0, total_iters=1, **kwargs):
        super().__init__(optimizer, initial_val=initial_val, total_iters=total_iters, **kwargs)
        self.end_val = end_val

    def get(self):
        pct = min(1.0, max(0.0, (self.last_epoch + 1) / self.total_iters))
        return self.initial_val + (self.end_val - self.initial_val) * pct


class _ConstantSchedule(Scheduler):
    def __init__(self, optimizer=None, val=None, initial_val=None, **kwargs):
        value = val if val is not None else initial_val
        super().__init__(optimizer, initial_val=value if value is not None else 0.0, **kwargs)
        self.val = self.initial_val

    def get(self):
        return self.val


class _ExponentialSchedule(Scheduler):
    def __init__(self, optimizer=None, initial_val=1.0, gamma=0.99, **kwargs):
        super().__init__(optimizer, initial_val=initial_val, **kwargs)
        self.gamma = gamma

    def get(self):
        return self.initial_val * (self.gamma ** max(0, self.last_epoch + 1))


class _StepSchedule(Scheduler):
    def __init__(self, optimizer=None, initial_val=1.0, step_size=1, gamma=0.1, **kwargs):
        super().__init__(optimizer, initial_val=initial_val, **kwargs)
        self.step_size = max(1, int(step_size))
        self.gamma = gamma

    def get(self):
        return self.initial_val * (self.gamma ** ((self.last_epoch + 1) // self.step_size))


class _MultiStepSchedule(Scheduler):
    def __init__(self, optimizer=None, initial_val=1.0, milestones=(), gamma=0.1, **kwargs):
        super().__init__(optimizer, initial_val=initial_val, **kwargs)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get(self):
        return self.initial_val * (self.gamma ** bisect_right(self.milestones, self.last_epoch + 1))


class _CosineSchedule(Scheduler):
    def __init__(self, optimizer=None, initial_val=1.0, end_val=0.0, total_iters=1, **kwargs):
        super().__init__(optimizer, initial_val=initial_val, total_iters=total_iters, **kwargs)
        self.end_val = end_val

    def get(self):
        pct = min(1.0, max(0.0, (self.last_epoch + 1) / self.total_iters))
        return self.end_val + 0.5 * (self.initial_val - self.end_val) * (1 + math.cos(math.pi * pct))


class _PiecewiseConstantSchedule(Scheduler):
    def __init__(self, optimizer=None, values=None, milestones=None, **kwargs):
        vals = list(values or [kwargs.pop("initial_val", 0.0)])
        super().__init__(optimizer, initial_val=vals[0], **kwargs)
        self.values = vals
        self.milestones = list(milestones or [])

    def get(self):
        idx = bisect_right(self.milestones, self.last_epoch + 1)
        return self.values[min(idx, len(self.values) - 1)]


class _PolynomialSchedule(Scheduler):
    def __init__(self, optimizer=None, initial_val=1.0, end_val=0.0, power=1.0, total_iters=1, **kwargs):
        super().__init__(optimizer, initial_val=initial_val, total_iters=total_iters, **kwargs)
        self.end_val = end_val
        self.power = power

    def get(self):
        pct = min(1.0, max(0.0, (self.last_epoch + 1) / self.total_iters))
        return self.end_val + (self.initial_val - self.end_val) * ((1 - pct) ** self.power)


class _LambdaSchedule(Scheduler):
    def __init__(self, optimizer=None, initial_val=1.0, lr_lambda=None, **kwargs):
        super().__init__(optimizer, initial_val=initial_val, **kwargs)
        self.lr_lambda = lr_lambda or (lambda step: 1.0)

    def get(self):
        return self.initial_val * self.lr_lambda(self.last_epoch + 1)


class _ChainedSchedule(Scheduler):
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self, *args, **kwargs):
        value = None
        for scheduler in self.schedulers:
            value = scheduler.step(*args, **kwargs)
        return value

    def state_dict(self):
        return {"schedulers": [scheduler.state_dict() for scheduler in self.schedulers]}


class _SequentialSchedule(_ChainedSchedule):
    pass
