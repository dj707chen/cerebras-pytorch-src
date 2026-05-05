"""Optimizers and schedulers for the compatibility layer."""

from __future__ import annotations

from collections import OrderedDict

from .._compat import optional_torch

torch = optional_torch()


class _MissingTorchOptimizer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Optimizer classes require PyTorch. Install dependencies with `uv sync`.")


if torch is None:
    Optimizer = _MissingTorchOptimizer
else:

    class _CerebrasMixin:
        def preinitialize(self):
            return None

        def apply(self, f):
            f(self)
            return self

        def visit_state(self, fn):
            for state in self.state.values():
                for key, value in list(state.items()):
                    state[key] = fn(value)
            return self

        def increment_global_step(self, p=None):
            self.global_step = getattr(self, "global_step", 0) + 1
            return self.global_step

    class Optimizer(_CerebrasMixin, torch.optim.Optimizer):
        pass


def _make_builtin(name, fallback="AdamW"):
    if torch is None:
        return type(name, (_MissingTorchOptimizer,), {})
    base = getattr(torch.optim, name, getattr(torch.optim, fallback))

    class Wrapped(_CerebrasMixin, base):
        pass

    Wrapped.__name__ = name
    Wrapped.__qualname__ = name
    return Wrapped


Adadelta = _make_builtin("Adadelta")
Adagrad = _make_builtin("Adagrad")
Adam = _make_builtin("Adam")
AdamW = _make_builtin("AdamW")
Adamax = _make_builtin("Adamax")
ASGD = _make_builtin("ASGD")
NAdam = _make_builtin("NAdam", "Adam")
RAdam = _make_builtin("RAdam", "Adam")
RMSprop = _make_builtin("RMSprop")
Rprop = _make_builtin("Rprop")
SGD = _make_builtin("SGD")
Adafactor = _make_builtin("Adafactor", "AdamW")


if torch is None:
    Lamb = _MissingTorchOptimizer
    Lion = _MissingTorchOptimizer
else:

    class Lamb(_CerebrasMixin, torch.optim.AdamW):
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False):
            self.adam = adam
            super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    class Lion(_CerebrasMixin, torch.optim.AdamW):
        def __init__(self, params, lr=0.0001, betas=(0.9, 0.99), weight_decay=0.0):
            super().__init__(params, lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay)


from . import lr_scheduler, scheduler, weight_decay_scheduler
from .lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CosineDecayLR,
    CyclicLR,
    ExponentialLR,
    InverseExponentialTimeDecayLR,
    InverseSquareRootDecayLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    MultiplicativeLR,
    OneCycleLR,
    PiecewiseConstantLR,
    PolynomialLR,
    SequentialLR,
    StepLR,
)
from .scheduler import Scheduler
from .weight_decay_scheduler import (
    ChainedWD,
    ConstantWD,
    CosineAnnealingWD,
    CosineAnnealingWarmRestartsWD,
    CosineDecayWD,
    CyclicWD,
    ExponentialWD,
    InverseExponentialTimeDecayWD,
    InverseSquareRootDecayWD,
    LambdaWD,
    LinearWD,
    MultiStepWD,
    MultiplicativeWD,
    OneCycleWD,
    PiecewiseConstantWD,
    PolynomialWD,
    SequentialWD,
    StepWD,
    WeightDecayScheduler,
)

__all__ = [
    "ASGD",
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "ChainedScheduler",
    "ChainedWD",
    "ConstantLR",
    "ConstantWD",
    "CosineAnnealingLR",
    "CosineAnnealingWD",
    "CosineAnnealingWarmRestarts",
    "CosineAnnealingWarmRestartsWD",
    "CosineDecayLR",
    "CosineDecayWD",
    "CyclicLR",
    "CyclicWD",
    "ExponentialLR",
    "ExponentialWD",
    "InverseExponentialTimeDecayLR",
    "InverseExponentialTimeDecayWD",
    "InverseSquareRootDecayLR",
    "InverseSquareRootDecayWD",
    "LRScheduler",
    "LambdaLR",
    "LambdaWD",
    "Lamb",
    "LinearLR",
    "LinearWD",
    "Lion",
    "MultiStepLR",
    "MultiStepWD",
    "MultiplicativeLR",
    "MultiplicativeWD",
    "NAdam",
    "OneCycleLR",
    "OneCycleWD",
    "Optimizer",
    "PiecewiseConstantLR",
    "PiecewiseConstantWD",
    "PolynomialLR",
    "PolynomialWD",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "Scheduler",
    "SequentialLR",
    "SequentialWD",
    "StepLR",
    "StepWD",
    "WeightDecayScheduler",
    "lr_scheduler",
    "scheduler",
    "weight_decay_scheduler",
]
