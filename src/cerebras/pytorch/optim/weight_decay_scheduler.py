"""Weight-decay scheduler aliases."""

from __future__ import annotations

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
    MultiStepLR,
    MultiplicativeLR,
    OneCycleLR,
    PiecewiseConstantLR,
    PolynomialLR,
    SequentialLR,
    StepLR,
)


class WeightDecayScheduler:
    param_group_key = "weight_decay"


def _wd(name, base):
    return type(name, (base, WeightDecayScheduler), {"param_group_key": "weight_decay"})


ConstantWD = _wd("ConstantWD", ConstantLR)
LinearWD = _wd("LinearWD", LinearLR)
ExponentialWD = _wd("ExponentialWD", ExponentialLR)
StepWD = _wd("StepWD", StepLR)
MultiStepWD = _wd("MultiStepWD", MultiStepLR)
CosineDecayWD = _wd("CosineDecayWD", CosineDecayLR)
CosineAnnealingWD = _wd("CosineAnnealingWD", CosineAnnealingLR)
CosineAnnealingWarmRestartsWD = _wd("CosineAnnealingWarmRestartsWD", CosineAnnealingWarmRestarts)
PolynomialWD = _wd("PolynomialWD", PolynomialLR)
PiecewiseConstantWD = _wd("PiecewiseConstantWD", PiecewiseConstantLR)
LambdaWD = _wd("LambdaWD", LambdaLR)
MultiplicativeWD = _wd("MultiplicativeWD", MultiplicativeLR)
InverseExponentialTimeDecayWD = _wd("InverseExponentialTimeDecayWD", InverseExponentialTimeDecayLR)
InverseSquareRootDecayWD = _wd("InverseSquareRootDecayWD", InverseSquareRootDecayLR)
CyclicWD = _wd("CyclicWD", CyclicLR)
OneCycleWD = _wd("OneCycleWD", OneCycleLR)
ChainedWD = _wd("ChainedWD", ChainedScheduler)
SequentialWD = _wd("SequentialWD", SequentialLR)
