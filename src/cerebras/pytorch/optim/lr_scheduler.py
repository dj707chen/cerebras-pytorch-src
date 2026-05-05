"""Learning-rate scheduler aliases."""

from __future__ import annotations

from .scheduler import (
    Scheduler,
    _ChainedSchedule,
    _ConstantSchedule,
    _CosineSchedule,
    _ExponentialSchedule,
    _LambdaSchedule,
    _LinearSchedule,
    _MultiStepSchedule,
    _PiecewiseConstantSchedule,
    _PolynomialSchedule,
    _SequentialSchedule,
    _StepSchedule,
)


class LRScheduler(Scheduler):
    param_group_key = "lr"

    def get_last_lr(self):
        return self.get_last_value()

    def get_lr(self):
        return self.get()


class ConstantLR(_ConstantSchedule, LRScheduler):
    param_group_key = "lr"


class LinearLR(_LinearSchedule, LRScheduler):
    param_group_key = "lr"


class ExponentialLR(_ExponentialSchedule, LRScheduler):
    param_group_key = "lr"


class StepLR(_StepSchedule, LRScheduler):
    param_group_key = "lr"


class MultiStepLR(_MultiStepSchedule, LRScheduler):
    param_group_key = "lr"


class CosineDecayLR(_CosineSchedule, LRScheduler):
    param_group_key = "lr"


class CosineAnnealingLR(_CosineSchedule, LRScheduler):
    param_group_key = "lr"


class CosineAnnealingWarmRestarts(_CosineSchedule, LRScheduler):
    param_group_key = "lr"


class PolynomialLR(_PolynomialSchedule, LRScheduler):
    param_group_key = "lr"


class PiecewiseConstantLR(_PiecewiseConstantSchedule, LRScheduler):
    param_group_key = "lr"


class LambdaLR(_LambdaSchedule, LRScheduler):
    param_group_key = "lr"


class MultiplicativeLR(_LambdaSchedule, LRScheduler):
    param_group_key = "lr"


class InverseExponentialTimeDecayLR(ExponentialLR):
    pass


class InverseSquareRootDecayLR(PolynomialLR):
    pass


class CyclicLR(CosineDecayLR):
    @property
    def base_val(self):
        return self.end_val

    @property
    def max_val(self):
        return self.initial_val


class OneCycleLR(CosineDecayLR):
    @property
    def max_val(self):
        return self.initial_val


class ChainedScheduler(_ChainedSchedule):
    pass


class SequentialLR(_SequentialSchedule):
    pass
