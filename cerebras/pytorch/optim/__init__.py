# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from cerebras.pytorch.optim.Adadelta import Adadelta
from cerebras.pytorch.optim.Adafactor import Adafactor
from cerebras.pytorch.optim.Adagrad import Adagrad
from cerebras.pytorch.optim.AdamBase import Adam, AdamBase, AdamW
from cerebras.pytorch.optim.Adamax import Adamax
from cerebras.pytorch.optim.ASGD import ASGD
from cerebras.pytorch.optim.Lamb import Lamb
from cerebras.pytorch.optim.Lion import Lion
from cerebras.pytorch.optim.lr_scheduler import (
    ChainedLR,
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
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PiecewiseConstantLR,
    PolynomialLR,
    ScalePerParamLR,
    SequentialLR,
    StepLR,
)
from cerebras.pytorch.optim.NAdam import NAdam
from cerebras.pytorch.optim.optimizer import Optimizer
from cerebras.pytorch.optim.RAdam import RAdam
from cerebras.pytorch.optim.RMSprop import RMSprop
from cerebras.pytorch.optim.Rprop import Rprop
from cerebras.pytorch.optim.scheduler import (
    ChainedScheduler as BaseChainedScheduler,
    ConstantScheduler,
    CosineAnnealingScheduler,
    CosineAnnealingWarmRestartsScheduler,
    CosineDecayScheduler,
    CyclicScheduler,
    ExponentialScheduler,
    InverseExponentialTimeDecayScheduler,
    InverseSquareRootDecayScheduler,
    LambdaScheduler,
    LinearScheduler,
    MultiplicativeScheduler,
    MultiStepScheduler,
    OneCycleScheduler,
    PiecewiseConstantScheduler,
    PolynomialScheduler,
    ScalePerParamScheduler,
    Scheduler,
    SequentialScheduler,
    StepScheduler,
)
from cerebras.pytorch.optim.SGD import SGD
from cerebras.pytorch.optim.weight_decay_scheduler import (
    ConstantWD,
    CosineAnnealingWD,
    CosineDecayWD,
    ExponentialWD,
    InverseExponentialTimeDecayWD,
    InverseSquareRootDecayWD,
    LinearWD,
    MultiplicativeWD,
    MultiStepWD,
    OneCycleWD,
    PiecewiseConstantWD,
    PolynomialWD,
    ScalePerParamWD,
    SequentialWD,
    StepWD,
    WDScheduler,
)
