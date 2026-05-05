# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Implementations for Cerebras specific weight decay schedulers"""

from typing import List, Optional, Union

import torch

from .scheduler import (
    ChainedScheduler,
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


class WeightDecayScheduler(Scheduler):
    @property
    def param_group_key(self):
        return "weight_decay"


class ConstantWD(WeightDecayScheduler, ConstantScheduler):
    """Maintains a constant weight decay for each parameter group (no decaying).

    Args:
        optimizer: The optimizer to schedule
        val: The weight decay value to maintain
        total_iters: The number of steps to decay for
    """


class PolynomialWD(WeightDecayScheduler, PolynomialScheduler):
    r"""Decays the weight decay of each parameter group using a polynomial function
    in the given `total_iters`.

    This class is similar to the `Pytorch PolynomialLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay
        end_val: The final weight decay
        total_iters: Number of steps to perform the decay
        power: Exponent to apply to "x" (as in y=mx+b),
            which is ratio of step completion (1 for linear)
            Default: 1.0 (only Linear supported at the moment)
        cycle: Whether to cycle

    .. _Pytorch PolynomialLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR
    """


class LinearWD(WeightDecayScheduler, LinearScheduler):
    """Alias for Polynomial Scheduler scheduler with a power of 1"""


class ExponentialWD(WeightDecayScheduler, ExponentialScheduler):
    r"""Decays the weight decay of each parameter group by `decay_rate` every step.

    This class is similar to the `Pytorch ExponentialLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay.
        total_iters: Number of steps to perform the decay
        decay_rate: The decay rate
        staircase: If True decay the weight decay at discrete intervals

    .. _Pytorch ExponentialLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR
    """


class InverseExponentialTimeDecayWD(
    WeightDecayScheduler, InverseExponentialTimeDecayScheduler
):
    r"""Decays the weight decay inverse-exponentially over time, as described
    in the `Keras InverseTimeDecay class`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay.
        step_exponent: Exponential weight decay.
        total_iters: Number of steps to perform the decay.
        decay_rate: The decay rate.
        staircase: If True decay the weight decay at discrete intervals.

    .. _Keras InverseTimeDecay class:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay
    """


class InverseSquareRootDecayWD(
    WeightDecayScheduler, InverseSquareRootDecayScheduler
):
    r"""Decays the weight decay inverse-squareroot over time, as described
    in the following equation:

    .. math::
        \begin{aligned}
            wd_t & = \frac{\text{scale}}{\sqrt{\max\{t, \text{warmup_steps}\}}}.
        \end{aligned}

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay.
        scale: Multiplicative factor to scale the result.
        warmup_steps: use initial_val for the first warmup_steps.
    """


class CosineDecayWD(WeightDecayScheduler, CosineDecayScheduler):
    r"""Applies the cosine decay schedule as described
    in the `Keras CosineDecay class`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay
        end_val: The final weight decay
        total_iters: Number of steps to perform the decay

    .. _Keras CosineDecay class:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
    """


class SequentialWD(WeightDecayScheduler, SequentialScheduler):
    r"""Receives the list of schedulers that is expected to be called sequentially
    during optimization process and milestone points that provides exact
    intervals to reflect which scheduler is supposed to be called at a given
    step.

    This class is similar to `Pytorch SequentialLR LRS`_.

    Args:
        optimizer: Wrapped optimizer
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _Pytorch SequentialLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR
    """


class PiecewiseConstantWD(WeightDecayScheduler, PiecewiseConstantScheduler):
    r"""Adjusts the weight decay to a predefined constant at each milestone and
    holds this value until the next milestone. Notice that such adjustment can
    happen simultaneously with other changes to the weight decays from outside
    this scheduler.

    Args:
        optimizer: The optimizer to schedule
        vals: List of weight decays to maintain before/during each
            milestone.
        milestones: List of step indices. Must be increasing.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        vals: List[float],
        milestones: List[int],
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            ConstantWD,
            vals=vals,
            milestones=milestones,
            param_group_tags=param_group_tags,
        )


class MultiStepWD(WeightDecayScheduler, MultiStepScheduler):
    r"""Decays the weight decay of each parameter group by gamma once the number of
    steps reaches one of the milestones. Notice that such decay can happen
    simultaneously with other changes to the weight decay from outside this
    scheduler.

    This class is similar to the `Pytorch MultiStepLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay.
        gamma: Multiplicative factor of weight decay decay.
        milestones: List of step indices. Must be increasing.

    .. _Pytorch MultiStepLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR
    """


class StepWD(WeightDecayScheduler, StepScheduler):
    r"""Decays the weight decay of each parameter group by gamma every `step_size`.
    Notice that such decay can happen simultaneously with other changes to the
    weight decay from outside this scheduler.

    This class is similar to the `Pytorch StepLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial val.
        step_size: Period of decay.
        gamma: Multiplicative factor of decay.

    .. _Pytorch StepLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
    """


class CosineAnnealingWD(WeightDecayScheduler, CosineAnnealingScheduler):
    r"""Set the weight decay of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial wd and
    :math:`T_{cur}` is the number of steps since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    Notice that because the schedule is defined recursively, the weight decay
    can be simultaneously modified outside this scheduler by other operators.
    If the weight decay is set solely by this scheduler, the weight decay at
    each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    This class is similar to the `Pytorch CosineAnnealingLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay.
        T_max: Maximum number of iterations.
        eta_min: Minimum weight decay.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    .. _Pytorch CosineAnnealingLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR
    """


class LambdaWD(WeightDecayScheduler, LambdaScheduler):
    r"""Sets the weight decay of each parameter group to the initial wd times a
    given function (which is specified by overriding `set_value_lambda`).

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay.
    """


class CosineAnnealingWarmRestartsWD(
    WeightDecayScheduler, CosineAnnealingWarmRestartsScheduler
):
    r"""Set the weight decay of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial wd, :math:`T_{cur}`
    is the number of steps since the last restart and :math:`T_{i}` is the number
    of steps between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    This class is similar to the `Pytorch CosineAnnealingWarmRestarts LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay.
        T_0: Number of iterations for the first restart.
        T_mult: A factor increases Ti after a restart. Currently T_mult must be
            set to 1.0
        eta_min: Minimum weight decay.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    .. _Pytorch CosineAnnealingWarmRestarts LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    """


class MultiplicativeWD(WeightDecayScheduler, MultiplicativeScheduler):
    r"""Multiply the weight decay of each parameter group by the supplied
    coefficient.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial weight decay.
        coefficient: Multiplicative factor of weight decay.
    """


class ChainedWD(WeightDecayScheduler, ChainedScheduler):
    r"""Chains list of weight decay schedulers.
    It takes a list of chainable weight decay schedulers and
    performs consecutive step() functions belonging to them by just one call.
    """


class CyclicWD(WeightDecayScheduler, CyclicScheduler):
    r"""Sets the weight decay of each parameter group according to
    cyclical weight decay policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical weight decay policy changes the weight decay after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by
        half each cycle.
    * "exp_range": A cycle that scales initial amplitude by
        :math:`\text{gamma}^{\text{cycle iterations}}` at each cycle iteration.

    This class is similar to the `Pytorch CyclicLR LRS`_.

    Args:
        optimizer: The optimizer to schedule.
        base_val: Initial weight decay which is the lower boundary in the cycle.
        max_val: Upper weight decay boundaries in the cycle.
        step_size_up: Number of training iterations in the increasing half of a
            cycle.
        step_size_down: Number of training iterations in the decreasing half of
            a cycle.
        mode: One of {'triangular', 'triangular2', 'exp_range'}.
        gamma: Constant in 'exp_range' scaling function:
            gamma**(cycle iterations).
        scale_mode: {'cycle', 'iterations'} Defines whether scale_fn is
            evaluated on cycle number or cycle iterations.

    .. _Cyclical Learning Rates for Training Neural Networks:
            https://arxiv.org/abs/1506.01186

    .. _Pytorch CyclicLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
    """


class OneCycleWD(WeightDecayScheduler, OneCycleScheduler):
    r"""Sets the weight decay of each parameter group according to the
    1cycle weight decay policy. The 1cycle policy anneals the learning
    rate from an initial weight decay to some maximum weight decay and then
    from that maximum weight decay to some minimum weight decay much lower
    than the initial weight decay.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    This scheduler is not chainable.

    This class is similar to the `Pytorch OneCycleLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: Initial weight decay. Compared with PyTorch,
            this is equivalent to max_val / div_factor.
        max_val: Upper weight decay boundaries in the cycle.
        total_steps: The total number of steps in the cycle.
        pct_start: The percentage of the cycle (in number of steps) spent
            increasing the weight decay.
        final_div_factor: Determines the minimum weight decay via
            min_val = initial_val/final_div_factor.
        three_phase: If True, use a third phase of the schedule to annihilate
            the weight decay
        anneal_strategy: Specifies the annealing strategy:
            "cos" for cosine annealing, "linear" for linear annealing.

    .. _Super-Convergence\:
        Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120

    .. _Pytorch OneCycleLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
    """


class ScalePerParamWD(WeightDecayScheduler, ScalePerParamScheduler):
    r"""Wrapper around the LRScheduler to scale the weight decay of
    each optimizer parameter group by the scaling factor `adjust_val`.
    weight decay scaling is proposed in the Maximal Update Parameterization work
    that aids one-shot hyperparameter transfer from a smaller base model to larger
    models.
    It also serves a generic use case of layer-wise/param_group-wise adaptation
    of the weight decay.
    This wrapper doesn't work with ChainedLR scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler: wrapped scheduler
    """
