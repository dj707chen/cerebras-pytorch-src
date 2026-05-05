# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Implementations for Cerebras specific learning rate schedulers"""

from typing import List, Optional, Union

import torch

from .scheduler import ChainedScheduler as BaseChainedScheduler
from .scheduler import (
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


class LRScheduler(Scheduler, torch.optim.lr_scheduler._LRScheduler):
    @property
    def param_group_key(self):
        return "lr"

    # aliasing torch.optim.lr_scheduler.LRScheduler methods for cross compatibilty

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler."""
        return self.get_last_value()

    def get_lr(self) -> List[float]:
        # Compute learning rate using chainable form of the scheduler
        return self.get()


class ConstantLR(LRScheduler, ConstantScheduler):
    r"""Maintains a constant learning rate for each parameter group (no decaying).

    Args:
        optimizer: The optimizer to schedule
        val: The learning_rate value to maintain
        total_iters: The number of steps to decay for
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        total_iters: int = None,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            val=learning_rate,
            total_iters=total_iters,
            param_group_tags=param_group_tags,
        )

    @property
    def val(self):
        return self.learning_rate

    @val.setter
    def val(self, value):
        self.learning_rate = value


class PolynomialLR(LRScheduler, PolynomialScheduler):
    r"""Decays the learning rate of each parameter group using a polynomial function
    in the given `total_iters`.

    This class is similar to the `Pytorch PolynomialLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        end_learning_rate: The final learning rate
        total_iters: Number of steps to perform the decay
        power: Exponent to apply to "x" (as in y=mx+b),
            which is ratio of step completion (1 for linear)
            Default: 1.0 (only Linear supported at the moment)
        cycle: Whether to cycle

    .. _Pytorch PolynomialLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        end_learning_rate: float,
        total_iters: int,
        power: float = 1.0,
        cycle: bool = False,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            end_val=end_learning_rate,
            total_iters=total_iters,
            power=power,
            cycle=cycle,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value

    @property
    def end_val(self):
        return self.end_learning_rate

    @end_val.setter
    def end_val(self, value):
        self.end_learning_rate = value


class LinearLR(LRScheduler, LinearScheduler):
    """Alias for Polynomial LR scheduler with a power of 1"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        end_learning_rate: float,
        total_iters: int,
        cycle: bool = False,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            end_val=end_learning_rate,
            total_iters=total_iters,
            cycle=cycle,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value

    @property
    def end_val(self):
        return self.end_learning_rate

    @end_val.setter
    def end_val(self, value):
        self.end_learning_rate = value


class ExponentialLR(LRScheduler, ExponentialScheduler):
    r"""Decays the learning rate of each parameter group by `decay_rate` every step.

    This class is similar to the `Pytorch ExponentialLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        total_iters: Number of steps to perform the decay
        decay_rate: The decay rate
        staircase: If True decay the learning rate at discrete intervals

    .. _Pytorch ExponentialLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        total_iters: int,
        decay_rate: float,
        staircase: bool = False,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            total_iters=total_iters,
            decay_rate=decay_rate,
            staircase=staircase,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class InverseExponentialTimeDecayLR(
    LRScheduler, InverseExponentialTimeDecayScheduler
):
    r"""Decays the learning rate inverse-exponentially over time, as described
    in the `Keras InverseTimeDecay class`_.

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        step_exponent: Exponential value.
        total_iters: Number of steps to perform the decay.
        decay_rate: The decay rate.
        staircase: If True decay the learning rate at discrete intervals.

    .. _Keras InverseTimeDecay class:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        step_exponent: int,
        total_iters: int,
        decay_rate: float,
        staircase: bool = False,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            step_exponent=step_exponent,
            total_iters=total_iters,
            decay_rate=decay_rate,
            staircase=staircase,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class InverseSquareRootDecayLR(LRScheduler, InverseSquareRootDecayScheduler):
    r"""Decays the learning rate inverse-squareroot over time, as described
    in the following equation:

    .. math::
        \begin{aligned}
            lr_t & = \frac{\text{scale}}{\sqrt{\max\{t, \text{warmup_steps}\}}}.
        \end{aligned}

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        scale: Multiplicative factor to scale the result.
        warmup_steps: use initial_learning_rate for the first warmup_steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float = 1.0,
        scale: float = 1.0,
        warmup_steps: int = 1.0,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            scale=scale,
            warmup_steps=warmup_steps,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class CosineDecayLR(LRScheduler, CosineDecayScheduler):
    r"""Applies the cosine decay schedule as described
    in the `Keras CosineDecay class`_.

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        end_learning_rate: The final learning rate
        total_iters: Number of steps to perform the decay

    .. _Keras CosineDecay class:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        end_learning_rate: float,
        total_iters: int,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            end_val=end_learning_rate,
            total_iters=total_iters,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value

    @property
    def end_val(self):
        return self.end_learning_rate

    @end_val.setter
    def end_val(self, value):
        self.end_learning_rate = value


class SequentialLR(LRScheduler, SequentialScheduler):
    r"""Receives the list of schedulers that is expected to be called sequentially
    during optimization process and milestone points that provides exact
    intervals to reflect which scheduler is supposed to be called at a given
    step.

    This class is a wrapper around the `Pytorch SequentialLR LRS`_.

    Args:
        optimizer: Wrapped optimizer
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _Pytorch SequentialLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers: List[Scheduler],
        milestones: List[int],
        last_epoch: int = -1,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
            last_epoch=last_epoch,
            param_group_tags=param_group_tags,
        )


class PiecewiseConstantLR(LRScheduler, PiecewiseConstantScheduler):
    r"""Adjusts the learning rate to a predefined constant at each milestone and
    holds this value until the next milestone. Notice that such adjustment can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler.

    Args:
        optimizer: The optimizer to schedule
        learning_rates: List of learning rates to maintain before/during each
            milestone.
        milestones: List of step indices. Must be increasing.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rates: List[float],
        milestones: List[int],
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            ConstantLR,
            vals=learning_rates,
            milestones=milestones,
            param_group_tags=param_group_tags,
        )


class MultiStepLR(LRScheduler, MultiStepScheduler):
    r"""Decays the learning rate of each parameter group by gamma once the number of
    steps reaches one of the milestones. Notice that such decay can happen
    simultaneously with other changes to the learning rate from outside this
    scheduler.

    This class is similar to the `Pytorch MultiStepLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        gamma: Multiplicative factor of learning rate decay.
        milestones: List of step indices. Must be increasing.

    .. _Pytorch MultiStepLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        gamma: float,
        milestones: List[int],
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            gamma=gamma,
            milestones=milestones,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class StepLR(LRScheduler, StepScheduler):
    r"""Decays the learning rate of each parameter group by gamma every `step_size`.
    Notice that such decay can happen simultaneously with other changes to the
    learning rate from outside this scheduler.

    This class is similar to the `Pytorch StepLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        step_size: Period of decay.
        gamma: Multiplicative factor of decay.

    .. _Pytorch StepLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        step_size: int,
        gamma: float,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            step_size=step_size,
            gamma=gamma,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class CosineAnnealingLR(LRScheduler, CosineAnnealingScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
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

    Notice that because the schedule is defined recursively, the learning rate
    can be simultaneously modified outside this scheduler by other operators.
    If the learning rate is set solely by this scheduler, the learning rate at
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
        initial_learning_rate: The initial learning rate.
        T_max: Maximum number of iterations.
        eta_min: Minimum learning rate.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    .. _Pytorch CosineAnnealingLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        T_max: int,
        eta_min: float = 0.0,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            T_max=T_max,
            eta_min=eta_min,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class LambdaLR(LRScheduler, LambdaScheduler):
    r"""Sets the learning rate of each parameter group to the initial lr times a
    given function (which is specified by overriding `set_value_lambda`).

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class CosineAnnealingWarmRestarts(
    LRScheduler, CosineAnnealingWarmRestartsScheduler
):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
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
        initial_learning_rate: The initial learning rate.
        T_0: Number of iterations for the first restart.
        T_mult: A factor increases Ti after a restart. Currently T_mult must be
            set to 1.0
        eta_min: Minimum learning rate.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    .. _Pytorch CosineAnnealingWarmRestarts LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class MultiplicativeLR(LRScheduler, MultiplicativeScheduler):
    r"""Multiply the learning rate of each parameter group by the supplied
    coefficient.

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        coefficient: Multiplicative factor of learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        coefficient: float,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            coefficient=coefficient,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value


class ChainedLR(LRScheduler, BaseChainedScheduler):
    r"""Chains list of learning rate schedulers.
    It takes a list of chainable learning rate schedulers and
    performs consecutive step() functions belonging to them by just one call.
    """

    def __init__(
        self,
        schedulers: List[Scheduler],
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            schedulers=schedulers,
            param_group_tags=param_group_tags,
        )


# alias to match torch.optim.lr_scheduler
class ChainedScheduler(ChainedLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CyclicLR(LRScheduler, CyclicScheduler):
    r"""Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
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
        base_lr: Initial learning rate which is the lower boundary in the cycle.
        max_lr: Upper learning rate boundaries in the cycle.
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

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: int = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_mode: str = "cycle",
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            base_val=base_lr,
            max_val=max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            mode=mode,
            gamma=gamma,
            scale_mode=scale_mode,
            param_group_tags=param_group_tags,
        )

    @property
    def base_val(self):
        return self.base_lr

    @base_val.setter
    def base_val(self, value):
        self.base_lr = value

    @property
    def max_val(self):
        return self.max_lr

    @max_val.setter
    def max_val(self, value):
        self.max_lr = value


class OneCycleLR(LRScheduler, OneCycleScheduler):
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    This scheduler is not chainable.

    This class is similar to the `Pytorch OneCycleLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: Initial learning rate. Compared with PyTorch,
            this is equivalent to max_lr / div_factor.
        max_lr: Upper learning rate boundaries in the cycle.
        total_steps: The total number of steps in the cycle.
        pct_start: The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
        final_div_factor: Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor.
        three_phase: If True, use a third phase of the schedule to annihilate
            the learning rate
        anneal_strategy: Specifies the annealing strategy:
            "cos" for cosine annealing, "linear" for linear annealing.

    .. _Super-Convergence\:
        Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120

    .. _Pytorch OneCycleLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        max_lr: float,
        total_steps: int = 1000,
        pct_start: float = 0.3,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        anneal_strategy: str = "cos",
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            initial_val=initial_learning_rate,
            max_val=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            anneal_strategy=anneal_strategy,
            param_group_tags=param_group_tags,
        )

    @property
    def initial_val(self):
        return self.initial_learning_rate

    @initial_val.setter
    def initial_val(self, value):
        self.initial_learning_rate = value

    @property
    def max_val(self):
        return self.max_lr

    @max_val.setter
    def max_val(self, value):
        self.max_lr = value


class ScalePerParamLR(LRScheduler, ScalePerParamScheduler):
    r"""Wrapper around the LRScheduler to scale the learning rate of
    each optimizer parameter group by the scaling factor `adjust_learning_rate`.
    Learning rate scaling is proposed in the Maximal Update Parameterization work
    that aids one-shot hyperparameter transfer from a smaller base model to larger
    models.
    It also serves a generic use case of layer-wise/param_group-wise adaptation
    of the learning rate.
    This wrapper doesn't work with ChainedLR scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler: wrapped scheduler
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Scheduler,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer,
            scheduler=scheduler,
            param_group_tags=param_group_tags,
        )

    @property
    def adjustment_scalars(self):
        return self.lr_adjustment_scalars

    @adjustment_scalars.setter
    def adjustment_scalars(self, value):
        self.lr_adjustment_scalars = value
