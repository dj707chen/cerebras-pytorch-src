# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings
import weakref
from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Optional, Union

import numpy
import torch

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.classes import retrieve_all_subclasses
from cerebras.pytorch.backend import current_backend_impl, use_cs


class _enable_get_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_called_within_step = False


class Scheduler(ABC):
    """
    Generic scheduler class for various optimizer params.

    Args:
        optimizer: The optimizer to schedule
        total_iters: Number of steps to perform the decay
        last_epoch: the initial step to start at
        param_group_tags: param group tags to target update for
    """

    def __init__(
        self, optimizer, total_iters, last_epoch=-1, param_group_tags=None
    ):
        # To be changed by the backend on setup
        self.device = None

        if param_group_tags is None:
            self.param_group_tags = None
        elif isinstance(param_group_tags, (list, tuple, set)):
            self.param_group_tags = set(param_group_tags)
            if len(self.param_group_tags) != len(param_group_tags):
                raise ValueError(
                    f"`param_group_tags` contains duplicate values. "
                    f"Tag list: {param_group_tags}."
                )
        else:
            self.param_group_tags = {param_group_tags}

        # filled with default dummy values
        self._last_value = [
            group[self.param_group_key] for group in optimizer.param_groups
        ]

        self.total_iters = total_iters

        # Attach optimizer
        if not isinstance(optimizer, cstorch.optim.Optimizer):
            raise TypeError(
                f"{type(optimizer).__name__} is not an a cstorch.optim.Optimizer"
            )
        self.optimizer = optimizer

        self._validate_tags_apply()

        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)

        # Perform an initial step
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

        backend = current_backend_impl()
        backend.setup_scheduler(self)

        if not isinstance(self.last_epoch, torch.Tensor):
            # The tensor representation of last_epoch
            self.last_epoch = torch.tensor(self.last_epoch, device=self.device)

        self._get_called_within_step = False

        self._post_init()

    def _post_init(self):
        if not use_cs():
            # For non-CS backends, we need to update the learning
            # rate on initialization
            self.update_last_value()

    def _validate_tags_apply(self):
        all_tags = set()
        for group in self.optimizer.param_groups:
            if self._should_apply_to_param_group(group):
                break
            all_tags.update(group.get("tags", set()))
        else:
            raise ValueError(
                f"Scheduler {self.__class__.__name__} was created with "
                f"`param_group_tags` {self.param_group_tags} but none of them "
                f"were applied to any params groups in the optimizer. Please "
                f"check the correctness of the param group tags. Available "
                f"tags: {all_tags}."
            )

    @property
    @abstractmethod
    def param_group_key(self):
        """
        Key of the param group value to modify. For example, \'lr\' or \'weight_decay\'.
        """

    def _should_apply_to_param_group(self, param_group):
        """
        Checks if param_group_tags conatins a match to a param group's tags.
        """
        if self.param_group_tags is None:
            return True
        if p_tags := param_group.get("tags", None):
            return bool(self.param_group_tags.intersection(p_tags))
        else:
            return False

    def get(self):
        if not self._get_called_within_step:
            warnings.warn(
                "To get the last value computed by the scheduler, "
                "please use `get_last_value()`."
            )

        val_tensor = self._get_closed_form()
        return [
            val_tensor if self._should_apply_to_param_group(group) else None
            for group in self.optimizer.param_groups
        ]

    @abstractmethod
    def _get_closed_form(self):
        pass

    def state_dict(self):
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key != "optimizer"
        }
        state.pop("device")
        return state

    def load_state_dict(self, state_dict):
        ignored_keys = [
            # In older releases, _step_count wasn't updated and always had value
            # 0. In this iteration of the LR schedulers, step_count is updated
            # during LRS construction and its value is used to determine if the
            # LR tensor should be computed. However, its value is still not used
            # in a substantial way (just for some warning messages in vanilla
            # PT). We pop it from the state_dict to avoid overriding the
            # increment during construction.
            "_step_count",
        ]
        ignored_kwargs = {
            key: state_dict.pop(key)
            for key in ignored_keys
            if key in state_dict
        }

        self.__dict__.update(state_dict)

        self._validate_tags_apply()

        state_dict.update(ignored_kwargs)

        # This causes `self` to be registered in the LR scheduler registry of
        # the optimizers. LR scheduler subclasses that act as a wrapper around
        # other LR schedulers may need to remove the wrapped LR schedulers from
        # the registery to avoid double registration.
        current_backend_impl().setup_scheduler(self)

    def increment_last_epoch(self):
        """Increments the last epoch by 1"""
        # Mark the tensor to conditionally skip its update in the case where
        # DLS is active and NaN/inf is detected in the gradients
        cstorch.amp.update_if_finite(self.optimizer, self.last_epoch)
        self.last_epoch += 1

    def step(self, *args, **kwargs):
        """
        Steps the scheduler and computes the latest value

        Only sets the last_epoch if running on CS
        """
        self.increment_last_epoch()
        self._step_count += 1

        if self._step_count == 1:
            # don't calculate the value on initialization
            # as we need to recalculate it anyways just before
            # optimizer step
            return

        self.update_last_value()
        self.update_groups(self._last_value)

    def update_last_value(self):
        with _enable_get_call(self):
            self._last_value = self.get()

        # Update the optimizer param groups so that the latest value is
        # available for use during optimizer step
        for param_group, val in zip(
            self.optimizer.param_groups, self._last_value
        ):
            if val is not None:
                param_group[self.param_group_key] = val

    @cstorch.step_closure
    def update_groups(self, values):
        """Update the optimizer groups with the latest values"""

        # Update the last lr values so that they are available after training
        # The _last_value values are tensors here to reflect the fact that
        # they are tensors during training
        # This should be compatible with cpu/gpu runs as well
        self._last_value = values

        # Update the optimizer param groups so that the latest value is
        # available for checkpointing
        for param_group, val in zip(
            self.optimizer.param_groups, self._last_value
        ):
            if val is not None:
                param_group[self.param_group_key] = val

    def get_last_value(self):
        """Return last computed value by current scheduler."""
        return self._last_value


class ConstantScheduler(Scheduler):
    """Maintains a constant value for each parameter group (no decaying).

    Args:
        optimizer: The optimizer to schedule
        val: The value to maintain
        total_iters: The number of steps to decay for
    """

    def __init__(
        self,
        optimizer,
        val: float,
        total_iters: int = None,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.val = val

        super().__init__(
            optimizer,
            total_iters=total_iters,
            param_group_tags=param_group_tags,
        )

    def _get_closed_form(self):
        return torch.tensor(self.val, device=self.device)


class PolynomialScheduler(Scheduler):
    r"""Decays the value of each parameter group using a polynomial function
    in the given `total_iters`.

    This class is similar to the `Pytorch PolynomialLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value
        end_val: The final value
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
        optimizer,
        initial_val: float,
        end_val: float,
        total_iters: int,
        power: float = 1.0,
        cycle: bool = False,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = initial_val
        self.end_val = end_val
        self.power = power
        self.cycle = cycle

        super().__init__(
            optimizer,
            total_iters=total_iters,
            param_group_tags=param_group_tags,
        )

    def _get_closed_form(self):
        diff = self.initial_val - self.end_val
        alpha = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        if self.cycle:
            alpha = torch.add(self.last_epoch, 1).div(self.total_iters).ceil()

        return torch.where(
            self.last_epoch >= self.total_iters,
            torch.tensor(
                self.end_val,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.sub(
                1,
                torch.div(self.last_epoch, torch.mul(self.total_iters, alpha)),
            )
            .pow(self.power)
            .mul(diff)
            .add(self.end_val)
            .float(),
        )


class LinearScheduler(PolynomialScheduler):
    """Alias for Polynomial Scheduler scheduler with a power of 1"""

    def __init__(
        self,
        optimizer,
        initial_val: float,
        end_val: float,
        total_iters: int,
        cycle: bool = False,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(
            optimizer=optimizer,
            initial_val=initial_val,
            end_val=end_val,
            total_iters=total_iters,
            power=1.0,
            cycle=cycle,
            param_group_tags=param_group_tags,
        )


class ExponentialScheduler(Scheduler):
    r"""Decays the value of each parameter group by `decay_rate` every step.

    This class is similar to the `Pytorch ExponentialLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value.
        total_iters: Number of steps to perform the decay
        decay_rate: The decay rate
        staircase: If True decay the value at discrete intervals

    .. _Pytorch ExponentialLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float,
        total_iters: int,
        decay_rate: float,
        staircase: bool = False,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = float(initial_val)
        self.decay_rate = decay_rate
        self.staircase = staircase

        super().__init__(
            optimizer,
            total_iters=total_iters,
            param_group_tags=param_group_tags,
        )

    def _get_closed_form(self):
        power = torch.div(self.last_epoch, self.total_iters)
        if self.staircase:
            power.floor_()
        return torch.pow(self.decay_rate, power).mul(self.initial_val)


class InverseExponentialTimeDecayScheduler(Scheduler):
    r"""Decays the value inverse-exponentially over time, as described
    in the `Keras InverseTimeDecay class`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value.
        step_exponent: Exponential value.
        total_iters: Number of steps to perform the decay.
        decay_rate: The decay rate.
        staircase: If True decay the value at discrete intervals.

    .. _Keras InverseTimeDecay class:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float,
        step_exponent: int,
        total_iters: int,
        decay_rate: float,
        staircase: bool = False,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = initial_val
        self.step_exponent = step_exponent
        self.decay_rate = decay_rate
        self.staircase = staircase

        super().__init__(
            optimizer,
            total_iters=total_iters,
            param_group_tags=param_group_tags,
        )

    def _get_closed_form(self):
        alpha = torch.div(
            torch.pow(self.last_epoch.float(), self.step_exponent),
            self.total_iters,
        )
        if self.staircase:
            alpha.floor_()
        return torch.div(
            torch.tensor(
                self.initial_val,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.mul(alpha, self.decay_rate).add(1.0),
        )


class InverseSquareRootDecayScheduler(Scheduler):
    r"""Decays the value inverse-squareroot over time, as described
    in the following equation:

    .. math::
        \begin{aligned}
            lr_t & = \frac{\text{scale}}{\sqrt{\max\{t, \text{warmup_steps}\}}}.
        \end{aligned}

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value.
        scale: Multiplicative factor to scale the result.
        warmup_steps: use initial_val for the first warmup_steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float = 1.0,
        scale: float = 1.0,
        warmup_steps: int = 1.0,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = initial_val
        self.scale = scale
        self.warmup_steps = warmup_steps

        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def _get_closed_form(self):
        return torch.div(
            torch.tensor(self.scale, dtype=torch.float32, device=self.device),
            torch.sqrt(
                torch.max(
                    torch.tensor(
                        self.warmup_steps,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    self.last_epoch,
                )
            ),
        ).mul(self.initial_val)


class CosineDecayScheduler(Scheduler):
    r"""Applies the cosine decay schedule as described
    in the `Keras CosineDecay class`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value
        end_val: The final value
        total_iters: Number of steps to perform the decay

    .. _Keras CosineDecay class:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
    """

    def __init__(
        self,
        optimizer,
        initial_val: float,
        end_val: float,
        total_iters: int,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = initial_val
        self.end_val = end_val

        super().__init__(
            optimizer,
            total_iters=total_iters,
            param_group_tags=param_group_tags,
        )

    def _get_closed_form(self):
        diff = self.initial_val - self.end_val
        # clip the steps to be at most total_iters

        step = torch.minimum(
            torch.tensor(
                self.total_iters, dtype=torch.float32, device=self.device
            ),
            self.last_epoch,
        )
        # where we are at the cosine curve
        progress = (
            torch.div(math.pi, self.total_iters).mul(step).cos().add(1).mul(0.5)
        )
        return torch.mul(progress, diff).add(self.end_val)


class SequentialScheduler(Scheduler):
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

    def __init__(
        self,
        optimizer,
        schedulers,
        milestones,
        last_epoch=-1,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(milestones, numpy.ndarray):
            milestones = milestones.tolist()
        if isinstance(milestones, (list, tuple)):
            if any(not isinstance(milestone, int) for milestone in milestones):
                raise TypeError(
                    f"Expected milestones to be a list of integers. "
                    f"Got: {[type(milestone) for milestone in milestones]}"
                )
        else:
            raise TypeError(
                f"Expected milestones to be a list of integers. "
                f"Got: {type(milestones)}"
            )

        for scheduler_idx in range(len(schedulers)):
            if schedulers[scheduler_idx].optimizer != optimizer:
                raise ValueError(
                    f"Sequential Schedulers expects all schedulers to belong "
                    f"to the same optimizer, but got schedulers at index "
                    f"{scheduler_idx} to be different than the optimizer "
                    f"passed in."
                )

            if schedulers[scheduler_idx].optimizer != schedulers[0].optimizer:
                raise ValueError(
                    f"Sequential Schedulers expects all schedulers to belong "
                    f"to the same optimizer, but got schedulers at index {0} "
                    f"and {scheduler_idx} to be different."
                )

            if (
                schedulers[scheduler_idx].param_group_key
                != self.param_group_key
            ):
                raise ValueError(
                    f"Sequential Schedulers expects all schedulers to have "
                    f" the same `param_group_key`, but got schedulers "
                    f"`param_group_key` at index {scheduler_idx} to be different "
                    f"than the Sequential Scheduler's `param_group_key`."
                )

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                f"Sequential Schedulers expects number of schedulers provided "
                f"to be one more than the number of milestone points, but got "
                f"number of schedulers {len(schedulers)} and the number of "
                f" milestones to be equal to {len(milestones)}"
            )
        self._schedulers = schedulers
        self._milestones = milestones

        for scheduler in schedulers:
            optimizer._schedulers_registry.pop(scheduler, None)
        super().__init__(
            optimizer,
            total_iters=None,
            last_epoch=last_epoch,
            param_group_tags=param_group_tags,
        )

    def _post_init(self):
        for idx, _scheduler in enumerate(self._schedulers):
            if _scheduler.param_group_tags != self.param_group_tags:
                raise ValueError(
                    f"Sequential Schedulers expects all schedulers to have "
                    f"the same `param_group_tags`, but found `param_group_tags` at index "
                    f"{idx} to be different than the Sequential "
                    f"Scheduler's `param_group_tags."
                )

        super()._post_init()

    def _get_closed_form(self):
        new_val = self._schedulers[0]._get_closed_form()
        for idx, milestone in enumerate(self._milestones):
            # If current global step is equal or greater than
            # the 'milestone', we will choose the corresponding
            # scheduler which is indexed 'idx+1' in 'self._schedulers`.
            # Otherwise, we will use the scheduler from previous iteration.
            res = torch.where(
                self.last_epoch < milestone,
                new_val,
                self._schedulers[idx + 1]._get_closed_form(),
            )
            new_val = res
        return new_val

    def increment_last_epoch(self, *args, **kwargs):
        """Increments the last_epoch of the scheduler whose milestone we are on"""
        super().increment_last_epoch()

        # All schedulers have already been incremented before the first step
        if self._step_count == 0:
            return

        # Mark the tensors to conditionally skip their update in the case where
        # DLS is active and NaN/inf is detected in the gradients
        for scheduler in self._schedulers:
            cstorch.amp.update_if_finite(self.optimizer, scheduler.last_epoch)
        self._schedulers[0].last_epoch += 1
        for idx, milestone in enumerate(self._milestones):
            # Increment next scheduler's last_epoch if we are on the correct milestone
            self._schedulers[idx + 1].last_epoch += torch.where(
                self.last_epoch <= milestone, 0, 1
            )

    def state_dict(self):
        s = super().state_dict()

        schedulers = s.pop("_schedulers")
        s["_schedulers"] = []
        for scheduler in schedulers:
            s['_schedulers'].append(scheduler.state_dict())

        return s

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        super().load_state_dict(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)
            self.optimizer._schedulers_registry.pop(self._schedulers[idx], None)


class PiecewiseConstantScheduler(SequentialScheduler):
    r"""Adjusts the value to a predefined constant at each milestone and
    holds this value until the next milestone. Notice that such adjustment can
    happen simultaneously with other changes to the value from outside
    this scheduler.

    Args:
        optimizer: The optimizer to schedule
        vals: List of values to maintain before/during each
            milestone.
        milestones: List of step indices. Must be increasing.
    """

    def __init__(
        self,
        optimizer,
        constant_cls,
        vals: List[float],
        milestones: List[int],
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        schedulers = []
        boundaries = [0]
        boundaries.extend(milestones)
        for val, b1, b2 in zip(vals, boundaries[:-1], boundaries[1:]):
            schedulers.append(
                constant_cls(
                    optimizer, val, b2 - b1, param_group_tags=param_group_tags
                )
            )
        # Final value
        schedulers.append(
            constant_cls(optimizer, vals[-1], param_group_tags=param_group_tags)
        )

        super().__init__(
            optimizer, schedulers, milestones, param_group_tags=param_group_tags
        )


class MultiStepScheduler(Scheduler):
    r"""Decays the value of each parameter group by gamma once the number of
    steps reaches one of the milestones. Notice that such decay can happen
    simultaneously with other changes to the value from outside this
    scheduler.

    This class is similar to the `Pytorch MultiStepLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value.
        gamma: Multiplicative factor of value decay.
        milestones: List of step indices. Must be increasing.

    .. _Pytorch MultiStepLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float,
        gamma: float,
        milestones: List[int],
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = initial_val
        self.gamma = gamma
        self.milestones = milestones
        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def _get_closed_form(self):
        new_val = torch.tensor(
            self.initial_val,
            dtype=torch.float32,
            device=self.device,
        )
        for milestone in self.milestones:
            res = torch.where(
                self.last_epoch < milestone,
                new_val,
                torch.mul(
                    torch.tensor(
                        self.gamma,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    new_val,
                ),
            )
            new_val = res
        return new_val


class StepScheduler(Scheduler):
    r"""Decays the value of each parameter group by gamma every `step_size`.
    Notice that such decay can happen simultaneously with other changes to the
    value from outside this scheduler.

    This class is similar to the `Pytorch StepLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value.
        step_size: Period of decay.
        gamma: Multiplicative factor of decay.

    .. _Pytorch StepLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float,
        step_size: int,
        gamma: float,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = float(initial_val)
        self.gamma = gamma
        self.step_size = step_size
        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def _get_closed_form(self):
        return torch.mul(
            torch.pow(
                torch.tensor(
                    self.gamma, dtype=torch.float32, device=self.device
                ),
                torch.div(self.last_epoch, self.step_size).floor_(),
            ),
            self.initial_val,
        )


class CosineAnnealingScheduler(Scheduler):
    r"""Set the value of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial value and
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

    Notice that because the schedule is defined recursively, the value
    can be simultaneously modified outside this scheduler by other operators.
    If the value is set solely by this scheduler, the value at
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
        initial_val: The initial value.
        T_max: Maximum number of iterations.
        eta_min: Minimum value.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    .. _Pytorch CosineAnnealingLR LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float,
        T_max: int,
        eta_min: float = 0.0,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = float(initial_val)
        self.T_max = float(T_max)
        self.eta_min = eta_min
        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def _get_closed_form(self):
        val_diff = self.initial_val - self.eta_min
        a = torch.div(
            torch.mul(torch.div(self.last_epoch, self.T_max), math.pi)
            .cos()
            .add(1),
            2,
        )
        return torch.add(torch.mul(a, val_diff), self.eta_min)


class LambdaScheduler(Scheduler):
    r"""Sets the value of each parameter group to the initial value times a
    given function (which is specified by overriding `set_value_lambda`).

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = initial_val
        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def set_value_lambda(self):  # pylint: disable=no-self-use
        """Sets learning lambda functions"""
        lambda1 = lambda epoch: torch.div(epoch, 30)
        lambda2 = lambda epoch: torch.pow(
            torch.tensor(0.95, dtype=torch.float32, device=epoch.device),
            epoch,
        )
        val_lambda = [lambda1, lambda2]
        return val_lambda

    def _get_closed_form(self):
        new_val = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=self.device,
        )
        val_lambda = self.set_value_lambda()
        for val in val_lambda:
            new_val = torch.mul(
                torch.mul(
                    torch.tensor(
                        self.initial_val,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    val(self.last_epoch),
                ),
                new_val,
            )
        return new_val


class CosineAnnealingWarmRestartsScheduler(Scheduler):
    r"""Set the value of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial val, :math:`T_{cur}`
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
        initial_val: The initial value.
        T_0: Number of iterations for the first restart.
        T_mult: A factor increases Ti after a restart. Currently T_mult must be
            set to 1.0
        eta_min: Minimum value.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    .. _Pytorch CosineAnnealingWarmRestarts LRS:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        if T_mult != 1.0:
            raise ValueError(
                f"Unsupported value of Parameters 'T_mult' for LR scheduler "
                f"type CosineAnnealingWarmRestarts, Only supported default "
                f"T_mult value: 1.0. "
            )
        self.initial_val = float(initial_val)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def _get_closed_form(self):
        tensor_t_i_1 = torch.tensor(
            self.T_0, dtype=torch.float32, device=self.device
        )

        tensor_t_cur_1 = self.last_epoch.float()
        tensor_t_cur_2 = torch.sub(
            torch.torch.mul(
                torch.div(self.last_epoch, self.T_0).floor_(), self.T_0
            ),
            self.T_0,
        )

        tensor_t_mul = torch.tensor(
            self.T_mult, dtype=torch.float32, device=self.device
        )
        nn = torch.mul(
            torch.div(self.last_epoch, self.T_0), tensor_t_mul.sub(1)
        ).add(1)
        n = torch.div(torch.log(nn), torch.log(tensor_t_mul)).floor_()

        tensor_t_i_3 = torch.pow(tensor_t_mul, n).mul(self.T_0)
        tensor_t_cur_3 = torch.sub(
            self.last_epoch,
            torch.div(
                torch.pow(tensor_t_mul, n).sub(1), tensor_t_mul.sub(1)
            ).mul(self.T_0),
        ).float()

        T_i = torch.where(tensor_t_mul == 1, tensor_t_i_1, tensor_t_i_3)
        T_cur = torch.where(
            self.last_epoch < self.T_0,
            tensor_t_cur_1,
            torch.where(tensor_t_mul == 1, tensor_t_cur_2, tensor_t_cur_3),
        )
        val_diff = self.initial_val - self.eta_min
        a = torch.div(
            torch.mul(torch.div(T_cur, T_i), math.pi).cos().add(1),
            2,
        )
        return torch.add(torch.mul(a, val_diff), self.eta_min)


class MultiplicativeScheduler(Scheduler):
    r"""Multiply the value of each parameter group by the supplied
    coefficient.

    Args:
        optimizer: The optimizer to schedule
        initial_val: The initial value.
        coefficient: Multiplicative factor of value.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_val: float,
        coefficient: float,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = initial_val
        self.coefficient = coefficient
        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def set_value_lambda(self):
        """Sets learning lambda functions"""
        val_lambda = lambda epoch: self.coefficient
        return val_lambda

    def _get_closed_form(self):
        new_val = None
        val_lambda = self.set_value_lambda()
        new_val = torch.mul(
            torch.pow(
                torch.tensor(
                    val_lambda(self.last_epoch),
                    dtype=torch.float32,
                    device=self.device,
                ),
                self.last_epoch,
            ),
            self.initial_val,
        )
        return new_val


class ChainedScheduler(Scheduler):
    r"""Chains list of schedulers.
    It takes a list of chainable schedulers and performs consecutive
    step() functions belonging to them by just one call.
    """

    def __init__(
        self,
        schedulers,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self._schedulers = list(schedulers)
        super().__init__(
            schedulers[0].optimizer,
            total_iters=None,
            param_group_tags=param_group_tags,
        )

    def _post_init(self):
        for i, scheduler in enumerate(self._schedulers):
            if scheduler.optimizer != self.optimizer:
                raise ValueError(
                    f"ChainedScheduler expects all schedulers to belong to the "
                    f"same optimizer, but got schedulers at index 0 and "
                    f"{i} to be different"
                )

            if scheduler.param_group_tags != self.param_group_tags:
                raise ValueError(
                    f"Chained Scheduler expects all schedulers to have the same "
                    f"`param_group_tags`, but found `param_group_tags` at index {i} to be "
                    f"different than the Chained Scheduler's `param_group_tags."
                )

            scheduler.last_epoch = self.last_epoch

            self.optimizer._schedulers_registry.pop(scheduler, None)

        super()._post_init()

    def _get_closed_form(self):
        new_value = self._schedulers[0]._get_closed_form()
        for scheduler in self._schedulers[1:]:
            new_value = torch.mul(
                new_value,
                torch.div(
                    scheduler._get_closed_form(),
                    scheduler.initial_val,
                ),
            )
        return new_value

    def state_dict(self):
        s = super().state_dict()

        schedulers = s.pop("_schedulers")
        s["_schedulers"] = []
        for scheduler in schedulers:
            s['_schedulers'].append(scheduler.state_dict())

        return s

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        super().load_state_dict(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)
            self._schedulers[idx].last_epoch = self.last_epoch
            self.optimizer._schedulers_registry.pop(self._schedulers[idx], None)


class CyclicScheduler(Scheduler):
    r"""Sets the value of each parameter group according to
    cyclical value policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical policy changes the value after every batch.
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
        base_val: Initial value which is the lower boundary in the cycle.
        max_val: Upper value boundaries in the cycle.
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
        base_val: float,
        max_val: float,
        step_size_up: int = 2000,
        step_size_down: int = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_mode: str = "cycle",
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.base_val = base_val
        self.max_val = max_val
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_mode = scale_mode

        if self.step_size_down == None:
            self.step_size_down = step_size_up

        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def _triangular_scale_fn(self, x):  # pylint: disable=no-self-use
        return 1.0

    def _triangular2_scale_fn(self, x):  # pylint: disable=no-self-use
        return torch.div(
            torch.tensor(1, dtype=torch.float32, device=x.device),
            torch.pow(
                torch.tensor(2, dtype=torch.float32, device=x.device),
                torch.sub(x, 1),
            ),
        )

    def _exp_range_scale_fn(self, x):
        return torch.pow(
            torch.tensor(self.gamma, dtype=torch.float32, device=x.device), x
        )

    def set_scale_fn(self):
        """Sets the scaling function"""
        scale_fn = None
        if self.mode == 'triangular':
            scale_fn = self._triangular_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            scale_fn = self._triangular2_scale_fn
            self.scale_mode = 'cycle'
        else:
            scale_fn = self._exp_range_scale_fn
            self.scale_mode = 'iterations'
        return scale_fn

    def _get_closed_form(self):
        scale_fn = self.set_scale_fn()
        total_size = self.step_size_up + self.step_size_down
        step_ratio = self.step_size_up / total_size
        cycle = torch.floor(torch.div(self.last_epoch, total_size).add(1))
        x = torch.sub(torch.div(self.last_epoch, total_size), cycle).add(1)
        scale_factor = torch.where(
            x <= step_ratio,
            torch.div(x, step_ratio),
            torch.div(torch.sub(x, 1), torch.sub(step_ratio, 1)),
        )

        base_height = torch.mul((scale_factor), (self.max_val - self.base_val))
        if self.scale_mode == "cycle":
            return torch.add(
                torch.mul(base_height, scale_fn(cycle)), self.base_val
            )
        else:
            return torch.add(
                torch.mul(base_height, scale_fn(self.last_epoch)),
                self.base_val,
            )


class OneCycleScheduler(Scheduler):
    r"""Sets the value of each parameter group according to the
    1cycle policy. The 1cycle policy anneals the learning
    rate from an initial value to some maximum value and then
    from that maximum value to some minimum value much lower
    than the initial value.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    This scheduler is not chainable.

    This class is similar to the `Pytorch OneCycleLR LRS`_.

    Args:
        optimizer: The optimizer to schedule
        initial_val: Initial value. Compared with PyTorch,
            this is equivalent to max_val / div_factor.
        max_val: Upper value boundaries in the cycle.
        total_steps: The total number of steps in the cycle.
        pct_start: The percentage of the cycle (in number of steps) spent
            increasing the value.
        final_div_factor: Determines the minimum value via
            min_val = initial_val/final_div_factor.
        three_phase: If True, use a third phase of the schedule to annihilate
            the value
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
        initial_val: float,
        max_val: float,
        total_steps: int = 1000,
        pct_start: float = 0.3,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        anneal_strategy: str = "cos",
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.initial_val = initial_val
        self.max_val = max_val
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.anneal_strategy = anneal_strategy
        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def _annealing_cos(self, start, end, pct):  # pylint: disable=no-self-use
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = torch.mul(pct, math.pi).cos().add(1)
        return torch.add(torch.mul(cos_out, ((start - end) / 2.0)), end)

    def _annealing_linear(self, start, end, pct):  # pylint: disable=no-self-use
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return torch.add(torch.mul(pct, (end - start)), start)

    def _get_closed_form(self):
        min_val = self.initial_val / self.final_div_factor
        if self.three_phase:
            milestones = [
                self.pct_start * self.total_steps - 1,
                2 * self.pct_start * self.total_steps - 2,
                self.total_steps - 1,
            ]
            val_start = [
                self.initial_val,
                self.max_val,
                self.initial_val,
            ]
            val_end = [self.max_val, self.initial_val, min_val]
        else:
            milestones = [
                self.pct_start * self.total_steps - 1,
                self.total_steps - 1,
            ]
            val_start = [self.initial_val, self.max_val]
            val_end = [self.max_val, min_val]

        if self.anneal_strategy == "cos":
            anneal_func = self._annealing_cos
        else:
            anneal_func = self._annealing_linear

        start_step = 0
        pct = torch.div(
            torch.sub(self.last_epoch, start_step),
            (milestones[0] - start_step),
        )
        val = anneal_func(val_start[0], val_end[0], pct)
        start_step = milestones[0]
        for idx, milestone in enumerate(milestones[1:]):
            pct = torch.div(
                torch.sub(self.last_epoch, start_step),
                (milestone - start_step),
            )
            val = torch.where(
                self.last_epoch > milestones[idx],
                anneal_func(val_start[idx + 1], val_end[idx + 1], pct),
                val,
            )
            start_step = milestone
        return val


class ScalePerParamScheduler(Scheduler):
    r"""Wrapper around the Scheduler to scale the value of
    each optimizer parameter group by the scaling factor `adjust_val`.
    Learning rate scaling is proposed in the Maximal Update Parameterization work
    that aids one-shot hyperparameter transfer from a smaller base model to larger
    models.
    It also serves a generic use case of layer-wise/param_group-wise adaptation
    of the value.
    This wrapper doesn't work with Chained scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler: wrapped scheduler
    """

    def __init__(
        self,
        optimizer,
        scheduler,
        param_group_tags: Optional[Union[str, List[str]]] = None,
    ):
        self.adjustment_scalars = [
            param_group.get(
                'adjust_learning_rate', param_group.get("adjust_val", 1.0)
            )
            for param_group in optimizer.param_groups
        ]
        self._scheduler_nested = scheduler

        optimizer._schedulers_registry.pop(scheduler, None)

        super().__init__(
            optimizer, total_iters=None, param_group_tags=param_group_tags
        )

    def _post_init(self):
        if self._scheduler_nested.param_group_tags != self.param_group_tags:
            raise ValueError(
                f"ScalePerParamScheduler expects the wrapped scheduler "
                f"to have the same `param_group_tags`, but found `param_group_tags` to be "
                f"different than the ScalePerParamScheduler's `param_group_tags."
            )
        super()._post_init()

    def state_dict(self):
        s = super().state_dict()
        s["_scheduler"] = self._scheduler_nested.state_dict()
        s.pop("_scheduler_nested", None)
        s["_scheduler"].pop("_get_called_within_step", None)
        return s

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        _scheduler_dict = state_dict.pop('_scheduler')
        state_dict['_scheduler'] = _scheduler_dict
        self._scheduler_nested.load_state_dict(_scheduler_dict)
        self.optimizer._schedulers_registry.pop(self._scheduler_nested, None)

    def increment_last_epoch(self, *args, **kwargs):
        """Increments the last_epoch of the scheduler whose milestone we are on"""
        super().increment_last_epoch()
        if self._step_count == 0:
            return
        self._scheduler_nested.increment_last_epoch()

    def get(self):
        if not self._get_called_within_step:
            warnings.warn(
                "To get the last value computed by the scheduler, "
                "please use `get_last_value()`."
            )
        val_tensor = self._get_closed_form()
        return [
            (
                val_tensor * self.adjustment_scalars[group_idx]
                if self._should_apply_to_param_group(group)
                else None
            )
            for group_idx, group in enumerate(self.optimizer.param_groups)
        ]

    def _get_closed_form(self):
        return self._scheduler_nested._get_closed_form()


__all__ = ["Scheduler"] + [
    cls.__name__
    for cls in retrieve_all_subclasses(
        Scheduler, condition=lambda subcls: subcls.__module__ == __name__
    )
]
