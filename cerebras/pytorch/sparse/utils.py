# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.classes import retrieve_all_subclasses
from cerebras.appliance.utils.typing import signature_matches_type_hint
from cerebras.pytorch.utils._decorators import lazy_tensor_lru_cache


class HyperParameterSchedule(ABC):
    """
    Base class for step-aware hyperparameters used in Sparsity Optimizers.
    """

    def __init__(self):
        # Wrap the compute method with a cached equivalent
        self.compute = lazy_tensor_lru_cache(maxsize=1)(self.compute)
        self.update = lazy_tensor_lru_cache(maxsize=1)(self.update)
        # Wrap the get_min_max_end method with a cached equivalent as it only
        # needs to be computed once
        self.get_min_max_end = lazy_tensor_lru_cache(maxsize=1)(
            self.get_min_max_end
        )

    @abstractmethod
    def compute(self, step: torch.Tensor) -> torch.Tensor:
        """
        Return a torch.Tensor with the value of the hyperparatmer at the given
        step.

        Args:
            step: int64 tensor holding current step

        Returns:
            torch.Tensor on the device of step with the value of the
                hyperparamter
        """

    def __call__(self, step: torch.Tensor) -> torch.Tensor:
        return self.compute(step)

    def update(self, is_update_step: torch.Tensor):
        """
        Given a boolean tensor indicating if this is an update step, update the
        internal state of this hyperparameter.

        Args:
            is_update_step: A boolean tensor indicating if this is an update step.
        """

    def cache_clear(self):
        self.compute.cache_clear()
        self.update.cache_clear()

    def visit_state(self, fn):
        """
        Applies a lambda to each stateful value.
        """

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def get_min_max_end(
        self, begin: int, end: int
    ) -> Tuple[float, float, float]:
        """
        Given a beginning and ending step, compute the statistics of this
        step-aware hyper parameter. Used for estimating memory requirements for
        dynamic sparsity.

        Return [min, max, ending]
        """
        # By default, assume monotonic behavior and sample the callable
        begin_value = self(torch.tensor(begin)).item()
        end_value = self(torch.tensor(end)).item()
        if begin_value < end_value:
            return (begin_value, end_value, end_value)
        else:
            return (end_value, begin_value, end_value)


class Constant(HyperParameterSchedule):
    """
    Constant at every step.
    """

    def __init__(self, value: float):
        """
        Args:
            value: The constant value of the hyperparameter
        """
        super().__init__()
        self.value = value

    def compute(self, step: torch.Tensor):
        return torch.tensor(self.value)


class Linear(HyperParameterSchedule):
    r"""
    Linear change from an initial value.

    :math:`y(step) = init + step \cdot slope`
    """

    def __init__(self, init: float, slope: float):
        """
        Args:
            init: The initial value of the hyperparameter
            slope: The rate of change of the hyperparameter
        """
        super().__init__()
        self.init = torch.tensor(init)
        self.slope = torch.tensor(slope)

    def compute(self, step: torch.Tensor):
        if not isinstance(step, torch.Tensor):
            step = torch.tensor(step, dtype=torch.float)
        return self.init + step * self.slope


class Exp(HyperParameterSchedule):
    """
    Exponential, approaching an asymptotic final value

    :math:`y(step) = final + (init-final) e^{step \cdot gamma}`
    """

    def __init__(self, init: float, gamma: float, final: float = 1):
        """
        Args:
            init: The initial value of the hyperparameter
            gamma: The rate of change of the hyperparameter
            final: The final value of the hyperparameter (Default: 1.0)
        """
        super().__init__()
        self.final = torch.tensor(final)
        self.scale = self.final - torch.tensor(init)
        self.gamma = torch.tensor(gamma)

    def compute(self, step: torch.Tensor):
        if not isinstance(step, torch.Tensor):
            step = torch.tensor(step, dtype=torch.float)
        return self.final - self.scale * torch.exp(step * self.gamma)


class Power(HyperParameterSchedule):
    """
    Power law.

    :math:`y(step) = init \cdot beta^{step}`
    """

    def __init__(self, init: float, beta: float):
        """
        Args:
            init: The initial value of the hyperparameter
            beta: The rate of change of the hyperparameter
        """
        super().__init__()
        self.init = torch.tensor(init)
        self.beta = torch.tensor(beta)

    def compute(self, step: torch.Tensor):
        if not isinstance(step, torch.Tensor):
            step = torch.tensor(step, dtype=torch.float)
        return self.init * torch.pow(self.beta, step)


class Cosine(HyperParameterSchedule):
    """
    Cosine function for oscilating between an initial (maximum) value down to a
    minimum and back to the maximum every period.

    :math:`y(step) = o + a \cdot \cos(step \cdot \pi / half\_period)`, where
    :math:`o = (init + minimum)/2` and :math:`a = init - o`.
    """

    def __init__(self, init: float, half_period: float, minimum: float = 0.0):
        """
        Args:
            init: The initial value of the hyperparameter
            half_period: The number of steps to complete a full cycle
            minimum: The minimum value of the hyperparameter
        """
        super().__init__()
        # cos(x) mean is 0, compute mean of (init+min)/2
        o = (minimum + init) / 2
        # cos(pi) max is 1, remove offset
        a = init - o

        self.amp = torch.tensor(a)
        self.offset = torch.tensor(o)
        self.freq = torch.tensor(torch.pi / half_period)

    def compute(self, step: torch.Tensor):
        if not isinstance(step, torch.Tensor):
            step = torch.tensor(step, dtype=torch.float)
        return self.amp * torch.cos(step * self.freq) + self.offset

    def get_min_max_end(
        self, begin: int, end: int
    ) -> Tuple[float, float, float]:
        min_value = (-self.amp + self.offset).item()
        max_value = (self.amp + self.offset).item()
        end_value = self(torch.tensor(end)).item()
        if max_value < min_value:
            # swap, amp must be negative
            min_value, max_value = max_value, min_value
        return (min_value, max_value, end_value)


class Cycling(HyperParameterSchedule):
    """
    Hyper parameter cycling between discrete values at update steps.
    """

    def __init__(self, values: List[float]):
        """
        Args:
            values: A list of discrete values to cycle through
        """
        super().__init__()
        self.values = values
        self.index = torch.tensor(0, dtype=torch.int64)

    def compute(self, step: torch.Tensor) -> torch.Tensor:
        if not isinstance(step, torch.Tensor):
            step = torch.tensor(step, dtype=torch.float)

        # Terrible unrolled version to work around stack limitations
        v = torch.tensor(self.values[0], device=step.device)
        for i, vi in enumerate(self.values):
            vi = torch.tensor(vi, device=step.device)
            v = torch.where(self.index == i, vi, v)
        return v

    def update(self, is_update_step: torch.Tensor):
        self.index = torch.where(
            is_update_step,
            torch.where(self.index == len(self.values) - 1, 0, self.index + 1),
            self.index,
        )

    def get_min_max_end(
        self, begin: int, end: int
    ) -> Tuple[float, float, float]:
        # Technically not an "end" since it cycles, so assume its cycled
        # completely by the end of dynamic updates.
        return (min(self.values), max(self.values), self.values[-1])

    def visit_state(self, fn):
        new_index = fn(self.index)
        if new_index is not None:
            self.index = new_index

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state):
        self.index = state.pop("index")


class Lambda(HyperParameterSchedule):
    """
    Invoke a user's lambda function of step to obtain the hyper parameter.
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Args:
            fn: A lambda function that takes a step and returns a hyperparameter
        """
        super().__init__()
        self.fn = fn

    def compute(self, step: torch.Tensor):
        return self.fn(step)

    def get_min_max_end(
        self, begin: int, end: int
    ) -> Tuple[float, float, float]:
        # Can't assess any statistics of a user provided lambda
        return None


HyperParameterCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
HyperParameterScheduleType = Union[
    int,
    float,
    List[int],
    List[float],
    Tuple,
    Dict,
    HyperParameterCallable,
    HyperParameterSchedule,
]


def make_hyperparam_schedule(schedule):
    """
    Given some user specified configuration, construct a HyperParameterSchedule
    object that is step aware.
    """
    if isinstance(schedule, HyperParameterSchedule):
        return schedule
    if isinstance(schedule, (int, float)):
        return Constant(schedule)
    if isinstance(schedule, (list, tuple)):
        return Cycling(schedule)
    if callable(schedule):
        signature = inspect.signature(schedule)
        if signature_matches_type_hint(signature, HyperParameterCallable):
            return Lambda(schedule)

    hyperparam_classes = {
        cls.__name__.lower(): cls
        for cls in retrieve_all_subclasses(HyperParameterSchedule)
        if not inspect.isabstract(cls)
    }

    if isinstance(schedule, dict):
        schedule = schedule.copy()
        typename = schedule.pop("type", None)
        if not typename:
            raise ValueError("Must specify `type`")
        cls = hyperparam_classes.get(typename.lower())
        return cls(**schedule)

    valid_types = sorted(hyperparam_classes.keys())
    raise ValueError(
        f"Unhandled {schedule}. Options are:\n"
        f"* int/float: ConstantHyperParameter\n"
        f"* list[int/float]: CyclingHyperParameter\n"
        f"* Callable: LambdaHyperParameter\n"
        f"* BaseHyperParameter: used as-is\n"
        f"* {{\"type\": ...}} as one of {valid_types}"
    )


class UpdateSchedule(ABC):
    @abstractmethod
    def is_update_step(self, step: torch.LongTensor) -> torch.BoolTensor:
        """
        Given a training step rankless tensor, return a rankless bool tensor if
        this is a sparsity update step.
        """

    def __call__(self, step: torch.LongTensor) -> torch.BoolTensor:
        return self.is_update_step(step)


class FreqSchedule(UpdateSchedule):
    """
    When schedulding sparsity update steps on a regular interval, this class
    allows configuring the start and stop step in addition to the update
    frequency.
    """

    def __init__(self, freq=1, start=0, stop=None):
        """
        Args:
            freq: The frequency of steps at which to update the sparsity pattern (Default: 1)
            start: The step at which to start updating the sparsity pattern (Default: 0)
            stop: The step at which to stop updating the sparsity pattern (Default: None)
        """
        super().__init__()
        self.start = start
        self.freq = freq
        self.stop = stop

    def is_update_step(self, step: torch.LongTensor) -> torch.BoolTensor:
        # First, check if this is (after offsetting from start) an update step
        # based on the frequency
        check_step = step
        if self.start:
            check_step = step - self.start
        is_update_step = check_step % self.freq == 0

        # Next add the bounds checking if applicable
        if self.start:
            is_update_step &= step >= self.start
        if self.stop is not None:
            is_update_step &= step < self.stop

        return is_update_step


class ListSchedule(UpdateSchedule):
    """
    When schedulding requires an irregular update cadence, explicit steps can
    be provided as a list.
    """

    def __init__(self, steps: Union[List[int], torch.Tensor]):
        """
        Args:
            steps: A list of steps at which to update the sparsity pattern
        """
        super().__init__()
        steps = tuple(steps)
        self.steps = steps
        self.start = min(steps)
        self.stop = max(steps)

    def is_update_step(self, step: torch.LongTensor) -> torch.BoolTensor:
        is_update_step = torch.tensor(False, device=step.device)
        for s in self.steps:
            is_update_step |= step == s
        return is_update_step


UpdateScheduleCallable = Callable[
    # torch.tensor(shape=[], dtype=int64) -> torch.tensor(shape=[], dtype=bool)
    [torch.LongTensor],
    torch.BoolTensor,
]
UpdateScheduleType = Union[Dict, UpdateScheduleCallable]


def make_update_schedule(update: UpdateScheduleType) -> UpdateScheduleCallable:
    """
    Instantiate a supported schedule type.
    """
    if isinstance(update, UpdateSchedule):
        return update
    if update is None:
        # always update
        return FreqSchedule(freq=1)
    elif isinstance(update, dict):
        update = update.copy()
        if "freq" in update:
            return FreqSchedule(**update)
        elif "steps" in update:
            return ListSchedule(update["steps"])
    elif callable(schedule):
        signature = inspect.signature(schedule)
        if signature_matches_type_hint(signature, ScheduleCallable):
            return schedule

    raise ValueError(
        f"Invalid `update`: {update}. Valid options are:\n"
        f"* None: Assume every step is an update step"
        f'* {{"freq": freq, "start": start, "stop": stop}}\n'
        f'* {{"steps": steps}}: List of specific update steps\n'
        f"* Callable: Used as-is\n"
    )


UnshaperCallable = Callable[[torch.Tensor], torch.Tensor]
ShaperReturn = Tuple[torch.Tensor, UnshaperCallable]
ShaperCallable = Callable[[torch.Tensor], ShaperReturn]


class ScoreShaper(ABC):
    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> ShaperReturn:
        """
        Given a tensor, such as a score or mask, reshape it so that the inner
        dimension is the one over which magnitudes should be compared.

        Args:
            tensor: Will be reshaped so that the inner dimension
        Returns:
            tuple containing:
                - reshaped ``tensor``
                - Callable to reverse this shaper.

        """


class ScoreFlattener(ScoreShaper):
    """
    Default ScoreShaper which everything is flattened, providing a global
    competition for magnitude. If only sub-portions of the weight should
    compete for magnitude, provide an alternative shaper object.
    """

    def __call__(self, tensor: torch.Tensor) -> ShaperReturn:
        def unshaper(ret: torch.Tensor) -> torch.Tensor:
            return ret.view(tensor.shape)

        return tensor.view(-1), unshaper


class OutputGroupScoreShaper(ScoreShaper):
    """
    A ScoreShaper interface when weights are logically shaped as
    [num_groups*out_per_group, insize], but need to be scored in a "balanced"
    fashion as [num_groups, out_per_group*insize]

    Examples:

        >>> # Common score used for the following examples
        >>> score=torch.tensor([[1.0, 2.0],
        ...                     [0.0, -1.0]])

        >>> # 50% sparsity, drops the 2 lowest magnitude
        >>> make_mask_topk_sparsity(
        ...     score=score,
        ...     sparsity=torch.tensor(0.5),
        ... )
        tensor([[ True,  True],
                [False, False]])

        >>> # 50% sparsity, but computed rowwise
        >>> make_mask_topk_sparsity(
        ...     score=score,
        ...     sparsity=torch.tensor(0.5),
        ...     score_shaper=OutputGroupScoreShaper(num_groups=2)
        ... )
        tensor([[False,  True],
                [ True, False]])
    """

    def __init__(self, num_groups):
        self.num_groups = num_groups

    def __call__(self, tensor: torch.Tensor) -> ShaperReturn:
        def unshaper(ret: torch.Tensor) -> torch.Tensor:
            return ret.view(tensor.shape)

        return tensor.view(self.num_groups, -1), unshaper


class InputGroupScoreShaper(ScoreShaper):
    """
    A ScoreShaper interface when weights are logically shaped as
    [outsize, num_groups*in_per_group], but need to be scored in a "balanced"
    fashion as [num_groups, outsize*in_per_group]

    Examples:

        >>> # Common score used for the following examples
        >>> score=torch.tensor([[1.0, 0.0],
        ...                     [2.0, -1.0]])

        >>> # 50% sparsity, drops the 2 lowest magnitude
        >>> make_mask_topk_sparsity(
        ...     score=score,
        ...     sparsity=torch.tensor(0.5),
        ... )
        tensor([[ True, False],
                [ True, False]])

        >>> # 50% sparsity, but computed columnwise
        >>> make_mask_topk_sparsity(
        ...     score=score,
        ...     sparsity=torch.tensor(0.5),
        ...     score_shaper=InputGroupScoreShaper(num_groups=2)
        ... )
        tensor([[False,  True],
                [ True, False]])
    """

    def __init__(self, num_groups):
        self.num_groups = num_groups

    def __call__(self, tensor: torch.Tensor) -> ShaperReturn:
        O, I = tensor.shape
        # Swap [O,I] -> [I, O] and flatten [N, I/N*O]
        ret = tensor.permute(1, 0).reshape(self.num_groups, -1)

        def unshaper(ret: torch.Tensor) -> torch.Tensor:
            # flatten [N, I/N*O] -> [I, O] then swap to [O, I]
            return ret.view(I, O).permute(1, 0).contiguous()

        return ret, unshaper


def make_mask_drop_minimum(
    score: torch.FloatTensor,
    mask: torch.BoolTensor,
    drop_fraction: torch.FloatTensor,
    score_shaper: Optional[ShaperCallable] = None,
) -> torch.BoolTensor:
    """
    Given a sparse ``score`` (with ``mask``), return a new ``torch.BoolTensor``
    the same shape as `mask` where a ``drop_fraction`` portion of the currently
    present (``mask==True``) connections are dropped (``mask==False``).

    The connections are dropped at positions corresponding to the `lowest`
    values of ``score``.

    Equivalently, a subset of ``mask`` is returned corresponding to the
    `highest` magnitude elements of ``score``.

    Args:
        score: Values used to evaluate which positions to drop
        mask: Current connections, same shape as ``score``
        drop_fraction: What fraction of current connections to drop
        score_shaper: If given, ``score`` (and ``mask``) will be interpreted as
            multiple independent subtensors. This can be used to ensure
            sparsity distribution is "balanced" or to produce blockwise
            sparsity. By default, ``score`` and ``mask`` are reinterpreted as
            1D tensors, yielding completely unstructured sparsity.

    Returns:
        New mask that has existing connections dropped. No connections will be
        regrown (unless drop_fraction is negative).
    """
    if not score_shaper:
        score_shaper = ScoreFlattener()
    score, unshape = score_shaper(score)

    # Compute total remaining dense elements kept after dropping a certain
    # fraction of current connections.
    keep_fraction = 1 - drop_fraction
    current_k = mask.sum().float()

    # Divide the dropping evenly among groups if the score has them.
    groups = 1
    for dim in score.size()[:-1]:
        groups *= dim
    current_k /= groups
    num_dense_elem = (keep_fraction * current_k).int()
    # Return the new mask and the number of dense elements (often needed for
    # make_mask_grow_maximum with target sparsity)
    new_mask = unshape(_make_mask_topk_k(score, num_dense_elem))
    return new_mask, num_dense_elem


def make_mask_grow_maximum(
    score: torch.FloatTensor,
    mask: torch.BoolTensor,
    sparsity: torch.FloatTensor,
    mask_nonzero: Optional[torch.IntTensor] = None,
    score_shaper: Optional[ShaperCallable] = None,
) -> torch.BoolTensor:
    """
    Given a sparse ``score`` (with ``mask``), return a new torch.BoolTensor the
    same shape as ``mask`` where some currently pruned connections are regrown
    (from those positions with the highest score) such that the returned mask
    has the given target sparsity.

    If ``mask`` is already less sparse (has more connections) than the target,
    none are regrown and the original mask is returned as-is. That is, the
    given ``mask`` should be `more` sparse than the target sparsity.

    Args:
        score: Values used to evaluate which positions to regrow
        mask: Current connections, same shape as ``score``
        drop_fraction: What fraction of current connections to drop
        mask_nonzero: If given, the number of nonzero elements currently in the
            mask, used to control the number of connections needing regrowth.
            If it is not given, will be computed as ``mask.nonzero().int()``.
            Since ``make_mask_grow_maximum`` is often used in conjunction with
            ``make_mask_drop_minimum``, this value is commonly available.
        score_shaper: If given, ``score`` (and ``mask``) will be interpreted as
            multiple independent subtensors. This can be used to ensure
            sparsity distribution is "balanced" or to produce blockwise
            sparsity. By default, ``score`` and ``mask`` are reinterpreted as
            1D tensors, yielding completely unstructured sparsity.

    Returns:
        New mask that has connections regrown necessary to reach (decrease to)
        the target sparsity.
    """
    # Ensure mask and grow_mask are in fact disjoint (i.e. this function _only_
    # grows) by disqualifying any non-pruned score elements.
    score = torch.where(mask, float('-inf'), score)

    if not score_shaper:
        score_shaper = ScoreFlattener()
    score, unshape = score_shaper(score)

    # Regrow connections to reach the target sparsity.
    density = 1 - sparsity
    numel = torch.tensor(score.size(dim=-1), dtype=torch.float)
    num_dense_elem = (density * numel).int()

    # The final mask needs a total of num_dense_elem connections and will be
    # the union of 2 disjoint masks mask|grow_mask, so compute the size of
    # grow_mask.
    if mask_nonzero is None:
        mask_nonzero = mask.sum().int()
    num_grow_elem = torch.clamp(num_dense_elem - mask_nonzero, min=0)

    # Find the positions of the highest magnitude score needed to reach the
    # target sparsity after regrowing.
    grow_mask = unshape(_make_mask_topk_k(score, num_grow_elem))

    # Return the combined mask and grow_mask
    return mask.logical_or(grow_mask)


def make_mask_topk_sparsity(
    score: torch.FloatTensor,
    sparsity: torch.FloatTensor,
    score_shaper: Optional[ShaperCallable] = None,
) -> torch.BoolTensor:
    """
    Given a dense ``score``, return a ``torch.BoolTensor`` which is True at
    positions corresponding to values in the top ``k =
    (1-sparsity)*score.numel()`` of ``score``.

    Args:
        score: Values used to evaluate which positions to keep.
        sparsity: rankless tensor in range [0,1] controlling fraction of the
            resulting mask that will be pruned.
        score_shaper: If given, ``score`` will be interpreted as multiple
            independent subtensors. This can be used to ensure sparsity
            distribution is "balanced" or to produce blockwise sparsity. By
            default, ``score`` is reinterpreted as a 1D tensor, yielding
            completely unstructured sparsity.

    Returns:
        ``mask`` with given ``sparsity``, keeping only the highest values from
        ``score``.

    Examples:

        >>> # Common score used for the following examples
        >>> score=torch.tensor([[1.0, 2.0],
        ...                     [0.0, -1.0]])

        >>> # 25% sparsity, drops the one lowest magnitude
        >>> make_mask_topk_sparsity(
        ...     score=score,
        ...     sparsity=torch.tensor(0.25),
        ... )
        tensor([[ True,  True],
                [ True, False]])

        >>> # 75% sparsity, drops the 3 lowest magnitude
        >>> make_mask_topk_sparsity(
        ...     score=score,
        ...     sparsity=torch.tensor(0.75),
        ... )
        tensor([[False,  True],
                [False, False]])
    """
    if not score_shaper:
        score_shaper = ScoreFlattener()
    score, unshape = score_shaper(score)

    density = 1 - sparsity
    numel = torch.tensor(score.size(dim=-1), dtype=torch.float)
    num_dense_elem = (density * numel).int()

    new_mask = _make_mask_topk_k(score, num_dense_elem)
    return unshape(new_mask)


def _make_mask_topk_k(
    score: torch.FloatTensor,
    num_dense_elem: torch.IntTensor,
) -> torch.BoolTensor:
    if cstorch.use_cs():
        # `torch.topk` uses a python integer for the `k` operand, which will
        # change throughout training. Even though this integer is computed from
        # tensors (the sparsity schedule), calling .item() on it breaks the
        # ability to trace the dataflow.
        # Since we only trace the program once, this prevents us from using
        # `torch.topk. Although even if it somehow did accept a traceable
        # tensor for `k`, the result would not be statically shaped, causing
        # other issues.

        # Instead, sort the whole tensor...
        indices = torch.sort(score, dim=-1, descending=True).indices
        # .. and mask off all but the first k indices, replacing them with the
        # largest(0th) index. This works even if num_dense_elem == numel.
        iota = torch.arange(
            indices.shape[-1],
            dtype=num_dense_elem.dtype,
            device=num_dense_elem.device,
        )
        in_topk = iota < num_dense_elem
        indices = torch.where(in_topk, indices, indices[..., 0:1])
    else:
        # CPU/GPU
        _, indices = torch.topk(score, num_dense_elem.item(), dim=-1)

    mask = torch.zeros_like(score, dtype=torch.bool)
    # expand necessary due to bug in TorchScript
    src_opt = torch.tensor(True, dtype=mask.dtype, device=mask.device).expand(
        mask.shape
    )
    mask = mask.scatter(-1, indices, src_opt)
    return mask
