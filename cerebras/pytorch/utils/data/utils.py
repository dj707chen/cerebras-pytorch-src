# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Utility and helper functions used by the Cerebras dataloader."""
import math
from dataclasses import dataclass
from typing import List, Optional
from warnings import warn

import numpy as np
import torch

import cerebras.pytorch as cstorch
from cerebras.appliance.data.dtypes import bf16, is_bf16
from cerebras.pytorch.utils._num import ceildiv
from cerebras.pytorch.utils.nest import visit_torch_tensors


@dataclass
class Schedule:
    """Generic schedule object that represents a collection of step intervals.

    Args:
        intervals: List of ranges.
    """

    @dataclass
    class Range:
        """A range of steps.

        Args:
            start: The starting index (inclusive).
            end: The end index (exclusive).
            step: the jump size.
            include_last: Whether the `end - 1` is included in the interval or not,
                regardless of whether it overlaps (start + step * N).
        """

        start: int
        end: int
        step: int
        include_last: bool

        def __post_init__(self):
            if not isinstance(self.start, int) or self.start < 0:
                raise ValueError(f"start ({self.start}) must an integer >= 0.")
            if not isinstance(self.end, int) or self.end <= self.start:
                raise ValueError(
                    f"end ({self.end}) must be an integer greater than "
                    f"start {self.start}."
                )
            if not isinstance(self.step, int) or self.step < 1:
                raise ValueError(
                    f"step ({self.step}) must be an integer greater than zero."
                )
            if not isinstance(self.include_last, bool):
                raise ValueError(
                    f"include_last must be a bool, got {type(self.include_last)}"
                )

        def match(self, index: int) -> bool:
            """Returns whether the given index belongs to this interval."""
            return (self.start <= index < self.end) and (
                ((index - self.start) % self.step == 0)
                or (index == self.end - 1 and self.include_last)
            )

        def __iter__(self):
            yield from self.range()
            if self.include_last:
                yield self.end

        def range(self):
            return range(self.start, self.end, self.step)

    intervals: List[Range]

    def __post_init__(self):
        if any(
            not isinstance(interval, Schedule.Range)
            for interval in self.intervals
        ):
            raise ValueError(f"interval must of type {Schedule.Range}")

        for interval1, interval2 in zip(self.intervals, self.intervals[1:]):
            if interval2.start < interval1.end:
                raise ValueError(
                    f"Intervals must be non-overlapping ranges, but got {self.intervals}."
                )

    def match(self, index: int) -> bool:
        """Returns whether the given index belongs to this schedule."""
        return any(interval.match(index) for interval in self.intervals)

    def __iter__(self):
        for interval in self.intervals:
            yield from interval

    def range(self):
        return [interval.range() for interval in self.intervals]


def compute_num_steps(
    dataloader: torch.utils.data.DataLoader,
    initial_step: int = 0,
    num_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
    num_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    grad_accum_steps: int = 1,
):
    """
    Computes the number of steps to execute on the system based on the
    provided step information.

    Args:
        dataloader: The dataloader itself which is used to determine the length
            of the dataset if available
        initial_step: The step to begin on. An error is thrown if the initial
            step exceeds the maximal steps calulated below
        num_steps: The number of steps to run
        max_steps: The maximum number of steps to run
        num_epochs: The number of epochs to run
        steps_per_epoch: The number of steps to run each epoch
        grad_accum_steps: The number of steps accumulate gradients before stepping

    Note:
        At least one of num_steps, max_steps, or num_epochs must be specified

    Returns:
        The calculated total number of steps to execute
    """

    def _check_steps(name, value, allow_none=False, allow_zero=False):
        if value is None:
            if not allow_none:
                raise ValueError(f"`{name}` cannot be None.")
        else:
            if not isinstance(value, int):
                raise ValueError(
                    f"`{name}` must be an integer, but got {type(value)}."
                )
            if value == 0 and not allow_zero:
                raise ValueError(f"`{name}` must be greater than zero.")
            if value < 0:
                raise ValueError(
                    f"`{name}` cannot be negative, but got {value}."
                )

    if num_epochs is not None and num_steps is not None:
        raise ValueError(
            "Only one of `num_epochs` or `num_steps` can be specified."
        )

    _check_steps(
        "initial_step", initial_step, allow_none=False, allow_zero=True
    )
    _check_steps("num_steps", num_steps, allow_none=True)
    _check_steps("max_steps", max_steps, allow_none=True)
    _check_steps("num_epochs", num_epochs, allow_none=True)
    _check_steps("steps_per_epoch", steps_per_epoch, allow_none=True)
    _check_steps("grad_accum_steps", grad_accum_steps, allow_none=False)

    try:
        # Dataset length is known
        dataloader_size = len(dataloader)
        assert dataloader_size > 0, "Dataloader does not generate any batches."
        if steps_per_epoch is not None:
            if steps_per_epoch > dataloader_size:
                raise ValueError(
                    f"The requested steps per epoch of {steps_per_epoch} "
                    f"exceeds total steps in an epoch, which is "
                    f"{dataloader_size}."
                )
        else:
            steps_per_epoch = dataloader_size

        # With grad accumulation, the global step is incremented every Nth
        # batch, so our effective steps per epoch needs to be adjusted.
        if grad_accum_steps > steps_per_epoch:
            raise ValueError(
                f"Gradient accumulation steps of {grad_accum_steps} is "
                f"greater than batches per epoch of {steps_per_epoch}."
            )

        steps_per_epoch //= grad_accum_steps
    except TypeError:
        # Dataset length is not known
        if num_epochs is not None:
            raise ValueError(
                "Specifying num_epochs for datasets with unknown length is "
                "not allowed. Please control training behavior through "
                "number of steps instead."
            )
        steps_per_epoch = 1

    # Calculate total steps
    total_steps = math.inf
    if num_epochs is not None:
        total_steps = min(total_steps, num_epochs * steps_per_epoch)
    if num_steps is not None:
        total_steps = min(total_steps, num_steps)
    if max_steps is not None:
        remaining_steps = max_steps - initial_step
        if remaining_steps <= 0:
            raise RuntimeError(
                f"Initial global step {initial_step} already exceeds "
                f"max step {max_steps}."
            )
        total_steps = min(total_steps, remaining_steps)

    # At least one of the above if blocks must have been true.
    # Adding an assert in case someone makes a mistake.
    if math.isinf(total_steps):
        raise ValueError(
            "One of num_epochs, num_steps, or max_steps must be provided"
        )

    if num_epochs is None:
        steps_per_epoch = total_steps

    # Override steps_per_epoch depending on the num_epochs computation
    num_epochs = ceildiv(total_steps, steps_per_epoch)
    steps_per_epoch = ceildiv(total_steps, num_epochs)

    return total_steps


def infer_batch_size(data, batch_size=None) -> Optional[int]:
    """Infers the batch size from a dataloader batch.

    Args:
        data: A nested structure of tensors.
        batch_size: The batch size to compare against.
            If None, the batch size is inferred from the data.
    Returns:
        If all tensors have the same batch size, it is returned.
        If inconsistent batch sizes are seen across tensors in the batch,
        None is returned in the CPU/GPU case and an error is raised in
        the CSX case.
    """
    inferred_batch_sizes = set(
        1 if len(tensor.size()) == 0 else tensor.size()[0]
        for _, tensor in visit_torch_tensors(data)
    )
    if len(inferred_batch_sizes) > 1:
        if cstorch.use_cs():
            raise RuntimeError(
                f"Only uniform batch sizes are supported in CS runs, but "
                f"the dataloader returned a batch with batch sizes "
                f"{inferred_batch_sizes}. "
            )
        warn(
            f"Detected non-uniform batch sizes within the same batch: "
            f"{inferred_batch_sizes}. While this is allowed in non-CSX "
            f"runs, it may throw off metrics such as rate profiling. "
            f"The run will proceed assuming no batch size."
        )
        return None

    if len(inferred_batch_sizes) == 1:
        inferred_batch_size = inferred_batch_sizes.pop()

        if batch_size is not None and inferred_batch_size != batch_size:
            if cstorch.use_cs():
                raise RuntimeError(
                    f"Only uniform batch sizes are supported in CS runs, but "
                    f"the dataloader returned two different batches with "
                    f"batch sizes {batch_size} and {inferred_batch_size}. "
                    f"Make sure to set `drop_last=True` in the dataloader."
                )
            else:
                warn(
                    f"Detected non-uniform batch sizes between batches "
                    f"({batch_size} vs {inferred_batch_size}). "
                    f"While this is allowed in non-CSX runs, it may throw off "
                    f"metrics such as rate profiling. "
                )

        return inferred_batch_size

    raise RuntimeError(
        "We could not detect any torch tensors in the input data "
        "returned by the dataloader. We expect the dataloader to "
        "return a nested dict/list/tuple of tensors. If there are "
        "custom types that internally hold tensors, we are not "
        "currently able to detect them. Please ensure that the "
        "dataloader returns tensors in the expected format."
    )


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a torch tensor to a numpy array."""
    if tensor.dtype == torch.bfloat16:
        assert bf16.itemsize == 2  # Sanity check
        return tensor.view(torch.int16).numpy().view(bf16)
    return tensor.numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    """Converts a numpy array to a torch tensor."""
    # Copy non-writeable array to make it writable for torch.from_numpy
    if not array.flags.writeable:
        array = array.copy()

    if is_bf16(array.dtype):
        return torch.from_numpy(array).view(torch.bfloat16)
    return torch.from_numpy(array)
