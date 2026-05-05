# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for implementing metrics."""
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import MethodType
from typing import Any

import torch

import cerebras.pytorch as cstorch


class _empty:
    """An empty object to use as a sentinel for caching metric results."""


class Metric(torch.nn.Module, ABC):
    """Base class for implementing metrics compatible with Cerebras WSC.

    This class is designed to be used as a base class for implementing metrics
    compatible with Cerebras Wafer-Scale Cluster, but they also work with CPU
    and GPU backends.

    To implement a new metric, subclass `Metric` and implement the following:
    - `reset`: This is to initialize the metric state.
    - `update`: This is to update the metric state at every iteration.
    - `compute`: This is to compute the final metric value based on the state.

    To use metrics, instantiate them and call them with the appropriate inputs.
    For example:
        >>> metric = MyMetric()
        >>> metric(input_1, input_2)  # Calls update and compute
        >>> metric.compute()  # Returns the final (cached) metric value
    """

    # Keep a registry of all metrics so we can reference them in the backend
    registry = defaultdict(list)

    def __init__(self, name: str):
        """Constructs a `Metric` instance.

        Args:
            name: The name of the metric. This is used to reference the metric
                and does not have to be unique.
        """
        super().__init__()

        self.name = name

        # Keeps track of total number of times the metric was updated
        self._num_updates = 0

        # Add the metric to the registry
        self.registry[self.name].append(self)

        # Cached result of the last compute call
        self._cached_result = _empty

        # Wrap reset, update and compute methods
        self._wrap_reset()
        self._wrap_update()
        self._wrap_compute()

        # Call reset to initialize metric state
        self.reset()

    @property
    def num_updates(self) -> int:
        """Returns the number of times the metric was updated."""
        return self._num_updates

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric state."""

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates the metric state."""

    @abstractmethod
    def compute(self) -> Any:
        """Computes and returns the current metric value."""

    def register_state(
        self, name: str, tensor: torch.Tensor, persistent: bool = False
    ) -> None:
        """Registers a state variable to the module.

        By default, metric state variables are non-persistent buffers that
        are not included in the module's state dictionary. To have them as part
        of the state dictionary, set `persistent=True`.

        Once registered, the state variable can be accessed as an attribute
        on the module by the given name.

        Args:
            name: The name of the state variable.
            tensor: The tensor to register.
            persistent: Whether this state is part of the module's `state_dict`.
        """
        self.register_buffer(name, tensor, persistent=persistent)

    def forward(self, *args, **kwargs) -> Any:
        """Updates and computes the metric value."""
        self.update(*args, **kwargs)
        return self.compute()

    def _wrap_reset(self) -> None:
        """Wraps the update method to clear the cache before running update."""
        reset = self.reset

        @wraps(reset)
        def wrapped_reset(self: Metric, *args, **kwargs):
            reset_result = reset(*args, **kwargs)
            self._num_updates = 0
            self._cached_result = _empty
            return reset_result

        self.reset = MethodType(wrapped_reset, self)

    def _wrap_update(self) -> None:
        """Wraps the update method to clear the cache before running update."""
        update = self.update

        @wraps(update)
        def wrapped_update(self: Metric, *args, **kwargs):
            self._cached_result = _empty
            update_result = update(*args, **kwargs)
            self._num_updates += 1
            return update_result

        self.update = MethodType(wrapped_update, self)

    def _wrap_compute(self) -> None:
        """Wraps the compute method to cache the result."""
        compute = self.compute

        @wraps(compute)
        def wrapped_compute(self: Metric):
            if self._cached_result is not _empty:
                return self._cached_result

            # Cache the result (which could be a lazy tensor) so if we call
            # compute() again without an explicit update(), we don't recompute
            # unnecessarily. User should not be explicitly calling compute()
            # or update() anyway, so this is mostly a sanity check. Instead,
            # they should be calling metric instance as a function which will
            # both update and compute.
            self._cached_result = compute()

            @cstorch.step_closure
            def cache_result(r):
                # Once the step closure runs, the cached result is no longer
                # a lazy tensor and has been materialized to a CPU value.
                # This allows further `compute()` calls with no `update()`
                # to return the cached result without unnecessary re-computing.
                self._cached_result = r

            cache_result(self._cached_result)

            return self._cached_result

        self.compute = MethodType(wrapped_compute, self)

    def __float__(self) -> float:
        """Returns the floating point representation of the metric value."""
        return float(self.compute())


def get_all_metrics():
    """Get all registered metrics."""
    # TODO: Deprecate this eventually as we don't want to be keeping global
    #       state if we can help it
    for name, metric_list in Metric.registry.items():
        for i, metric in enumerate(metric_list):
            yield f"{name}.{i}" if len(metric_list) > 1 else name, metric
