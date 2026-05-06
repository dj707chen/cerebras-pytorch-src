# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import copy
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from cerebras.pytorch.backend import current_backend_impl

from .step_closures import step_closure
from .tracker import RateTracker


class Profiler:
    """Class for tracking and querying various activities during execution.

    Once started, activities accessible as attributes of the profiler object.

    By default, the following activities are tracked:
        - total_samples: the total number of samples processed
        - rate: Smoothed samples/second of all the samples added since last
              queried
        - global_rate: Non-smoothed samples/second since the beginning of when
              the profiler was instantiated
        - total_time: Number of seconds since the profiler was instantiated

    Args:
        outdir: The directory where the performance data will be stored.
        activities: A list of activities to track. If None, the default rate tracker activity will
            be used.
    """

    def __init__(
        self,
        outdir: str,
        activities: Optional[List[Type["Activity"]]] = None,
    ):
        self.outdir = outdir

        self._activity_classes = (
            activities if activities is not None else [RateTrackerActivity]
        )
        self._activities: Optional[Dict[str, Activity]] = None

        names = set()
        for activity in self._activity_classes:
            if not issubclass(activity, Activity):
                raise TypeError(
                    f"Expected activity to be a subclass of Activity, got {type(activity)}"
                )

            name = getattr(activity, "name", None)
            if name is None:
                raise ValueError(f"Activity {activity} does not have a name")

            if name in names:
                raise ValueError(
                    f"Activity with name \"{name}\" already exists."
                )

            if name in self.__dict__:
                raise ValueError(
                    f"\"{name}\" is a reserved keyword. Please use a different name "
                    f"for the activity."
                )

            names.add(name)

    def __enter__(self):
        self._activities = {
            activity.name(): activity() for activity in self._activity_classes
        }
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._save_perf()
        except Exception as e:  # pylint: disable=broad-except
            logging.error(f"Failed to save performance data:\n{e}")
            if exc_type is None:
                # Only raise if there was no exception already
                raise e

    def __getattr__(self, name):
        if self._activities and name in self._activities:
            activity = self._activities[name]
            activity()  # Calls compute
            return activity  # Returns the activity instance

        return super().__getattribute__(name)

    def step(self, batch_size):
        """Updates all of the profiler's activities with the given batch size."""
        if self._activities is None:
            raise RuntimeError(
                f"Attempting to step the profiler when it has not been started. "
            )
        for activity in self._activities.values():
            activity.update(batch_size)

    def _save_perf(self):
        """Saves the performance data to the outdir."""

        if len(self._activities) == 1:
            perf_data = list(self._activities.values())[0]()
        else:
            perf_data = {
                name: activity() for name, activity in self._activities
            }

        os.makedirs(self.outdir, exist_ok=True)
        with open(os.path.join(self.outdir, "performance.json"), "w") as f:
            json.dump(perf_data, f, sort_keys=True, indent=4)


class Activity(ABC):
    """Defines a single activity that can be profiled."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the unique name of this activity."""

    @abstractmethod
    def update(self, batch_size) -> None:
        """Update the current activity with a batch."""

    @abstractmethod
    def compute(self) -> Any:
        """Compute the current value of this activity."""

    def __call__(self) -> Any:
        return self.compute()

    def __str__(self) -> str:
        return str(self.compute())


class RateTrackerActivity(Activity):
    def __init__(self):
        self._tracker = RateTracker()
        self._cached_result: Optional[dict] = None
        self.reset_time()

    # Total number of samples processed so far
    total_samples = property(
        fget=lambda self: self._compute_property("total_samples")
    )
    # Elapsed time so far, in seconds
    total_time = property(
        fget=lambda self: self._compute_property("total_time")
    )
    # Smoothed samples/second of all the samples added since last queried
    rate = property(fget=lambda self: self._compute_property("rate"))
    # Non-smoothed samples/second since the beginning of when the executor context was entered
    global_rate = property(
        fget=lambda self: self._compute_property("global_rate")
    )
    # Non-smoothed samples/second since the beginning of when the executor context was entered
    samples_per_sec = property(
        fget=lambda self: self._compute_property("samples_per_sec")
    )
    # Real flops utilization for the run
    flops_utilization = property(
        fget=lambda self: self._compute_property("flops_utilization")
    )

    def _compute_property(self, key):
        if self._cached_result is None:
            raise RuntimeError(
                f"Attempting to access \"{key}\" from {self.__class__.__name__} before it's "
                f"computed. You can compute the activity by calling it (i.e., `activity()`), or by "
                f"registering it to a profiler object which will automatically call the activity "
                f"when appropriate."
            )
        return self._cached_result[key]

    @classmethod
    def name(cls) -> str:
        return "rate_tracker"

    @step_closure
    def reset_time(self):
        """Reset the tracker's start time to the current time."""
        # We reset the tracker's start time inside a step closure here so that
        # the time is reset after compile and execute setup is done.
        # TODO: add an offset of 1 so that the time isn't ~0 when the first
        #       rate/global_rate is computed
        self._tracker.reset_time(offset=0)

    def update(self, batch_size):
        if self._tracker and batch_size:
            self._tracker.add(batch_size)
        self._cached_result = None

    def compute(self):
        if self._cached_result is None:
            global_rate = self._tracker.global_rate()
            self._cached_result = {
                "total_samples": self._tracker.total_count,
                "total_time": self._tracker.elapsed_seconds(),
                "rate": self._tracker.rate(),
                "global_rate": global_rate,
                "samples_per_sec": global_rate,
                "flops_utilization": None,
            }
            backend = current_backend_impl()
            if (
                backend.is_csx
                and backend.appliance
                and backend.appliance.compile_resp
            ):
                algo_flops = backend.appliance.compile_resp.perf_info.algo_flops
                supported_flops = (
                    backend.appliance.compile_resp.perf_info.supported_flops
                )
                if algo_flops and supported_flops:
                    self._cached_result["flops_utilization"] = round(
                        (
                            (float(algo_flops * global_rate) / supported_flops)
                            * 100
                        ),
                        3,
                    )

        return copy.deepcopy(self._cached_result)
