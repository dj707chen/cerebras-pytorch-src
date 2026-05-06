# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""A tool for benchmarking dataloaders."""

import abc
import functools
import inspect
import time
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

import numpy as np
import psutil
import torch
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from cerebras.appliance.log import TRACE, VERBOSE
from cerebras.appliance.log import logger as _logger
from cerebras.appliance.utils.memory import get_process_memory_full_info
from cerebras.appliance.utils.process import map_processes
from cerebras.appliance.utils.units import (
    bytes_to_human,
    convert_time_unit,
    time_to_human,
)
from cerebras.pytorch.utils.tracker import RateTracker

timedelta = np.timedelta64
logger = _logger.getChild("benchmark.dataloader")


@dataclass
class TensorSpec:
    """Specification for a single tensor.

    Args:
        shape: Shape of the tensor.
        dtype: Data type of the tensor.
    """

    shape: torch.Size
    dtype: torch.dtype

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "TensorSpec":
        """Constructs a tensor spec from a tensor returned by dataloader.

        Args:
            tensor: Tensor returned by dataloader.
        Returns:
            Tensor spec for the tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"DataLoader returned a non-tensor of type `{type(tensor)}` "
                f"and value: {tensor}"
            )
        return TensorSpec(tensor.shape, tensor.dtype)

    def __str__(self):
        return f"TensorSpec(shape={tuple(self.shape)}, dtype={self.dtype})"

    def __hash__(self):
        return hash((tuple(self.shape), self.dtype))


@dataclass
class BatchSpec:
    """Specification for a single batch of data.

    Args:
        tensors: Flattened list of tensor specs in the batch.
        spec: PyTree structure of the batch.
    """

    tensors: List[TensorSpec]
    spec: TreeSpec

    @staticmethod
    def from_batch(batch) -> "BatchSpec":
        """Constructs a batch spec from a batch returned by dataloader.

        Args:
            batch: Batch returned by dataloader.
        Returns:
            Batch spec for the batch.
        """
        tensors, spec = tree_flatten(batch)
        tensors = [TensorSpec.from_tensor(tensor) for tensor in tensors]
        return BatchSpec(tensors, spec)

    def __hash__(self):
        return hash((tuple((hash(t) for t in self.tensors)), repr(self.spec)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BatchSpec)
            and self.tensors == other.tensors
            and self.spec == other.spec
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


@dataclass
class BatchSpecOccurence:
    """Dataclass for storing occurence of a unique batch spec.

    Args:
        epoch_step: Epoch step at which the batch spec first occured.
        epoch: Epoch at which the batch spec first occured.
        count: Total number of times the batch spec occured across all epochs.
    """

    epoch_step: int
    epoch: int
    count: int


@dataclass
class BatchMetrics:
    """Metrics for a single batch of a dataloader experiment.

    Args:
        epoch_step: Epoch step at which the batch was generated.
        global_step: Global step at which the batch was generated.
        local_rate: Local rate (in steps/second) at the sampling frequency.
            This is the instantaneous rate (relative to previous batch) at
            which the batch was generated.
        global_rate: global rate (in steps/second) at the sampling frequency.
            This is the global rate since the start of the iterating epochs.
        profile_activities: Dictionary of profile activities and their values.
        sampling_time_ns: Time at which the batch was sampled.
    """

    epoch_step: int
    global_step: int
    local_rate: float
    global_rate: float
    profile_activities: Dict[str, Any] = field(default_factory=dict)
    sampling_time_ns: int = field(default_factory=time.time_ns)


@dataclass
class EpochMetrics:
    """Metrics for a single epoch of a dataloader experiment.

    Args:
        iterator_creation: Time to create the dataloader iterator.
        iteration_time: Time to iterate the entire epoch excluding the creation
            of the iterator.
        total_steps: Total number of steps in the epoch.
        batch_metrics: List of metrics for batches generated in the epoch.
        start_time_ns: Time at which the epoch started.
        end_time_ns: Time at which the epoch ended.
    """

    iterator_creation: timedelta = field(default_factory=lambda: timedelta(0))
    iteration_time: timedelta = field(default_factory=lambda: timedelta(0))
    total_steps: int = 0
    batch_metrics: List[BatchMetrics] = field(default_factory=list)
    start_time_ns: int = field(default_factory=time.time_ns)
    end_time_ns: int = 0

    @property
    def total_time(self) -> timedelta:
        """Returns the total time to create and iterate the epoch."""
        return self.iterator_creation + self.iteration_time


@dataclass
class Metrics:
    """Metrics for a single dataloader experiment.

    Args:
        dataloader_build_time: Time to build the dataloader.
        epoch_metrics: List of metrics for each epoch.
        batch_specs: Mapping between unique batch specs found and their
            occurences.
        total_time: Total time to iterate through all epochs.
        global_rate: Overall global rate in steps/second.
        is_partial: Whether the metrics are partial. This can happen if
            the benchmark is interrupted in the middle of execution.
        start_time_ns: Time at which the experiment started.
        end_time_ns: Time at which the experiment ended.
    """

    dataloader_build_time: timedelta = field(
        default_factory=lambda: timedelta(0)
    )
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    batch_specs: Dict[BatchSpec, BatchSpecOccurence] = field(
        default_factory=dict
    )
    total_time: timedelta = field(default_factory=lambda: timedelta(0))
    global_rate: float = 0.0
    is_partial: bool = True
    start_time_ns: int = field(default_factory=time.time_ns)
    end_time_ns: int = 0

    @property
    def total_steps(self) -> int:
        """Returns the total number of steps across all epochs."""
        return sum(epoch.total_steps for epoch in self.epoch_metrics)

    @property
    def global_sample_rate(self) -> Optional[float]:
        """Returns the overall global rate in samples/second.

        Note that this value only exists if all batches have the exact same
        structure, dtypes, and shapes. Otherwise, this value is None.
        """
        if len(self.batch_specs) != 1:
            return None

        batch_spec = next(iter(self.batch_specs))
        if not batch_spec.tensors[0].shape:
            return None

        batch_size = batch_spec.tensors[0].shape[0]
        for tensor_spec in batch_spec.tensors[1:]:
            if not tensor_spec.shape or batch_size != tensor_spec.shape[0]:
                return None

        return self.global_rate * batch_size


def benchmark_dataloader(
    input_fn: Callable[..., Iterable],
    num_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    sampling_frequency: Optional[int] = None,
    profile_activities: Optional[List[str]] = None,
    print_metrics: bool = True,
) -> Metrics:
    """Utility to benchmark a dataloader.

    Args:
        input_fn: Function that creates and returns a dataloader.
        num_epochs: Number of epochs to iterate over the dataloader. If None,
            the dataloader is only iterated for one epoch.
        steps_per_epoch: Number of steps to iterate over the dataloader in each
            epoch. If None, the dataloader is iterated in its entirety.
        sampling_frequency: Frequency at which to sample metrics. If None, a
            default value of 100 (i.e. every 100 steps) is used. First step of
            each epoch is always sampled.
        profile_activities: List of optional activities to profile. If None, no
            extra activities are profiled. Note that these may incur additional
            overhead and could affect overall performance of the dataloader,
            especially if the sampling frequency is high.
        print_metrics: Whether to pretty print the final metrics to console.
    Returns:
        Metrics for the dataloader experiment.
    """
    try:
        # pylint: disable=unused-import
        import pandas as pd  # noqa
    except ImportError:
        if print_metrics:
            raise ImportError(
                "Printing metrics requires pandas. "
                "Please install pandas to use pretty printing."
            )

    if profile_activities is None:
        profile_activities = set()
    if not isinstance(profile_activities, (list, tuple, set)):
        raise TypeError(
            f"Invalid profile_activities type: {type(profile_activities)}. "
            f"Expected a list, tuple, or set."
        )
    available_profilers = Profiler.get_profilers()
    profilers = ProfilerCollection()
    for activity in profile_activities:
        if activity not in available_profilers:
            raise ValueError(
                f"Invalid profile activity: {activity}. "
                f"Available profiling activities are: "
                f"{list(available_profilers.keys())}."
            )
        profilers[activity] = available_profilers[activity]()
    if profilers:
        logger.warning(
            f"Profiling activities could negatively impact performance of the "
            f"dataloader, especially at high sampling frequencies. Please note "
            f"that the reported metrics may not be fully representative of the "
            f"actual performance of the dataloader."
        )

    if num_epochs is None:
        num_epochs = 1
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, but got {num_epochs}.")

    if sampling_frequency is None:
        sampling_frequency = 100
    if sampling_frequency < 0:
        raise ValueError(
            f"sampling_frequency must be non-negative, "
            f"but got {sampling_frequency}."
        )

    if steps_per_epoch is not None and steps_per_epoch <= 0:
        raise ValueError(
            f"steps_per_epoch must be positive, but got {steps_per_epoch}."
        )

    metrics = Metrics()
    global_step = 0

    profilers.on_start()

    try:
        logger.verbose("Building dataloader")
        global_timer = _Tracker()
        try:
            dataloader = input_fn()
        except Exception as e:
            raise RuntimeError("Failed to create the dataloader.") from e
        metrics.dataloader_build_time = global_timer.elapsed()
        logger.verbose("Dataloader built")

        global_timer.reset()
        for epoch in range(1, num_epochs + 1):
            epoch_str = f"Epoch={epoch}"

            logger.verbose(f"{epoch_str}: Starting")

            epoch_metrics = EpochMetrics()
            metrics.epoch_metrics.append(epoch_metrics)

            epoch_timer = _Tracker()

            profilers.on_epoch_start(epoch)

            logger.verbose(f"{epoch_str}: Creating dataloader iterator")
            try:
                dataloader_iter = iter(dataloader)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create dataloader iterator at epoch {epoch}."
                ) from e
            epoch_metrics.iterator_creation = epoch_timer.elapsed()
            logger.verbose(f"{epoch_str}: Dataloader iterator created")

            logger.verbose(f"{epoch_str}: Iterating over dataloader")
            epoch_timer.reset()
            batch_timer = _Tracker()
            epoch_step = 0
            while steps_per_epoch is None or epoch_step < steps_per_epoch:
                profilers.on_step_start(epoch, epoch_step)

                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    if epoch_step == 0:
                        raise RuntimeError(
                            f"Dataloader returned no batches at epoch {epoch}."
                        ) from None
                    break
                except Exception as e:
                    raise RuntimeError(
                        f"Dataloader ran into error when generating batch at "
                        f"epoch step {epoch_step + 1}, "
                        f"global step {global_step + 1}."
                    ) from e

                epoch_step += 1
                global_step += 1
                batch_timer.add(1)
                global_timer.add(1)
                step_str = f"{epoch_str}, Step={epoch_step}"

                if logger.isEnabledFor(TRACE):
                    logger.trace(f"{step_str}: {pformat(batch)}")

                if epoch_step == 1 or (
                    sampling_frequency > 0
                    and epoch_step % sampling_frequency == 0
                ):
                    profile_activites = {}
                    for activity, profiler in profilers.items():
                        profile_activites[activity] = profiler.profile()

                    batch_metrics = BatchMetrics(
                        epoch_step=epoch_step,
                        global_step=global_step,
                        local_rate=batch_timer.global_rate(),
                        global_rate=global_timer.global_rate(),
                        profile_activities=profile_activites,
                    )
                    epoch_metrics.batch_metrics.append(batch_metrics)

                    if logger.isEnabledFor(VERBOSE):
                        values = [
                            f"LocalRate={batch_metrics.local_rate:.2f} steps/s",
                            f"GlobalRate={batch_metrics.global_rate:.2f} steps/s",
                        ]
                        for activity, val in profile_activites.items():
                            values.append(
                                f"{profilers[activity].display_name()}="
                                f"{profilers[activity].to_human(val)}"
                            )
                        logger.verbose(f"{step_str}: " + ", ".join(values))

                try:
                    batch_spec = BatchSpec.from_batch(batch)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse batch at epoch step {epoch_step}, "
                        f"global step {global_step}."
                    ) from e

                if batch_spec not in metrics.batch_specs:
                    metrics.batch_specs[batch_spec] = BatchSpecOccurence(
                        epoch_step=epoch_step,
                        epoch=epoch,
                        count=1,
                    )
                else:
                    metrics.batch_specs[batch_spec].count += 1

                epoch_metrics.total_steps = epoch_step
                epoch_metrics.iteration_time = epoch_timer.elapsed()

                batch_timer.reset()
                profilers.on_step_end(epoch, epoch_step)

            epoch_metrics.end_time_ns = time.time_ns()

            profilers.on_epoch_end(epoch)

            logger.verbose(f"{epoch_str}: Finished iterating dataloader")

        metrics.is_partial = False
    except Exception as e:
        logger.exception(
            "Dataloader benchmark failed. Returning partial results."
        )
        raise
    except KeyboardInterrupt:
        logger.warning(
            "Dataloader benchmark interrupted. Returning partial results."
        )
    finally:
        metrics.total_time = global_timer.elapsed()
        metrics.global_rate = global_timer.global_rate()
        metrics.end_time_ns = time.time_ns()

        profilers.on_end()

    if print_metrics and global_step > 0:
        pprint_metrics(metrics)

    if len(metrics.batch_specs) > 1:
        logger.warning(
            f"Running on Cerebras Wafer-Scale Cluster requires all batches "
            f"to have the same PyTree structure with tensors at the same "
            f"index having consistent shape and dtype, but found "
            f"{len(metrics.batch_specs)} unique batch specs. This DataLoader "
            f"may not be compatible for running on Cerebras Wafer-Scale "
            f"Cluster."
        )

    if len(metrics.batch_specs) == 1:
        batch_sizes = set()
        for tensor in next(iter(metrics.batch_specs)).tensors:
            if tensor.shape:
                batch_sizes.add(tensor.shape[0])
            else:
                batch_sizes.add(None)
        if len(batch_sizes) > 1:
            logger.warning(
                f"Running on Cerebras Wafer-Scale Cluster requires all tensors "
                f"in a batch to have the same batch size, but found "
                f"{len(batch_sizes)} unique batch sizes. This DataLoader is "
                f"not be compatible for running on Cerebras Wafer-Scale "
                f"Cluster."
            )

    return metrics


def pprint_metrics(metrics: Metrics) -> None:
    """Pretty prints the metrics to console.

    Args:
        metrics: Metrics to print.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Pretty printing requires pandas. "
            "Please install pandas to use pretty printing."
        )

    print()
    print("##################################################################")
    print("################## Dataloader Benchmark Results ##################")
    if metrics.is_partial:
        print(
            "######################## (Partial Results) #######################"
        )
    print("##################################################################")
    print()

    print(f"Overall statistics")
    print(f"------------------")
    print(
        f"  Dataloader creation time: "
        f"{time_to_human(metrics.dataloader_build_time, 'ns')}"
    )
    print(f"  Total iteration time: {time_to_human(metrics.total_time, 'ns')}")
    print(f"  Total steps: {metrics.total_steps}")
    throughput_str = f"Total throughput: {metrics.global_rate:.2f} steps/s"
    if metrics.global_sample_rate is not None:
        throughput_str += f" ({metrics.global_sample_rate:.2f} samples/s)"
    print(f"  {throughput_str}")
    print()
    print()

    available_profilers = Profiler.get_profilers()

    epoch_data = []
    profile_columns = []
    epoch_idx = []
    for i, epoch_metrics in enumerate(metrics.epoch_metrics):
        if not epoch_metrics.batch_metrics:
            continue

        epoch_idx.append(f"Epoch {i + 1}")

        data = []

        batch_df = pd.DataFrame(
            epoch_metrics.batch_metrics,
            index=range(1, len(epoch_metrics.batch_metrics) + 1),
        )
        profile_df = batch_df["profile_activities"].apply(pd.Series)

        data.append(epoch_metrics.total_steps)
        data.append(len(epoch_metrics.batch_metrics))
        data.append(time_to_human(epoch_metrics.iterator_creation, "ns"))
        data.append(time_to_human(epoch_metrics.iteration_time, "ns"))
        for attr in ["local_rate", "global_rate"]:
            for agg in ["min", "mean", "max"]:
                data.append(getattr(batch_df[attr], agg)())

        for activity in profile_df.columns:
            for agg in ["min", "mean", "max"]:
                val = getattr(profile_df[activity], agg)()
                val = available_profilers[activity].to_human(val)
                data.append(val)

            name = available_profilers[activity].display_name()
            if name not in profile_columns:
                profile_columns.append(name)

        epoch_data.append(data)

    epoch_cols = pd.MultiIndex.from_tuples(
        [
            ("Steps", ""),
            ("Sample Points", ""),
            ("Iterator Creation", ""),
            ("Epoch Iteration", ""),
        ]
    ).union(
        pd.MultiIndex.from_product(
            [
                ["Local steps/s", "Global steps/s"] + profile_columns,
                ["min", "avg", "max"],
            ]
        ),
        sort=False,
    )
    epoch_df = pd.DataFrame(epoch_data, columns=epoch_cols, index=epoch_idx)
    epoch_df.dropna(axis=1, how="all", inplace=True)
    epoch_df.update(
        epoch_df.select_dtypes(include=[timedelta]).apply(
            lambda col: col.astype(timedelta).apply(
                lambda v: time_to_human(v, "ns")
            )
        )
    )
    print(f"Epoch statistics")
    print(f"----------------")
    print(epoch_df.to_string(float_format="{:.2f}".format))
    print()
    print()

    print(f"Unique Batch Specs")
    print(f"-------------------")
    for batch_spec, occurence in sorted(
        metrics.batch_specs.items(),
        key=lambda i: (i[1].epoch, i[1].epoch_step),
    ):
        print(
            f"  First occurence: "
            f"epoch={occurence.epoch}, epoch step={occurence.epoch_step}"
        )
        print(f"  Total number of occurences: {occurence.count}")
        batch = tree_unflatten(
            [str(t) for t in batch_spec.tensors], batch_spec.spec
        )
        print("  PyTree:")
        print("    " + pformat(batch, indent=2).replace("\n", "\n    "))
        print()


class _Tracker(RateTracker):
    """A simple tracker for measuring elapsed time."""

    def elapsed(self) -> timedelta:
        """Returns the elapsed time in nanoseconds."""
        return timedelta(
            convert_time_unit(super().elapsed_seconds(), "s", "ns", precision=0)
        )


class Profiler(abc.ABC):
    """Base class for implementing profilers for dataloader benchmarking.

    Profilers are used to profile various activities during dataloader
    benchmarking. Some pre-defined profilers are available in this file, but
    custom profilers can be implemented by subclassing this class and
    implementing all abstract methods. Profilers are registered by their
    `name()` which must be unique across all registered profilers. The
    name of the profiler can be passed to `benchmark_dataloader` method in
    the `profile_activities` argument to enable profiling of that activity.

    Profilers are instantiated once per dataloader benchmarking run. Profiler
    hooks, such as `on_epoch_start`, `on_epoch_end`, etc. are called at points
    of interest in the benchmarking run. The `profile()` method which is the
    main method that returns the actual profiling data is called at the
    sampling frequency. The profiling data returned by `profile()` is stored
    in the `BatchMetrics` and is printed along with other predefined metrics.
    """

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Returns the unique name of the profiler."""
        raise NotImplementedError

    @abc.abstractmethod
    def profile(self) -> Any:
        """Samples the profiler at the sampling frequency."""
        raise NotImplementedError

    @classmethod
    def to_human(cls, val: Any) -> str:
        """Converts the profiler value to a human-readable string."""
        return val

    @classmethod
    def display_name(cls) -> str:
        """Returns the display name of the profiler."""
        return cls.name()

    def on_start(self) -> None:
        """Hook called at the start of the benchmarking run."""

    def on_end(self) -> None:
        """Hook called at the end of the benchmarking run."""

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the start of each epoch.

        Args:
            epoch: The 1-indexed epoch number at which the hook is called.
        """

    def on_epoch_end(self, epoch: int) -> None:
        """Hook called at the end of each epoch.

        Args:
            epoch: The 1-indexed epoch number at which the hook is called.
        """

    def on_step_start(self, epoch: int, step: int) -> None:
        """Hook called at the start of each step.

        Args:
            epoch: The 1-indexed epoch number at which the hook is called.
            step: The 1-indexed step number at which the hook is called.
        """

    def on_step_end(self, epoch: int, step: int) -> None:
        """Hook called at the end of each step.

        Args:
            epoch: The 1-indexed epoch number at which the hook is called.
            step: The 1-indexed step number at which the hook is called.
        """

    @classmethod
    def get_profilers(cls) -> Dict[str, Type["Profiler"]]:
        """Return all currently-registerd concrete subclasses of this class.

        Note that this method only returns concrete implementations and ignores
        any subclasses that are abstract.

        Returns:
            Mapping between profiler name and profiler class.
        """
        profilers = dict()

        to_visit = [cls]
        while to_visit:
            this = to_visit.pop()
            if not inspect.isabstract(this):
                if this.name() in profilers:
                    raise ValueError(
                        f"Profiler with name `{this.name()}` already exists."
                    )
                profilers[this.name()] = this
            to_visit.extend(this.__subclasses__())

        return profilers


def _collection_decorator(func: Callable) -> Callable:
    """Decorator to call a function on all profilers in the collection."""

    @functools.wraps(func)
    def wrapper(self: "ProfilerCollection", *args, **kwargs):
        for name, profiler in self.items():
            try:
                getattr(profiler, func.__name__)(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    f"Failed to run {func.__name__} on profiler `{name}`."
                    f"Ran into error: {e}"
                )

    return wrapper


class ProfilerCollection(Dict[str, Profiler]):
    """A custom dictionary for storing profilers."""

    def __setitem__(self, key: str, value: Profiler) -> None:
        if not isinstance(key, str):
            raise ValueError(
                f"Key must be a string, but got type `{type(key)}` with value "
                f"`{key}`."
            )
        if not isinstance(value, Profiler):
            raise ValueError(
                f"Value must be a Profiler, but got type `{type(value)}` with."
            )
        return super().__setitem__(key, value)

    @_collection_decorator
    def on_start(self) -> None:
        """Calls `on_start` on all profilers in the collection."""

    @_collection_decorator
    def on_end(self) -> None:
        """Calls `on_end` on all profilers in the collection."""

    @_collection_decorator
    def on_epoch_start(self, epoch: int) -> None:
        """Calls `on_epoch_start` on all profilers in the collection."""

    @_collection_decorator
    def on_epoch_end(self, epoch: int) -> None:
        """Calls `on_epoch_end` on all profilers in the collection."""

    @_collection_decorator
    def on_step_start(self, epoch: int, step: int) -> None:
        """Calls `on_step_start` on all profilers in the collection."""

    @_collection_decorator
    def on_step_end(self, epoch: int, step: int) -> None:
        """Calls `on_step_end` on all profilers in the collection."""


class _MemoryProfiler(Profiler):
    """Base class for profiling memory usage."""

    # pylint: disable=no-self-use
    @functools.lru_cache(maxsize=1)
    def _get_process_memory_full_info(self) -> psutil._pslinux:
        """A cached version of `get_process_memory_full_info`."""
        return get_process_memory_full_info()

    def on_step_end(self, epoch: int, step: int) -> None:
        """Clears the cached memory usage after each step."""
        self._get_process_memory_full_info.cache_clear()

    @classmethod
    def to_human(cls, val: int) -> str:
        return bytes_to_human(val, fmt="{value:.2f}{unit}")


class MemoryUSSProfiler(_MemoryProfiler):
    """Class for profiling USS memory usage.

    USS stands for Unique Set Size and is the amount of memory that is unique
    to the process.
    """

    @classmethod
    def name(cls) -> str:
        return "memory_uss"

    @classmethod
    def display_name(cls) -> str:
        return "Memory USS"

    def profile(self) -> int:
        return self._get_process_memory_full_info().uss


class MemoryPSSProfiler(_MemoryProfiler):
    """Class for profiling PSS memory usage.

    PSS stands for Proportional Set Size and is the amount of memory shared
    with other processes, divided by the number of processes sharing that
    memory, plus the amount of memory unique to the process.
    """

    @classmethod
    def name(cls) -> str:
        return "memory_pss"

    @classmethod
    def display_name(cls) -> str:
        return "Memory PSS"

    def profile(self) -> int:
        return self._get_process_memory_full_info().pss


class MemorySwapProfiler(_MemoryProfiler):
    """Class for profiling swap memory usage."""

    @classmethod
    def name(cls) -> str:
        return "memory_swap"

    @classmethod
    def display_name(cls) -> str:
        return "Swap"

    def profile(self) -> int:
        return self._get_process_memory_full_info().swap


class _IOProfiler(Profiler):
    """Base class for profiling IO usage."""

    # pylint: disable=no-self-use
    @functools.lru_cache(maxsize=1)
    def _get_process_used_io(self) -> psutil._pslinux.pio:
        """A cached version of `get_process_used_io`."""

        def map_fn(process: psutil.Process):
            nonlocal infos
            infos.append(process.io_counters())

        infos = []

        map_processes(map_fn, include_children=True)

        result = {key: 0 for key in infos[0]._fields}
        for info in infos:
            for key in result:
                result[key] += getattr(info, key)

        return type(infos[0])(**result)

    def on_step_end(self, epoch: int, step: int) -> None:
        """Clears the cached IO usage after each step."""
        self._get_process_used_io.cache_clear()

    @classmethod
    def to_human(cls, val: int) -> str:
        return bytes_to_human(val, fmt="{value:.2f}{unit}")


class IOReadCharsProfiler(_IOProfiler):
    """Class for profiling IO read chars usage.

    This counts the number of bytes the process caused to be read from storage.
    This includes things like tty IO and it is unaffected by whether or not
    actual physical disk IO was required (e.g., the read might have been
    satisfied from pagecache).
    """

    def __init__(self):
        self._last_read_chars = 0

    @classmethod
    def name(cls) -> str:
        return "io_read_chars"

    @classmethod
    def display_name(cls) -> str:
        return "IO Read Chars"

    def on_epoch_start(self, epoch: int) -> None:
        if epoch == 1:
            self._last_read_chars = self._get_process_used_io().read_chars
            self._get_process_used_io.cache_clear()

    def profile(self) -> int:
        read_chars = self._get_process_used_io().read_chars
        res = read_chars - self._last_read_chars
        self._last_read_chars = read_chars
        return res
