# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Data pipes for transforming framework inputs to Cerebras hardware inputs."""
import copy
import os
import pprint
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import dill
import numpy as np
import torch
from typing_extensions import Self

import cerebras.pytorch as cstorch
from cerebras.appliance import log
from cerebras.appliance.data.conversions import np_dtype_from_rtfx_dtype
from cerebras.pytorch.core.constants import INPUT_NAME_PREFIX
from cerebras.pytorch.distributed import WorkerState
from cerebras.pytorch.utils.data.dataloader import (
    DataLoaderCheckpoint,
    RestartableDataLoader,
)
from cerebras.pytorch.utils.nest import visit_torch_tensors
from cerebras.pytorch.utils.tracker import RateTracker


class DataPipeStage(ABC):
    """Base class for all data pipe stages."""

    @abstractmethod
    def __iter__(self):
        """Returns an iterator for iteraing the data pipeline."""
        raise NotImplementedError


@log.named_class_logger("data_pipe.FnWrapper")
class FnWrapper(DataPipeStage, log.ClassLogger):
    """Wraps the callable to create a DataPipeStage."""

    def __init__(self, source_fn: Callable, count: Optional[int] = None):
        """Constructs a `FnWrapper` instance.

        Args:
            source_fn: The function called every iteration to generate data.
                The function can raise a StopIteration to signal that there are
                no more data available.
            count: Maximum number of iterations to run. If None, it generates
                data indefinitely. Defaults to None.
        """
        self._source_fn = source_fn
        self._count = count

        if self._count is not None and self._count < 0:
            raise ValueError(f"Count must be positive, but got {self._count}")

    def __iter__(self):
        count = 0
        try:
            while self._count is None or count < self._count:
                yield self._source_fn()
                count += 1
        except StopIteration:
            pass


@log.named_class_logger("data_pipe.OperatorWrapper")
class OperatorWrapper(DataPipeStage, log.ClassLogger):
    """Wraps the callable to create a DataPipeStage."""

    def __init__(self, source: DataPipeStage, operator: Callable):
        """Applies an operator to the output of a previous datapipe.

        Args:
            source: The iterator to access previous datapipe output.
            operator: Function that acts on that output.
        """
        self._source = source
        self._operator = operator

    def __iter__(self):
        for data in self._source:
            yield self._operator(data)


@log.named_class_logger("data_pipe.Repeater")
class Repeater(DataPipeStage, log.ClassLogger):
    def __init__(
        self, source: Iterable, count: Optional[int] = None
    ) -> Iterable:
        """Repeats the source iterator many (possibly infinite) times.

        Args:
            source: The source iterable for generating data.
            count: Count for iterating over source. If None, it iterates
                indefinitely. Defaults to None.
        """
        self._source = source
        self._count = count

    def __iter__(self):
        batch_generated = False
        epoch = 0
        gen = iter(self._source)
        while self._count is None or epoch < self._count:
            try:
                yield next(gen)
                batch_generated = True
            except StopIteration:
                if not batch_generated:
                    raise RuntimeError(
                        f"Dataloader did not generate any batches at epoch "
                        f"{epoch}."
                    )
                epoch += 1
                self.logger.info(
                    f"Iterator was exhausted. Iterating from scratch for epoch {epoch}"
                )
                gen = iter(self._source)
                batch_generated = False


@log.named_class_logger("data_pipe.NameRemapper")
class NameRemapper(DataPipeStage, log.ClassLogger):
    def __init__(self, source: DataPipeStage, mapping: Dict[str, Any]):
        self._source = source
        self._mapping = mapping

    def __iter__(self):
        for data in self._source:
            remapped = {}
            for old, new in self._mapping.items():
                if old not in data:
                    raise RuntimeError(
                        f"Runtime expects a tensor with name \"{old}\" in the batch, but only "
                        f"found: {list(data.keys())}"
                    )
                remapped[new] = data[old]
            yield remapped


@log.named_class_logger("data_pipe.MegaBatcher")
class MegaBatcher(DataPipeStage, log.ClassLogger):
    """Class for creating a megabatch from multiple smaller batches.

    For performance reasons, we send megabatches instead of one batch at a time.
    """

    def __init__(
        self,
        source: DataPipeStage,
        block_sizes: Union[int, List[int]],
        pad_to_largest_block: bool = True,
    ):
        """Constructs a `MegaBatcher` instance.

        Args:
            source: The source data pipe to cycle through.
            block_sizes: Number of samples to put in each block. If a list, it cycles through the
                list to generate blocks of different sizes at each iteration.
            pad_to_largest_block: Whether to zero-pad all blocks to have the same size as the
                largest block in `block_sizes`.
        """
        self._source = source
        self._buffered_data: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._tensor_specs: Dict[str, Tuple[Tuple[int], np.dtype]] = dict()
        self._samples_available = 0
        self._pad_to_largest_block = pad_to_largest_block

        self._block_sizes = []
        self._block_idx = 0
        self._largest_block_size = 0
        self.set_block_sizes(block_sizes)

    def set_block_sizes(self, block_sizes: Union[int, List[int]]):
        """Changes the block sizes to generate without resetting cached data."""
        if isinstance(block_sizes, int):
            block_sizes = [block_sizes]

        if not all(isinstance(b, int) and b >= 0 for b in block_sizes):
            raise ValueError(
                f"Block sizes must be >= 0, but got {block_sizes}."
            )

        self._block_sizes = block_sizes.copy()
        self._block_idx = 0
        self._largest_block_size = max(self._block_sizes)

    def __iter__(self):
        self._reset()

        try:
            for data in self._source:
                self._add_batch(data)
                while self._has_mega_batch():
                    yield self._get_mega_batch()
        finally:
            if self._samples_available:
                self.logger.warning(
                    f"MegaBatcher is discarding {self._samples_available} cached samples "
                    f"during iterator shutdown"
                )

    @property
    def _block_size(self) -> int:
        """Returns the current block size."""
        return self._block_sizes[self._block_idx]

    def _reset(self):
        self._buffered_data = defaultdict(list)
        self._tensor_specs = dict()
        self._samples_available = 0
        self._block_idx = 0

    def _add_batch(self, data: Dict[str, np.ndarray]):
        """Adds a new batch and returns the next mega batch if available.

        Args:
            data: A batch of data returned from the dataloader.
        Returns:
            The next available mega batch. If enough samples are not buffered
            to create a mega batch, it returns None.
        """
        infer_specs = not self._tensor_specs

        batch_size = infer_block_size(data)
        for name, tensor in data.items():
            self._buffered_data[name].append(tensor)
            if infer_specs:
                self._tensor_specs[name] = (tensor.shape, tensor.dtype)
        self._samples_available += batch_size

    def _has_mega_batch(self) -> bool:
        """Returns true if enough samples are buffered to create a block."""
        return self._samples_available >= self._block_size

    def _get_mega_batch(self) -> Dict[str, np.ndarray]:
        """Constructs a megabatch from buffered samples and returns it.

        Returns:
            The next available mega batch.
        Raises:
            AssertionError if enough samples are not available.
        """
        assert self._has_mega_batch(), "Not enough samples available."

        mega_batch = {}

        if self._block_size:
            # Pair of (batch_idx, tensor_idx) to split on in order to get a block
            split_indices = None
            for name, buffered_batches in self._buffered_data.items():
                if buffered_batches[0].shape[0] != self._block_size:
                    # Create split indices on first tensor and reuse for the rest
                    if split_indices is None:
                        remaining = self._block_size
                        for batch_idx, tensor in enumerate(buffered_batches):
                            if tensor.shape[0] >= remaining:
                                split_indices = (batch_idx + 1, remaining)
                                break
                            remaining -= tensor.shape[0]
                        assert split_indices is not None

                    # Grab batches that will constitute this block
                    batch_list = buffered_batches[: split_indices[0]]
                    batch_list[-1] = batch_list[-1][: split_indices[1]]

                    # Delete batches that were used to create the block
                    del buffered_batches[: split_indices[0] - 1]
                    buffered_batches[0] = buffered_batches[0][
                        split_indices[1] :
                    ]
                    if buffered_batches[0].shape[0] == 0:
                        del buffered_batches[0]
                else:
                    batch_list = [buffered_batches[0]]
                    del buffered_batches[0]

                # Add padding if necessary
                if (
                    self._pad_to_largest_block
                    and self._block_size < self._largest_block_size
                ):
                    batch_list.append(
                        np.zeros(
                            (
                                self._largest_block_size - self._block_size,
                                *batch_list[0].shape[1:],
                            ),
                            dtype=batch_list[0].dtype,
                        )
                    )

                # Create the block
                mega_batch[name] = np.concatenate(batch_list)
        else:
            bs = self._largest_block_size if self._pad_to_largest_block else 0
            mega_batch = {
                name: np.zeros(shape=(bs, *t[0][1:]), dtype=t[1])
                for name, t in self._tensor_specs.items()
            }

        # Decrement available samples
        self._samples_available -= self._block_size
        # Switch to the next block size
        self._block_idx = (self._block_idx + 1) % len(self._block_sizes)

        return mega_batch


@log.named_class_logger("data_pipe.TensorNamer")
class TensorNamer(DataPipeStage, log.ClassLogger):
    """Names incoming tensors."""

    def __init__(self, source: DataPipeStage):
        """Constructs a `TensorNamer` instance.

        Args:
            source: The source data pipe to cycle through.
        """
        self._source = source

    def __iter__(self):
        for idx, data in enumerate(self._source, start=1):
            batch = {}
            for scope, tensor in visit_torch_tensors(
                data, scope=[INPUT_NAME_PREFIX]
            ):
                name = "_".join(scope)

                # If scalar, convert to a 1D tensor
                if len(tensor.shape) == 0:
                    tensor = tensor.unsqueeze(0)

                batch[name] = cstorch.to_numpy(tensor.to("cpu"))

            if not batch:
                raise ValueError(f"Batch data at index {idx} is empty.")

            yield batch


@log.named_class_logger("data_pipe.BlockValidator")
class BlockValidator(DataPipeStage, log.ClassLogger):
    """Validates the incoming data blocks."""

    def __init__(self, source: DataPipeStage, golden_spec: "SampleSpec"):
        """Constructs a `BlockValidator` instance.

        Args:
            source: The source data pipe to cycle through.
            golden_spec: The spec describing the expected tensors.
        """
        self._source = source
        self._golden_spec = golden_spec

    def __iter__(self):
        for idx, data in enumerate(self._source, start=1):
            for tensor_spec in self._golden_spec.tensors:
                if (
                    tensor_spec.source not in data
                    or data[tensor_spec.source].dtype != tensor_spec.dtype
                    or list(data[tensor_spec.source].shape[1:])
                    != list(tensor_spec.shape[1:])
                ):
                    expected = {
                        t.source: {"shape": t.shape, "dtype": t.dtype}
                        for t in self._golden_spec.tensors
                    }
                    actual = {
                        name: {"shape": t.shape, "dtype": t.dtype}
                        for name, t in data.items()
                    }
                    raise ValueError(
                        f"Execution on CS-X currently requires the input batches "
                        f"to have the same tensor names, shapes, and dtypes. "
                        f"However, batch returned at step {idx} by the "
                        f"input pipeline that does not match the compiler's spec: "
                        f"{expected} from compiler artifacts vs. {actual} received."
                    )

            yield data


@log.named_class_logger("data_pipe.SampleSkipper")
class SampleSkipper(DataPipeStage, log.ClassLogger):
    """Skips the first N samples of data."""

    def __init__(self, source: DataPipeStage, count: int):
        """Constructs a `SampleSkipper` instance.

        Args:
            source: The source data pipe to cycle through.
            count: Number of initial samples to skip.
        """
        self._source = source
        assert count >= 0, f"Expected non-negative skip count, got {count}."
        self._count = count
        self._buffer = None
        self._samples_seen = None
        self._samples_available = None
        self._block_size = None

    def __iter__(self):
        self._reset()

        if self._count <= 0:
            yield from self._source
            return

        for data in self._source:
            block = self._get_next_block(data)
            if block is not None:
                yield block

    def _reset(self):
        self._buffer = defaultdict(list)
        self._samples_seen = 0
        self._samples_available = 0
        self._block_size = None

    def _get_next_block(self, data: Dict[str, np.ndarray]):
        if self._block_size == None:
            self._block_size = infer_block_size(data)
        else:
            curr_block_size = infer_block_size(data)
            assert self._block_size == curr_block_size, (
                f"Non-uniform block sizes between tensors not supported, "
                f"previous block size was {self._block_size} but currently "
                f"seeing {curr_block_size}"
            )

        self._samples_seen += self._block_size
        if self._samples_seen <= self._count:
            return None

        offset = max(0, self._block_size - (self._samples_seen - self._count))
        self._enqueue(data, offset)
        if self._samples_available < self._block_size:
            return None

        return self._dequeue()

    def _enqueue(self, data: Dict[str, np.ndarray], offset: int):
        for name, tensor in data.items():
            self._buffer[name].append(tensor[offset:])
        self._samples_available += self._block_size - offset

    def _dequeue(self) -> Dict[str, np.ndarray]:
        block = dict()
        for name, tensors in self._buffer.items():
            # pylint: disable=unnecessary-dict-index-lookup
            block[name], *(self._buffer[name]) = np.split(
                np.concatenate(tensors), [self._block_size]
            )
        self._samples_available -= self._block_size
        return block


@log.named_class_logger("data_pipe.SampleSaver")
class SampleSaver(DataPipeStage, log.ClassLogger):
    """Saves the source stream data to file(s)."""

    def __init__(self, source: DataPipeStage, outdir: Optional[str] = None):
        """Constructs a `SampleSaver` instance.

        Args:
            source: The source data pipe to cycle through.
            outdir: Output directory to save streams to. If None, CWD is used.
        """
        self._source = source
        self._outdir = Path(
            outdir or os.getcwd(), f"streams_npz_{str(uuid.uuid4())[:8]}"
        ).resolve()
        self._mb_per_file = 10  # Buffer size before flushing samples to file

        self._buffer = defaultdict(list)
        self._buffer_size = 0
        self._curr_file_idx = 0

    def __iter__(self):
        self._outdir.mkdir(parents=True, exist_ok=True)

        try:
            for data in self._source:
                self._add_to_buffer(data)
                if self._is_buffer_full():
                    self._flush_to_file()
                yield data
        finally:
            # Flush final samples
            self._flush_to_file()

    @property
    def _curr_file(self) -> Path:
        return self._outdir.joinpath(
            f"streams.{str(self._curr_file_idx).zfill(5)}.npz"
        )

    def _add_to_buffer(self, data: Dict[str, np.ndarray]):
        for name, tensor in data.items():
            self._buffer[name].append(tensor)
            self._buffer_size += tensor.size * tensor.itemsize

    def _is_buffer_full(self) -> bool:
        return self._buffer_size >= self._mb_per_file * 1e6

    def _flush_to_file(self):
        if not self._buffer:
            return

        self.logger.debug(f"Flushing samples to {self._curr_file}")

        arrays = {}
        for name, tensors in self._buffer.items():
            arrays[name] = np.concatenate(tensors)

        np.savez(str(self._curr_file), **arrays)

        self._buffer.clear()
        self._buffer_size = 0
        self._curr_file_idx += 1


@log.named_class_logger("data_pipe.DataLoaderCheckpointer")
class DataLoaderCheckpointer(DataPipeStage, log.ClassLogger):
    """Checkpoints the DataLoader state at frequent intervals."""

    def __init__(
        self,
        source: DataPipeStage,
        dataloader: RestartableDataLoader,
        worker_checkpoint: DataLoaderCheckpoint,
        batch_size: int,
        checkpoint_schedule: Iterable[int],
    ):
        """Constructs a `DataLoaderCheckpointer` instance.

        Args:
            source: The source data pipe to cycle through.
            dataloader: `torch.utils.data.DataLoader` object for the run
            worker_checkpoint: Data checkpoint holding cluster info for current run
            batch_size: Per-box batch size for this WRK
            checkpoint_schedule: Iterable that yields the steps at which to take checkpoints.
        """
        assert isinstance(dataloader, RestartableDataLoader)

        self._source = source
        self._dataloader = dataloader
        self._per_box_batch_size = batch_size
        self._default_worker_checkpoint = copy.deepcopy(worker_checkpoint)
        self._checkpoint_step_map: Dict[int, List[int]] = (
            self._create_checkpoint_step_map(checkpoint_schedule)
        )
        self._iteration = 0
        self.logger.debug(
            f"Checkpoint step map is:\n{pprint.pformat(self._checkpoint_step_map)}"
        )

        self._ckpt_outdir = Path(
            f"dataloader_checkpoints_{str(uuid.uuid4())[:8]}"
        ).resolve()
        self._ckpt_outdir.mkdir()

    def __iter__(self):
        self._iteration = 0
        # Save state at global steps where worker iter is 0
        self._maybe_save_state(self._iteration)

        for data in self._source:
            if (
                batch_size := infer_block_size(data)
            ) != self._per_box_batch_size:
                raise ValueError(
                    f"DataLoader restartability requires that the batch size returned in the "
                    f"workers to be the per-CSX batch size, which is {self._per_box_batch_size} "
                    f"in this case. However, we encountered a batch with size {batch_size}. "
                    f"Please make sure to call \"cstorch.distributed.get_streaming_batch_size()\" "
                    f"in the DataLoader to use the correct batch size when the DataLoader is "
                    f"instantiated in the workers."
                )

            self._maybe_save_state(self._iteration + 1)
            self._iteration += 1  # Update after having saved the state
            yield data

    def state_dict(self, step: int) -> Optional[DataLoaderCheckpoint]:
        """Returns the state dict at given checkpoint step.

        Args:
            step: A preconfigured checkpoint step.
        Returns:
            The checkpoint at the given step. If Iterator hasn't reached the required iteration
            to generate a checkpoint for this step, None is returned.
        """
        worker_iter = self._compute_worker_iteration_count(
            step,
            self._default_worker_checkpoint.num_workers_per_csx,
            self._default_worker_checkpoint.global_worker_id,
        )
        if step not in self._checkpoint_step_map.get(worker_iter, []):
            all_ckpt_steps = []
            for steps in self._checkpoint_step_map.values():
                all_ckpt_steps.extend(steps)
            all_ckpt_steps.sort()
            raise RuntimeError(
                f"Dataloader state_dict was requested at step {step}, but "
                f"this step was not preconfigured as a checkpoint step. "
                f"Preconfigured checkpoint steps are: {all_ckpt_steps}."
            )

        # If iterator has reached the requested iteration, we must have
        # already saved the state, so return it. Otherwise, we haven't
        # reached the desired step yet, so return None.
        if worker_iter <= self._iteration:
            return self._load_cached_state(step)
        else:
            return None

    def _maybe_save_state(self, iteration: int) -> None:
        """Optionally saves state for this CSX Worker.

        This saves for all global checkpoint steps where the worker should have performed
        `iteration` steps.

        Args:
            iteration: This is the step at which to save the dataloader state. Note that this is
                called *after* the dataloader has generated `iteration` batches.
        """
        checkpoint_steps_list = self._checkpoint_step_map.get(iteration, [])
        # Return if no checkpoint is to be saved at current iteration
        if not checkpoint_steps_list:
            return

        # Save state in data checkpoint per global checkpoint step
        for checkpoint_step in checkpoint_steps_list:
            # Create a fresh worker state object
            state = copy.deepcopy(self._default_worker_checkpoint)
            state.user_state_dict = None
            state.worker_step = iteration
            state.samples_streamed = iteration * self._per_box_batch_size
            state.appliance_step = checkpoint_step

            # Configure API with current worker state info
            WorkerState.configure(state)

            # Call user-defined `state_dict` method
            state.user_state_dict = self._dataloader.state_dict()

            # Now save ckpt file
            self.logger.debug(
                f"Saving state at iteration {iteration} for checkpoint step {checkpoint_step}"
            )
            self._cache_state(state, checkpoint_step)

    def _dump_filepath(self, step: int) -> Path:
        return self._ckpt_outdir.joinpath(f"step_{step}.pkl")

    def _cache_state(self, state: DataLoaderCheckpoint, step: int) -> None:
        """Pickles the object and saves it to a file."""
        filepath = self._dump_filepath(step)
        try:
            with open(filepath, 'wb') as f:
                dill.dump(state, f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save dataloader checkpoint file {filepath} "
                f"due to error: {e}. Please ensure that the provided state is "
                f"picklable using the `dill` package."
            ) from e

    def _load_cached_state(self, step: int) -> DataLoaderCheckpoint:
        """Loads the object from a file."""
        filepath = self._dump_filepath(step)
        try:
            with open(filepath, 'rb') as f:
                return dill.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read dataloader checkpoint file {filepath} "
                f"due to error: {e}"
            )

    def _create_checkpoint_step_map(
        self,
        checkpoint_schedule: Iterable[int],
    ) -> Dict[int, List[int]]:
        """Creates a mapping from each iteration of this worker to
        the list of all appliance checkpoint steps at which the worker
        will have completed that iteration. This map will be used to
        capture worker state via a data checkpoint for all checkpoint steps.

        Args:
            checkpoint_schedule: Iterable that yields the steps at which to take checkpoints.
        Returns:
            `dict` mapping each worker iteration to the list of appliance
            checkpoint steps. Steps per each iteration are sorted in increasing order.
        """
        ckpt_step_map = defaultdict(set)
        for ckpt_step in checkpoint_schedule:
            worker_iter = self._compute_worker_iteration_count(
                ckpt_step,
                self._default_worker_checkpoint.num_workers_per_csx,
                self._default_worker_checkpoint.global_worker_id,
            )
            ckpt_step_map[worker_iter].add(ckpt_step)

        return {k: sorted(v) for k, v in ckpt_step_map.items()}

    @staticmethod
    def _compute_worker_iteration_count(
        total_steps: int,
        num_workers_per_csx: int,
        global_worker_id: int,
    ) -> int:
        """Compute the number of iterations for a worker.

        Args:
            total_steps: Total number of iterations for the run.
            num_workers_per_csx: Number of workers per CSX.
            global_worker_id: Index of the worker to compute the number of
                iterations for in the global pool of all workers. It is assumed that
                workers for CS-X 0 are first in the global pool, followed by workers
                for CS-X 1, and so on.
        Returns:
            The number of iterations for the given worker.
        """

        # Each worker streams a local batch to the CSX it's connected to in a round
        # robin fashion. So at each update step (i.e., global batch), all workers
        # with the same local rank must have streamed a batch to each CSX, otherwise
        # we get a stall.
        worker_num_iters = total_steps // num_workers_per_csx
        remain_steps = total_steps % num_workers_per_csx
        local_rank = global_worker_id % num_workers_per_csx
        if local_rank < remain_steps:
            worker_num_iters += 1
        return worker_num_iters


@log.named_class_logger("data_pipe.RateProfiler")
class RateProfiler(DataPipeStage, log.ClassLogger):
    def __init__(self, source: DataPipeStage):
        """Constructs a `RateProfiler` instance.

        Args:
            source: The source data pipe to cycle through.
        """
        self._source = source

    def __iter__(self):
        tracker = RateTracker()
        source_iter = iter(self._source)
        self.logger.info(
            f"| Streamer IteratorCreationTime={tracker.elapsed_seconds()} seconds"
        )

        tracker.reset()
        try:
            while True:
                data = next(source_iter)
                tracker.add(1)
                self._log_rate(tracker)
                yield data
        except StopIteration:
            pass
        finally:
            self._log_rate(tracker, final=True)

    def _log_rate(self, tracker: RateTracker, final=False):
        if final or tracker.total_count == 1 or tracker.total_count % 100 == 0:
            level = log.INFO
        elif self._should_log_trace():
            level = log.TRACE
        else:
            level = None

        if level is not None:
            self.logger.log(
                level,
                f"| Streamer "
                f"Step={tracker.total_count}, "
                f"Rate={tracker.rate():.2f} steps/sec, "
                f"GlobalRate={tracker.global_rate():.2f} steps/sec",
            )


@dataclass
class SampleSpec:
    """A dataclass for holding info about a block of tensors.

    Args:
        tensors: List of tensors in this block.
    """

    @dataclass
    class TensorSpec:
        """A dataclass for holding info about tensors.

        Args:
            source: Name of the tensor.
            dtype: Datatype of the tensor.
            shape: Shape of the tensor.
            tensor_id: Integer identifier of tensor.
            shm_name: Name of the shared memory to use for this tensor.
        """

        source: str
        dtype: np.dtype
        shape: Tuple[int]
        tensor_id: int
        shm_name: Optional[str] = None

        def __post_init__(self):
            if isinstance(self.dtype, str):
                self.dtype = np_dtype_from_rtfx_dtype(self.dtype)
            assert isinstance(self.dtype, np.dtype)

    tensors: List[TensorSpec]

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        assert list(obj.keys()) == ["tensors"]
        tensors = []
        for item in obj["tensors"]:
            tensors.append(cls.TensorSpec(**item))
        return cls(tensors)


def infer_block_size(data: Dict[str, Union[torch.Tensor, np.ndarray]]) -> int:
    """Checks the block size of the incoming data and verifies that it is
        of uniform size throughout the different features
    Args:
        data: the data to be streamed, for which the block size is calculated
    Returns:
        The block size of the data.
    """
    block_size = None

    for name, tensor in data.items():
        if block_size == None:
            block_size = tensor.shape[0]
            assert block_size > 0, f"Batch size of tensor \"{name}\" is zero."
        else:
            assert block_size == tensor.shape[0], (
                f"Non-uniform block sizes between tensors not supported, "
                f"saw block sizes {block_size} and {tensor.shape[0]}"
            )

    assert block_size is not None, "Batch data is empty."

    return block_size
