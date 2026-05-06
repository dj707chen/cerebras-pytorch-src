# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Samplers for dataloading."""

import itertools
import math
from typing import Iterable, List, Optional, Union

import numpy as np
import torch

import cerebras.pytorch as cstorch

# Arbitrary object to use as a placeholder for padding indices
pad_index = object()


class DistributedSampler(torch.utils.data.Sampler):
    """CSX sampler for map-style datasets.

    This sampler handles sharding, batching, and skipping of map style datasets
    intended for use on CSX. Sharding is performed in such a way that data order
    is independent of the number of systems being used and the number of workers
    per system.
    """

    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        shuffle: bool = True,
        seed: Optional[int] = None,
        start_index: int = 0,
        shard: bool = True,
        batch_size: Optional[int] = None,
        drop_last: bool = True,
        num_samples: Optional[int] = None,
        pad_last: bool = False,
    ):
        """Constructs a `DistributedSampler` instance.

        Args:
            data_source: Dataset to sample from.
            shuffle: Whether or not to shuffle the dataset.
            seed: The RNG seed used when shuffling is on.
            start_index: The index of the first sample to yield.
            shard: Whether or not to shard the dataset across data streamer nodes.
            batch_size: The batch size to use to compute sharded indices
                and group samples into batches. If `None`, no batching will be
                performed. When running on CS-X, this must always be the global
                batch size. The sampler will internally handle appropriately
                sharding batches across workers when necessary.
            num_samples: The number of samples to extend or shrink each logical epoch to.
                In multi-epoch training, it is common to set this to the total number of
                samples that you plan to train on and enable shuffling so that epochs
                are not sequential but instead shuffled together for potentially improved
                convergence. If shuffling is off, epoch is shrunk (if num_samples < len(dataset))
                or extended by appending copies of the dataset (if num_samples > len(dataset))
                to fill the requested num_samples. It is not recommended to set this
                setting when shuffle is off. If None, One epoch is sampled from by querying
                the length of the dataset.
            pad_last: Whether to enable padding of the last batch so that the last batch
                has the same batch size as the rest of the batches. Only used if
                `batch_size` is not `None` and `drop_last` is `False`.
        """
        if cstorch.use_cs() and not drop_last and not pad_last:
            raise ValueError(
                f"Running on CSX requires all batches to have the same batch size. "
                f"Setting `drop_last=False` and `pad_last=False` violates this "
                f"condition as the last batch may have a smaller batch size. "
                f"Please set `drop_last=True` or `pad_last=True`."
            )
        if cstorch.use_cs() and batch_size is None and shard:
            raise ValueError(
                f"Running on CSX but batch_size is None and shard is enabled. When "
                f"running on CSX, the batch_size must be the global batch size."
            )
        self.sampler = ShuffleSampler(
            data_source,
            shuffle=shuffle,
            seed=seed,
            start_index=start_index,
            num_samples=num_samples,
        )
        if batch_size is not None:
            self.sampler = BatchSampler(
                self.sampler, batch_size, drop_last, pad_last
            )
        if shard:
            self.sampler = ShardedSampler(self.sampler)

        self.kwargs = {
            "data_source": data_source,
            "shuffle": shuffle,
            "seed": seed,
            "shard": shard,
            "batch_size": batch_size,
            "drop_last": drop_last,
        }

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

    def set_state(self, start_index):
        """Sets the state of the sampler to continue deterministically from a prior run.

        Args:
            start_index: the total number of samples streamed globally across
                all workers from a previous run.
        """
        self.__init__(**self.kwargs, start_index=start_index)


class ShuffleSampler(torch.utils.data.Sampler):
    """A sampler that handles shuffling and skipping."""

    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        start_index: int = 0,
    ):
        """Constructs a `ShuffleSampler` instance.

        Args:
            data_source: Dataset to sample from.
            num_samples: The number of samples to extend or shrink each logical epoch to.
                In multi-epoch training, it is common to set this to the total number of
                samples that you plan to train on and enable shuffling so that epochs
                are not sequential but instead shuffled together for potentially improved
                convergence. If shuffling is off, epoch is shrunk (if num_samples < len(dataset))
                or extended by appending copies of the dataset (if num_samples > len(dataset))
                to fill the requested num_samples. It is not recommended to set this
                setting when shuffle is off. If None, One epoch is sampled from by querying
                the length of the dataset.
            shuffle: Whether or not to shuffle the dataset.
            seed: The RNG seed used when shuffling is on.
            start_index: The index of the first sample to yield.
        """
        self._data_source = data_source
        self._num_samples = num_samples
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )
        self._num_samples_frozen = self.num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = start_index // self.num_samples
        self.start_index = start_index - self.num_samples * self.epoch

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self._data_source)
        return self._num_samples

    def __iter__(self):
        if self.num_samples != self._num_samples_frozen:
            raise RuntimeError(
                f"Data source passed into Sampler must have the same length "
                f"every epoch. Original length was {self._num_samples_frozen}, "
                f"new length is {self.num_samples}"
            )

        data_source_len = len(self._data_source)

        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed + self.epoch)
            if self.num_samples > data_source_len:
                epochs = math.ceil(self.num_samples / data_source_len)
                perm = torch.cat(
                    [torch.arange(data_source_len) for _ in range(epochs - 1)]
                )
                perm = torch.cat(
                    (perm, torch.randperm(data_source_len, generator=gen))
                )
                perm = perm[: self.num_samples]
                indices = torch.randperm(self.num_samples, generator=gen)
                perm = perm[indices]
            else:
                perm = torch.randperm(data_source_len, generator=gen)
                perm = perm[: self.num_samples]
            perm = perm[self.start_index :]
            yield from perm.tolist()
        else:
            yield from map(
                lambda x: x % data_source_len,
                range(self.start_index, self.num_samples),
            )

        self.epoch += 1
        self.start_index = 0

    def __len__(self):
        return self.num_samples - self.start_index


class ShardedSampler(torch.utils.data.Sampler):
    """A sampler for handling sharding across CSX devices.

    When running on CSX, `batch_size` must be the global batch size. In that case, when running
    on the user node, returned batches are of the global batch size. But when running on the
    worker nodes, the returned batches are of the local streaming batch sizes of each CSX,
    in order. See :py:meth:`~cstorch.distributed.get_streaming_batch_size` for more details.

    For example, assume the source sampler returns batches of size 10, there are 3 CSX's, and
    2 workers per CSX. When running on the USR node, this sampler returns batches of size 10.
    But when running on the worker nodes, this sampler returns the following batch sizes depending
    on which CSX this worker is streaming to. Here, the table cells refer to indices of the batch
    samples returned by each worker at each global step.

        global_batch_idx | 0    | 1     | 2     | 3     | ...
        -----------------------------------------------------
        worker 0, csx 0  | 0-4  |       | 20-24 |       | ...
        worker 1, csx 0  |      | 10-14 |       | 30-34 | ...
        worker 0, csx 1  | 4-7  |       | 24-27 |       | ...
        worker 1, csx 1  |      | 14-17 |       | 34-37 | ...
        worker 0, csx 2  | 7-10 |       | 27-30 |       | ...
        worker 1, csx 2  |      | 17-20 |       | 37-40 | ...

    This ordering takes into account the natural ordering of batches streamed to each CSX
    in a multi-CSX run.
    """

    def __init__(
        self, data_source: Union[torch.utils.data.Sampler[int], Iterable[int]]
    ):
        """Constructs a `ShardedSampler` instance.

        Args:
            data_source: Base sampler that can be any iterable that yields data.
        """
        self._data_source = data_source
        self._starting_group = 0

        if cstorch.distributed.is_streamer():
            cluster_spec = cstorch.distributed.service_resolver().cluster_spec
            task_spec = cluster_spec.task()

            self._num_groups = cluster_spec.num_workers_per_csx
            self._group_idx = task_spec.local_rank
            self._local_idx = task_spec.wse_id
        else:
            self._num_groups = 1
            self._group_idx = 0
            self._local_idx = 0

    def __iter__(self):
        if cstorch.distributed.is_streamer():
            global_batch_size = None
            global_batch_idx = None

            for global_batch_idx, global_batch in enumerate(
                self._data_source, start=self._starting_group
            ):
                if global_batch_size is None or global_batch_size != len(
                    global_batch
                ):
                    global_batch_size = len(global_batch)
                    local_batch_sizes = self._get_local_batch_sizes(
                        global_batch_size
                    )
                    indices = np.cumsum([0] + local_batch_sizes)

                if global_batch_idx % self._num_groups == self._group_idx:
                    yield global_batch[
                        indices[self._local_idx] : indices[self._local_idx + 1]
                    ]

            if global_batch_idx is not None:
                self._starting_group = (global_batch_idx + 1) % self._num_groups
        else:
            yield from self._data_source

    def __len__(self):
        num_global_batches = len(self._data_source)
        l = num_global_batches // self._num_groups

        effective_group_idx = self._group_idx - self._starting_group
        if effective_group_idx < 0:
            effective_group_idx += self._num_groups
        if effective_group_idx < (num_global_batches % self._num_groups):
            l += 1

        return l

    @staticmethod
    def _get_local_batch_sizes(global_batch_size: int) -> List[int]:
        local_batch_sizes = []
        if cstorch.distributed.is_streamer():
            cluster_spec = cstorch.distributed.service_resolver().cluster_spec
            local_batch_sizes = [None] * cluster_spec.num_csx
            for task in cluster_spec.tasks:
                local_batch_size = cstorch.distributed.get_streaming_batch_size(
                    global_batch_size, task.rank
                )
                if local_batch_sizes[task.wse_id] is None:
                    local_batch_sizes[task.wse_id] = local_batch_size
                elif local_batch_sizes[task.wse_id] != local_batch_size:
                    raise ValueError(
                        f"Local batch sizes of all CSX's must be equal, but got "
                        f"multiple value for CSX {task.wse_id}: "
                        f"{local_batch_sizes[task.wse_id]} vs. {local_batch_size}"
                    )
            if any(bs is None for bs in local_batch_sizes):
                raise ValueError(
                    f"Some CSX's do not have a batch size: {local_batch_sizes}"
                )
            if sum(local_batch_sizes) != global_batch_size:
                raise ValueError(
                    f"Local batch sizes ({local_batch_sizes}) do not sum up to "
                    f"the global batch size ({global_batch_size})."
                )
        else:
            local_batch_sizes = cstorch.distributed.get_streaming_batch_size(
                global_batch_size
            )

        return local_batch_sizes


class BatchSampler(torch.utils.data.Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    This sampler is a slight modification of the PyTorch BatchSampler such that any samples not
    yielded at the end of an epoch when `drop_last=True` will be yielded at the start of the next
    epoch. It also exposes a `pad_last` argument to pad incomplete batches with a special
    `pad_index` value.
    """

    def __init__(
        self,
        sampler: Union[torch.utils.data.Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
        pad_last: bool,
    ):
        """Constructs a `BatchSampler` instance.

        Args:
            sampler: Base sampler that can be any iterable that yields something.
            batch_size: Size of each batch to collate the values yields from the sampler. When
                running on CSX, this must always be the global batch size.
            drop_last: If True, sampler will drop the last batch if its size is less than
                batch_size.
            pad_last: If True and drop_last is False, sampler will pad the last incomplete batch
                with special `pad_index` objects to make the last batch the same size as all other
                batches.
        """
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pad_last = pad_last
        self.leftover_samples = []

        if len(self.sampler) < self.batch_size:
            self.leftover_samples = [s for s in self.sampler]

    def __iter__(self):
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = itertools.chain(self.leftover_samples, self.sampler)
            while True:
                try:
                    batch = []
                    for _ in range(self.batch_size):
                        batch.append(next(sampler_iter))
                    yield batch
                except StopIteration:
                    self.leftover_samples = batch
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                if self.pad_last:
                    while idx_in_batch < self.batch_size:
                        batch[idx_in_batch] = pad_index
                        idx_in_batch += 1
                    yield batch
                else:
                    yield batch[:idx_in_batch]

    def __len__(self):
        if self.drop_last:
            return (
                len(self.sampler) + len(self.leftover_samples)
            ) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
