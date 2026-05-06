# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset classes for use with PyTorch DataLoaders."""

from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

import torch
from torch.utils._pytree import SUPPORTED_NODES, tree_flatten, tree_unflatten
from torch.utils.data import IterDataPipe

LeafT = Union[torch.Tensor, Callable[[int], torch.Tensor]]
SampleSpecT = Union[
    LeafT,
    List["SampleSpecT"],
    Tuple["SampleSpecT", ...],
    Dict[str, "SampleSpecT"],
    OrderedDict[str, "SampleSpecT"],
    NamedTuple,
]
SampleT = Union[
    torch.Tensor,
    List["SampleT"],
    Tuple["SampleT", ...],
    Dict[str, "SampleT"],
    OrderedDict[str, "SampleT"],
    NamedTuple,
]


# pylint: disable=abstract-method
class SyntheticDataset(IterDataPipe):
    """A synthetic dataset that generates samples from a `SampleSpec`."""

    def __init__(
        self, sample_spec: SampleSpecT, num_samples: Optional[int] = None
    ):
        """Constructs a `SyntheticDataset` instance.

        A synthetic dataset can be used to generate samples on the fly with
        an expected dtype/shape but without needing to create a full-blown
        dataset. This is especially useful for compile validation.

        Args:
            sample_spec: Specification of the samples to generate. This can be
                a nested structure of one of the following types:
                    - `torch.Tensor`: A tensor to be cloned.
                    - `Callable`: A callable that takes the sample index and
                        returns a tensor.
                Supported data structures for holding the above leaf nodes are
                `list`, `tuple`, `dict`, `OrderedDict`, and `NamedTuple`.
            num_samples: Total size of the dataset. If None, the dataset will
                generate samples indefinitely.
        """
        super().__init__()

        self._leaf_nodes, self._spec_tree = tree_flatten(sample_spec)
        if not self._leaf_nodes:
            raise ValueError(
                f"`sample_spec` must be a non-empty python tree of "
                f"`torch.Tensor` or `Callable`."
            )

        for item in self._leaf_nodes:
            if not isinstance(item, (torch.Tensor, Callable)):
                raise ValueError(
                    f"`sample_spec` is expected to contain a python tree of "
                    f"`torch.Tensor`, or `Callable`, but got an item of type "
                    f"`{type(item)}`. Note that supported data structures for "
                    f"holding leaf nodes in the tree are "
                    f"{', '.join(str(x) for x in SUPPORTED_NODES)}."
                )

        if isinstance(num_samples, int):
            if num_samples <= 0:
                raise ValueError(
                    f"`num_samples` must be a positive integer, but got "
                    f"`{num_samples}`."
                )
            self._num_samples = num_samples
        elif num_samples is None:
            self._num_samples = None
        else:
            raise TypeError(
                f"`num_samples` must be a positive integer or None, but got a "
                f"value of type `{type(num_samples)}`."
            )

    def __iter__(self) -> Iterator[SampleT]:
        """Returns an iterator for generating samples."""
        index = 0
        while self._num_samples is None or index < self._num_samples:
            sample_flat = []
            for item in self._leaf_nodes:
                if isinstance(item, torch.Tensor):
                    sample_flat.append(item.clone())
                elif callable(item):
                    sample_flat.append(item(index))
                else:
                    raise TypeError(
                        f"Invalid type for leaf node: {type(item)}."
                    )

            yield tree_unflatten(sample_flat, self._spec_tree)
            index += 1

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        if self._num_samples is None:
            raise TypeError(
                f"`{self.__class__.__name__}` does not have a length because "
                f"`num_samples` was not provided."
            )
        return self._num_samples
