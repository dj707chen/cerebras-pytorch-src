"""Dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from .._compat import require_torch, tree_map
from ..backend import current_torch_device


class DataLoader:
    """Wrapper that constructs an iterable from an input function."""

    def __init__(self, input_fn: Callable[..., Iterable], *args, **kwargs):
        self.input_fn = input_fn
        self.args = args
        self.kwargs = kwargs
        self._state: Dict[str, Any] = {}

    def _build(self):
        return self.input_fn(*self.args, **self.kwargs)

    def __iter__(self):
        device = current_torch_device()

        def move(item):
            return item.to(device) if hasattr(item, "to") else item

        for batch in self._build():
            yield tree_map(move, batch)

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state_dict=None, *args, **kwargs):
        self._state = dict(state_dict or {})


class SyntheticDataset:
    """Dataset that repeats tensors or calls per-index factories."""

    def __init__(self, sample_spec, num_samples: Optional[int] = None):
        self.sample_spec = sample_spec
        self.num_samples = num_samples

    def __len__(self):
        if self.num_samples is None:
            raise TypeError("SyntheticDataset with num_samples=None is unbounded")
        return self.num_samples

    def __getitem__(self, index):
        def make(item):
            if callable(item):
                return item(index)
            if hasattr(item, "clone"):
                return item.clone()
            return item

        return tree_map(make, self.sample_spec)


class DataExecutor:
    """Single-pass executor over a dataloader."""

    def __init__(self, dataloader, num_steps: Optional[int] = None, **kwargs):
        self.dataloader = dataloader
        self.num_steps = num_steps
        self.kwargs = kwargs
        self.profiler = None

    def __iter__(self):
        for step, batch in enumerate(self.dataloader):
            if self.num_steps is not None and step >= self.num_steps:
                break
            yield batch


class RestartableDataLoader:
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict, strict: bool = True):
        return None

    def aggregate_state_dict(self, worker_states):
        return {f"worker_{idx}": state for idx, state in enumerate(worker_states)}

    def deaggregate_state_dict(self, aggregated_state_dict, strict: bool = True):
        return aggregated_state_dict.get("worker_0", {})


@dataclass
class DataLoaderCheckpoint:
    step: int = 0
    state_dict: Optional[Dict[str, Any]] = None


def get_worker_state():
    from ..distributed import get_worker_state as _get_worker_state

    return _get_worker_state()


def torch_dataloader(*args, **kwargs):
    torch = require_torch()
    return torch.utils.data.DataLoader(*args, **kwargs)
