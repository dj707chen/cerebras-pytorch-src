"""TensorBoard utility shims."""

from __future__ import annotations

from collections import defaultdict


class SummaryWriter:
    def __init__(self, *args, **kwargs):
        try:
            from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
        except Exception as exc:
            raise RuntimeError("TensorBoard support requires torch and tensorboard") from exc
        self._impl = TorchSummaryWriter(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def add_tensor(self, tag, tensor, global_step=None, walltime=None):
        return self._impl.add_histogram(tag, tensor, global_step=global_step, walltime=walltime)


class SummaryReader:
    Tags = "tags"
    Scalars = "scalars"

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self._scalars = defaultdict(list)

    def reload(self):
        return self

    @property
    def scalar_names(self):
        return list(self._scalars)

    @property
    def tensor_names(self):
        return []

    @property
    def text_summary_names(self):
        return []

    @property
    def scalar_groups(self):
        return dict(self._scalars)

    def read_scalar(self, name):
        return self._scalars.get(name, [])

    def read_tensor(self, name):
        return []

    def read_text_summary(self, name):
        return []
