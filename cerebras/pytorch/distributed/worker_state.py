# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Class defining worker state info to be exposed via user-facing API."""
from typing import Optional

from cerebras.pytorch.utils.data.dataloader import DataLoaderCheckpoint


class WorkerState:
    """Holds Worker state info."""

    _state: DataLoaderCheckpoint = None

    @classmethod
    def configure(cls, worker_ckpt: DataLoaderCheckpoint):
        assert isinstance(worker_ckpt, DataLoaderCheckpoint), (
            "Expected `worker_ckpt` to be of type `DataLoaderCheckpoint`, "
            f"got {type(worker_ckpt)}."
        )
        cls._state = worker_ckpt

    @classmethod
    def _check(cls):
        return cls._state is not None

    @classmethod
    def get_worker_state(cls) -> Optional[DataLoaderCheckpoint]:
        try:
            if cls._check():
                return cls._state
            else:
                raise RuntimeError(
                    "Unable to fetch CSX Worker state: `cstorch.distributed.get_worker_state()` "
                    "must only be called once inside your dataloader's implementation of `state_dict`."
                )
        finally:
            cls._state = None
