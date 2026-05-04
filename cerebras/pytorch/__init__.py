# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Cerebras PyTorch (cstorch) — PyTorch extensions for Cerebras Wafer-Scale Cluster.
"""

_generating_docs = False

from cerebras.pytorch.backend import (
    Backend,
    BackendType,
    backend,
    current_backend,
    current_backend_impl,
    current_torch_device,
    get_backend_args,
    use_cs,
)
from cerebras.pytorch.core.compile import compile, trace
from cerebras.pytorch.saver import (
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
    save,
    load,
)
from cerebras.pytorch.utils.data.utils import from_numpy, to_numpy
from cerebras.pytorch.utils.data.dataset import SyntheticDataset
from cerebras.pytorch.utils.step_closures import (
    checkpoint_closure,
    step_closure,
)
from cerebras.pytorch.distributed import get_worker_state
from cerebras.pytorch.utils.data.dataloader import (
    DataLoader,
    DataLoaderCheckpoint,
    RestartableDataLoader,
)

from cerebras.pytorch import amp
from cerebras.pytorch import distributed
from cerebras.pytorch import metrics
from cerebras.pytorch import optim
from cerebras.pytorch import sparse
from cerebras.pytorch import utils
import cerebras.pytorch.saver as saver
