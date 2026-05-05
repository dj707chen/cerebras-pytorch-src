# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
The revamped Cerebras PyTorch package.
"""
import os
import warnings

import torch

# True if we're autogenerating docs
# This environment variable should only ever be set in the documentation repository
# when autogenerating docs from the docstrings in this package
_generating_docs = bool(
    os.environ.get("GENERATING_CEREBRAS_PYTORCH_DOCS") == "1"
)

from . import experimental

# pylint: disable=redefined-builtin
from .backend import backend, current_backend, current_torch_device, use_cs
from .core.compile import compile, trace
from .core.device import device
from .core.name_scope import (
    add_debug_name,
    get_debug_name,
    name_scope,
    set_debug_scope,
)
from .core.tensor import tensor
from .saver import load, save
from .saver.storage import full, full_like, ones, ones_like, zeros, zeros_like
from .utils.constant import make_constant
from .utils.data.data_executor import current_executor
from .utils.data.utils import from_numpy, to_numpy
from .utils.pol import pol
from .utils.step_closures import checkpoint_closure, step_closure


def summarize_scalar(*args, **kwargs):
    warnings.warn(
        "cstorch.summarize_scalar is deprecated and will be removed in a future "
        "release. Please create a SummaryWriter and write to it directly:\n\n"
        "\timport cerebras.pytorch.utils.tensorboard\n\n"
        "\twriter = cstorch.utils.tensorboard.SummaryWriter(log_dir='./log_dir')\n"
        "\twriter.add_scalar(...)\n\n"
        "Note, writing to a SummaryWriter should only occur inside a step closure."
    )

    from .utils import tensorboard

    tensorboard.summarize_scalar(*args, **kwargs)


def summarize_tensor(*args, **kwargs):
    warnings.warn(
        "cstorch.summarize_tensor is deprecated and will be removed in a future "
        "release. Please create a SummaryWriter and write to it directly:\n\n"
        "\timport cerebras.pytorch.utils.tensorboard\n\n"
        "\twriter = cstorch.utils.tensorboard.SummaryWriter(log_dir='./log_dir')\n"
        "\twriter.add_tensor(...)\n\n"
        "Note, writing to a SummaryWriter should only occur inside a step closure."
    )

    from .utils import tensorboard

    tensorboard.summarize_tensor(*args, **kwargs)


# isort: off
from . import (
    amp,
    core,
    distributed,
    metrics,
    nn,
    optim,
    profiler,
    sparse,
    utils,
)

if not _generating_docs:
    # Import backends here to avoid circular imports
    from .backends import backends

    # Reset all backend flags to their default values
    # This handles properly setting the default values for all flags
    # without running into circular import issues
    backends.reset()

# isort: on


__all__ = [
    "amp",
    "backend",
    "backends",
    "checkpoint_closure",
    "compile",
    "current_backend",
    "current_executor",
    "current_torch_device",
    "experimental",
    "from_numpy",
    "full",
    "full_like",
    "load",
    "metrics",
    "nn",
    "ones",
    "ones_like",
    "optim",
    "save",
    "step_closure",
    "summarize_scalar",
    "summarize_tensor",
    "to_numpy",
    "trace",
    "use_cs",
    "utils",
    "zeros",
    "zeros_like",
]


cirh = torch.ops.cirh

if not _generating_docs:
    from ._version import __version__
    from .lib import cerebras_pytorch_lib
else:
    # There will be no version file when generating docs
    __version__ = None
    cerebras_pytorch_lib = None
