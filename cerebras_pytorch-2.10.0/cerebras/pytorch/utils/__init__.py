# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Emulates torch.utils."""


def __getattr__(name):
    if name == "CSConfig":
        raise AttributeError(
            f"\"{name}\" is no longer available."
            "Please pass in a \"cstorch.distributed.ClusterConfig\" instance "
            "to the backend constructor instead. For example,\n\n"
            "\tbackend = cstorch.backend(\n"
            "\t\t\"CSX\", cluster_config=cstorch.distributed.ClusterConfig(...)\n"
            "\t)\n"
        )
    # Note: this warning does not show up if the module is correctly imported
    elif name == "tensorboard":
        from warnings import warn

        warn(
            f"cstorch.utils.tensorboard must be imported before it can be accessed.\n\n"
            f"\timport cerebras.pytorch.utils.tensorboard\n\n"
            f"\t# or\n\n"
            f"\tfrom cerebras.pytorch.utils.tensorboard import *\n\n"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from . import benchmark
from .constant import make_constant
from .data import (
    DataExecutor,
    DataLoader,
    RestartableDataLoader,
    SyntheticDataset,
)

__all__ = ["DataLoader", "SyntheticDataset"]
