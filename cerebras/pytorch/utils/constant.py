# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch


def make_constant(tensor: torch.Tensor) -> torch.Tensor:
    """Create a constant node from the tensor in the graph."""
    from cerebras.pytorch.backend import use_cs
    from cerebras.pytorch.lib import cerebras_pytorch_lib

    if not use_cs():
        return tensor
    return cerebras_pytorch_lib.make_constant(tensor)
