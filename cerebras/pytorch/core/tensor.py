# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
from warnings import warn

import torch


def tensor(*args, **kwargs) -> torch.Tensor:
    """
    Constructs a torch.Tensor using the provided arguments.
    If a backend has been initialized, the tensor will be moved to the
    backend's device and tracked by the backend as a stateful tensor
    """
    from cerebras.pytorch.backend import current_backend_impl

    backend = current_backend_impl(raise_exception=False)

    if backend is None:
        warn(
            f"Attempting to construct a stateful tensor without a "
            f"backend initialized. This tensor will not be tracked."
        )

        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], torch.Tensor)
        ):
            return args[0]
        return torch.tensor(*args, **kwargs)

    if backend.is_tracing:
        raise RuntimeError(
            f"Cannot construct a stateful tensor while tracing "
            f"the forward/backward pass. Please only construct "
            f"stateful tensors during initialization."
        )

    with backend.device:
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], torch.Tensor)
        ):
            out = args[0]
        else:
            out = torch.tensor(*args, **kwargs)

        if out.device.type != backend.device.type:
            out = out.to(backend.torch_device)

        if backend.is_csx:
            from cerebras.pytorch.lib import cerebras_pytorch_lib

            unique_id = cerebras_pytorch_lib.get_unique_id(out)
            backend.detached_stateful[f"tensor_{unique_id}"] = out

    return out
