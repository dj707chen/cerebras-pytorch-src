# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import logging
from contextlib import contextmanager

import torch

from cerebras.pytorch.backend import current_backend_impl

from ._amp_state import _amp_state


@contextmanager
def autocast(dtype: torch.dtype = None):
    """Context manager that invokes torch.autocast() if on CPU/GPU"""

    backend = current_backend_impl()

    if backend.backend_type.is_csx:
        logging.debug(
            "autocast() has no effect on CSX runs. "
            "Just call cstorch.amp.set_half_dtype(dtype=...) to "
            "set the half dtype and the CSX backend will handle "
            "autocasting automatically."
        )
        yield None
    else:
        if dtype is None:
            dtype = _amp_state.half_dtype

        if backend.backend_type.is_cpu and dtype != torch.bfloat16:
            raise ValueError(
                "Mixed precision on CPU is only supported with bfloat16. "
                "Please call cstorch.amp.set_half_dtype(torch.bfloat16) "
                "or pass the dtype to cstorch.amp.autocast(dtype=torch.bfloat16)"
            )

        with torch.autocast(
            backend.device.torch_device.type, dtype=dtype
        ) as ctx:
            yield ctx
