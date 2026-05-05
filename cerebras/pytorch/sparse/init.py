# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Sparsity mask initialization methods and helpers, invoked by
:py:class:`~cerebras.pytorch.sparse.SparsityAlgorithm`.
"""
import inspect
from typing import Callable, Optional, Union

import numpy as np
import torch

from cerebras.appliance.utils.typing import signature_matches_type_hint

from .utils import ScoreShaper, make_mask_topk_sparsity

InitMethodCallable = Callable[
    [
        torch.nn.Parameter,
        torch.FloatTensor,
        Optional[ScoreShaper],
        Optional[torch.device],
    ],
    torch.BoolTensor,
]
InitMethodType = Union[str, InitMethodCallable]


def random(
    p: torch.nn.Parameter,
    sparsity: torch.FloatTensor,
    score_shaper: Optional[ScoreShaper] = None,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """
    Uniformly random sparsity pattern.

    A score tensor with the same shape as the parameter is randomly generated
    with values between 0.0 and 1.0. The mask is then created by taking the
    :py:func:`top-k <cerebras.pytorch.sparse.utils.make_mask_topk_sparsity>` of
    the score tensor, where k is determined by the sparsity level.

    """
    if device is None:
        device = p.device

    # Move sparsity to device so we can use it to trace random initialization
    sparsity = sparsity.to(device)
    score = torch.rand_like(p, device=device)
    return make_mask_topk_sparsity(score, sparsity, score_shaper)


def topk(
    p: torch.nn.Parameter,
    sparsity: torch.FloatTensor,
    score_shaper: Optional[ScoreShaper] = None,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """
    Prune lowest magnitude weights.
    """
    if device is None:
        device = p.device

    # Move sparsity to the device so we can use it to trace topk
    sparsity = sparsity.to(device)
    score = p.to(device).abs()
    return make_mask_topk_sparsity(score, sparsity, score_shaper)


def from_zeros(
    p: torch.nn.Parameter,
    sparsity: torch.FloatTensor,
    score_shaper: Optional[ScoreShaper] = None,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """
    Any zeros currently in the weights represent pruned connections.
    NOTE: Doesn't actualy honor the configured sparsity.
    """
    if device is None:
        device = p.device

    return p.to(device) != 0


def checkerboard(
    p: torch.nn.Parameter,
    sparsity: torch.FloatTensor,
    score_shaper: Optional[ScoreShaper] = None,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """
    Mostly for stress and performance testing, creates a sparsity mask that is
    maximally distributed in a checkerboard across the weight.
    """
    density = 1 - sparsity.item()
    # Create a row with a uniformly distributed sparsity pattern
    col = p.shape[-1]
    # Alocate padding for potential rolling to still result in balance.
    padding = int(np.ceil(col / density + 1e-5))
    # [ 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 ]
    steps = torch.floor(torch.arange(col + padding) * density + 1e-5)
    # [ F,   F,   T,   F,   F,   T,   F,   F]
    mask = steps[1:] != steps[:-1]
    if len(p.shape) == 2:
        row = p.shape[0]
        # Now evenly distribute this over the rows as well by rolling each
        # This offset computation is equivalent to `-np.nonzero(mask)[0][0]`
        # but is more efficient, and more importantly allows torch.roll
        # to be traceable.
        offset = -int(np.floor(1 / density - 1e-5))
        mask = torch.stack([torch.roll(mask, x * offset) for x in range(row)])
    # Trim off padding columns and return
    return mask[..., :col].clone()


def _noop_compile_only(
    p: torch.nn.Parameter,
    sparsity: torch.FloatTensor,
    score_shaper: Optional[ScoreShaper] = None,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """
    "init" method that doesn't init to be used only with compile_only. This
    avoids computing masks on the CPU that aren't ultimately used.
    """
    return torch.empty_like(p, dtype=torch.bool)


def make_init_method(init_method: InitMethodType) -> InitMethodCallable:
    """
    Returns the corresponding init method callable for the given `init_method`.

    Args:
        init_method: The method to use to initialize the sparsity mask.
            This can be a string or a callable. If a string, it must be one of
                -  ":py:func:`~cerebras.pytorch.sparse.init.random`": Randomly initialize the mask
                -  ":py:func:`~cerebras.pytorch.sparse.init.topk`": prune the lowest magnitude weights
                -  ":py:func:`~cerebras.pytorch.sparse.init.from_zeros`": Any zeros in the weights represent pruned connections
                -  ":py:func:`~cerebras.pytorch.sparse.init.checkerboard`": Creates a sparsity mask that is maximally distributed across the weight

            If a callable, it must have the signature:

            .. code-block:: python

                def init_method(
                    param: torch.Tensor,
                    sparsity: float,
                    scope_shaper: Optional[ScoreShaper] = None,
                    device: Optional[torch.device] = None
                ) -> torch.Tensor:

            where
                - ``param`` is the original dense parameter
                - ``sparsity`` is the sparsity level
                - ``scope_shaper`` is an optional callable that can be used to reshape the mask
                - ``device`` is optionally the device to use to initialize the mask
    """
    from cerebras.pytorch.backend import current_backend_impl, use_cs

    if use_cs() and current_backend_impl().compile_only:
        return _noop_compile_only

    init_methods = {
        "random": random,
        "topk": topk,
        "from_zeros": from_zeros,
        "checkerboard": checkerboard,
    }
    init_method_error = (
        f'Unknown `init_method`: "{init_method}". Valid options are one '
        f'of the built-in {list(init_methods.keys())} or a function with '
        f'signature {InitMethodCallable}.'
    )
    if isinstance(init_method, str):
        if init_method not in init_methods:
            raise ValueError(init_method_error)
        init_method = init_methods[init_method]
    elif callable(init_method):
        signature = inspect.signature(init_method)
        if not signature_matches_type_hint(signature, InitMethodCallable):
            raise ValueError(init_method_error)
    else:
        raise ValueError(init_method_error)
    return init_method
