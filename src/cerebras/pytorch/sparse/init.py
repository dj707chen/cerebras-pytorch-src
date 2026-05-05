"""Sparsity mask initialization functions."""

from __future__ import annotations

from typing import Callable

from .._compat import require_torch


def _shape(param):
    return getattr(param, "shape", param)


def random(param, sparsity, score_shaper=None, device=None):
    torch = require_torch()
    keep = 1.0 - float(sparsity or 0.0)
    return torch.rand(_shape(param), device=device or getattr(param, "device", None)) < keep


def topk(param, sparsity, score_shaper=None, device=None):
    torch = require_torch()
    scores = param.detach().abs().flatten() if hasattr(param, "detach") else torch.rand(_shape(param), device=device).flatten()
    keep = max(0, int(round(scores.numel() * (1.0 - float(sparsity or 0.0)))))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if keep:
        mask[scores.topk(keep).indices] = True
    return mask.reshape(_shape(param))


def from_zeros(param, sparsity, score_shaper=None, device=None):
    torch = require_torch()
    if hasattr(param, "detach"):
        return param.detach().ne(0)
    return torch.ones(_shape(param), dtype=torch.bool, device=device)


def checkerboard(param, sparsity, score_shaper=None, device=None):
    torch = require_torch()
    numel = int(param.numel()) if hasattr(param, "numel") else 1
    mask = torch.arange(numel, device=device or getattr(param, "device", None)) % 2 == 0
    return mask.reshape(_shape(param))


def make_init_method(method) -> Callable:
    if callable(method):
        return method
    methods = {
        "random": random,
        "topk": topk,
        "from_zeros": from_zeros,
        "zeros": from_zeros,
        "checkerboard": checkerboard,
    }
    try:
        return methods[str(method).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown sparsity init method: {method!r}") from exc
