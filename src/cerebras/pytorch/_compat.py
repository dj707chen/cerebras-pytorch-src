"""Small compatibility helpers shared by the clean-room implementation."""

from __future__ import annotations

import importlib
from typing import Any


def optional_torch() -> Any:
    """Return the imported torch module, or None when torch is unavailable."""

    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def require_torch() -> Any:
    """Import torch or raise a focused runtime error."""

    torch = optional_torch()
    if torch is None:
        raise RuntimeError(
            "This API requires PyTorch. Install project dependencies with "
            "`uv sync` before using tensor, optimizer, or dataloader features."
        )
    return torch


def tree_map(fn, value):
    """Apply fn to every leaf of a small Python container tree."""

    if isinstance(value, dict):
        return type(value)((key, tree_map(fn, item)) for key, item in value.items())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(tree_map(fn, item) for item in value))
    if isinstance(value, tuple):
        return tuple(tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [tree_map(fn, item) for item in value]
    return fn(value)
