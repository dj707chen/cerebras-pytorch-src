"""NumPy conversion helpers."""

from __future__ import annotations

from ._compat import require_torch, tree_map


def from_numpy(value):
    torch = require_torch()
    return tree_map(lambda item: torch.from_numpy(item) if hasattr(item, "__array__") else item, value)


def to_numpy(value):
    def convert(item):
        if hasattr(item, "detach") and hasattr(item, "cpu") and hasattr(item, "numpy"):
            return item.detach().cpu().numpy()
        return item

    return tree_map(convert, value)
