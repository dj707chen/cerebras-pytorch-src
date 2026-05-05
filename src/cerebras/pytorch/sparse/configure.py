"""Sparsity configuration helpers."""

from __future__ import annotations


def default_sparse_param_filter(name, param):
    if name is None:
        return True
    lowered = name.lower()
    return lowered.endswith("weight") or "weight" in lowered
