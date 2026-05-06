# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch


class ConditionalUpdateManager:
    """Mark tensors and update them on context manager exit if condition is true"""

    def __init__(self):
        self.condition = None
        self.marked_tensors = []

    def mark_tensor(self, tensor):
        if not self.marked_tensors:
            raise RuntimeError(
                f"Failed to mark tensor {tensor}. Please ensure that "
                f"the ConditionalUpdateManager context is entered before "
                f"attempting to mark any tensors."
            )
        self.marked_tensors[-1][tensor] = tensor.clone()

    def set_condition(self, condition):
        self.condition = condition

    def __enter__(self):
        self.marked_tensors.append(torch.utils.weak.WeakTensorKeyDictionary())
        return self

    def __exit__(self, *exc):
        if not self.marked_tensors:
            raise RuntimeError(
                f"Stack for marked tensors is empty on __exit__, please "
                f"ensure that every __exit__ call for "
                f"ConditionalUpdateManager has a corresponding preceding "
                f"__enter__ call."
            )
        if isinstance(self.condition, torch.Tensor):
            with torch.no_grad():
                for tensor, unchanged_tensor in self.marked_tensors[-1].items():
                    tensor.copy_(
                        torch.where(
                            self.condition,
                            tensor,
                            unchanged_tensor,
                        )
                    )
        self.marked_tensors.pop()

        if not self.marked_tensors:
            self.condition = None

        return False


def update_if_finite(optimizer, tensor):
    """Wrapper used to mark tensors without explicitly referencing amp stash"""
    if hasattr(optimizer, "_amp_stash"):
        optimizer._amp_stash.dls_update_manager.mark_tensor(tensor)


def isfinite(optimizer):
    if hasattr(optimizer, "_amp_stash"):
        return getattr(optimizer._amp_stash, "isfinite", True)
    return True
