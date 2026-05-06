# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
"""Kernel annotation utilities."""

from dataclasses import dataclass
from typing import Optional

from cerebras.pytorch.backend import use_cs
from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation


@dataclass
class KernelConfig(AnnotationMode.Config):
    """Kernel configuration."""

    kernel: Optional[str] = None
    enable_fwd: bool = True
    enable_bwd: bool = True

    def __post_init__(self):
        if not isinstance(self.kernel, (str, type(None))):
            raise TypeError(
                f"Expected `kernel` to be {str}, got {type(self.kernel)}."
            )


def get_attribute(config: KernelConfig, is_backward: bool):
    """Returns kernel attribute"""
    return AnnotationMode.Attribute('kernel', config.kernel)


def kernel_annotater(kernel: Optional[str] = None):
    """Return an annotating function which wraps the given function."""
    if not use_cs():
        return lambda fn: fn

    return create_annotation(KernelConfig, get_attribute, kernel=kernel)
