# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
"""Device annotation utilities."""

from dataclasses import dataclass
from typing import Literal, get_args

from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation
from cerebras.pytorch.core.constants import (
    ACTIVATION_HOST_ANNOTATION,
    WAFER_ANNOTATION,
    WEIGHT_HOST_ANNOTATION,
)

DeviceType = Literal[
    WEIGHT_HOST_ANNOTATION, ACTIVATION_HOST_ANNOTATION, WAFER_ANNOTATION
]


@dataclass
class AnnotationConfig(AnnotationMode.Config):
    device: DeviceType

    def __post_init__(self):
        if self.device not in get_args(DeviceType):
            raise TypeError(
                f"Expected `device` to be one of {get_args(DeviceType)}."
            )


def get_attribute(config, is_backward: bool):
    """Returns device_annotation attribute"""
    return AnnotationMode.Attribute('device_annotation', config.device)


# Define function decorator.
def device_annotation(device: DeviceType, enable_fwd=True, enable_bwd=True):
    """Enables device annotation for the wrapped function.

    Args:
        deviceType: Device to allocate the function on. Can be wgth (weight host), acth (activation host), or wafer.
        enable_fwd: Apply annotation to forward operations
        enable_bwd: Apply annotation to backwards operations

    Returns:
        An annotating function which wraps the given function.
    """
    return create_annotation(
        AnnotationConfig,
        get_attribute,
        device=device,
        enable_fwd=enable_fwd,
        enable_bwd=enable_bwd,
    )
