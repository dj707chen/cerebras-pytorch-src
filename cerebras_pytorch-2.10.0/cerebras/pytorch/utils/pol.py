# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cerebras.appliance.utils.descriptor import Descriptor
from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation
from cerebras.pytorch.lib import cerebras_pytorch_lib


@dataclass
class PolConfig(AnnotationMode.Config):
    """Represents POL config"""

    level: Optional[int]
    bwd_level: Optional[int]

    def __post_init__(self):
        for name in ["level", "bwd_level"]:
            val = getattr(self, name)
            if val is not None and val not in range(0, 3):
                raise ValueError(
                    f"POL must be None or an integer in range [0, 3) but got "
                    f"{'forward' if name == 'level' else 'backward'} level {val}."
                )


def get_attribute(config: PolConfig, is_backward: bool):
    """Returns POL attribute"""
    level = config.level
    if is_backward and config.bwd_level is not None:
        level = config.bwd_level
    return AnnotationMode.Attribute('pol', level)


def pol(
    level: Optional[int] = None,
    bwd_level: Optional[int] = None,
    enable_fwd: bool = True,
    enable_bwd: bool = True,
):
    """Enables POL annotation for the wrapped function."""
    return create_annotation(
        PolConfig,
        get_attribute,
        level=level,
        bwd_level=bwd_level,
        enable_fwd=enable_fwd,
        enable_bwd=enable_bwd,
    )


def current_pol():
    """Returns the current POL level within context."""
    curr_pol_level = cerebras_pytorch_lib.get_annotation("pol")
    if curr_pol_level is None:
        from cerebras.pytorch.backends import backends

        return backends.csx.precision.optimization_level
    return curr_pol_level


class POL(Descriptor):
    def sanitize(self, value: int) -> int:
        if value not in range(0, 3):
            raise ValueError(
                f"{self.name} must be an integer in range [0, 3). Got {value}"
            )

        from cerebras.pytorch.lib import cerebras_pytorch_lib

        cerebras_pytorch_lib.set_pol(value)
        return value
