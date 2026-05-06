# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
"""num_kv_groups annotation utilities."""

from dataclasses import dataclass
from typing import Optional

from cerebras.pytorch.backend import use_cs
from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation


@dataclass
class GroupsConfig(AnnotationMode.Config):
    """num_kv_groups configuration."""

    num_kv_groups: Optional[int] = None
    enable_fwd: bool = True
    enable_bwd: bool = True

    def __post_init__(self):
        if not isinstance(self.num_kv_groups, (int, type(None))):
            raise TypeError(
                f"Expected `num_kv_groups` to be {int}, got {type(self.num_kv_groups)}."
            )


def get_attribute(config: GroupsConfig, is_backward: bool):
    """Returns num_kv_groups attribute"""
    return AnnotationMode.Attribute('num_kv_groups', config.num_kv_groups)


def groups_annotater(num_kv_groups: Optional[int] = None):
    """Return an annotating function which wraps the given function."""
    if not use_cs():
        return lambda fn: fn

    return create_annotation(
        GroupsConfig, get_attribute, num_kv_groups=num_kv_groups
    )
