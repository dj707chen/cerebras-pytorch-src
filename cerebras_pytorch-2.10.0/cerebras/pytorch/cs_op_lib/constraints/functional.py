# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from ..errors import ConstraintValidationError
from .base import Constraint


def is_valid(constraint: Constraint):
    """
    Returns a function that checks if a particle satisfies a given constraint.
    """

    def is_valid_particle(particle):
        try:
            constraint.validate(particle)
            return True
        except ConstraintValidationError:
            return False

    return is_valid_particle
