# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from ..data_containers import Particle
from ..errors import ConstraintValidationError
from .base import Constraint


class NullConstraint(Constraint):
    """
    A constraint that always passes.
    This is used internally as a placeholder for constraints that are not yet implemented.
    It is not intended for public use.

    """

    def validate(self, particle: "Particle") -> None:
        pass

    def __str__(self) -> str:
        return "No constraints"

    def enumerate(self):
        yield from (self,)


class UnsupportedOpConstraint(Constraint):
    """
    A constraint that raises an error when validated.
    This is used when the op is entirely not supported.
    """

    def validate(self, particle: "Particle") -> None:
        raise ConstraintValidationError("This op is not supported.")

    def __str__(self) -> str:
        return "Unsupported op constraint"

    def enumerate(self):
        yield from (self,)
