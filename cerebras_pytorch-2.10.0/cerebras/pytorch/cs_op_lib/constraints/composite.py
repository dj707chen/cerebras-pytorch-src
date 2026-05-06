# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from typing import Iterator

from ..data_containers import Particle
from ..errors import ConstraintValidationError
from .base import Constraint


class CompositeConstraint(Constraint):
    """
    Abstract base class for composite constraints.
    """

    def __init__(self, *constraints: Constraint):
        self.constraints = constraints

    def enumerate(self) -> Iterator[Constraint]:
        """
        Enumerates the constraints contained within this composite constraint.
        """
        for constraint in self.constraints:
            yield from constraint.enumerate()  # Recursively yield nested constraints


class AndConstraint(CompositeConstraint):
    """
    A composite constraint that validates all contained constraints with logical AND.
    """

    def validate(self, particle: "Particle"):
        failed_constraints = []
        for constraint in self.constraints:
            try:
                constraint.validate(particle)
            except ConstraintValidationError:
                failed_constraints.append(constraint)

        if failed_constraints:
            raise ConstraintValidationError(
                ", ".join(str(c) for c in failed_constraints)
            )

    def __str__(self) -> str:
        return f"({' AND '.join(str(c) for c in self.constraints) })"


And = AndConstraint


class OrConstraint(CompositeConstraint):
    """
    A composite constraint that validates any one of the contained constraints with logical OR.
    """

    def validate(self, particle: "Particle") -> None:
        failed_constraints = []
        for constraint in self.constraints:
            try:
                constraint.validate(particle)
                return  # Succeeds if any one constraint passes
            except ConstraintValidationError:
                failed_constraints.append(constraint)

        raise ConstraintValidationError(
            " OR\n".join(str(c) for c in failed_constraints)
        )

    def __str__(self) -> str:
        return f"({' OR '.join(str(c) for c in self.constraints) })"


Or = OrConstraint
