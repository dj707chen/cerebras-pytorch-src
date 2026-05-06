# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Iterator

from ..data_containers import Particle


class Constraint(ABC):
    """
    Abstract base class for constraints.

    A ConstraintValidationError should be raised if the constraint is not satisfied when calling validate.
    """

    @abstractmethod
    def validate(self, particle: "Particle") -> None:
        """
        Validates the constraint with the given particle.
        Should raise a ConstraintValidationError if the constraint is not satisfied.
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a human-readable representation of the constraint.
        """

    @abstractmethod
    def enumerate(self) -> Iterator["Constraint"]:
        """
        Enumerates the constraints contained within this constraint.
        """

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the constraint.
        """
        return f"<{self.__class__.__name__}: {self}>"
