# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from typing import Iterator, Sequence, Union

from ..data_containers import MaterializedOperand, Operand, Particle
from ..errors import ConstraintValidationError
from .base import Constraint


class OperandConstraint(Constraint):
    """
    Abstract base class for constraints that operate on specific operands.
    """

    def __init__(
        self, applies_to: Union[str, Sequence[str]] = "all", *args, **kwargs
    ):
        """
        Arguments:
            applies_to: Specifies which operands the constraint applies to.
                        Can be "all" or a list of operand names.
        """

        self.applies_to = applies_to
        super().__init__(*args, **kwargs)

    def validate(self, particle: "Particle") -> None:
        for operand in particle:
            if self.applies_to == "all" or operand.name in self.applies_to:
                if isinstance(operand, MaterializedOperand):
                    self.validate_materialized_operand(operand)
                else:
                    self.validate_operand(operand)

    @abstractmethod
    def validate_operand(self, operand: "Operand") -> None:
        """
        Validates the constraint with the given non-materialized operand.
        Constraints defined here should be mutually exclusive with those defined
        in validate_materialized_operand.

        MaterializedOperands have value which contains the actual data along with metadata,
        while regular Operands do not have value and contain only metadata.

        For constraints that can be defined with respect to either a regular Operand
        or a MaterializedOperand, the regular Operand should be preferred, since
        this allows more constraints to be validated before materialization.

        Should raise a ConstraintValidationError if the constraint is not satisfied.
        """

    @abstractmethod
    def validate_materialized_operand(
        self, operand: "MaterializedOperand"
    ) -> None:
        """
        Validates the constraint with the given materialized operand.
        Constraints defined here should be mutually exclusive with those defined
        in validate_operand.

        MaterializedOperands have value which contains the actual data along with metadata,
        while regular Operands do not have value and contain only metadata.

        For constraints that can be defined with respect to either a regular Operand
        or a MaterializedOperand, the regular Operand should be preferred, since
        this allows more constraints to be validated before materialization.

        Should raise a ConstraintValidationError if the constraint is not satisfied.
        """

    def enumerate(self) -> Iterator["Constraint"]:
        yield from (self,)


class UnsupportedOperandConstraint(OperandConstraint):
    """
    Constraint for checking if an unsupported operand is present in
    the particle or input.
    """

    def validate_operand(self, operand: "Operand") -> None:
        pass

    def validate_materialized_operand(
        self, operand: "MaterializedOperand"
    ) -> None:
        # The value of the operand itself must be None for unsupported operands
        if operand.value is not None:
            raise ConstraintValidationError(
                f"Unsupported operand {operand.name}"
            )

    def __str__(self) -> str:
        return f"Unsupported operand {self.applies_to}"


class OperandValueInRangeConstraint(OperandConstraint):
    """
    Constraint for checking if an operand is within a specified range.
    """

    def __init__(self, min, max, *args, **kwargs):
        self.min, self.max = min, max
        super().__init__(*args, **kwargs)

    def validate_operand(self, operand: "Operand") -> None:
        pass

    def validate_materialized_operand(
        self, operand: "MaterializedOperand"
    ) -> None:
        err = ConstraintValidationError(
            f"Materialized operand {operand.name} is not in range {self.min} to {self.max}"
        )
        if operand.value is None:
            raise err
        if not (self.min <= operand.value <= self.max):
            raise err

    def __str__(self) -> str:
        return f"Operand value in range {self.min} to {self.max} for {self.applies_to}"
