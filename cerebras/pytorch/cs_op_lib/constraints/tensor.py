# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from typing import Set

from ..attributes import DataType, Likeness
from ..data_containers import MaterializedTensorOperand, TensorOperand
from ..errors import ConstraintValidationError
from .operand import OperandConstraint


class TensorConstraint(OperandConstraint):
    """
    Abstract base class for constraints that operate on input tensors.
    """

    def validate_operand(self, operand: "TensorOperand") -> None:
        if not isinstance(operand, TensorOperand):
            raise ConstraintValidationError(
                f"Expected a tensor operand, got {type(operand)}"
            )
        self.validate_tensor(operand)

    def validate_materialized_operand(
        self, operand: "MaterializedTensorOperand"
    ) -> None:
        if not isinstance(operand, MaterializedTensorOperand):
            raise ConstraintValidationError(
                f"Expected a materialized tensor operand, got {type(operand)}"
            )
        self.validate_materialized_tensor(operand)

    @abstractmethod
    def validate_tensor(self, tensor: "TensorOperand") -> None:
        """
        Validates the constraint with the given tensor.
        Note that here only the non-materialized TensorOperand is passed to the method.
        Only metadata is available at this point (e.g., dtype, shape, etc.).

        For constraints that can be defined with respect to either a regular TensorOperand
        or a MaterializedTensorOperand, the TensorOperand should be preferred, since
        this allows more constraints to be validated before materialization.

        Should raise a ConstraintValidationError if the constraint is not satisfied.
        """

    @abstractmethod
    def validate_materialized_tensor(
        self, tensor: "MaterializedTensorOperand"
    ) -> None:
        """
        Validates the constraint with the given tensor.
        Note that here the MaterializedTensorOperand is passed to the method.
        The actual tensor data is available at this point.

        For constraints that can be defined with respect to either a regular TensorOperand
        or a MaterializedTensorOperand, the TensorOperand should be preferred, since
        this allows more constraints to be validated before materialization.

        Should raise a ConstraintValidationError if the constraint is not satisfied.
        """


class DataTypeConstraint(TensorConstraint):
    """
    Constraint that checks if the data types of the input tensors are one of the supported data types.
    """

    def __init__(self, supported_datatypes: Set["DataType"], *args, **kwargs):
        self.supported_datatypes = supported_datatypes
        super().__init__(*args, **kwargs)

    def validate_tensor(self, tensor: "TensorOperand") -> None:
        if tensor.dtype not in self.supported_datatypes:
            raise ConstraintValidationError(str(self))

    def validate_materialized_tensor(
        self, tensor: "MaterializedTensorOperand"
    ) -> None:
        pass

    def __str__(self) -> str:
        return f"Data type of tensor {self.applies_to} must be : {self.supported_datatypes}"


class LikenessConstraint(TensorConstraint):
    """
    Constraint that checks if the likeness of the input tensors are one of the supported likeness.
    """

    def __init__(self, supported_likeness: Set["Likeness"], *args, **kwargs):
        self.supported_likeness = supported_likeness
        super().__init__(*args, **kwargs)

    def validate_tensor(self, tensor: "TensorOperand") -> None:
        """Verify if the likeness of the tensor is one of the supported likeness."""
        raise NotImplementedError

    def validate_materialized_tensor(
        self, tensor: "MaterializedTensorOperand"
    ) -> None:
        pass

    def __str__(self) -> str:
        return f"Likeness of tensor {self.applies_to} must be : {self.supported_likeness}"


class RankConstraint(TensorConstraint):
    """
    Constraint that checks if the rank of the input tensors is within a specified range.
    """

    def __init__(self, max_rank: int, min_rank: int = 1, *args, **kwargs):
        self.max_rank = max_rank
        self.min_rank = min_rank
        super().__init__(*args, **kwargs)

    def validate_tensor(self, tensor: "TensorOperand") -> None:
        if tensor.rank > self.max_rank or tensor.rank < self.min_rank:
            raise ConstraintValidationError(str(self))

    def validate_materialized_tensor(
        self, tensor: "MaterializedTensorOperand"
    ) -> None:
        pass

    def __str__(self) -> str:
        return f"Rank of tensor {self.applies_to} must be between {self.min_rank} and {self.max_rank}"
