# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import logging
import re
from dataclasses import dataclass
from dataclasses import fields as get_dataclass_fields
from typing import Any, Iterable, Optional, Sequence

from .attributes import DataType, Likeness

LOGGER = logging.getLogger(__name__)


@dataclass
class Operand:
    """
    A simple dataclass to represent a named operand.
    """

    name: str

    @classmethod
    def field_names(cls):
        return [field.name for field in get_dataclass_fields(cls)]

    def materialize(self, value: Any):
        """
        Materialize the operand with the given value.
        """
        return MaterializedOperand(
            name=self.name,
            value=value,
        )

    @classmethod
    def from_mlir(cls, name: str, value: "torch_mlir.ir.Value"):
        asm = str(value.type)
        try:
            if "!torch.vtensor" in asm:
                return MaterializedTensorOperand.from_mlir(name, value)
            else:
                return MaterializedOperand.from_mlir(name, value)
        except Exception as e:
            LOGGER.debug(
                f"Failed to create operand from MLIR value: {e.__class__.__name__}: {e}."
            )
            return cls(name=name)


@dataclass
class MaterializedOperand(Operand):
    """
    A simple dataclass to represent an operand that has been materialized.
    """

    value: Any

    def materialize(self, value: Any):
        """
        Materialize the operand with the given value.
        """
        raise ValueError("Cannot materialize a materialized operand.")

    def to_dict(self) -> dict:
        """
        Convert the operand to a dictionary representation.
        """
        return {self.name: self.value}

    @classmethod
    def from_mlir(cls, name: str, value: "torch_mlir.ir.Value"):
        """
        Create a MaterializedOperand from an MLIR value.
        """
        return cls(
            name=name,
            value=value.owner.opview.value.value,
        )


@dataclass
class TensorOperand(Operand):
    """
    A simple dataclass to represent a tensor operand.
    """

    dtype: DataType
    rank: int
    likeness: Optional[Likeness]

    def materialize(self, value: "torch.Tensor"):
        """
        Materialize the operand with the given value.
        """
        return MaterializedTensorOperand(
            name=self.name,
            dtype=self.dtype,
            rank=self.rank,
            likeness=self.likeness,
            shape=value.shape,
            value=value,
        )


@dataclass
class MaterializedTensorOperand(MaterializedOperand, TensorOperand):
    """
    A simple dataclass to represent a materialized tensor operand.
    """

    shape: Sequence[int]

    @classmethod
    def from_mlir(cls, name: str, value: "torch_mlir.ir.Value"):
        asm = str(value.type)
        # Regex to capture shape (as a comma-separated list of numbers) and dtype
        # Example: !torch.vtensor<[1, 2, 3],f32>
        pattern = r"^!torch\.vtensor<\[(.*?)\],(.*?)>$"
        match = re.match(pattern, asm.strip())
        if not match:
            raise ValueError(
                f"Type ASM does not match the expected format: {asm}"
            )

        shape_str, dtype = match.groups()

        # Split the shape string by commas and convert each to int
        shape = []
        if shape_str.strip():
            shape = [int(dim.strip()) for dim in shape_str.split(",")]

        # Rank is simply the length of the shape list
        rank = len(shape)

        return cls(
            name=name,
            dtype=DataType.from_str(dtype),
            rank=rank,
            shape=shape,
            likeness=None,
            value=value,
        )


class Particle:
    """
    A container for a set of operands. The operands can be accessed by index or by name.
    Order is preserved.
    """

    def __init__(self, operands: Optional[Iterable[Operand]] = None):
        if operands is None:
            operands = []
        self._operands = list(operands)
        if not all(isinstance(x, Operand) for x in self._operands):
            raise ValueError(
                "All items must be of type Operand. "
                f"Offending items: {[operand for operand in self._operands if not isinstance(operand, Operand)]}"
            )
        self._by_name = {operand.name: operand for operand in self._operands}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._operands[key]
        elif isinstance(key, str):
            return self._by_name[key]
        raise TypeError("Key must be an int or a string.")

    def __len__(self):
        return len(self._operands)

    def __iter__(self):
        return iter(self._operands)

    def __repr__(self):
        return f"Particle({self._operands})"

    def is_materialized(self) -> bool:
        """
        Check if all operands are materialized.
        """
        return all(
            isinstance(operand, MaterializedOperand)
            for operand in self._operands
        )

    def to_dict(self) -> dict:
        """
        Convert the particle to a dictionary representation.
        """
        if not self.is_materialized():
            raise ValueError(
                "Cannot convert a non-materialized particle to a dictionary."
            )

        merged_dict = {}
        for operand in self._operands:
            merged_dict.update(operand.to_dict())
        return merged_dict

    @classmethod
    def from_mlir(
        cls, names: Iterable[str], operands: Iterable["torch_mlir.ir.Value"]
    ):
        """
        Create a Particle from MLIR values"
        """
        import torch_mlir

        if not all(
            isinstance(value, torch_mlir.ir.Value) for value in operands
        ):
            raise ValueError("All items must be of type torch_mlir.ir.Value.")

        return cls(
            Operand.from_mlir(name, value)
            for name, value in zip(names, operands)
        )
