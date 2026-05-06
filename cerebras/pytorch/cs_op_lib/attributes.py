# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum, auto, unique

import numpy as np


class ConciseReprEnum(Enum):
    """
    A base class for enums that provides a custom __repr__ method.
    """

    def __repr__(self):
        """
        Returns a concise string representation of the enum member.
        Example: <EnumName.MemberName>
        """
        return f"<{self.__class__.__name__}.{self.name}>"


@unique
class DataType(ConciseReprEnum):
    """This class is used to represent a generic data-type suitable for a various
    platforms, such as PyTorch, Tensorflow, Numpy and Cerebras Systems.
    """

    f16 = auto()
    f32 = auto()
    f64 = auto()
    bf16 = auto()
    cb16 = auto()  # Cerebras custom, not supported in this library yet.
    u8 = auto()
    u16 = auto()
    u32 = auto()
    u64 = auto()
    i8 = auto()
    i16 = auto()
    i32 = auto()
    i64 = auto()
    bool = auto()
    c64 = auto()
    c128 = auto()

    def is_float(self):
        return self in {
            DataType.f16,
            DataType.f32,
            DataType.f64,
            DataType.bf16,
            DataType.cb16,
        }

    def is_signed_integer(self):
        return self in {DataType.i8, DataType.i16, DataType.i32, DataType.i64}

    def is_unsigned_integer(self):
        return self in {DataType.u8, DataType.u16, DataType.u32, DataType.u64}

    def is_integer(self):
        return self.is_unsigned_integer() or self.is_signed_integer()

    @staticmethod
    def from_str(s):
        return {
            "f16": DataType.f16,
            "f32": DataType.f32,
            "f64": DataType.f64,
            "bf16": DataType.bf16,
            "cb16": DataType.cb16,
            "u8": DataType.u8,
            "u16": DataType.u16,
            "u32": DataType.u32,
            "u64": DataType.u64,
            "i8": DataType.i8,
            "i16": DataType.i16,
            "i32": DataType.i32,
            "i64": DataType.i64,
            "c64": DataType.c64,
            "c128": DataType.c128,
            "bool": DataType.bool,
        }[s]

    def numpy(self):
        return {
            DataType.f16: np.float16,
            DataType.f32: np.float32,
            DataType.f64: np.float64,
            DataType.u8: np.uint8,
            DataType.u16: np.uint16,
            DataType.u32: np.uint32,
            DataType.u64: np.uint64,
            DataType.i8: np.int8,
            DataType.i16: np.int16,
            DataType.i32: np.int32,
            DataType.i64: np.int64,
            DataType.c64: np.complex64,
            DataType.c128: np.complex128,
            DataType.bool: np.bool_,
        }[self]

    def torch(self) -> "torch.dtype":
        import torch

        return {
            DataType.f16: torch.float16,
            DataType.f32: torch.float32,
            DataType.f64: torch.float64,
            DataType.bf16: torch.bfloat16,
            DataType.u8: torch.uint8,
            # PyTorch does not support other unsigned int types: https://github.com/pytorch/pytorch/issues/58734
            DataType.i8: torch.int8,
            DataType.i16: torch.int16,
            DataType.i32: torch.int32,
            DataType.i64: torch.int64,
            DataType.c64: torch.complex64,
            DataType.c128: torch.complex128,
            DataType.bool: torch.bool,
        }[self]

    @staticmethod
    def from_torch(dtype) -> "DataType":
        import torch

        return {
            torch.float16: DataType.f16,
            torch.float32: DataType.f32,
            torch.float64: DataType.f64,
            torch.bfloat16: DataType.bf16,
            torch.uint8: DataType.u8,
            torch.int8: DataType.i8,
            torch.int16: DataType.i16,
            torch.int32: DataType.i32,
            torch.int64: DataType.i64,
            torch.complex64: DataType.c64,
            torch.complex128: DataType.c128,
            torch.bool: DataType.bool,
        }[dtype]

    @staticmethod
    def from_numpy(dtype) -> "DataType":
        # Handle both numpy type classes and dtype objects
        if hasattr(dtype, "type"):
            dtype = dtype.type
        return {
            np.float16: DataType.f16,
            np.float32: DataType.f32,
            np.float64: DataType.f64,
            np.uint8: DataType.u8,
            np.uint16: DataType.u16,
            np.uint32: DataType.u32,
            np.uint64: DataType.u64,
            np.int8: DataType.i8,
            np.int16: DataType.i16,
            np.int32: DataType.i32,
            np.int64: DataType.i64,
            np.complex64: DataType.c64,
            np.complex128: DataType.c128,
            np.bool_: DataType.bool,
        }[dtype]

    def sizeof(self) -> int:
        """Returns the size of the data type in bytes."""
        return {
            DataType.f16: 2,
            DataType.f32: 4,
            DataType.f64: 8,
            DataType.bf16: 2,
            DataType.cb16: 2,
            DataType.bool: 1,
            DataType.u8: 1,
            DataType.u16: 2,
            DataType.u32: 4,
            DataType.u64: 8,
            DataType.i8: 1,
            DataType.i16: 2,
            DataType.i32: 4,
            DataType.i64: 8,
            DataType.c64: 8,
            DataType.c128: 16,
        }[self]


@unique
class Likeness(ConciseReprEnum):
    WGT = auto()
    ACT = auto()


@unique
class Architecture(ConciseReprEnum):
    EIN = auto()
    FYN = auto()
    SDR = auto()
    HBG = auto()
