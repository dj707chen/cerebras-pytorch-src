# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch

from ..attributes import DataType
from ..constraints.operand import OperandValueInRangeConstraint
from ..ulp import UlpStrategy
from .base import CsOp


class SubTensorOp(CsOp):
    op_func = torch.ops.aten.sub
    name = "sub.Tensor"
    constraint = OperandValueInRangeConstraint(min=1, max=1, applies_to="alpha")
    ulp_strategy = UlpStrategy(
        {
            DataType.f16: 1,
            DataType.f32: 1,
            DataType.i16: 1,
            DataType.i32: 1,
        }
    )
    operands = ["self", "other", "alpha"]
