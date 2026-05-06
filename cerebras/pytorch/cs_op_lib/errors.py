# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause


class ConstraintValidationError(Exception):
    """
    Custom exception to report failed constraint.
    """

    def __init__(self, constraint: str):
        message = f"Validation failed for constraint: {constraint}"
        super().__init__(message)
        self.constraint = constraint


class OpNotFoundError(Exception):
    """
    Custom exception to report that an operation was not found.
    """

    def __init__(self, op_name: str):
        message = f"Op not found in the Cerebras Op Library: {op_name}"
        super().__init__(message)
        self.op_name = op_name
