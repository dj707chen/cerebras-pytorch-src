# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from collections import UserDict

import numpy as np

from .attributes import DataType


def get_ulp_diff(input: np.ndarray, other: np.ndarray) -> int:
    """
    Computes the ULP difference between two numpy arrays.
    The two arrays are flattened so we are not limited
    by the shape of the arrays (np.testing.assert_array_max_ulp
    adds a dimension before comparing).
    """
    if input.dtype != other.dtype:
        raise ValueError("dtype of input and other must match")
    return np.testing.assert_array_max_ulp(
        input.flatten(),
        other.flatten(),
        maxulp=np.inf,
        dtype=input.dtype,
    )


class UlpStrategy(UserDict):
    """
    A strategy for determining the ulp (unit in the last place) value for numeric"
    checking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in self.keys():
            if not isinstance(key, DataType):
                raise ValueError(f"Key {key} must be a DataType instance.")
            if not isinstance(self[key], int):
                raise ValueError(f"Value for {key} must be an int.")

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            raise KeyError(
                f"UlpStrategy does not contain an entry for {key}. "
                f"Available keys: {list(self.keys())}"
            ) from e

    def check(self, output: np.ndarray, reference: np.ndarray) -> bool:
        """
        Check if the output is within the ulp of the reference value.

        Args:
            output (np.ndarray): The output value.
            reference (np.ndarray): The reference value.
        Returns:
            bool: True if the output is within the ulp of the reference value,
                False otherwise.
        """
        # Cast the reference to the same dtype as the output
        reference = reference.astype(output.dtype)

        # Calculate the ulp difference
        ulp_diff = get_ulp_diff(output, reference)

        # Check if the ulp difference is within the allowed range
        return ulp_diff <= self[DataType.from_numpy(output.dtype)]
