# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import fnmatch
import inspect
import os
import re
from contextlib import contextmanager
from typing import Callable, List, Union

import torch.nn as nn

from cerebras.appliance.utils.typing import signature_matches_type_hint

# An object to signifiy an argument that's unspecified by the caller
UNSPECIFIED = object()

FilterCallable = Callable[[str, nn.Parameter], bool]


@contextmanager
def override_env_vars(**kwargs):
    """Temporarily override env variables from kwargs.

    Args:
        kwargs: List of key/value pairs to set.
    """
    old_values = {k: os.environ.get(k, None) for k in kwargs}

    try:
        for k, v in kwargs.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in old_values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def convert_glob_to_regex(f: str):
    """
    Converts the given glob string f to a type of regex which can then be used with .match() to
    check if the string matches the regex returned
    """
    if isinstance(f, str):
        return re.compile(fnmatch.translate(f))
    else:
        raise TypeError(
            f"filter must be a string or list of strings, "
            f"got {type(filter)}[{type(f)}]"
        )


def make_param_filter(
    param_filter: Union[str, List[str], FilterCallable]
) -> FilterCallable:
    """
    Returns the corresponding filter for parameters for the given `param_filter`.
    Args:
        param_filter: Either a string or a list of strings which are glob expressions or a
        callable which represents the filter itself
    Returns:
        A callable method that when given a parameter will return whether the filter matches it.
    """
    if callable(param_filter):
        signature = inspect.signature(param_filter)
        if not signature_matches_type_hint(signature, FilterCallable):
            raise ValueError(
                f'Unknown `param_filter`: "{param_filter}". Valid options are a string, or a '
                f'list of strings representing the filter representing the filter, or a '
                f'function with signature {FilterCallable}.'
            )
        return param_filter

    param_filters = list(
        map(
            convert_glob_to_regex,
            [param_filter] if isinstance(param_filter, str) else param_filter,
        )
    )

    def glob_expression_param_filter(name: str, param: nn.Parameter) -> bool:
        return any(filter.fullmatch(name) for filter in param_filters)

    return glob_expression_param_filter
