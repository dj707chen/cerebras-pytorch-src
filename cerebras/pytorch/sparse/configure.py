# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration helpers for constructing SparsityOptimizer objects.
"""

import inspect
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Union
from warnings import warn

import torch

from cerebras.appliance.utils.classes import retrieve_all_subclasses

from .base import SparsityAlgorithm
from .group import Group

SparseParamFilterType = Callable[[str, torch.nn.Parameter], bool]
# Can be a single regex, a list of regex, or a dict of regex -> config
ParamNamePatternsType = Union[str, List[str], Dict[str, dict]]

LOGGER = logging.getLogger("cerebras.sparsity")


def default_sparse_param_filter(name: str, param: torch.nn.Parameter) -> bool:
    """
    Return True if the given parameter should be sparse.

    Only returns true if the parameter is > 1D and not an embedding or norm or
    lm_head or pe_helper.

    Args:
        name: Name of the parameter
        param: The parameter itself
    """

    # By default, sparsify params that are > 1D and not embedding or norm.
    name = name.lower()
    if (
        param is None
        or len(param.shape) <= 1
        or "embedding" in name
        or "norm" in name
        or "lm_head" in name
        or "pe_helper" in name
    ):
        return False
    return True


def flatten_sparsity_params(kwargs):
    """
    Config classes package sparsity related params in a sub dict.
    ALthough, if we use native yaml config, they come unrolled.
    This utility unwraps the sparsity related params(if present)
    into an unroller sparsity param dict for consistency.
    Args:
        kwargs : Input args
    Returns:
        Flattened dict
    """

    if isinstance(kwargs, (int, float, list, tuple)):
        return kwargs

    if 'groups' in kwargs:
        kwargs = kwargs.get('groups', {})
    else:
        return kwargs  # No need to flatten if no groups present

    if 'groups' in kwargs:
        # Remove the 'groups' key from the flattened dictionary
        del kwargs['groups']

    if isinstance(kwargs, dict):
        additional_dict = kwargs.get('params', {})
        flattened_dict = kwargs.copy()

        for key, value in additional_dict.items():
            new_key = f"{key}"
            flattened_dict[new_key] = value

        if 'params' in flattened_dict:
            # Remove the 'params' key from the flattened dictionary
            del flattened_dict['params']
        return flattened_dict
    elif isinstance(kwargs, list):
        param_list = []
        for param in kwargs:
            additional_dict = param.get('params', {})
            flattened_dict = param.copy()

            for key, value in additional_dict.items():
                new_key = f"{key}"
                flattened_dict[new_key] = value

            if 'params' in flattened_dict:
                # Remove the 'params' key from the flattened dictionary
                del flattened_dict['params']
            param_list.append(flattened_dict)
        return param_list
    else:
        return kwargs


def map_sparsity_algorithm(algorithm: str):
    """
    Map the sparsity type to a valid sparsity class

    Args:
        sparsity_type: Type of sparsity optimizer to construct.
        kwargs: Passed along to the chosen sparsity optimizer ``__init__``.

    """
    sparsity_algorithms = {
        cls.__name__.lower(): cls
        for cls in retrieve_all_subclasses(SparsityAlgorithm)
        if not inspect.isabstract(cls) and not isinstance(cls, Group)
    }

    # Ensure we have a known sparsity optimizer.
    sparsity_cls = sparsity_algorithms.get(algorithm.lower())

    if not sparsity_cls:
        raise ValueError(
            f"Unsupported sparsity algorithm: {algorithm}. "
            f"Supported types: {sorted(supported_sparsity_types.keys())}"
        )

    return sparsity_cls


def configure(config: Union[float, dict, List[dict]]) -> Group:
    config = flatten_sparsity_params(config)
    if isinstance(config, (int, float)):
        # Configure static sparsity and return
        return configure({"sparsity": config})
    elif isinstance(config, (list, tuple)):
        sparsity = Group()
        for item in config:
            # configure returns Group, so we extend the top-level group
            # with the sub-groups
            sparsity.extend(configure(item))
        return sparsity
    elif isinstance(config, dict):
        config = deepcopy(config)

        if "algorithm" not in config and "type" in config:
            warn(
                "The 'type' key is deprecated, please use 'algorithm' instead",
            )
            config["algorithm"] = config.pop("type")

        # If no algorithm is specified, assume static sparsity
        sparsity_algorithm = config.pop("algorithm", "static")
        param_filter = config.pop("param_filter", None)

        sparsity_cls = map_sparsity_algorithm(sparsity_algorithm)

        # Allow "schedule" to be used as an alias of "sparsity"
        if "schedule" in config:
            if "sparsity" in config:
                raise ValueError(
                    "Cannot specify both 'sparsity' and 'schedule' in the same config"
                )
            config["sparsity"] = config.pop("schedule")

        if config["sparsity"] is None:
            return None

        # TODO: handle more validation, inspect signature
        sparsity = sparsity_cls(**config)

        group = Group()

        if param_filter is None:
            group.add(default_sparse_param_filter, sparsity)
        elif isinstance(param_filter, (list, tuple, str)):
            group.add(param_filter, sparsity)
        else:
            raise TypeError(
                f"filter must be a string or list of strings, "
                f"got {type(param_filter)}"
            )

        return group
