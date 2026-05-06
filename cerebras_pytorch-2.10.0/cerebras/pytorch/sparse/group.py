# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union
from warnings import warn

import torch

import cerebras.pytorch as cstorch
from cerebras.pytorch.sparse.base import SparsityAlgorithm
from cerebras.pytorch.utils.utils import convert_glob_to_regex


class Group(SparsityAlgorithm):
    """
    Group sparsity algorithm. This algorithm allows for multiple sparsity
    algorithms to be applied to different groups of parameters.

    For example:

    .. code:: python

        sparsity = cstorch.sparse.Group({
            "fc1.*": cstorch.sparse.Static(sparsity=0.5),
            "fc2.*": cstorch.sparse.GMP(
                schedule=[0.3, 0.4, 0.5],
                update: {"freq": 100}
            ),
        })
        sparsity.add("fc3.*", cstorch.sparse.RigL(sparsity=0.5))

        model.apply(sparsity)
        optimizer.apply(sparsity)

    The group sparsity algorithm will apply the sparsity algorithms to the
    parameters that match the filter. If a parameter name matches multiple
    filters, the first filter that matches will be used.
    """

    @dataclass
    class Filter:
        filter: Callable[[str, torch.Tensor], bool]
        algorithm: SparsityAlgorithm

    def __init__(self, groups: Dict[str, SparsityAlgorithm] = None):
        """
        Args:
            groups: A dictionary of filter -> algorithm pairs.
                See :py:meth:`~cerebras.pytorch.sparse.Group.add` for more details.
        """
        super().__init__(sparsity=None)

        self._groups = []

        if groups is not None:
            for group_filter, algorithm in groups.items():
                self.add(group_filter, algorithm)

    @property
    def num_sparse_params(self):
        return sum(len(g.algorithm.sparse_params) for g in self._groups)

    @property
    def sparsity(self):
        raise NotImplementedError(
            "Group sparsity algorithm does not have a sparsity level. "
            "You can access the sparsity of nested sparsity algorithms by "
            "indexing the Group object, i.e. group[0].sparsity"
        )

    def __getitem__(self, index) -> SparsityAlgorithm:
        """Returns the algorithm at the given index."""
        return self._groups[index].algorithm

    def add(
        self,
        filter: Union[str, Callable[[str, torch.Tensor], bool]],
        algorithm: SparsityAlgorithm,
    ):
        """
        Add a sparsity algorithm to the group.

        Args:
            filter: A string, list of strings, or callable that takes a
                parameter name and a parameter tensor and returns True if the
                parameter should be sparsified.

                If one or more strings are provided, the filter will match if
                any of the strings match the parameter name. The strings may
                contain glob patterns, e.g. "fc1.*" will match all parameters
                in the "fc1" module.

            algorithm: An instance of :py:class:`~cerebras.pytorch.sparse.SparsityAlgorithm`
        """
        if not isinstance(algorithm, SparsityAlgorithm):
            raise TypeError(
                f"algorithm must be an instance of SparsityAlgorithm, got {type(algorithm)}"
            )
        elif isinstance(algorithm, Group):
            raise TypeError(
                f"algorithm must be not be Group sparsity algorithm. "
                f"If you want to merge groups, use the extend method."
            )

        if isinstance(filter, str):
            filter = [filter]

        if isinstance(filter, (list, tuple)):
            filter_re = list(map(convert_glob_to_regex, filter))

            filter = lambda name, _: any(
                f.match(name) is not None for f in filter_re
            )
            self._groups.append(Group.Filter(filter, algorithm))

        elif callable(filter):
            self._groups.append(Group.Filter(filter, algorithm))
        else:
            raise TypeError(
                f"filter must be a string or callable, got {type(filter)}"
            )

        self.sparse_params.update(algorithm.sparse_params)

    def extend(self, group: "Group"):
        """
        Extend the group with the filters and algorithms from another group.

        Args:
            group: An instance of :py:class:`~cerebras.pytorch.sparse.Group`
        """
        if not isinstance(group, Group):
            raise TypeError(
                f"group must be an instance of Group, got {type(group)}"
            )

        for g in group._groups:
            self.add(g.filter, g.algorithm)

    def sparsify_parameter(
        self, module: torch.nn.Module, name: str, param: torch.Tensor
    ) -> None:
        if param is None:
            # Parameter is None, nothing to sparsify
            return
        if self.get_sparse_params(param):
            # Parameter is already sparsified
            return
        if getattr(param, "requires_dense", False):
            # Parameter has been marked as not sparsifiable
            return

        for group in self._groups:
            if group.filter(name, param):
                logging.debug(f"Sparsity filter matched: {name}")
                group.algorithm.sparsify_parameter(module, name, param)
                # Update the sparse_params
                self.sparse_params.update(group.algorithm.sparse_params)
                return
            else:
                logging.debug(f"Sparsity filter did *not* match: {name}")

    def sparsify_module(self, module):
        if len(self._groups) == 0:
            raise RuntimeError(
                "No groups were added to the Group sparsity algorithm"
            )

        super().sparsify_module(module)

        if sum(len(g.algorithm.sparse_params) for g in self._groups) == 0:
            warn(
                "No parameters were sparsified in the module. "
                "This is likely due to the parameter filter not matching any "
                "parameters in the module"
            )

    def _forward_pre_hook(self, module, input):
        for group in self._groups:
            group.algorithm._forward_pre_hook(module, input)

    def sparsify_optimizer(self, optimizer):
        super().sparsify_optimizer(optimizer)

        # Call sparsify optimizer on each algorithm
        # so that it can apply any optimizer hooks
        for group in self._groups:
            group.algorithm.sparsify_optimizer(optimizer)

    def update(self, optimizer: Optional[cstorch.optim.Optimizer] = None):
        for group in self._groups:
            group.algorithm.update(optimizer)

    def register_target_sparsity_hook(self, hook):
        return [
            group.algorithm.register_target_sparsity_hook(hook)
            for group in self._groups
        ]

    def register_computed_sparsity_hook(self, hook):
        return [
            group.algorithm.register_computed_sparsity_hook(hook)
            for group in self._groups
        ]

    def visit_state(self, f):
        for group in self._groups:
            group.algorithm.visit_state(f)

    def state_dict(self):
        return [group.algorithm.state_dict() for group in self._groups]

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict):
            state_dict = [state_dict]
        if isinstance(state_dict, list):
            if len(state_dict) != len(self._groups):
                raise ValueError(
                    f"Expected a list of {len(self._groups)} state_dicts for "
                    f"the Group sparsity algorithm but got {len(state_dict)}."
                )
            for s, group in zip(state_dict, self._groups):
                group.algorithm.load_state_dict(s)
