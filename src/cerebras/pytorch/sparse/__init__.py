"""Sparsity helpers and algorithm shells."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

from .._compat import optional_torch, require_torch
from . import configure, init, utils


class _HookHandle:
    def __init__(self, hooks, hook):
        self._hooks = hooks
        self._hook = hook

    def remove(self):
        if self._hook in self._hooks:
            self._hooks.remove(self._hook)


class SparseParameter:
    def __init__(self, param, mask=None, name: Optional[str] = None):
        self.param = param
        self.mask = mask
        self.name = name


class SparsityAlgorithm:
    def __init__(self, sparsity, init_method="random"):
        self._sparsity = sparsity
        self.init_method = init.make_init_method(init_method)
        self._sparse_params: Dict[Any, SparseParameter] = {}
        self._target_hooks = []
        self._computed_hooks = []

    @property
    def num_sparse_params(self) -> int:
        return len(self._sparse_params)

    @property
    def sparsity(self):
        return self._sparsity

    def get_sparse_params(self, obj):
        if obj in self._sparse_params:
            return self._sparse_params[obj]
        if hasattr(obj, "parameters"):
            return (self._sparse_params[p] for p in obj.parameters() if p in self._sparse_params)
        if hasattr(obj, "param_groups"):
            params = [p for group in obj.param_groups for p in group.get("params", [])]
            return (self._sparse_params[p] for p in params if p in self._sparse_params)
        return None

    def initialize(self):
        return self

    def csx_annotate_sparsity(self, param):
        return param

    def sparsify_parameter(self, module, name, param):
        mask = self.init_method(param, self._target_sparsity(param), None, getattr(param, "device", None))
        sparse_param = SparseParameter(param=param, mask=mask, name=name)
        self._sparse_params[param] = sparse_param
        return sparse_param

    def apply(self, obj):
        if hasattr(obj, "param_groups"):
            return self.sparsify_optimizer(obj)
        if hasattr(obj, "named_parameters"):
            return self.sparsify_module(obj)
        return obj

    def sparsify_module(self, module):
        for name, param in module.named_parameters(recurse=True):
            if configure.default_sparse_param_filter(name, param):
                self.sparsify_parameter(module, name, param)
        return module

    def prune_weight(self, sparse_param):
        if sparse_param.mask is not None and hasattr(sparse_param.param, "data"):
            sparse_param.param.data.mul_(sparse_param.mask.to(sparse_param.param.device))
        return sparse_param

    def _grad_hook(self, p, grad):
        sparse_param = self._sparse_params.get(p)
        if sparse_param is None or sparse_param.mask is None:
            return grad
        return grad * sparse_param.mask.to(grad.device)

    def sparsify_optimizer(self, optimizer):
        return optimizer

    def update(self, optimizer=None):
        for sparse_param in self._sparse_params.values():
            self.prune_weight(sparse_param)
        return self

    def register_target_sparsity_hook(self, hook):
        self._target_hooks.append(hook)
        return _HookHandle(self._target_hooks, hook)

    def register_computed_sparsity_hook(self, hook):
        self._computed_hooks.append(hook)
        return _HookHandle(self._computed_hooks, hook)

    def visit_state(self, fn):
        self._sparsity = fn(self._sparsity)
        return self

    def state_dict(self):
        return {"sparsity": self._sparsity, "num_sparse_params": self.num_sparse_params}

    def load_state_dict(self, state_dict):
        self._sparsity = state_dict.get("sparsity", self._sparsity)

    def _target_sparsity(self, param=None):
        if hasattr(self._sparsity, "compute"):
            return self._sparsity.compute()
        if isinstance(self._sparsity, dict) and param in self._sparsity:
            value = self._sparsity[param]
            return value.compute() if hasattr(value, "compute") else value
        return float(self._sparsity or 0.0)


class Static(SparsityAlgorithm):
    pass


class DynamicSparsityAlgorithm(SparsityAlgorithm):
    def __init__(self, sparsity=None, schedule=None, update=None, **kwargs):
        super().__init__(sparsity if sparsity is not None else schedule, kwargs.pop("init_method", "random"))
        self.update_schedule = utils.make_update_schedule(update)

    @property
    def is_update_step(self):
        torch = optional_torch()
        value = self.update_schedule.compute() if hasattr(self.update_schedule, "compute") else True
        return torch.tensor(bool(value)) if torch is not None else bool(value)

    def update_mask(self, p, mask, sparsity):
        raise NotImplementedError


class GMP(DynamicSparsityAlgorithm):
    pass


class SET(DynamicSparsityAlgorithm):
    def __init__(self, drop_fraction=0.3, **kwargs):
        self.drop_fraction = drop_fraction
        super().__init__(**kwargs)


class RigL(DynamicSparsityAlgorithm):
    def __init__(self, drop_fraction=0.3, balance_in_groups=None, balance_out_groups=None, **kwargs):
        self.drop_fraction = drop_fraction
        self.balance_in_groups = balance_in_groups
        self.balance_out_groups = balance_out_groups
        super().__init__(**kwargs)


class Group(SparsityAlgorithm):
    def __init__(self, groups=None):
        super().__init__(0.0)
        self.groups = OrderedDict(groups or {})

    def add(self, pattern, algorithm):
        self.groups[pattern] = algorithm
        return self

    def extend(self, groups):
        self.groups.update(groups)
        return self

    def apply(self, obj):
        for algorithm in self.groups.values():
            algorithm.apply(obj)
        return obj


__all__ = [
    "DynamicSparsityAlgorithm",
    "GMP",
    "Group",
    "RigL",
    "SET",
    "SparseParameter",
    "SparsityAlgorithm",
    "Static",
    "configure",
    "init",
    "utils",
]
