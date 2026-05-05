# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""The Cerebras base optimizer class"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterable, Union

import torch
from torch.optim.optimizer import ParamsT
from torch.utils.hooks import RemovableHandle
from torch.utils.weak import WeakIdKeyDictionary

from cerebras.pytorch import _generating_docs
from cerebras.pytorch.backend import current_backend_impl

if _generating_docs:
    ParamsT = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

    class torch:
        class optim:
            class Optimizer:
                pass


class Optimizer(torch.optim.Optimizer, ABC):
    """
    The abstract Cerebras base optimizer class.

    Enforces that the `preinitialize` method is implemented
    wherein the optimizer state should be initialized ahead of time
    """

    def __init__(
        self,
        params: ParamsT,
        defaults: Dict[str, Any],
        enable_global_step: bool = False,
    ):
        """
        Args:

            params: Specifies what Tensors should be optimized.
            defaults: a dict containing default values of optimization options
                (used when a parameter group doesn’t specify them).
            enable_global_step: If True, the optimizer will keep track of the
                global step for each parameter.
        """
        super().__init__(params, defaults)
        self.enable_global_step = enable_global_step

        self.backend = current_backend_impl()

        with self.backend.device:
            self.preinitialize()

            if enable_global_step:
                for group in self.param_groups:
                    for p in group["params"]:
                        self.state[p]["step"] = torch.tensor(
                            0.0, dtype=torch.float32
                        ).to(p.device)

        self._schedulers_registry = WeakIdKeyDictionary()

        self.backend.register_optimizer(self)

        self._optimizer_zero_grad_pre_hooks = OrderedDict()
        self._optimizer_zero_grad_post_hooks = OrderedDict()

        for param_group in self.param_groups:
            if param_group.get("tags", None):
                if isinstance(param_group["tags"], (list, tuple)):
                    param_group["tags"] = set(param_group["tags"])
                elif not isinstance(param_group["tags"], set):
                    param_group["tags"] = {param_group["tags"]}

    def increment_global_step(self, p):
        """
        Increases the global steps by 1 and returns the current
        value of global step tensor in torch.float32 format.
        """
        if "step" not in self.state[p]:
            raise RuntimeError(
                "No global step in the state. "
                "Please pass in `enable_global_step=True` "
                "to initialize the global step"
            )

        self.state[p]["step"] += 1.0
        return self.state[p]["step"]

    def state_dict(self, *args, **kwargs):
        s = super().state_dict(*args, **kwargs)

        return s

    def load_state_dict(self, state_dict):
        with self.backend.device:
            super().load_state_dict(state_dict)

    def register_zero_grad_pre_hook(self, hook) -> RemovableHandle:
        r"""Register an optimizer zero_grad pre hook which will be called before
        optimizer zero_grad. It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._optimizer_zero_grad_pre_hooks)
        self._optimizer_zero_grad_pre_hooks[handle.id] = hook
        return handle

    def register_zero_grad_post_hook(self, hook) -> RemovableHandle:
        r"""Register an optimizer zero_grad post hook which will be called after
        optimizer zero_grad. It should have the following signature::

            hook(optimizer, args, kwargs)

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._optimizer_zero_grad_pre_hooks)
        self._optimizer_zero_grad_pre_hooks[handle.id] = hook
        return handle

    def zero_grad(self, *args, **kwargs):
        """
        Runs the optimizer zero_grad method and calls any pre and post hooks
        """
        for pre_hook in self._optimizer_zero_grad_pre_hooks.values():
            result = pre_hook(self, args, kwargs)
            if result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    args, kwargs = result
                else:
                    raise RuntimeError(
                        f"{pre_hook} must return None or a tuple of "
                        f"(new_args, new_kwargs), but got {result}."
                    )

        super().zero_grad(*args, **kwargs)

        for post_hook in self._optimizer_zero_grad_post_hooks.values():
            post_hook(self, args, kwargs)

    def apply(self, f):
        """Calls the function on self"""
        if not callable(f):
            # If the function is not callable, check if it has an apply
            # method and call it, supplying self as the argument.
            f_apply = getattr(f, "apply", None)
            if f_apply is not None and callable(f_apply):
                return f_apply(self)

            raise TypeError(
                f"Expected a callable as the argument to apply. "
                f"Got: {type(f)}"
            )

        return f(self)

    def visit_state(self, fn):
        """
        Applies a lambda to each stateful value.
        """
        for state in self.state.values():
            for key, val in state.items():
                new_val = fn(val)
                if new_val is not None:
                    state[key] = new_val

    @abstractmethod
    def preinitialize(self):
        """
        The optimizer state must be initialized ahead of time in order
        to capture the full compute graph in the first iteration. This method
        must be overriden to perform the state preinitialization
        """

    @abstractmethod
    def step(self, closure=None):
        """
        Perform the optimizer step itself. Note, there should be no new state
        being created in this function. All state must be created ahead of time in
        `preinitialize` and only updated in this method.
        """
