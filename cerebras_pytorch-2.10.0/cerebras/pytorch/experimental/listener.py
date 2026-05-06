# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Callable, Dict

import torch
from torch.utils.hooks import RemovableHandle

from cerebras.pytorch.backend import current_backend_impl
from cerebras.pytorch.core.compile import (
    register_trace_fn_post_hook,
    register_trace_fn_pre_hook,
)
from cerebras.pytorch.core.function_mode import (
    register_function_mode_forward_hook,
    register_function_mode_forward_pre_hook,
)

TracedTensorHook = Callable[[torch.Tensor, str], None]
_global_traced_tensor_hooks: Dict[int, TracedTensorHook] = OrderedDict()


def register_traced_tensor_hook(hook: TracedTensorHook) -> RemovableHandle:
    """Register a hook to be called when a tensor is created during tracing.

    Tracing here referes to the `cstorch.trace()` context.

    Args:
        hook: Callable that will be called with the signature below. The name is a unique
            name derived from the order of operations and scopes during tracing.
                "hook(tensor: torch.Tensor, name: str) -> None"
    Returns:
        A handle which can be used to remove the hook.
    """
    handle = RemovableHandle(_global_traced_tensor_hooks)
    _global_traced_tensor_hooks[handle.id] = hook
    return handle


class _ListenerMode:
    """Class that listens to tensors captured during tracing and calls registered hooks on them."""

    def __init__(self):
        self._unique_tensor_names = None
        self._is_tracing = False
        self._last_context = None

        self._handles = [
            register_trace_fn_pre_hook(self._trace_fn_pre_hook),
            register_trace_fn_post_hook(self._trace_fn_post_hook),
            register_function_mode_forward_pre_hook(self._forward_pre_hook),
            register_function_mode_forward_hook(self._forward_hook),
        ]

    def _trace_fn_pre_hook(self):
        """Hook that is called before tracing begins every iteration."""
        self._is_tracing = True

        # We want to keep uniqueness among all traces in the same iteration.
        backend = current_backend_impl()
        new_context = (hash(backend.run_context), backend.run_context.iteration)
        if new_context != self._last_context:
            self._unique_tensor_names = defaultdict(int)
            self._last_context = new_context

    def _trace_fn_post_hook(self):
        """Hook that is called after tracing ends every iteration."""
        self._is_tracing = False

    def _forward_pre_hook(self, func, types, args, kwargs):
        """Pre-forward hook that hooks `grad_fn` in order to capture tensors from bwd pass."""
        if not self._is_tracing or self._is_blacklisted_op(func):
            return

        def hook_bwd(tensor: torch.Tensor):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.grad_fn is None
                or not hasattr(tensor, '_func_name')
            ):
                return

            def set_grad_fn_hook(tensors, func_name):
                backend = current_backend_impl()

                def grad_fn_hook(bwd_tensor):
                    if (
                        not isinstance(bwd_tensor, torch.Tensor)
                        or bwd_tensor.device.type != backend.device.type
                        or hasattr(bwd_tensor, '_has_bwd_hook')
                    ):
                        return

                    self._apply_listeners(bwd_tensor, func_name)

                _ = torch.utils._pytree.tree_map(grad_fn_hook, tensors)

            # Mark tensor to avoid multiple `grad_fn` registration for
            # the same tensor.
            if hasattr(tensor, '_has_bwd_hook'):
                return

            tensor._has_bwd_hook = True

            tensor_func_name = tensor._func_name
            tensor.grad_fn.register_hook(
                lambda inputs, outputs: set_grad_fn_hook(
                    inputs, tensor_func_name
                )
            )

        _ = torch.utils._pytree.tree_map(hook_bwd, (args, kwargs))

    def _forward_hook(self, func, types, args, kwargs, res):
        """Post-forward hook that applies listeners to the resulting tensors of fwd pass."""
        if not self._is_tracing or self._is_blacklisted_op(func):
            return

        _ = torch.utils._pytree.tree_map(
            lambda arg: self._apply_listeners(arg, func.__name__), res
        )

        def save_bwd_name(arg):
            arg._func_name = func.__name__
            return arg

        # Annotate tensors with the function name which is used
        # in `grad_fn` hooks for tensor naming.
        _ = torch.utils._pytree.tree_map_only(torch.Tensor, save_bwd_name, res)

    def _apply_listeners(self, arg: torch.Tensor, func_name: str):
        """Apply registered listeners for a given tensor."""
        backend = current_backend_impl()
        if (
            not isinstance(arg, torch.Tensor)
            or arg.device.type != backend.torch_device.type
        ):
            return arg

        scope_name = deepcopy(backend.current_scope_name)
        scope_name.scope_func = func_name

        name = str(scope_name)

        # In case we have the same operation called several times
        # within a module, the resulting tensors will have the same
        # name. So we make these tensor names unique by addind `.index`
        counter = self._unique_tensor_names[name]
        self._unique_tensor_names[name] += 1

        if counter:
            name = f"{name}.{counter}"

        # Uncomment to annotate tensors. For debug purposes only.
        # backend.set_attribute(arg, "tensor_name", name)

        for hook in _global_traced_tensor_hooks.values():
            hook(arg, name)

    @staticmethod
    def _is_blacklisted_op(func):
        """Filter noisy ops."""
        return (
            # Avoid `__get__` like tensor accessors.
            re.match(r"__[_a-zA-Z0-9]+__", func.__name__)
            # Skip annoying tensor movements and copy-like ops.
            or func.__name__ in ["clone", "copy", "detach"]
        )


# Instantiate a listener mode on import to register all hooks
_listener_mode = _ListenerMode()
