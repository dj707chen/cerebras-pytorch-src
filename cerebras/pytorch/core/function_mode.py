# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle

_global_function_mode_forward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_function_mode_forward_error_hooks: Dict[int, Callable] = OrderedDict()
_global_function_mode_forward_hooks: Dict[int, Callable] = OrderedDict()


def register_function_mode_forward_pre_hook(
    hook: Callable[..., None]
) -> RemovableHandle:
    """
    Register pre forward hook for the function mode.
    Args:
        hook: a callback that is being called before operation execution.
        For the callback signature see `__function_mode__`.
    Returns:
        handle: a handle that can be used to delete registered hook.
    Example:
        def forward_pre_hook(func, types, args, kwargs) -> None:
            ...
        handle = register_function_mode_forward_pre_hook(forward_pre_hook)
        ...
        handle.remove()
    """
    handle = RemovableHandle(_global_function_mode_forward_pre_hooks)
    _global_function_mode_forward_pre_hooks[handle.id] = hook
    return handle


def register_function_mode_forward_error_hook(
    hook: Callable[..., None]
) -> RemovableHandle:
    """
    Register forward error hook for the function mode. This will run if
    one of the function calls leads to an error.
    Note: Best practice is to only append to the original message and reraise it.

    Args:
        hook: a callback that is being called before operation execution.
        For the callback signature see `__function_mode__`.
    Returns:
        handle: a handle that can be used to delete registered hook.
    Example:
        def forward_error_hook(func, types, args, kwargs, error) -> None:
            ...
        handle = register_function_mode_forward_error_hook(forward_error_hook)
        ...
        handle.remove()
    """
    handle = RemovableHandle(_global_function_mode_forward_error_hooks)
    _global_function_mode_forward_error_hooks[handle.id] = hook
    return handle


def register_function_mode_forward_hook(
    hook: Callable[..., None]
) -> RemovableHandle:
    """
    Register forward hook for the function mode.
    Args:
        hook: a callback that is being called after operation execution.
        For the callback signature see `__function_mode__`.
    Returns:
        handle: a handle that can be used to delete registered hook.
    Example:
        def forward_hook(func, types, args, kwargs, res) -> None:
            ...
        handle = register_function_mode_forward_hook(forward_hook)
        ...
        handle.remove()
    """
    handle = RemovableHandle(_global_function_mode_forward_hooks)
    _global_function_mode_forward_hooks[handle.id] = hook
    return handle


class CerebrasFunctionMode(TorchFunctionMode, ABC):
    """
    Function Mode allows to capture tensor operations on
    the python-level. The main goal of this class is to
    provide a single function mode which is running on
    `step_fn` and allows to register hooks.
    Note: function mode doesn't capture operations from
    the bwd pass, since these ops are being created on
    C++ level, so they are not visible to the pytorch
    function mode.
    """

    def __torch_function__(
        self,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[Dict] = None,
    ):
        """Hook operations from the forward pass"""
        if not kwargs:
            kwargs = {}

        for fwd_pre_hook in _global_function_mode_forward_pre_hooks.values():
            out = fwd_pre_hook(func, types, args, kwargs)
            if out is not None and isinstance(out, tuple) and len(out) == 4:
                func, types, args, kwargs = out

        try:
            res = func(*args, **kwargs)
        except Exception as e:
            # handle e if possible otherwise, we'll reraise the original error
            for (
                fwd_error_hook
            ) in _global_function_mode_forward_error_hooks.values():
                fwd_error_hook(func, types, args, kwargs, e)
            raise

        for fwd_hook in _global_function_mode_forward_hooks.values():
            fwd_hook(func, types, args, kwargs, res)

        return res


class CerebrasFunctionModeContext(ABC):
    """
    Function Mode Context allows us to create a context where the hooks that are overridden in
    child classes will automatically be added in on enter and will be removed on exit.

    This would look like as follows (with ctx being an instance of CerebrasFunctionModeContext)
    with ctx:
        ...
    where ... will run with the pre and forward hooks provided.
    """

    @abstractmethod
    def forward_pre_hook(
        self,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[Dict] = None,
    ) -> Optional[Tuple[Callable, Sequence, Sequence, Optional[Dict]]]:
        """A hook that is called before an operation is dispatched

        This hook may optionally return a tuple of the form
        (func, types, args, kwargs) to override the original
        operation that was dispatched.
        """

    @abstractmethod
    def forward_hook(
        self,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[Dict] = None,
        result: Any = None,
    ):
        """A hook that is called after an operation is executed"""

    def __enter__(self):
        """On __enter__, or on `with ctx` register hooks"""
        self._handles = [
            register_function_mode_forward_pre_hook(self.forward_pre_hook),
            register_function_mode_forward_hook(self.forward_hook),
        ]

    def __exit__(self, *args):
        """On __exit__, or when the `with ctx` scope ends, unregister hooks"""
        for handle in self._handles:
            handle.remove()

        self._handles = None
