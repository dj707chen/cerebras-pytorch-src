# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils._pytree import tree_map

from cerebras.pytorch.backend import current_backend_impl
from cerebras.pytorch.core.function_mode import (
    register_function_mode_forward_hook,
    register_function_mode_forward_pre_hook,
)
from cerebras.pytorch.lib import cerebras_pytorch_lib


class SkipBwdAnnotation:
    """
    SkipBwdAnnotation class provides utilities for tensors annotation
    enabling/disabling withing a context. Should be used at the beginning
    of annotation (to prevent out of scope tensors annotation) and at the
    end (to restore original tensor state).
    """

    @dataclass
    class TensorState:
        tensor: torch.Tensor
        orig_value: bool

    def __init__(self, *args, **kwargs):
        """Constructs a `SkipBwdAnnotation` instance.

        Args:
            args: Function args captured by `pol` decorator. We
                explicitly disable annotation for them since they are
                out of annotation scope.
            kwargs: Function kwargs captured by `pol` decorator.
                We explicitly disable annotation for them since they
                are out of annotation scope.
        """
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        # Original func arguments shouldn't be annotated as they
        # are out of scope, so we mark them with special flag.
        self.func_args_state = self.disable_annotation(self.args, self.kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self.enable_annotation(self.func_args_state)

    @staticmethod
    def disable_annotation(*args, **kwargs):
        """Disable bwd pass annotation for given tensor"""

        def disable(arg: torch.Tensor):
            if isinstance(arg, torch.Tensor):
                orig_value = getattr(arg, '_skip_grad_fn_hooks', None)
                arg._skip_grad_fn_hooks = True
                return SkipBwdAnnotation.TensorState(arg, orig_value)
            return None

        return tree_map(
            lambda x: disable(x),
            (args, kwargs),
        )

    @staticmethod
    def enable_annotation(self, *args, **kwargs):
        """Reset the annotation state for given tensor"""

        def enable(arg_state: SkipBwdAnnotation.TensorState):
            if not isinstance(arg_state, SkipBwdAnnotation.TensorState):
                return

            arg = arg_state.tensor
            if arg_state.orig_value is not None:
                arg._skip_grad_fn_hooks = arg_state.orig_value
            else:
                del arg._skip_grad_fn_hooks

        tree_map(
            lambda x: enable(x),
            (args, kwargs),
        )


class ReverseHooksExecutor:
    """
    Execute hooks in specific order. See [Note: bwd pass annotation]
    for more details.
    """

    def __init__(self):
        self._hooks: List[Callable] = []

    def __call__(self, *args, **kwargs):
        for hook in reversed(self._hooks):
            hook(*args, **kwargs)

    def register_hook(self, hook: Callable):
        self._hooks.append(hook)


def get_hooks_executor(arg: torch.Tensor) -> ReverseHooksExecutor:
    """
    Hook `grad_fn` for given tensor using ReverseHooksExecutor.
    See [Note: bwd pass annotation] for more details.
    Args:
        arg: tensor to hook.
    """
    if hasattr(arg, '_grad_fn_hooks_executor'):
        return arg._grad_fn_hooks_executor

    reverse_executor = ReverseHooksExecutor()
    arg._grad_fn_hooks_executor = reverse_executor
    arg.grad_fn.register_prehook(reverse_executor)
    return reverse_executor


class AnnotationMode(ABC):
    """
    [Note: tensors annotation]
    AnnotationMode class allows to annotate tensors created
    within its context with attributes including operations
    from both fwd/bwd passes. At the core of this class we use
    TorchFunctionMode context manager to hook ops execution and
    capture all LTC tensors. In addition to function mode we use
    LTC backend functionality to set attributes to the IR nodes.

    To capture operations from fwd or bwd pass we use next approach:

       1. Capture operation execution in __torch_function__
       2. Before op execution we set attribute to the LTC backend
          and notify it to start listening for the ops, so every
          created op will be annotated with given attribute.
       3. Then we execute operation and notify LTC backend to
          stop listen for new operations.
          See start_annotation/end_annotation for more details.

    The only difference between fwd/bwd ops is that we can easily
    hook fwd ops, but to capture bwd ops (tensors) we need to hook
    backward method call which constructs a graph underneath. So, for
    bwd ops we hook tensor.grad_fn (start/end) functions that notify
    LTC backend when to start annotation and when to end.
    See [Note: bwd pass annotation] for more details.

    It would be nice to have only AnnotationMode for annotation,
    however, we can not hook `grad_fn` for the last tensors created
    by the last operation, since AOT applies after we return tensor
    from __torch_function__ dispatcher.
    To handle that we provide `annotate` function, which returns a
    decorator. See `pol` function for more details.

    Annotation example:

      @cstorch.pol(level=1)
      def forward(self, x: torch.Tensor):
          return self.fc(x)

    `pol` decorator captures function tensors and disables annotation
    for them, since they are not in scope. At the same time, it captures
    resulting tensors and explicitly enables bwd ops annotation.
    """

    @dataclass
    class Config:
        """Base class for annotation config"""

        enable_fwd: bool
        enable_bwd: bool

    @dataclass
    class Attribute:
        """Represents annotation attribute"""

        name: str
        value: Optional[Any]

    @property
    @abstractmethod
    def config(self):
        """Global config getter"""
        ...

    @config.setter
    @abstractmethod
    def config(self, config: AnnotationMode.Config):
        """Global config setter"""
        ...

    @abstractmethod
    def get_attribute(self, config: AnnotationMode.Config):
        """Returns annotation attribute"""
        ...

    def enable_annotation(
        self,
        attr: AnnotationMode.Attribute,
        is_backward: bool,
        enable_fwd: bool,
        enable_bwd: bool,
    ):
        """This function notifies LTC backend to start listening for the operations"""
        if (
            (is_backward and not enable_bwd)
            or (not is_backward and not enable_fwd)
            or attr.value is None
        ):
            return

        cerebras_pytorch_lib.enable_annotation(attr.name, attr.value)

    def disable_annotation(self, attr: AnnotationMode.Attribute):
        """This function notifies LTC backend to stop listening for the operations"""
        cerebras_pytorch_lib.disable_annotation(attr.name)

    def hook_tensor_grad_fn(self, arg: torch.Tensor, grad_fns: list):
        """Hooks `tensor.grad_fn` function with start/end annotation hooks"""
        if not isinstance(arg, torch.Tensor) or arg.grad_fn is None:
            return None

        grad_fns.append(get_hooks_executor(arg))

        # Set grad_fn begin/end hooks to annotate tensors created within backward pass.
        attr = self.get_attribute(self.config, is_backward=True)

        if hasattr(arg, '_skip_grad_fn_hooks'):
            skip_hooks = arg._skip_grad_fn_hooks
            if isinstance(skip_hooks, bool):
                if skip_hooks:
                    return None
            elif attr.name in skip_hooks:
                return None

        # If `arg` doesn't have corresponding input `grad_fn` function, we don't
        # know when to stop bwd pass annotation, so we skip annotation in that case.
        if not hasattr(arg, "_input_grad_fns") or len(arg._input_grad_fns) == 0:
            return

        # [Note: bwd pass annotation]
        # ReverseHooksExecutor allows to manage hooks execution order.
        # This is critical for `bwd` pass annotation, since we
        # may have two hooks (`__enter__` ad `__exit__`) registered
        # for the same `grad_fn` function, so we need to run exit hook first,
        # and only then execute enter hook.
        #
        # For example, let's look at the annotation process for this
        # sequence of ops:
        #
        #   input->linear->reshape (fwd pass)
        #
        # First, we hit `linear` in `__torch_function__` and apply bwd
        # annotation for `input` tensor (in our case it is input tensor,
        # so no `grad_fn` annotation will be applied).
        # Then we annotate ops from `linear` fwd pass and save `grad_fn`
        # from `input` tensor to the tensor produced by `linear` op. This
        # will give us information about when bwd pass for `linear` op
        # starts and when it ends.
        # Then we hit `reshape` in `__torch_function__` and apply bwd
        # annotation for the input tensor, so in our case it will be an
        # output of `linear` op. Inside bwd annotation we hook `grad_fn`
        # for the input tensor (this will mark start of the bwd pass) and
        # then we hook `input_grad_fn` (this will mark end of the bwd pass).
        # The same `grad_fn` annotation process will be applied to the
        # `reshape` op later. So, given that we annotate `linear` op first,
        # we will register `__enter__` hook first, and then we will register
        # `__exit__` hook for the same tensor to mark the end of bwd pass for
        # the `reshape` op.
        # The problem with such ordering of hooks is that `__exit__` removes
        # attribute from the backend, so we need to execute hooks in reverse
        # order.
        hq = get_hooks_executor(arg)
        # Since bwd sub-graphs execution happens after fwd pass (where we may have
        # nested annotation levels), we store some config fields to preserve their original value.
        enable_fwd, enable_bwd = self.config.enable_fwd, self.config.enable_bwd
        hq.register_hook(
            lambda x: self.enable_annotation(
                attr,
                is_backward=True,
                enable_fwd=enable_fwd,
                enable_bwd=enable_bwd,
            )
        )

        # Original fwd operation may take several input tensors and `_input_grad_fns`
        # will have all available `grad_fn` functions. However, in order to mark the
        # end of bwd pass we may annotate any of these functions, so we take the first
        # one.
        arg._input_grad_fns[0].register_hook(
            lambda x: self.disable_annotation(attr)
        )

        # The same tensor may be consumed by may different operations, however
        # we need to annotate only once, so we mark a tensor with `_skip_grad_fn_hooks`
        # to show that hooks were already set for these tensor.
        if not hasattr(arg, '_skip_grad_fn_hooks'):
            arg._skip_grad_fn_hooks = set()
        arg._skip_grad_fn_hooks.add(attr.name)

    def enable_bwd_annotation(
        self,
        args: Sequence[Any] = (),
        kwargs: Optional[Dict] = None,
    ):
        """
        Applies bwd ops annotation for the args and kwargs and
        collect `grad_fn` from given tensors.
        """
        input_grad_fns = []
        tree_map(
            lambda x: self.hook_tensor_grad_fn(x, input_grad_fns),
            (args, kwargs),
        )
        return input_grad_fns

    def __init__(self, config: AnnotationMode.Config):
        """Constructs a `AnnotationMode` instance"""
        super().__init__()
        self.local_config = config

    def __enter__(self):
        # In case we have nested annotation levels, we need to preserve original
        # global annotation config, so we can restore it at the exit.
        self.orig_config = self.config
        self.config = self.local_config

        # [Note: global config]
        # In case of nested annotation levels, we use root AnnotationMode for annotation
        # and registering hooks, and in the nested AnnotationMode instances we just update
        # global config and enable nested annotation.

        if self.orig_config is not None:
            self.disable_annotation(
                self.get_attribute(self.orig_config, is_backward=False)
            )
            self.attr = self.get_attribute(self.config, is_backward=False)
            self.enable_annotation(
                self.attr,
                is_backward=False,
                enable_fwd=self.config.enable_fwd,
                enable_bwd=self.config.enable_bwd,
            )
            return self

        def forward_pre_hook(func, types, args, kwargs):
            # Annotate ops from the bwd pass.
            self.input_grad_fns = self.enable_bwd_annotation(args, kwargs)

        def forward_hook(func, types, args, kwargs, res):
            # [Note: bwd pass annotation]
            # To have correct annotation for bwd pass ops, we need
            # to hook `grad_fn` for the resulting tensor and for the
            # input tensor. This pair of `grad_fn` hooks represents
            # start of bwd pass for given op and end of bwd pass.
            # We can not set `grad_fn` hook for resulting tensor here
            # since autograd will be applied after tensor is returned
            # from `__torch_function__`. Given that, we postpone
            # `grad_fn` hooks to the next `__torch_function__` call,
            # where the resulting tensor becomes an input to following
            # operation, so autograd was already applied, and we have
            # `grad_fn` in place.
            # For that purpose, we need to save information about input
            # tensor `grad_fn` functions, so we can access them later in
            # `enable_bwd_annotation` function.
            def save_grad_fn(x: torch.Tensor):
                if not isinstance(x, torch.Tensor):
                    return x
                x._input_grad_fns = self.input_grad_fns
                return x

            tree_map(lambda x: save_grad_fn(x), res)
            self.input_grad_fns = None

        self.fwd_pre_hook_handle = register_function_mode_forward_pre_hook(
            forward_pre_hook
        )
        self.fwd_hook_handle = register_function_mode_forward_hook(forward_hook)

        # Annotate ops from the fwd pass.
        self.attr = self.get_attribute(self.config, is_backward=False)
        self.enable_annotation(
            self.attr,
            is_backward=False,
            enable_fwd=self.config.enable_fwd,
            enable_bwd=self.config.enable_bwd,
        )

    def __exit__(self, *args, **kwargs):
        # Restore original global config.
        # See [Note: global config] for more details.
        self.config = self.orig_config

        self.disable_annotation(self.attr)

        if self.config is None:
            self.fwd_pre_hook_handle.remove()
            self.fwd_hook_handle.remove()
        else:
            # nested annotation, re-enable outer annotation
            self.enable_annotation(
                self.get_attribute(self.config, is_backward=False),
                is_backward=False,
                enable_fwd=self.config.enable_fwd,
                enable_bwd=self.config.enable_bwd,
            )


def annotate(
    annotation_mode: AnnotationMode,
):
    backend = current_backend_impl(False)
    if not backend or backend.torch_device.type != "lazy":
        backend_name = f"{backend.torch_device}" if backend else "None"
        warnings.warn(
            f"Setting {annotation_mode.__class__.__name__} for \"{backend_name}\" "
            f"device doesn't have any effect."
        )
        if not backend:
            return lambda fn: fn

    def _(func):
        @wraps(func)
        def inner(*args, **kwargs):
            with SkipBwdAnnotation(args, kwargs) as skip_annotation:
                with annotation_mode:
                    out = func(*args, **kwargs)
                    # Annotate last output tensors not captured
                    # by the torch function mode.
                    annotation_mode.enable_bwd_annotation(out)
                    # To avoid annotations override (case with nested annotation levels),
                    # we disable annotation for the `out` tensor as it can be used
                    # by operations in the parent annotation context manager.
                    skip_annotation.disable_annotation(out)
                    return out

        return inner

    return _


annotation_registry: Dict[AnnotationMode.Config, AnnotationMode] = {}


def create_annotation(
    annotation_config: AnnotationMode.Config,
    get_attribute_fn: Callable,
    **kwargs,
):
    """Create annotation decorator."""
    if annotation_config not in annotation_registry:

        class GenericAnnotationMode(AnnotationMode):
            _annotation_config: Optional[AnnotationMode.Config] = None

            def __init__(
                self, config: AnnotationMode.Config, get_attribute_fn: Callable
            ):
                super().__init__(config)
                self.get_attribute_fn = get_attribute_fn

            @property
            def config(self):
                return GenericAnnotationMode._annotation_config

            @config.setter
            def config(self, config: AnnotationMode.Config):
                GenericAnnotationMode._annotation_config = config

            def get_attribute(
                self, config: AnnotationMode.Config, is_backward: bool
            ):
                return self.get_attribute_fn(config, is_backward)

        annotation_registry[annotation_config] = GenericAnnotationMode

    return annotate(
        annotation_mode=annotation_registry[annotation_config](
            annotation_config(**kwargs), get_attribute_fn
        )
    )
