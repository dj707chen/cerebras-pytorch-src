# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Provides the fundamental and helper functions
required to compile a model for a Cerebras system
"""
from contextlib import nullcontext
from functools import wraps
from inspect import ismethod
from types import MethodType
from typing import Union

import torch

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.descriptor import Descriptor
from cerebras.pytorch.backend import (
    Backend,
    current_backend,
    current_backend_impl,
)
from cerebras.pytorch.core.function_mode import (
    CerebrasFunctionMode,
    CerebrasFunctionModeContext,
)
from cerebras.pytorch.core.name_scope import ModuleNamesGenerator
from cerebras.pytorch.lib import cerebras_pytorch_lib
from cerebras.pytorch.utils.step_closures import RepeatStepClosure


def compile(  # pylint: disable=redefined-builtin
    model: torch.nn.Module,
    backend: Union[str, Backend, None] = None,
):
    """Prepares the PyTorch module for tracing.

    This method prepares the module by moving it to the device so that it can be
    compiled after the first trace. Note that parameter initialization must be
    done before calling this method since post this call, the parameters are
    moved to the device.

    Args:
        model: The PyTorch module to be compiled.
        backend: The Cerebras backend to use to compile. If None, the current
            backend is used. If not current backend is set, the CPU backend is
            initialized and used. Defaults to None.
    Returns:
        A pseudo-module that almost acts like the original module but does not
        have any of the property accessor or private methods of the original
        module. It can be called `module(*args, **kwargs)` to run the forward
        pass, similar to the original module.
    """
    if backend is None:
        backend = current_backend(raise_exception=False, raise_warning=False)
        if backend is None:
            backend = cstorch.backend("cpu")
    elif isinstance(backend, str):
        backend = cstorch.backend(backend)
    elif isinstance(backend, Backend):
        curr_backend = current_backend(
            raise_exception=False, raise_warning=False
        )
        if backend is not curr_backend:
            raise RuntimeError(
                f"Compile got a different backend than the currently "
                f"initialized backend. "
            )
    else:
        raise TypeError(
            f"Expected backend to be one of str, Backend or None. "
            f"Got: {type(backend)}"
        )

    if (
        hasattr(model, "cerebras_device")
        and model.cerebras_device != backend.device
    ):
        raise RuntimeError(
            f"Attempting to compile a model using a different backend "
            f"than what was used to initialize its parameters. "
            f"Please make sure that you are using the same backend "
            f"in initialization and compilation. "
        )

    # pylint: disable=protected-access
    cs_backend_impl = backend._impl
    active_model_id = cs_backend_impl.setup_model(model)

    # Replace the apply method of all submodules with a custom apply method
    # that checks if the argument is callable and calls it if it is.
    # Otherwise, it checks if the argument has an apply method and calls it instead
    def wrap_apply(module):
        module_apply = module.apply

        @wraps(module_apply)
        def custom_apply(_self, f):
            if not callable(f):
                # If the function is not callable, check if it has an apply
                # method and call it, supplying self as the argument.
                f_apply = getattr(f, "apply", None)
                if f_apply is not None and callable(f_apply):
                    return f_apply(_self)

                raise TypeError(
                    f"Expected a callable as the argument to apply. "
                    f"Got: {type(f)}"
                )

            return module_apply(f)

        module.apply = MethodType(custom_apply, module)

    for submodule in model.modules():
        wrap_apply(submodule)

    @wraps(model.__call__)
    def compiled_forward(*args, **kwargs):
        cs_backend_impl.register_active_model(model, active_model_id)
        return cs_backend_impl.forward(model, *args, **kwargs)

    # Add aliases to the compiled forward
    for name in dir(model):
        method = getattr(model, name)
        if not name.startswith("_") and ismethod(method):
            setattr(compiled_forward, name, method)

    compiled_forward.device = cs_backend_impl.torch_device

    return compiled_forward


def trace(step_fn: callable) -> callable:
    """A decorator that wraps the training/evaluation step function for tracing.

    Any operation that is meant to be executed on the Cerebras Wafer-Scale
    Cluster must be wrapped with this decorator. This includes the forward pass,
    backward pass, optimizer steps, and more.

    For example, the following code snippet shows how to wrap a training step
    that does the forward and backward pass and optimizer step:

    ::

        @cstorch.trace
        def training_step(batch, model, optimizer, loss_fn):
            features, labels = batch
            outputs = model(features)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss

    Args:
        step_fn: The training/evaluation step function to be wrapped.
    Returns:
        The wrapped training/evaluation step function.
    """
    outputs = None
    ctx = None

    @wraps(step_fn)
    def generated_trace_fn(*args, **kwargs):
        nonlocal outputs
        nonlocal ctx

        backend = current_backend_impl()
        if (
            not backend.is_csx
            or cstorch.backends.csx.debug.retrace_every_iteration
            or not backend.in_run_context
            or not backend.run_context.traced.is_set()
        ):
            # change functionality if it's a CSX backend
            # if it's CSX - must not change the graph
            if backend.is_csx:
                if cstorch.backends.csx.debug.retrace_every_iteration:
                    # if this is the beginning of a new DataExecutor
                    # i.e. run_context is its initial step or there is no context
                    # create a new context
                    if ctx is None or backend.run_context.is_initial_step:
                        ctx = CheckFunctionArgs()
                else:
                    ctx = RepeatStepClosure()

            with ctx or nullcontext():
                with CerebrasFunctionMode(), ModuleNamesGenerator():
                    outputs = step_fn(*args, **kwargs)

                # Set force=True to mark the outputs as if they were added to a
                # step closure. This ensures that if caller passes these outputs
                # to a step closure, we don't get duplicates.
                backend.mark_output(outputs, force=True)

            if backend.in_run_context:
                backend.run_context.traced.set()

        return outputs

    return generated_trace_fn


class RetraceEveryIteration(Descriptor):
    """
    Descriptor class to allow for the retrace_every_iteration flag to be set
    """

    def sanitize(self, retrace_every_iteration):
        cerebras_pytorch_lib.retrace_every_iteration(retrace_every_iteration)
        return retrace_every_iteration


class CheckFunctionArgs(CerebrasFunctionModeContext):
    """Provides a context that will add function mode hooks to check whether functions and their
    arguments will stay consistent throughout graph compilations.

    This class implements a generator pattern where on the first iteration, we capture all function
    calls (i.e. function name, args, kwargs) that are in the IR graph (i.e. operations that return
    lazy tensors) and on future iterations ensures that those function calls all match
    """

    def __init__(self):
        # trace_ops: stores the first graph compilation
        self.trace_ops = []
        # generator marks whether we're wanting to store information
        # or we want to move to making sure the ops are handled correctly
        self.generator = None

    def transform_arg(self, arg):
        """transforms the arg so we don't save a reference directly to it"""
        if isinstance(arg, torch.Tensor):
            return cerebras_pytorch_lib.get_tensor_hash(arg)
        elif callable(arg):
            return arg.__name__

        # try hashing it if we can - reduces the memory constraints
        try:
            return hash(arg)
        except:
            # if it's not hashable, we can just ignore it, as it's probably not important
            return None

    def transform_func_call(self, func_call):
        """transforms an entire func call to avoid storing lazy tensors"""
        func, args, kwargs = func_call
        args, kwargs = torch.utils._pytree.tree_map(
            self.transform_arg, (args, kwargs)
        )
        return (func.__name__, args, kwargs)

    def capture(self):
        """First iteration of dataexecutor means capturing all trace ops"""
        while True:
            func_call = yield
            # if it's the final call, from __exit__ we stop
            if func_call is None:
                break

            self.trace_ops.append(self.transform_func_call(func_call))

    def raise_error(self, error_str):
        raise RuntimeError(
            f"{error_str}\n"
            f"This will cause a second compile which is currently not allowed.\n"
            f"Ensure that all controls lead to the same graph.\n"
            f"Read https://docs.cerebras.net/en/latest/wsc/tutorials/cstorch-limitations.html#ahead-of-time-aot-compilation"
            f" for more information."
        )

    def get_current_step(self):
        return current_backend_impl().run_context.user_iteration

    def compare(self):
        """After the first iteration, compare all trace ops with the first iteration's trace ops"""
        for first_func, first_args, first_kwargs in self.trace_ops:
            func_call = yield

            # this is the case where the num of ops in 2nd iteration < 1st iteration
            if func_call is None:
                self.raise_error(
                    (
                        f"Traced graph at iteration {self.get_current_step()} has fewer operations "
                        f"than the traced graph at the first iteration."
                    )
                )

            func, args, kwargs = self.transform_func_call(func_call)

            if not (
                first_func == func
                and len(args) == len(first_args)
                and all(
                    arg == first_arg for arg, first_arg in zip(args, first_args)
                )
                and set(kwargs.keys()) == set(first_kwargs.keys())
                and all(
                    value == first_kwargs[key] for key, value in kwargs.items()
                )
            ):
                # figure out which one is different
                str_reason = ""
                if first_func != func:
                    str_reason = "functions"
                elif len(args) != len(first_args) or any(
                    arg != first_arg for arg, first_arg in zip(args, first_args)
                ):
                    str_reason = "args passed in"
                else:
                    str_reason = "kwargs passed in"

                error_str = (
                    f"At this line, we encountered a function call that was different from "
                    f"the first iteration.\n"
                    f"Function call `{func}` on iteration {self.get_current_step()} differs from "
                    f"the first compilation function call `{first_func}` because "
                    f"the {str_reason} are different."
                )

                self.raise_error(error_str)

        next_call = yield

        # this should be the final yield call and lead to None
        if next_call is not None:
            # in this case, this means that the num of ops in the 2nd iteration > 1st iteration
            self.raise_error(
                (
                    f"Traced graph at iteration {self.get_current_step()} has more operations "
                    f"than the traced graph at the first iteration."
                )
            )

    def forward_pre_hook(self, func, types, args, kwargs):
        pass

    def forward_hook(self, func, types, args, kwargs, result):
        # ignore it if it's a get function
        if func.__name__ == '__get__':
            return

        # we require a check to ensure that the output is a lazy tensor
        # so that we only check operations that are actually baked into the graph
        # i.e. operations that have a result that is a lazy tensor
        if not isinstance(result, torch.Tensor) or result.device.type != "lazy":
            return

        self.generator.send((func, args, kwargs))

    def __enter__(self):
        super().__enter__()

        if not self.trace_ops:
            self.generator = self.capture()
        else:
            self.generator = self.compare()

        # must send none as the first
        self.generator.send(None)

    def __exit__(self, *args):
        super().__exit__(*args)
        try:
            self.generator.send(None)
        except StopIteration:
            # it should get to here, because we should have equal amount of ops in first and consecutive calls
            pass
