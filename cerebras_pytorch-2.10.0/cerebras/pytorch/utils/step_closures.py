# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers and decorators for using step closures"""
from functools import wraps
from typing import Callable, List

from cerebras.appliance import logger
from cerebras.pytorch.backend import current_backend_impl


class RepeatStepClosure:
    """Contols whether or not to repeat the step closure by default"""

    default: bool = False

    def __enter__(self):
        # Any step closures added within this context will be repeatedly
        # added back to the queue and run every iteration
        RepeatStepClosure.default = True

    def __exit__(self, *args):
        RepeatStepClosure.default = False


class StepClosureContext:
    """Keeps track of whether or not we're inside a step closure"""

    step_closure_stack: List[str] = []

    @classmethod
    def wrap(cls, closure):
        @wraps(closure)
        def wrapped_closure(*args, **kwargs):
            try:
                cls.step_closure_stack.append(closure.__name__)
                return closure(*args, **kwargs)
            finally:
                cls.step_closure_stack.pop()

        return wrapped_closure


def step_closure(closure: Callable) -> Callable:
    """Decorator to automatically wrap a function call in a step closure.

    Step closures are queued and all run at the end of each step. This is to
    ensure that the tensor arguments to the closures are computed and are
    retrieved before they are used.

    Usage:

    ::

        @step_closure
        def closure(...):
            ...
        ...
        closure(...)

    Args:
        closure: The function to wrap in a step closure.
    Returns:
        The wrapped function.
    """

    @wraps(closure)
    def inner(*args, **kwargs):
        backend = current_backend_impl(raise_exception=False)
        if backend:
            backend.add_step_closure(
                StepClosureContext.wrap(closure),
                args,
                kwargs,
                run_async=False,
                repeat=RepeatStepClosure.default,
            )
        else:
            closure(*args, **kwargs)

    inner.is_step_closure = True

    return inner


def checkpoint_closure(closure: Callable) -> Callable:
    """Decorator to wrap function so it is only ever called on checkpoint steps.

    With this decorator, the closure may be called at any time. But it will only
    ever run if on a checkpoint step, as configured by setting
    `checkpoint_steps` when creating a `DataExecutor`.

    An example of a function that would benefit from using this decorator
    is a function that saves the checkpoint. It ensures that checkpoints are
    only saved on steps on which the checkpoint is available to be retrieved
    from the Cerebras wafer-scale cluster.

    Example Usage:

    ::

        @checkpoint_closure
        def save_checkpoint(...):
            ...
            cstorch.save(...)
        ...
        executor = cstorch.utils.data.DataExecutor(..., checkpoint_steps=100)
        for batch in executor:
            ...
            # Always call save checkpoint
            # But save checkpoint only actually runs on checkpoint steps
            save_checkpoint(...)

    Args:
        closure: The function to wrap in a step closure that only runs on
            checkpoint steps.
    Returns:
        The wrapped function.
    """

    @wraps(closure)
    def checkpoint_step_closure(*args, **kwargs):
        backend = current_backend_impl()

        def closure_wrapper(*args, **kwargs):
            if len(backend.data_executor_stack) == 0:
                raise RuntimeError(
                    "Cannot fetch a checkpoint outside of an execution context. "
                    "Please make all calls to any checkpoint closures inside "
                    "the training loop."
                )

            # Only call the function if is an initial checkpoint or on a checkpoint step
            if (
                backend.run_context.is_pre_initial_step
                or backend.run_context.is_checkpoint_step
            ):
                closure(*args, **kwargs)
            else:
                logger.debug(
                    f"Skipping calling checkpoint closure `{closure.__name__}` "
                    f"on non-checkpoint step {backend.run_context.user_iteration}."
                )

        backend.add_step_closure(
            StepClosureContext.wrap(closure_wrapper),
            args,
            kwargs,
            run_async=False,
            repeat=RepeatStepClosure.default,
        )

    return checkpoint_step_closure
