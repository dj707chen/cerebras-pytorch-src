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
