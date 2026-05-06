# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Provides utilities around operation/module name scopes.
"""
import contextlib
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ScopeName:
    scope_name: Optional[str] = None
    scope_type: Optional[str] = None
    scope_func: Optional[str] = None

    def __repr__(self):
        scope_name_args = []
        if self.scope_name:
            scope_name_args.append(self.scope_name)
        if self.scope_type:
            scope_name_args.append(self.scope_type)
        if self.scope_func:
            scope_name_args.append(self.scope_func)

        return ".".join(scope_name_args)


@contextlib.contextmanager
def name_scope(name: str, raw: bool = False):
    """
    A context manager that names operations in PyTorch

    Example usage:
    ```
    class Model(nn.Module):

        def __init__(self):
            self.fc1 = nn.Linear(10, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 2)

        def forward(self, x):
            x = self.fc1(x)
            with cstorch.name_scope("my_scope") as scope:
                x = self.fc2(self.relu(x))
            return F.log_softmax(x, dim=1)
    ```
    """
    from cerebras.pytorch.backend import current_backend_impl

    backend_impl = current_backend_impl()
    with backend_impl.name_scope(name):
        yield


def set_debug_scope(name: str):
    """
    Set global state such that all traced operation will get a unique debug
    name starting with the given scope name, even those created during
    autograd.

    Note that the debug scope must be cleared before mark_step to avoid error.

    Args:
        name: The name of the debug scope to use, or None to clear scope.
    """
    from cerebras.pytorch.lib.cerebras_pytorch_lib import set_scope_name

    if not name:
        name = ""
    set_scope_name(name)


def add_debug_name(module: torch.nn.Module, root_name: Optional[str] = None):
    """
    Adds an invasive _debug_name string attribute to the module and all its
    children. This name will contain the full "." seperated module hierarchy
    name starting from the given module as the root.
    """

    def add_name(module, name):
        module._debug_name = name  # pylint: disable=protected-access
        for cname, child in module.named_children():
            add_name(child, ".".join(filter(len, (name, cname))))

    add_name(module, name=root_name or "")


class ModuleNamesGenerator:
    """
    A context manager that enables modules names generation
    based on the module class name and class instance id.
    This is primarily used to name unnamed modules during the
    step function tracing process, so each module gets a unique name.
    The names generated within this context are not cached and
    will be regenerated on each entry to the context to guaranty
    uniqueness and correct naming aligned with the tracing order.
    """

    entered_context = False

    @dataclass
    class InstanceInfo:
        num_instances: int = 0

    registry = []

    @classmethod
    def get_or_create_instances_info(cls, module_cls):
        """
        Attaches and returns the InsanceInfo object for the given class
        that counts the number of class instances created within the context.
        The attached info object lifetime is limited to the context.
        """
        if not cls.entered_context:
            raise RuntimeError(
                "Attempting to access module instance info "
                f"outside of \"{cls.__name__}\" context"
            )

        if hasattr(module_cls, "_instances_info"):
            return module_cls._instances_info

        item = cls.InstanceInfo()
        module_cls._instances_info = item
        return item

    @classmethod
    def get_name(cls, module):
        """
        Generates or returns a generated module name. The name format is:
        "<module_class_name>_<class_instance_id>". The name is cached
        inside the module for the lifetime of the context.
        """
        if not cls.entered_context:
            raise RuntimeError(
                "Attempting to access generated module name "
                f"outside of \"{cls.__name__}\" context"
            )

        if hasattr(module, "_generated_debug_name"):
            return module._generated_debug_name

        # Get/create the class instance info for the module class to count
        # the number of class instances.
        module_cls = module.__class__
        instances_info = cls.get_or_create_instances_info(module_cls)
        instances_info.num_instances += 1

        # Generate the module name and cache it inside the module.
        name = f"{module_cls.__name__}_{instances_info.num_instances}"
        module._generated_debug_name = name
        cls.registry.append(module)
        return name

    def __enter__(self):
        if self.__class__.entered_context:
            raise RuntimeError(
                f"Attempting to enter \"{self.__class__.__name__}\" "
                "context while already in it"
            )

        self.__class__.entered_context = True
        return self

    def __exit__(self, *args, **kwargs):
        for module in self.__class__.registry:
            cls = module.__class__
            if hasattr(cls, "_instances_info"):
                del cls._instances_info
            del module._generated_debug_name

        self.__class__.registry = []
        self.__class__.entered_context = False


def get_debug_name(module: torch.nn.Module) -> str:
    """
    Returns either the _debug_name string attribute of the module, or
    generates a new model name based on the class name and instance id
    for un-named modules.
    """
    if hasattr(module, "_debug_name") and module._debug_name:
        return module._debug_name

    if ModuleNamesGenerator.entered_context:
        return ModuleNamesGenerator.get_name(module)

    # Since we set a global fwd module hook that triggers `get_debug_name`
    # call, we plumb the debug name with the module class name.
    return module.__class__.__name__
