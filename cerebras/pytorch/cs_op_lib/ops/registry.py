# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from ..errors import OpNotFoundError


class CsOpRegistry:
    """
    Registry to store op_name -> CsOp subclass.
    """

    _registry = {}

    @classmethod
    def register(cls, op_cls):
        from .base import CsOp

        if not issubclass(op_cls, CsOp):
            raise TypeError(f"Expected a subclass of CsOp, got {op_cls}")
        cls._registry[op_cls.op_name()] = op_cls

    @classmethod
    def get_op_class(cls, op_name: str):
        op_cls = cls._registry.get(op_name)
        if op_cls is None:
            raise OpNotFoundError(op_name)
        return op_cls

    @classmethod
    def get_all_ops(cls):
        return cls._registry.values()


def register_all_ops(definitions_module=None):
    """
    Register all ops in the registry.
    If definitions_module is None, it will register ops from the default definitions module.
    `definitions_module` can be a module object or a string with the module name.
    """
    import importlib
    import inspect

    from .base import CsOp

    if isinstance(definitions_module, str):
        definitions_module = importlib.import_module(definitions_module)
    elif definitions_module is None:
        from . import definitions as definitions_module

    for _, op_cls in inspect.getmembers(definitions_module, inspect.isclass):
        # Check if the class is a subclass of CsOp
        if op_cls is CsOp or not issubclass(op_cls, CsOp):
            continue
        CsOpRegistry.register(op_cls)
