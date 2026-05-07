# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

try:
    from . import cerebras_pytorch_lib
except ImportError:
    # In case cerebras_pytorch_pylib is not available we mock
    # it with a dummy class which raises an execption if the
    # library was accessed. At the same time we want to keep
    # this library importable.
    class CatchAllMethodCalls(type):
        """Meta class to catch all method calls and raise error."""

        def __getattribute__(cls, attr):
            raise RuntimeError(
                "cerebras_pytorch_lib is not expected to be called on the "
                "server side. Unless explicitly called by user code, this "
                "points to a bug in the framework. Please contact Cerebras "
                "support."
            )

    class cerebras_pytorch_lib(metaclass=CatchAllMethodCalls):
        """Dummy class to catch all method calls and raise error."""
