# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

try:
    from . import cerebras_pytorch_lib
except ImportError:
    # Native library not available (e.g. macOS/ARM64).
    # Provide a recursive no-op stub so that module-level class
    # definitions (descriptors, flags) that reference the lib can
    # complete without error.  Actual CSX operations will fail at
    # runtime with clear messages from higher-level code.

    _LIB_MISSING_MSG = (
        "cerebras_pytorch_lib native extension is not available. "
        "CSX operations require the native library."
    )

    class _Stub:
        """Recursive no-op stub for cerebras_pytorch_lib."""

        def __getattr__(self, name):
            return _Stub()

        def __setattr__(self, name, value):
            pass

        def __bool__(self):
            return False

        def __call__(self, *args, **kwargs):
            return _Stub()

        def __repr__(self):
            return "<cerebras_pytorch_lib stub>"

    class _StubModule:
        """Module-level stub that returns _Stub for any attribute."""

        def __getattr__(self, name):
            return _Stub()

        def __setattr__(self, name, value):
            pass

    cerebras_pytorch_lib = _StubModule()
