# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Directory containing the implementations of the various API backends."""
import inspect
from enum import Enum, auto
from typing import Optional
from warnings import warn

import torch


class BackendType(Enum):
    """
    The enum class used to distinguish which Cerebras backend to use.
    """

    CPU = auto()
    GPU = auto()
    # synonyms
    CSX = auto()
    WSE = auto()  # deprecated

    @property
    def is_cpu(self):
        """Returns True if the backend is for the CPU."""
        return self == BackendType.CPU

    @property
    def is_gpu(self):
        """Returns True if the backend is for the GPU."""
        return self == BackendType.GPU

    @property
    def is_csx(self):
        """Returns True if the backend is for the Cerebras wafer scaler cluster."""
        return self in (BackendType.CSX, BackendType.WSE)

    @staticmethod
    def from_str(backend_type: str):
        assert isinstance(backend_type, str)
        backend_type = backend_type.upper()
        if backend_type not in BackendType.__members__:
            raise ValueError(
                f"Invalid Cerebras PyTorch backend type specified. "
                f"Expected one of {list(BackendType.__members__)}. "
                f"Got {backend_type}. "
            )
        return BackendType[backend_type]


class BackendMeta(type):
    """
    The metaclass for Backend to ensure only one backend class is ever
    instantiated.
    """

    instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.instance:
            cls.instance[cls] = super(BackendMeta, cls).__call__(
                *args, **kwargs
            )
        else:
            raise RuntimeError(
                f"Cannot instantiate multiple backends. "
                f"A backend with type {cls.instance[cls].backend_type.name} "
                f"has already been instantiated.\n"
                f"Use cstorch.backend() to access the existing backend, "
                f"or use cstorch.backend().torch_device to access the "
                f"current backend's torch device."
            )
        return cls.instance[cls]


class Backend(metaclass=BackendMeta):
    """Externally facing Cerebras backend class."""

    # Only if True, initialize the backend implementation
    _init_impl: bool = True

    def __init__(self, backend_type: BackendType, *args, **kwargs):
        assert isinstance(backend_type, BackendType)
        self.backend_type = backend_type

        if not self._init_impl:
            return

        if self.backend_type == BackendType.CSX:
            from .ltc_backend import PyTorchLtcBackendImpl

            self._impl = PyTorchLtcBackendImpl(
                self.backend_type, *args, **kwargs
            )

        elif self.backend_type == BackendType.CPU:
            from .cpu_backend import CpuBackendImpl

            self._impl = CpuBackendImpl(self.backend_type, *args, **kwargs)

        elif self.backend_type == BackendType.GPU:
            from .gpu_backend import GpuBackendImpl

            self._impl = GpuBackendImpl(self.backend_type, *args, **kwargs)

        else:
            raise ValueError(
                f"{self.backend_type.name} backend not yet supported. "
                f"Supported backends include: CSX, CPU, GPU"
            )

    @property
    def artifact_dir(self):
        """Returns the artifact directory being used by the backend."""
        return self._impl.artifact_dir

    @artifact_dir.setter
    def artifact_dir(self, value):
        """Sets the artifact directory for the backend."""
        self._impl.artifact_dir = value

    @property
    def device(self):
        """Returns the Cerebras device being used by the backend."""
        return self._impl.device

    @property
    def torch_device(self):
        """Returns the underlying PyTorch device being used by the backend."""
        return self._impl.device.torch_device

    @property
    def is_tracing(self):
        """Returns True if the backend is currently tracing a model."""
        return self._impl.is_tracing

    @property
    def is_e2e_execution(self):
        """Returns True if the backend is currently tracing a model"""
        return self._impl.is_e2e_execution

    @property
    def cluster_config(self):
        """Returns the cluster config if the backend is a CSX backend, otherwise None."""
        return getattr(self._impl, "cluster_config", None)

    # alias properties from backend type
    is_cpu = property(lambda self: self.backend_type.is_cpu)
    is_gpu = property(lambda self: self.backend_type.is_gpu)
    is_csx = property(lambda self: self.backend_type.is_csx)


def get_backend_args(backend_type: str):
    """
    Get the arguments for the backend class with the given backend type.

    Args:
        backend_type: The type of backend to get the arguments for.
            Must be one of "CSX", "CPU", "GPU"
    """
    if isinstance(backend_type, str):
        backend_type = BackendType.from_str(backend_type)

    if backend_type == BackendType.CSX:
        from .ltc_backend import PyTorchLtcBackendImpl as BackendImpl
    elif backend_type == BackendType.CPU:
        from .cpu_backend import CpuBackendImpl as BackendImpl
    elif backend_type == BackendType.GPU:
        from .gpu_backend import GpuBackendImpl as BackendImpl
    else:
        raise ValueError(
            f"{backend_type.name} backend not yet supported. "
            f"Supported backends include: CSX, CPU, GPU"
        )

    return inspect.signature(BackendImpl.__init__).parameters


# backend() queries the current backend, while backend(str, ...) creates a new one
def backend(backend_type: Optional[str] = None, *args, **kwargs):
    """
    Instantiates a backend with the given type.

    Args:
        backend_type: The type of backend to instantiate. One of "CSX", "CPU", "GPU"
            If no backend_type is provided, returns the current backend if it exists.

        args: Positional arguments to pass to the backend implementation
        kwargs: Keyword arguments to pass to the backend implementation
    """
    if backend_type is None:
        # if other args are given return a type error
        if len(args) != 0 or len(kwargs) != 0:
            raise TypeError(
                "Expected backend_type when constructing a backend. "
                "Either provide one of \"CSX\", \"CPU\", \"GPU\" to create a new backend, "
                "or call with no arguments to get the instance of the current backend, "
                "if it exists."
            )

        return current_backend(raise_warning=False)

    if isinstance(backend_type, str):
        backend_type = BackendType.from_str(backend_type)
    elif not isinstance(backend_type, BackendType):
        raise TypeError(
            f"Expected backend_type to be of type BackendType, "
            "or a string representing the backend type. "
            f"Got: {type(backend_type)}"
        )

    return Backend(backend_type, *args, **kwargs)


def current_backend(raise_exception: bool = True, raise_warning: bool = True):
    """DEPRECATED: Use cstorch.backend() instead.
    Gets instance of the current backend.

    Args:
        raise_exception: If True, raise an exception if no backend has been
            instantiated. Otherwise return None
    """
    if raise_warning:
        warn(
            "cstorch.current_backend() is deprecated and will be removed in a future release. "
            "Use cstorch.backend() instead to access the current backend.",
            DeprecationWarning,
        )

    if Backend not in BackendMeta.instance:
        if raise_exception:
            raise RuntimeError(
                "No active Cerebras backend found. Please make sure that "
                "your model has been prepared for compilation.\n"
                "You can do this using a call to:\n\n"
                "\tcompiled_model = cstorch.compile(model, backend=...)\n\n"
                "Or by explicitly instantiating a backend, e.g.\n\n"
                "\tbackend = cstorch.backend(...)"
            )
        return None
    return BackendMeta.instance[Backend]


def current_torch_device():
    """
    Gets the torch device of the current backend.

    Returns torch.device('cpu') if no backend has been initialized yet
    """
    _backend = current_backend(raise_exception=False, raise_warning=False)

    if _backend is None:
        return torch.device("cpu")

    # pylint: disable=protected-access
    return _backend._impl.torch_device


def current_backend_impl(raise_exception: bool = True):
    """Returns the implementation of the current backend class.

    Args:
        raise_exception: If True, raise an exception if no backend has been
            instantiated.
    Returns:
        The backend implementation if one exists, otherwise None.
    Raises:
        RuntimeError: If no backend has been instantiated and `raise_exception`
            is True.
    """

    _backend = current_backend(
        raise_exception=raise_exception, raise_warning=False
    )

    if _backend is None:
        return None
    # pylint: disable=protected-access
    return _backend._impl


def use_cs():
    """Returns True if the active device is a CSX device."""

    _backend = current_backend(raise_exception=False, raise_warning=False)
    return _backend is not None and _backend.is_csx
