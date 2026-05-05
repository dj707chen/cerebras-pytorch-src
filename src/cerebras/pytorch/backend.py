"""Backend and execution helpers for a CPU/GPU-friendly cstorch facade."""

from __future__ import annotations

import functools
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ._compat import optional_torch, require_torch


@dataclass
class Backend:
    """Minimal backend object compatible with the documented public API."""

    backend_type: str
    artifact_dir: str = "cerebras_logs"
    device: Any = "cpu"
    compile_only: bool = False
    validate_only: bool = False

    @property
    def is_csx(self) -> bool:
        return self.backend_type.upper() == "CSX"

    def __enter__(self) -> "Backend":
        set_backend(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


_CURRENT_BACKEND: Optional[Backend] = None


def _torch_device(name: str):
    torch = optional_torch()
    if torch is None:
        return name
    return torch.device(name)


def _make_backend(backend_type: str, *args, **kwargs) -> Backend:
    if args:
        raise TypeError("backend accepts keyword configuration in this implementation")

    kind = backend_type.upper()
    artifact_dir = kwargs.pop("artifact_dir", None) or str(Path.cwd() / "cerebras_logs")
    compile_only = bool(kwargs.pop("compile_only", False))
    validate_only = bool(kwargs.pop("validate_only", False))

    if kind == "CPU":
        device = _torch_device("cpu")
    elif kind == "GPU":
        torch = optional_torch()
        device = torch.device("cuda") if torch is not None and torch.cuda.is_available() else _torch_device("cpu")
    elif kind == "CSX":
        device = _torch_device("cpu")
    else:
        raise ValueError("backend_type must be one of 'CSX', 'CPU', or 'GPU'")

    backend = Backend(
        backend_type=kind,
        artifact_dir=artifact_dir,
        device=device,
        compile_only=compile_only,
        validate_only=validate_only,
    )
    for key, value in kwargs.items():
        setattr(backend, key, value)
    return backend


def set_backend(value: Backend) -> Backend:
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = value
    return value


def backend(backend_type: Optional[Union[str, Backend]] = None, *args, **kwargs) -> Optional[Backend]:
    """Instantiate or retrieve the active backend."""

    if backend_type is None:
        return _CURRENT_BACKEND
    if isinstance(backend_type, Backend):
        return set_backend(backend_type)
    return set_backend(_make_backend(backend_type, *args, **kwargs))


def current_backend(raise_exception: bool = True, raise_warning: bool = True) -> Optional[Backend]:
    if raise_warning:
        warnings.warn("current_backend is deprecated; use backend() instead", DeprecationWarning, stacklevel=2)
    if _CURRENT_BACKEND is None and raise_exception:
        raise RuntimeError("No Cerebras PyTorch backend has been initialized")
    return _CURRENT_BACKEND


def current_torch_device():
    active = _CURRENT_BACKEND
    if active is None:
        return _torch_device("cpu")
    return active.device


def use_cs() -> bool:
    active = _CURRENT_BACKEND
    return bool(active and active.is_csx)


def compile(model, backend: Optional[Union[str, Backend]] = None):
    """Prepare a module for execution on the selected backend.

    CPU/GPU compatibility mode simply moves modules with a ``to`` method to the
    selected torch device and returns the original object.
    """

    active = globals()["backend"](backend) if backend is not None else _CURRENT_BACKEND
    if active is None:
        active = globals()["backend"]("CPU")
    if hasattr(model, "to"):
        return model.to(active.device)
    return model


def trace(step_fn: Callable) -> Callable:
    @functools.wraps(step_fn)
    def wrapped(*args, **kwargs):
        return step_fn(*args, **kwargs)

    wrapped.__cerebras_traced__ = True
    return wrapped


def _creation(fn_name: str, *args, **kwargs):
    torch = require_torch()
    return getattr(torch, fn_name)(*args, **kwargs)


def full(shape, value, dtype=None):
    return _creation("full", shape, value, dtype=dtype)


def full_like(other, value, dtype=None):
    return _creation("full_like", other, value, dtype=dtype or getattr(other, "dtype", None))


def ones(shape, dtype=None):
    return _creation("ones", shape, dtype=dtype)


def ones_like(other, dtype=None):
    return _creation("ones_like", other, dtype=dtype or getattr(other, "dtype", None))


def zeros(shape, dtype=None):
    return _creation("zeros", shape, dtype=dtype)


def zeros_like(other, dtype=None):
    return _creation("zeros_like", other, dtype=dtype or getattr(other, "dtype", None))


def save(obj: dict, checkpoint_file: str) -> None:
    Path(os.fspath(checkpoint_file)).parent.mkdir(parents=True, exist_ok=True)
    torch = optional_torch()
    if torch is not None:
        torch.save(obj, checkpoint_file)
        return
    with open(checkpoint_file, "wb") as fh:
        pickle.dump(obj, fh)


def load(checkpoint_file, map_location=None, **kwargs):
    torch = optional_torch()
    if torch is not None:
        return torch.load(checkpoint_file, map_location=map_location, **kwargs)
    with open(checkpoint_file, "rb") as fh:
        return pickle.load(fh)


def step_closure(fn: Optional[Callable] = None):
    if fn is None:
        return lambda inner: inner
    return fn


def checkpoint_closure(fn: Optional[Callable] = None):
    if fn is None:
        return lambda inner: inner
    return fn
