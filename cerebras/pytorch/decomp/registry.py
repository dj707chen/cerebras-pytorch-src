# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Union

import torch
from torch._decomp import _add_op_to_registry, _convert_out_params

from cerebras.pytorch.core.compile import register_trace_fn_pre_hook
from cerebras.pytorch.utils.call_once import call_once

aten = torch._ops.ops.aten


def register_decomposition(
    op: Union[torch._ops.OpOverloadPacket, torch._ops.OpOverload],
    device: Literal["CSX", "CPU"] = "CSX",
) -> Callable:
    """
    A decorator registers a decomposition function. Once registered, decomposition will be
    used in all CSX/CPU runs.

    Args:
        op: PyTorch representation of an op.

    Usage examples:

        1. Custom decompositions. Add `register_decomposition` decorator to
        a decomposition function to make it available for CSX/CPU flow.

            @register_decomposition(torch._ops.ops.aten.clamp_min)
            def clamp_min(x):
                return torch.clamp(self, min=min)

        2. Register existing Pytorch decomposition by calling `register_decomposition` function.

            from torch._decomp.decompositions import _weight_norm_interface
            register_decomposition(torch._ops.ops.aten._weight_norm_interface)(_weight_norm_interface)

    In both cases make sure that decomposition is registered before backend instantiation.

    Note: for internal development please use `_decompositions.py` to add new custom decompositions,
    and register existing Pytorch decompositions.
    """
    return _REGISTRY.register_decomposition(op, device)


@dataclass(unsafe_hash=True)
class _DecompKey:
    """
    Key to uniquely identify a decomposition.

    Args:
        op: PyTorch object for an op's specific overload.
        device: Device on which decomposition is registered.
    """

    op: torch._ops.OpOverload
    device: str


@dataclass
class _DecompRegistrationDetails:
    """
    Stores important details of a decomposition to be registered.

    Args:
        impl: Implementation of the decomposition.
        traces: The stack traces from the location decomp decorator was called.
    """

    impl: Callable
    traces: List[str] = field(
        default_factory=lambda: traceback.format_stack()[:-1]
    )

    @property
    def stacktrace(self):
        """Return full stack trace as a string."""
        return "".join(self.traces)


class Registry:
    """
    Gobal registry to handle all decompositions infrastructure.

    Args:
        decompositions: Mapping between an op to their decomposition details.
        applied_decompositions: Flag to check if decompositions have been applied.
        device_dispatch: Translation between cstorch device to PyTorch device.
        deny_list: Blacklist of ops for custom decompositions.
    """

    device_dispatch = {"CSX": "AutogradLazy", "CPU": "cpu"}
    deny_list = {
        aten.detach,
        aten.lift,
        aten.lift_fresh,
    }

    def __init__(self):
        self.decompositions = dict()
        self.applied_decompositions = False

    def register_decomposition(
        self,
        op: Union[torch._ops.OpOverloadPacket, torch._ops.OpOverload],
        device: Literal["CSX", "CPU"] = "CSX",
    ) -> Callable:
        """
        Register a decomposition for a given op and device.

        Args:
            op: PyTorch op object or it's specific overloa.
            device: Device on which decomposition is registered.
        """
        if self.applied_decompositions:
            raise RuntimeError(
                "register_decomposition can only be called before the first trace."
            )

        device = device.upper()
        if device not in self.device_dispatch:
            raise ValueError(
                f"Registering decompositions for device {device} is not supported. "
                f"Supported devices are: {self.device_dispatch.keys()}"
            )

        def decomp_decorator(fn: Callable) -> Callable:
            filtered_overloads = dict()
            # this call will elegantly handle both OpOverload and OpOverloadPacket
            # and populate our dict with all "valid" overloads
            _add_op_to_registry(filtered_overloads, op, lambda: None)

            # _convert_out_params changes "out" parameter to appropriate name passed in out_wrapper
            decomp = _DecompRegistrationDetails(_convert_out_params(fn))
            for overload in filtered_overloads:
                if overload._overloadpacket in self.deny_list:
                    raise RuntimeError(
                        f"Decomposition for aten.{overload._opname} is not supported."
                    )
                elif overload in self.deny_list:
                    raise RuntimeError(
                        f"Decomposition for aten.{overload._opname}.{overload._overloadname} is not supported."
                    )

                decomp_key = _DecompKey(overload, device)
                if decomp_key in self.decompositions:
                    raise RuntimeError(
                        f"A decomposition for aten.{overload._opname}.{overload._overloadname} "
                        f"has already been registered on {device}, "
                        f"here is the location of the first registration:\n"
                        f"{self.decompositions[decomp_key].stacktrace}"
                    )

                self.decompositions[decomp_key] = decomp

            return fn

        return decomp_decorator

    def remove_decomposition(
        self,
        op: Union[torch._ops.OpOverloadPacket, torch._ops.OpOverload],
        device: Literal["CSX", "CPU"] = "CSX",
    ):
        """
        Removes a decomposition for a given op and device.

        Args:
            op: PyTorch op object or it's specific overloa.
            device: Device on which decomposition is registered.
        """
        if self.applied_decompositions:
            raise RuntimeError(
                "remove_decomposition() can only be called before the first trace."
            )

        device = device.upper()
        if device not in self.device_dispatch:
            raise ValueError(
                f"Removing decompositions for device {device} is not supported. "
                f"Supported devices are: {self.device_dispatch.keys()}"
            )

        filtered_overloads = dict()
        _add_op_to_registry(filtered_overloads, op, lambda: None)

        for overload in filtered_overloads:
            decomp_key = _DecompKey(overload, device)
            if decomp_key not in self.decompositions:
                raise RuntimeError(
                    f"Decomposition for aten.{overload._opname}.{overload._overloadname} "
                    f"on {device} has not been registered."
                )

            del self.decompositions[decomp_key]

    @call_once()
    def _apply_all_registered_decompositions(self):
        """
        Apply all registered PyTorch and custom decompositions.
        Should only be called once before the first trace.
        """
        for key, decomp in self.decompositions.items():
            torch.library.register_kernel(
                key.op, self.device_dispatch[key.device], decomp.impl
            )

        self.applied_decompositions = True


_REGISTRY = Registry()

decompositions_hook = register_trace_fn_pre_hook(
    _REGISTRY._apply_all_registered_decompositions
)


# This will register all internal decompositions
import cerebras.pytorch.decomp._decompositions  # noqa
