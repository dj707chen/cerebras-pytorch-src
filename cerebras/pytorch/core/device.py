# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Cerebras device class."""
import inspect
import os
import sys
from contextlib import ExitStack
from typing import Optional
from uuid import uuid4
from warnings import warn
from weakref import finalize

import torch
from torch.nn.modules.module import (
    register_module_buffer_registration_hook,
    register_module_parameter_registration_hook,
)

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.descriptor import Descriptor
from cerebras.pytorch.backend import current_backend_impl, current_torch_device
from cerebras.pytorch.core.function_mode import (
    CerebrasFunctionMode,
    CerebrasFunctionModeContext,
)
from cerebras.pytorch.lib import cerebras_pytorch_lib


class Device:
    """Base Cerebras device class."""

    def __init__(self, device_type: str):
        # for cpu and lazy we only have 1 device, and C++/python have
        # different defaults for index (C++ is -1, python is None)
        # the gpu/cuda implementation uses a different torch.device constructor
        self.torch_device = torch.device(device_type, index=0)

        # Number of contexts that have been entered
        self._stack_size = 0
        self._exit_stack = None

    @property
    def artifact_dir(self):
        return current_backend_impl().artifact_dir

    @property
    def type(self):
        """Returns the type of the torch device."""
        return self.torch_device.type

    def move_to_device(self, struct):
        """Moves all tensors in the provided structure to the torch device."""

        def move(tensor):
            if isinstance(tensor, (torch.nn.Module, torch.Tensor)):
                return tensor.to(self.torch_device)
            return tensor

        with self:
            # pylint: disable=protected-access
            return torch.utils._pytree.tree_map(move, struct)

    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return f"device(type='{str(self)}')"

    def __enter__(self):
        self._stack_size += 1

        # If this is the first time we are entering the context manager,
        # we need to enter the generator context
        if self._stack_size == 1:
            self._exit_stack = ExitStack()
            self._exit_stack.__enter__()
            self._exit_stack.enter_context(CerebrasFunctionMode())
            self._exit_stack.enter_context(GeneratorContext())
            self._exit_stack.enter_context(self.torch_device)

        return self

    def __exit__(self, *args):
        self._stack_size -= 1

        # If this is the last time we are exiting the context manager,
        # we need to exit the generator context
        if self._stack_size == 0:
            self._exit_stack.__exit__(*args)
            self._exit_stack = None


class CPUDevice(Device):
    """Cerebras CPU device subclass."""

    def __init__(self):
        super().__init__("cpu")


class GPUDevice(Device):
    """Cerebras GPU device subclass."""

    def __init__(
        self,
        enable_distributed: bool = False,
        dist_backend: str = "nccl",
        init_method: str = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"GPU was specified as the target device, but "
                f"CUDA is not available. Please make sure you have a "
                f"PyTorch installation with CUDA enabled to run on "
                "GPU"
            )

        if enable_distributed:
            import torch.distributed as dist

            # This environment variable is provided by torchrun
            if "LOCAL_RANK" not in os.environ:
                raise RuntimeError(
                    "Distibuted training was enabled, "
                    "but the script was not run using torchrun. "
                    "Please invoke the training script using torchrun, e.g.\n\n"
                    "\ttorchrun run.py <cli_arguments>"
                )

            dist.init_process_group(
                backend=dist_backend, init_method=init_method
            )

            import logging

            logging.info(
                f"Initialized distributed process group for rank {dist.get_rank()}"
            )

            world_size = dist.get_world_size()
            if world_size == 1:
                warn(
                    "Distributed training was enabled, but only "
                    "1 GPU was detected."
                )
            rank = dist.get_rank()
            self.torch_device = torch.device(rank)
            # call destroy_process_group when the device is destructed
            self._finalizer = finalize(self, dist.destroy_process_group)
        else:
            if torch.cuda.device_count() > 1:
                warn(
                    "Distributed training was not enabled even though "
                    "more than 1 GPU is available."
                )
            self.torch_device = torch.device('cuda')

        self._stack_size = 0


class LazyDevice(Device):
    """Cerebras Lazy device subclass."""

    def __init__(self):
        super().__init__("lazy")

        from cerebras.pytorch.lib import cerebras_pytorch_lib

        self.unique_id = uuid4().hex

        # pylint: disable=c-extension-no-member
        self.config = cerebras_pytorch_lib.appliance_file_config

        # Keep previous state for config.enabled field
        self.config_enabled = None

        # If True, clean up any artifacts on clean exit
        # This includes the device data directory and any appliance data
        # files that were created during the run
        # The only time we would want this to be False is when we are
        # inspecting the initial state after a run in a test.
        self._clean_on_exit = True

        self._functional_tensors = GatherFunctionalTensorContext()

    @property
    def functional_tensors(self):
        return self._functional_tensors.functional_tensors

    def __copy__(self):
        return super().__copy__()

    def __eq__(self, other):
        return self.unique_id == other.unique_id

    def __str__(self) -> str:
        return "CSX"

    def parameter_hook(self, module, name, param):
        """Wraps parameter in a Cerebras parameter.

        Note: Only used if tracing initialization
        """
        if not isinstance(param, cstorch.nn.Parameter):
            assert param.device.type == self.type

            p = cstorch.nn.Parameter(param)
            self._functional_tensors.add(p, name)
            return p

        self._functional_tensors.add(param, name)
        return param

    def buffer_hook(self, module, name, buffer):
        """Wraps buffer in a Cerebras buffer.

        Note: Only used if tracing initialization
        """
        if isinstance(buffer, torch.Tensor) and not isinstance(
            buffer, cstorch.nn.Buffer
        ):
            assert buffer.device.type == self.type

            b = cstorch.nn.Buffer(buffer)
            self._functional_tensors.add(b, name)
            return b

        if buffer is not None:
            self._functional_tensors.add(buffer, name)
        return buffer

    def move_to_device(self, struct):
        if isinstance(struct, torch.nn.Module):
            with self:
                struct.apply(self._move_module_to_device)
            return struct

        return super().move_to_device(struct)

    def _move_module_to_device(self, module: torch.nn.Module):
        # pylint: disable=protected-access
        for iterator, accessor, cstorch_type in [
            (module.named_parameters, module._parameters, cstorch.nn.Parameter),
            (module.named_buffers, module._buffers, cstorch.nn.Buffer),
        ]:
            parameter_names = [name for name, _ in iterator(recurse=False)]
            for param_name in parameter_names:
                param = accessor.pop(param_name)

                if not isinstance(param, cstorch_type):
                    device_param = param.to(self.torch_device)
                    del param  # Remove the in-memory copy
                    param = cstorch_type(device_param)
                else:
                    param = param.to(self.torch_device)

                self._functional_tensors.add(param, param_name)

                accessor[param_name] = param

        def load_state_dict_pre_hook(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if local_metadata.get("assign_to_params_buffers", False):
                raise RuntimeError(
                    f"Cannot call model.load_state_dict(..., assign=True) on a model "
                    f"that has already been compiled. Please call with assign=False."
                )

        module._register_load_state_dict_pre_hook(load_state_dict_pre_hook)

    def __enter__(self) -> "LazyDevice":
        """Enables appliance data.

        `tensor.to("lazy")` will copy the tensor to the Cerebras backend which
        is stored in an ApplianceData struct. This tensor is either backed by
        host memory or file. However, if the config is not enabled and
        `tensor.to("lazy")` is called, the tensor content is immediately dropped
        since we don't expect to use the tensor content.

        For example, as we don't support initialization inside a
        `cstorch.trace`, we don't expect to use the tensor content and
        drop the tensors as we don't wrap that function in a `with device`
        context. But for model parameters or optimizer state, we need to keep
        the tensor content (which contains initialization values) and hence we
        wrap those in a `with device` context.
        """
        self._stack_size += 1

        # If this is the first time we are entering the context manager,
        # we need to enable appliance data and register the module hook.
        if self._stack_size == 1:
            # Enable appliance data
            self.config_enabled = self.config.enabled
            self.config.enabled = not self.config.drop_data

            self._exit_stack = ExitStack()
            self._exit_stack.__enter__()

            self._exit_stack.enter_context(CerebrasFunctionMode())
            self._exit_stack.enter_context(GeneratorContext())

            if self.config.lazy_initialization:
                self._exit_stack.enter_context(self.torch_device)
                self._exit_stack.enter_context(self._functional_tensors)
                self._exit_stack.enter_context(
                    register_module_parameter_registration_hook(
                        self.parameter_hook
                    )
                )
                self._exit_stack.enter_context(
                    register_module_buffer_registration_hook(self.buffer_hook)
                )
            else:
                warn(
                    "Lazy initialization is disabled. "
                    "Weights will be initialized eagerly."
                )

        return self

    def __exit__(self, *args) -> None:
        """Disables appliance data."""
        self._stack_size -= 1

        # If this is the last time we are exiting the context manager,
        # we need to disable appliance data and remove the module hook.
        if self._stack_size == 0:
            # Sync all functional tensors.
            from cerebras.pytorch.lib import cerebras_pytorch_lib

            sorted_tensors = sorted(
                list(self._functional_tensors.functional_tensors.keys()),
                key=lambda t: cerebras_pytorch_lib.get_unique_id(t),
            )
            for tensor in sorted_tensors:
                cerebras_pytorch_lib.sync_functional_tensor(tensor)

            # Restore the previous state
            if self.config_enabled is not None:
                self.config.enabled = self.config_enabled
                self.config_enabled = None

            self._exit_stack.__exit__(*args)
            self._exit_stack = None


class GeneratorContext(CerebrasFunctionModeContext):
    """
    Context manager that assigns a new generator to all functions that
    accept a generator as an argument and don't have a generator argument
    already provided.
    """

    @classmethod
    def is_generator(cls, arg) -> bool:
        """Returns True if the argument is a generator type."""
        if isinstance(arg, torch.Type):
            return arg.kind() == "GeneratorType"
        if arg.type.kind() == "GeneratorType":
            return True
        if arg.type.kind() == "OptionalType":
            return any(
                cls.is_generator(c) for c in arg.real_type.containedTypes()
            )
        return False

    def get_generator_arguments(self, func, args, kwargs):
        """
        Extracts all arguments that are of generator type.

        This is done either via parsing the function schema or by
        inspectng the function signature. This appears to handle
        all tested cases.
        """
        op_name = f"aten::{func.__name__}"
        overloads = torch._C._jit_get_operation(op_name)[1]

        if overloads is not None:
            # Here we try to get the argument information from the
            # function schema.

            # NOTE: "low_generator" is only needed for randint which can have
            #       a variable number of positional arguments.
            for overload in ("", "generator", "low_generator"):
                if overload in overloads:
                    schema = torch._C._get_schema(op_name, overload)

                    if arg_names := [
                        a.name
                        for i, a in enumerate(getattr(schema, "arguments", []))
                        if (
                            self.is_generator(a)
                            and (len(args) <= i or args[i] is None)
                            and kwargs.get(a.name) is None
                        )
                    ]:
                        yield from arg_names
                        return

        try:
            # If we can't get the argument information from the
            # function schema, we try to get it from the function
            # signature.
            for i, (name, p) in enumerate(
                inspect.signature(func).parameters.items()
            ):
                if (
                    (
                        p.annotation is torch.Generator
                        or p.annotation == Optional[torch.Generator]
                    )
                    and (len(args) <= i or args[i] is None)
                    and kwargs.get(name) is None
                ):
                    yield name
        except ValueError:
            # If there is no signature available, then we can't handle
            # this function. Assume no generator args and move on.
            pass

    def construct_generator(self):
        # Use the current torch device unless running in a lazy context.
        # In that case, use the CPU device.
        device = current_torch_device()
        if device.type == "lazy":
            device = torch.device("cpu")

        # Create a new generator for each generator argument
        g = torch.Generator(device=device)
        # We use the default generator to seed the new generator.
        # This ensures that the new generator is seeded deterministically
        # according to the global seed
        g.manual_seed(
            torch.randint(
                sys.maxsize,
                (1,),
                device="cpu",
                generator=torch.default_generator,
            ).item()
        )
        return g

    def forward_pre_hook(self, func, types, args, kwargs):
        # Special implementations for:
        # - torch.rand_like
        # - torch.randint_like
        # - torch.randn
        # - torch.randn_like
        # as they don't have a generator argument in their default schema even
        # though they accept a generator argument.

        if func is torch.rand_like:
            generator = self.construct_generator()

            def func(*args, **kwargs):
                # github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorFactories.cpp#L1090-L1104
                out = torch.empty_like(*args, **kwargs)
                out.uniform_(0.0, 1.0, generator=generator)
                return out

        elif func is torch.randint_like:
            generator = self.construct_generator()

            def func(*args, **kwargs):
                # github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorFactories.cpp#L1090-L1104
                if len(args) == 2:
                    low = 0
                    input, high = args
                elif len(args) == 3:
                    input, low, high = args
                else:
                    # This is an invalid case.
                    # Let torch raise the proper exception
                    return torch.randint_like(*args, **kwargs)

                out = torch.empty_like(input, **kwargs)
                out.random_(low, high, generator=generator)
                return out

        # NOTE: Technically torch.randn does have a generator argument in the
        #       schema, but we don't have proper lowering support for it in
        #       Torch-MLIR.
        # TODO(SW-179348): Remove this special case once we have proper lowering support
        elif func is torch.randn:
            generator = self.construct_generator()

            def func(*args, **kwargs):
                # github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorFactories.cpp#L1250-L1264
                out = torch.empty(*args, **kwargs)
                out.normal_(0.0, 1.0, generator=generator)
                return out

        elif func is torch.randn_like:
            generator = self.construct_generator()

            def func(*args, **kwargs):
                # github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorFactories.cpp#L1306-L1320
                out = torch.empty_like(*args, **kwargs)
                out.normal_(0.0, 1.0, generator=generator)
                return out

        else:
            for arg_name in self.get_generator_arguments(func, args, kwargs):
                # Note, we don't need to check if the argument is already
                # provided as get_generator_arguments only returns arguments
                # that are not already provided.
                kwargs[arg_name] = self.construct_generator()

        return func, types, args, kwargs

    def forward_hook(self, func, types, args, kwargs, result):
        pass


class GatherFunctionalTensorContext(CerebrasFunctionModeContext):
    """
    Context manager that stores all functional tensor results that it encounters.
    """

    def __init__(self):
        self.functional_tensors = torch.utils.weak.WeakTensorKeyDictionary()

    def add(self, tensor, name=None):
        if isinstance(
            tensor, torch.Tensor
        ) and torch._C._functorch.is_functionaltensor(tensor):
            self.functional_tensors[tensor] = name

    def forward_pre_hook(self, func, types, args, kwargs):
        pass

    def forward_hook(self, func, types, args, kwargs, result):
        self.add(result)


def device(backend_type: str, *args, **kwargs):
    from cerebras.pytorch.backend import BackendType

    backend_type = BackendType.from_str(backend_type)

    if backend_type.is_cpu:
        return CPUDevice(*args, **kwargs)
    if backend_type.is_gpu:
        return GPUDevice(*args, **kwargs)
    if backend_type.is_csx:
        return LazyDevice(*args, **kwargs)


class Config(Descriptor):
    """
    Device config descriptor class to allow for various flag to be set.
    """

    def __set_name__(self, owner, name):
        if not hasattr(cerebras_pytorch_lib.appliance_file_config, name):
            raise AttributeError(
                f"Config attribute '{name}' does not exist in "
                f"appliance_file_config"
            )

        super().__set_name__(owner, name)

    def sanitize(self, value):
        setattr(
            cerebras_pytorch_lib.appliance_file_config,
            self._name,
            value,
        )
        return value
