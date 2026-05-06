# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Contains pre hooks that checks tensor function calls"""


import torch

from cerebras.pytorch.core.function_mode import (
    register_function_mode_forward_error_hook,
)


def add_checks(device_type):
    # helper function that prints out the type for an argument, and more for a tensor
    def get_arg_str(arg):
        tensor_str = ""
        if isinstance(arg, torch.Tensor):
            tensor_str = f"(shape={arg.shape}, dtype={arg.dtype}, device={arg.device.type})"
        return f"\t - {type(arg).__name__}{tensor_str}"

    # helper function that checks whether or not the there's a mix of tensors on different devices being used
    # returns false if there are multiple tensors not on the same device
    def are_tensors_on_same_device(all_args, is_tensor):
        # Requirements to raise a warning:
        # 1. Has to be an operation on >= 1 tensor, where a tensor is of type torch.Tensor and is not a scalar (not of shape ())
        # 2. At least one of the tensors are of type self.device.type and at least one type that is not
        all_tensors = list(filter(is_tensor, all_args))

        return all(
            tensor.device.type != device_type for tensor in all_tensors
        ) or all(tensor.device.type == device_type for tensor in all_tensors)

    # helper function that gets the string for all the arguments' types
    def get_all_arg_str(all_args, get_arg_str):
        return "\n".join(map(get_arg_str, all_args))

    # helper function that gets the string for all the arguments'
    def get_all_arg_types_str(all_args):
        return ", ".join(type(arg).__name__ for arg in all_args)

    # function that returns the squiggles string which marks the incorrect arguments
    def get_squiggles_str(all_args, is_improper_arg):
        return "  ".join(
            ("~" if is_improper_arg(arg) else " ") * len(type(arg).__name__)
            for arg in all_args
        )

    # helper function that gets all the wrong tensors' indices
    def get_wrong_tensor_idxs_str(all_args, is_improper_tensor):
        return ", ".join(
            str(idx)
            for idx, arg in enumerate(all_args)
            if is_improper_tensor(arg)
        )

    def get_all_args(args, kwargs):
        all_args, _ = torch.utils._pytree.tree_flatten((args, kwargs))
        return all_args

    # add in a hook that throws an error if the tensor types don't match the host device
    def check_tensor_types_hook(func, types, args, kwargs, error):
        # add a check to ensure that the error we get is in fact the multiple tensor types error
        if not isinstance(
            error, RuntimeError
        ) or "Input tensor is not a lazy tensor" not in str(error):
            return

        def is_tensor(arg):
            return isinstance(arg, torch.Tensor) and arg.shape != ()

        all_args = get_all_args(args, kwargs)

        if not are_tensors_on_same_device(all_args, is_tensor):

            def is_improper_tensor(arg):
                # if the tensor is a scalar, it doesn't lead to an abort
                return is_tensor(arg) and arg.device.type != device_type

            # re-raise the runtime error
            raise RuntimeError(
                f"Detected tensor arguments with incompatible devices in the following function call.\n"
                f"Please look at the stack trace for the originating function call.\n"
                f"\t{func.__name__}({get_all_arg_types_str(all_args)})\n"
                f"\t{' ' * len(func.__name__)} {get_squiggles_str(all_args, is_improper_tensor)}\n"
                f"Arguments {get_wrong_tensor_idxs_str(all_args, is_improper_tensor)} "
                f"of this function are tensors with incompatible devices. "
                f"The arguments to `{func.__name__}` are shown below:\n"
                f"{get_all_arg_str(all_args, get_arg_str)}\n"
                f"Please ensure that all your tensor arguments are on the same torch device. "
                f"To get the torch device for the current backend, use `cstorch.current_torch_device()`.",
            ) from error

    def check_tensor_functionality_types_hook(func, types, args, kwargs, error):
        # if it's not a runtime error, then we know it's not because
        # the tensor functionality types are different, so we stop
        if not isinstance(
            error, RuntimeError
        ) or "mutating a non-functional tensor with a functional tensor is not allowed" not in str(
            error
        ):
            return

        # require that the function be an in-place operation
        if not (
            (hasattr(func, "_schema") and func._schema.name[-1] == "_")
            or (hasattr(func, "__name__") and func.__name__[-1] == "_")
            or ("out" in kwargs)
        ):
            return

        all_args = get_all_args(args, kwargs)

        if not are_tensors_on_same_device(
            all_args, lambda arg: isinstance(arg, torch.Tensor)
        ):

            def is_improper_tensor(arg):
                # if the tensor is a scalar, it doesn't lead to an abort
                return (
                    isinstance(arg, torch.Tensor)
                    and arg.device.type != device_type
                )

            raise RuntimeError(
                f"Detected tensor arguments with incompatible devices in the following in-place function call.\n"
                f"See below for the PyTorch specific operation. Please look at the stack trace for the originating function call.\n"
                f"\t{func.__name__}({get_all_arg_types_str(all_args)})\n"
                f"\t{' ' * len(func.__name__)} {get_squiggles_str(all_args, is_improper_tensor)}\n"
                f"Arguments {get_wrong_tensor_idxs_str(all_args, is_improper_tensor)} "
                f"of this function are tensors with incompatible devices. "
                f"The arguments to `{func.__name__}` are shown below:\n"
                f"{get_all_arg_str(all_args, get_arg_str)}\n"
                f"Please ensure that all your tensor arguments are on the same device for an in-place function call.\n"
            )

    register_function_mode_forward_error_hook(check_tensor_types_hook)
    register_function_mode_forward_error_hook(
        check_tensor_functionality_types_hook
    )
