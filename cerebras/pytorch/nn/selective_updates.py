# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from functools import partial
from typing import Callable, List, Union
from weakref import ref

import torch
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.typing import signature_matches_type_hint
from cerebras.pytorch.utils.utils import FilterCallable, make_param_filter

InitMethodCallable = Callable[
    [torch.nn.Parameter],
    torch.BoolTensor,
]
InitMethodType = Union[str, InitMethodCallable]


def make_outlier(lowerThr: float = 0.75, upThr: float = 2.25):
    def outlier(param: torch.nn.Parameter) -> torch.BoolTensor:
        """
        A default mask that only applies gradients to weights that are `upThr` standard deviations away
        from the mean on a per-channel basis (along the output dimension).

        Args:
            param: The parameter to create the mask from and will eventually apply that mask to.
            lowerThr: Lower threshold as a float
            upThr: Upper threshold that determines the range of which to apply gradients to weights.

        Returns:
            A boolean tensor which represents the mask to apply.
        """
        vrstd = torch.std(param.abs(), dim=0, keepdim=True)
        mask = param.abs() >= upThr * vrstd
        return mask

    return outlier


def make_init_method(init_method: InitMethodType) -> InitMethodCallable:
    """
    Returns the corresponding init method callable for the given `init_method`.

    Args:
        init_method: Either a string dictating a default initialization method or a callable
            which will be a mask function to indicate which gradients should remain active even
            during a selective gradient run.

    Returns:
        A callable method that when given a parameter will return a mask as a Boolean tensor.
    """

    init_methods = {"outlier": make_outlier()}

    if isinstance(init_method, str):
        if init_method not in init_methods:
            raise ValueError(
                f'Unknown `init_method`: "{init_method}". Valid options are one '
                f'of the built-in {list(init_methods.keys())} or a function with '
                f'signature {InitMethodCallable}.'
            )
        return init_methods[init_method]

    elif callable(init_method):
        signature = inspect.signature(init_method)
        if signature_matches_type_hint(signature, InitMethodCallable):
            return init_method

    raise ValueError(
        f'Passed in argument `init_method` is not a valid option. Valid options are one '
        f'of the built-in {list(init_methods.keys())} or a function with '
        f'signature {InitMethodCallable}.'
    )


class SelectiveGrad:
    """
    SelectiveGrad class can be applied to any torch module and will allow that module to
    choose whether or not its weights will have active gradients through a provided mask function.

    Example usage:

    .. code:: python

        model = Model()
        selective_grad = cstorch.nn.utils.SelectiveGrad("fc.*", "outlier")
        model.apply(selective_grad)

    The user can also provide their own mask method such as:

    .. code:: python

        model = Model()
        def get_mask():
            requires_grad_val, _ = torch.topk(param, 2)
            return param > torch.min(requires_grad_val)
        selective_grad = cstorch.nn.utils.SelectiveGrad("fc.*", get_mask)
        model.apply(selective_grad)
    """

    def __init__(
        self,
        param_filter: Union[str, List[str], FilterCallable],
        init_method: InitMethodType = "outlier",
    ):
        """
        Constructs a `SelectiveGrad` instance given which parameters to affect and a given mask.

        The mask is a function that chooses which gradients to affect. Affected gradients are ones
        whose weights are included in the gradient calculation. Unaffected weights are not used to
        calculate the gradient, and are not updated after calculating the new gradient. Whether or
        not these weights are affected are marked by the boolean elements in the mask tensor.

        Args:
            param_filter: A string or list of strings, in glob format, or a callable that takes in
                the parameter's name and the parameter itself, represents a filter for which
                module/submodule and parameters/buffers this selective gradient applies to. Any
                parameters that do not match these filters will not have its gradients affected.
            init_method: A string representing a default mask, or a function that takes in
                a parameter and returns a boolean tensor that represents which specific weights in a
                parameter should have its gradients affected. If the boolean tensor has an element
                marked as True, its gradient will be recalculated and its weight adjusted.
        """
        self._param_filter = make_param_filter(param_filter)
        self._selective_grad = torch.utils.weak.WeakIdKeyDictionary()
        self._init_method = make_init_method(init_method)
        self._optimizer_step_pre_hook = None
        self._optimizer_step_post_hook = None

    def apply(self, obj: torch.nn.Module):
        """
        Applies an instance of SelectiveGrad to the module and its submodule recursively, as well as
        set up any global optimizer hooks. By running apply() we can set up any model for selective
        gradient updates.

        Args:
            obj: A torch module to have the current SelectiveGrad to be applied.
        """
        if isinstance(obj, torch.nn.Module):
            # get the optimizer to be globally registered
            self._selectively_update_optimizer()
            return self._selectively_update_module(obj)

        raise TypeError(
            f"Expected torch.nn.Module or cstorch.optim.Optimizer, "
            f"but got {type(obj)}"
        )

    def _selectively_update_module(self, module: torch.nn.Module):
        """
        Sets up the torch module itself to recursively look through its parameters and find any that
        matches the filter. If it matches the filter, it then registers a hook to zero out the
        unneeded gradients.

        Args:
            module: The torch module to filter through and set up a hook to zero out the gradients.
        """

        def get_members_fn(submodule: torch.nn.Module):
            return (
                (k, (submodule, p)) for k, p in submodule._parameters.items()
            )

        def grad_hook(
            selective_grad: _SelectiveGradParameter, grad: torch.Tensor
        ):
            zero = torch.zeros_like(grad)
            return torch.where(selective_grad.requires_grad, grad, zero)

        # Recursively get all parameters in the module as well as the module
        # that owns them.
        for name, (submodule, param) in module._named_members(
            get_members_fn, recurse=True
        ):
            # skip if the parameter is none
            if param is None:
                continue

            # filter out any parameters that are not what we're looking for
            if self._param_filter(name, param) and not getattr(
                param, "_selective_grad", None
            ):
                selective_grad = _SelectiveGradParameter(
                    submodule, name, self._init_method
                )
                param._selective_grad = selective_grad
                param.register_hook(partial(grad_hook, selective_grad))
                self._selective_grad[selective_grad] = name

    def _get_selective_grads(self, optimizer: torch.optim.Optimizer):
        """
        Given an optimizer, finds the parameters which were previously marked from
        _selective_update_module which will be the ones that need to have its weights removed for
        gradient calculation.
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Expected torch.optim.Optimizer but got {type(optimizer)}"
            )

        return (
            selectively_updated_weight
            for group in optimizer.param_groups
            for param in group["params"]
            if (
                selectively_updated_weight := getattr(
                    param, "_selective_grad", None
                )
            )
            and selectively_updated_weight.name in self._selective_grad.values()
        )

    def _selectively_update_optimizer(self):
        """
        Given that the hooks are not yet set, apply pre and post global optimizer hooks which will
        look through all the marked parameters and apply the logic to zero out the weights for
        gradient calculation and add them back in after the calculation.
        """
        if self._optimizer_step_pre_hook is not None:
            return

        @torch.no_grad()
        def optimizer_pre_step(
            optimizer: torch.optim.Optimizer, *args, **kwargs
        ):
            for selective_grad in self._get_selective_grads(optimizer):
                p = selective_grad.param

                zero = torch.zeros_like(p)
                selective_grad.original = torch.where(
                    selective_grad.requires_grad, zero, p
                )
                selective_grad.param.mul_(selective_grad.requires_grad)

                # also mask the other optimizer params/buffers
                for name, s in optimizer.state[p].items():
                    if s.shape == p.shape:
                        s.mul_(selective_grad.requires_grad)
                        # Mark the masked tensor to be the value that GradScaler
                        # restores to if DLS detects non-fininte grads. Note that
                        # GradScaler may have already marked the pre-masked state,
                        # so this is overriding it with the masked version.
                        cstorch.amp.update_if_finite(optimizer, s)

        @torch.no_grad()
        def optimizer_post_step(
            optimizer: torch.optim.Optimizer, *args, **kwargs
        ):
            for selective_grad in self._get_selective_grads(optimizer):
                selective_grad.param.add_(selective_grad.original)
                del selective_grad.original

        self._optimizer_step_pre_hook = register_optimizer_step_pre_hook(
            optimizer_pre_step
        )
        self._optimizer_step_post_hook = register_optimizer_step_post_hook(
            optimizer_post_step
        )


class _SelectiveGradParameter:
    """
    Representation of a parameter that has some weights' gradients selectively deactivated.

    This class does not own the original parameter or the mask, rather it registers the mask with
    the module that owns the parameter and provides convenience to manipulate the parameter or mask.
    """

    def __init__(
        self, module: torch.nn.Module, name: str, init_method: InitMethodType
    ):
        self._module_ref = ref(module)
        self.name = name
        self._param_name = name.rsplit(".", 1)[-1]
        self._requires_grad_name = f"{self._param_name}_requires_grad"

        requires_grad = init_method(self.param)
        self.module.register_buffer(
            self._requires_grad_name, requires_grad, persistent=True
        )

    @property
    def module(self):
        m = self._module_ref()
        if m is None:
            raise ValueError(f"Attempting to access mask after module deleted")
        return m

    @property
    def param(self):
        return self.module._parameters[self._param_name]

    @property
    def requires_grad(self):
        return self.module._buffers[self._requires_grad_name]
