# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, Optional, Union, final
from warnings import warn
from weakref import WeakValueDictionary, ref

import torch
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle

import cerebras.pytorch as cstorch
from cerebras.pytorch.backend import current_backend_impl
from cerebras.pytorch.utils.weak import DefaultWeakIdKeyDictionary

from .init import InitMethodType, make_init_method
from .utils import HyperParameterSchedule, make_hyperparam_schedule


class SparsityAlgorithm(ABC):
    """Base class for all sparsity algorithms.

    This class is responsible for sparsifying parameters and registering hooks
    to apply the sparsity pattern to the parameters before forward and to the
    gradients after backward. It also registers hooks to update the sparsity
    pattern after each optimizer step.

    .. warning::

        The way that sparse parameters are represented in the cerebras.pytorch API
        is via a mask tensor. This mask tensor is multiplied inplace to the original
        dense parameter before forward and to the gradients after backward. However,
        this is not the way that sparse parameters are represented on a Cerebras
        system. There, sparse parameters are handled natively in CSR format. As
        such, there is no mask tensor that can be referenced on the system side.
        What this means is that using the mask tensor haphazardly can lead to
        compile failures. Even if compile succeeds, any operations performed on
        the mask can be very computationally expensive. Having said that, there
        are several operations on masks that are supported on the Cerebras
        system. Please see the usage in the prepackaged algorithms as a guide
        for when and how it is acceptable to use the mask.

    """

    _sparsity_algorithm_count = defaultdict(int)

    def __init__(
        self,
        sparsity: Union[float, HyperParameterSchedule, None],
        init_method: InitMethodType = "random",
    ):
        """Constructs a `SparsityAlgorithm` instance.

        Args:
            sparsity: The sparsity level to use for the algorithm. This can be
                a float or a :py:class:`~cerebras.pytorch.sparse.utils.HyperParameterSchedule`.
                If a dictionary is passed in, then it is automatically converted to a
                :py:class:`~cerebras.pytorch.sparse.utils.HyperParameterSchedule`

            init_method: The method to use to initialize the sparsity mask.
                See :py:func:`~cerebras.pytorch.sparse.init.make_init_method` for more details.
        """
        count = SparsityAlgorithm._sparsity_algorithm_count[self.__class__]
        self.name = f"sparsity_{self.__class__.__name__.lower()}_{count}"
        SparsityAlgorithm._sparsity_algorithm_count[self.__class__] += 1

        if sparsity is not None:
            self.sparsity = sparsity

        self.init_method = make_init_method(init_method)

        self.sparse_modules = torch.utils.weak.WeakIdKeyDictionary()
        self.sparse_optimizers = torch.utils.weak.WeakIdKeyDictionary()

        self.sparse_params = WeakValueDictionary()

        self._backend = current_backend_impl()
        self._backend.setup_sparsity(self)

        self.autoupdate = True

        self._target_sparsity_hooks = OrderedDict()
        self._computed_sparsity_hooks = OrderedDict()

    @property
    def num_sparse_params(self) -> int:
        """Return the number of parameters that have been sparsified by this algorithm."""
        return len(self.sparse_params)

    def get_sparse_params(
        self, obj: Union[torch.Tensor, torch.nn.Module, torch.optim.Optimizer]
    ) -> Union["SparseParameter", Generator["SparseParameter", None, None]]:
        """Get all sparse parameters that were sparsified by this algorithm.

        Args:
            obj: The object to get sparse parameters from.
        Returns:
            If obj is a Tensor, returns the sparse parameter associated with that tensor (if any).
            If obj is a Module, returns an iterator over all sparse parameters of the module
                and its submodules recursively.
            If obj is an Optimizer, returns an iterator over all sparse parameters associated
                with the optimize param groups.
        """
        if isinstance(obj, torch.Tensor):
            return getattr(obj, "_sparse_param", None)
        elif isinstance(obj, torch.nn.Module):
            return (
                sparse_param
                for _, param in obj.named_parameters()
                if (sparse_param := getattr(param, "_sparse_param", None))
                and sparse_param.name in self.sparse_params
            )
        elif isinstance(obj, torch.optim.Optimizer):
            return (
                sparse_param
                for group in obj.param_groups
                for param in group["params"]
                if (sparse_param := getattr(param, "_sparse_param", None))
                and sparse_param.name in self.sparse_params
            )
        else:
            raise TypeError(
                f"Expected torch.nn.Module or torch.optim.Optimizer, "
                f"but got {type(obj)}"
            )

    def initialize(self) -> None:
        """Initialize the sparsity pattern for all parameters sparsified by this algorithm."""
        for sparse_param in self.sparse_params.values():
            sparse_param.initialize()

    def csx_annotate_sparsity(self, param: "SparseParameter") -> None:
        """Annotate the parameter with hints about the sparsity pattern.

        These hints are used as performance hints for the Cerebras compiler.

        Args:
            param: The sparse parameter to annotate with hints.
        """

    @property
    def sparsity(self) -> Dict[torch.Tensor, HyperParameterSchedule]:
        """Return the mapping between a parameter and its sparsity schedule."""
        if not hasattr(self, "_sparsity"):

            def default_error():
                raise ValueError(
                    f"{self.__class__.__name__} sparsity algorithm expected "
                    f"`sparsity` to be specified, but got none."
                )

            self._sparsity = DefaultWeakIdKeyDictionary(default_error)

        return self._sparsity

    @sparsity.setter
    def sparsity(self, sparsity: Union[float, HyperParameterSchedule, None]):
        """Create a mapping between a parameter and its sparsity schedule.

        If a mapping already exists, it will be updated.
        """
        if isinstance(sparsity, dict) and any(
            isinstance(k, torch.Tensor) for k in sparsity
        ):

            def default_error():
                raise KeyError("No sparsity schedule found for parameter")

            self._sparsity = DefaultWeakIdKeyDictionary(
                default_error,
                {p: make_hyperparam_schedule(s) for p, s in sparsity.items()},
            )
        else:
            # If a mapping exists, this will effectively just set the default
            # schedule and keep the previously existing schedules
            prev = getattr(self, "_sparsity", {})
            default_schedule = make_hyperparam_schedule(sparsity)
            self._sparsity = DefaultWeakIdKeyDictionary(
                lambda: default_schedule, prev
            )

    def sparsify_parameter(
        self, module: torch.nn.Module, name: str, param: torch.Tensor
    ) -> None:
        """Initialize the mask for a parameter in the given module.

        Args:
            module: The module that owns the parameter
            name: The full name of the parameter
            param: The parameter to initialze the sparsity mask for.
        """
        if param is None:
            # Parameter is None, nothing to sparsify
            return
        if self.get_sparse_params(param):
            # Parameter already sparsified
            return
        if getattr(param, "requires_dense", False):
            # Parameter has been marked as not sparsifiable
            return

        # This simple scalar computation does not need to be traced
        with torch.device("cpu"):
            # Get the sparsity schedule for the given parameter and then
            # call it with a step of 0 to get the initial sparsity value
            sparsity = self.sparsity[param](getattr(self, "step", 0))
            # Ensure that the sparsity level is valid
            sparsity = torch.clamp(sparsity, min=0.0, max=1.0)

        init_method = partial(self.init_method, sparsity=sparsity)
        sparse_param = SparseParameter(module, name, init_method)
        param._sparse_param = sparse_param

        # Keep a reference to the sparse parameter so that we can query them later on
        self.sparse_params[name] = sparse_param

    @final
    def apply(
        self, obj: Union[torch.nn.Module, cstorch.optim.Optimizer]
    ) -> None:
        """Sparsify the passed in object.

        .. note::

            This is called implicitly when calling ``module.apply(sparsity)``
            or ``optimizer.apply(sparsity)``

        Args:
            obj: a ``torch.nn.Module`` or a ``cstorch.optim.Optimizer`` object
                to sparsify.
        """
        if isinstance(obj, torch.nn.Module):
            self.sparsify_module(obj)
        elif isinstance(obj, cstorch.optim.Optimizer):
            self.sparsify_optimizer(obj)
        else:
            raise TypeError(
                f"Expected torch.nn.Module or cstorch.optim.Optimizer, "
                f"but got {type(obj)}"
            )

    def sparsify_module(self, module: torch.nn.Module) -> None:
        """Sparsify the ``torch.nn.Module`` object.

        Args:
            module: the ``torch.nn.Module`` object to sparsify
        """

        def get_members_fn(submodule):
            if getattr(submodule, "requires_dense", False):
                # Module has been marked as not sparsifiable
                return ()

            if submodule in self.sparse_modules or getattr(
                submodule, "is_sparse", False
            ):
                # Already applied sparsity for this module
                warn(f"Module {submodule} has already been sparsified.")
                return ()

            self.sparse_modules[submodule] = True
            submodule.is_sparse = True

            return (
                (k, (submodule, p)) for k, p in submodule._parameters.items()
            )

        pre_sparsification_count = self.num_sparse_params

        # Recursively get all parameters in the module as well as the module
        # that owns them.
        for name, (submodule, param) in module._named_members(
            get_members_fn, recurse=True
        ):
            self.sparsify_parameter(submodule, name, param)

        if self.num_sparse_params == pre_sparsification_count:
            warn(f"No parameters were sparsified in module {module}")
            # No parameters were sparsified, so no need to register
            # a forward pre hook
            return

        module.register_forward_pre_hook(self._forward_pre_hook)

        with self._backend.device:
            if (
                self._backend.is_csx
                and not self._backend.device.config.lazy_initialization
            ):
                # We need to move the masks to the device if we are doing
                # eager initialization
                self._backend.device.move_to_device(module)

            self.visit_state(lambda x: x.to(self._backend.torch_device))

    def _forward_pre_hook(self, module: torch.nn.Module, args: Any):
        """Hook the given module to apply sparsity patterns.

        The sparsity pattern is applied to both the parameters before `forward()`
        call and gradients after `backward()` call.

        Args:
            module: The module that `forward()` is called on.
            args: Positional arguments passed to the `forward()` call.
        """
        for sparse_param in self.get_sparse_params(module):
            # Clear sparse param's internal state
            sparse_param.clear()
            # Annotate the sparse param with hints for the Cerebras compiler
            if cstorch.use_cs():
                self.csx_annotate_sparsity(sparse_param)

            self.prune_weight(sparse_param)

    @torch.no_grad()
    def prune_weight(self, sparse_param: "SparseParameter"):
        """Prune the dense weight and register a hook to prune the gradients.

        .. note::

            This is called automatically in a module forward pre-hook.

        """
        p = sparse_param.param
        sparse_param.prune(p, sparse_param.name)
        if p.requires_grad and sparse_param.grad_hook is None:
            sparse_param.grad_hook = p.register_hook(
                partial(self._grad_hook, p)
            )

    def _grad_hook(self, p: torch.Tensor, grad: torch.Tensor):
        """Hook to prune the gradients after backward().

        .. note::

            This is called automatically in the parameter's backward grad hook.

        Args:
            p: The original parameter.
            grad: The gradient of the parameter.
        """
        # In the case there any NaNs in the unused gradients that correspond to
        # zero'd out weights, we use a selection to replace these NaNs with
        # zeros. (multiplying with the mask would preserve them).
        # DLS will skip a weight update if there is a NaN in the gradient, but
        # we only want this to happen if there is a NaN in gradients
        # corresponding to non-zero weights. This is the behavior of the CS2
        # which doesn't even compute the full gradients on most steps.
        zero = torch.zeros_like(grad)

        # Return modified gradient.
        with SparseParameter.disable_mask_access_warning():
            return torch.where(p.mask, grad, zero)

    def sparsify_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Sparsify the ``torch.optim.Optimizer`` object.

        Args:
            optimizer: the ``torch.optim.Optimizer`` object to sparsify
        """
        if optimizer in self.sparse_optimizers or getattr(
            optimizer, "is_sparse", False
        ):
            # Already applied sparsity for this optimizer
            return

        self.sparse_optimizers[optimizer] = True
        optimizer.is_sparse = True

        if len(self.sparse_optimizers) > 1:
            # TODO: Support multiple optimizers
            # This is not a high priority as we never really use
            # more than one optimizer in practice
            raise RuntimeError(
                "Sparsifying multiple optimizers using the same sparsity "
                "algorithm is not supported."
            )

        def prune_optimizer_states(optimizer, args, kwargs):
            params = list(self.get_sparse_params(optimizer))
            if len(params) == 0:
                raise RuntimeError(
                    "Detected that optimizer.apply(sparsity) was called "
                    "but model.apply(sparsity) was not.\n"
                    "Please call model.apply(sparsity)."
                )

            for sparse_param in params:
                p = sparse_param.param
                for name, s in optimizer.state[p].items():
                    # sparsify all optimizer state tensors that match the
                    # original parameter's shape and doesn't require dense
                    if s.shape == p.shape and not getattr(
                        s, "requires_dense", False
                    ):
                        sparse_param.prune(s, name)
                        # Mark the pruned tensor to be the value that GradScaler
                        # restores to if DLS detects non-fininte grads. Note that
                        # GradScaler may have already marked the state pre-pruning,
                        # so this is overriding it with the pruned version, with
                        # the assumption that `prune()` modifies `s` in-place.
                        cstorch.amp.update_if_finite(optimizer, s)

        # Only prune optimizer state if optimizer step is called
        optimizer.register_step_pre_hook(prune_optimizer_states)

        def step_post_hook(optimizer, args, kwargs):
            # The weights and optimizer state were just updated. In case we
            # _decrease_ sparsity in the update instead of increasing it, prune
            # the weights using the current weight masks
            for sparse_param in self.get_sparse_params(optimizer):
                sparse_param.prune()

            if self.autoupdate:
                self.update(optimizer)

        optimizer.register_step_post_hook(step_post_hook)

    @abstractmethod
    def update(self, optimizer: Optional[cstorch.optim.Optimizer] = None):
        """Update the parameter's sparsity masks.

        Args:
            optimizer: The optimizer that is being used to update the sparse parameters.
        """

    def register_target_sparsity_hook(
        self, hook: Callable[[str], torch.Tensor]
    ) -> RemovableHandle:
        r"""Register a hook which will be called when a new target sparsity
        is computed. It should have the following signature:

            hook(sparsity, name, target)

        ``sparsity`` argument is the sparsity instance being used.
        ``name`` is the name of the group of parameters that the target sparsity
        is being computed for.
        ``target`` is the computed target sparsity value.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._target_sparsity_hooks)
        self._target_sparsity_hooks[handle.id] = hook
        return handle

    def register_computed_sparsity_hook(
        self, hook: Callable[[str], torch.Tensor]
    ) -> RemovableHandle:
        r"""Register a hook which will be called when a new sparsity mask
        is computed. It should have the following signature:

            hook(sparsity, name, computed)

        ``sparsity`` argument is the sparsity instance being used.
        ``name`` is the name of the parameter that the mask belongs to.
        ``computed`` is the calculated sparsity level of the newly computed mask.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._computed_sparsity_hooks)
        self._computed_sparsity_hooks[handle.id] = hook
        return handle

    def visit_state(self, f: Callable):
        """Apply a callable to the stateful tensors."""

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a dictionary of all stateful tensors."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load the state of all stateful tensors."""


class SparseParameter:
    """Representation of a sparse parameter.

    This class does not own the original parameter or the mask. It registers
    the mask with the module that owns the parameter and provides convenient
    accessors and modifiers for the mask.
    """

    DISABLE_MASK_ACCESS_WARNING = False

    @staticmethod
    @contextmanager
    def disable_mask_access_warning():
        prev = SparseParameter.DISABLE_MASK_ACCESS_WARNING
        try:
            SparseParameter.DISABLE_MASK_ACCESS_WARNING = True
            yield
        finally:
            SparseParameter.DISABLE_MASK_ACCESS_WARNING = prev

    def __init__(
        self, module: torch.nn.Module, name: str, init_method: InitMethodType
    ):
        # Save a weak reference to the module so that we can access it
        # without creating a reference cycle.
        self._module_ref = ref(module)
        self.name = name
        self.param_name = name.rsplit(".", 1)[-1]
        self.mask_name = f"{self.param_name}_mask"

        self.init_method = init_method

        self._backend = current_backend_impl()
        with self._backend.device:
            placeholder = cstorch.ones_like(self.param, dtype=torch.bool).to(
                self._backend.torch_device
            )
            module.register_buffer(self.mask_name, placeholder, persistent=True)

        self._initialized = False

        def load_state_dict_pre_hook(state_dict, *args, **kwargs):
            # If we are loading the mask from a checkpoint, then
            # consider the mask as already initialized
            if f"{self.name}_mask" in state_dict:
                self._initialized = True

        module._register_load_state_dict_pre_hook(load_state_dict_pre_hook)

        self.grad_hook = None

        # Keep track of all tensors that were sparsified by this mask
        self._pruned_tensors = torch.utils.weak.WeakTensorKeyDictionary()
        # Keep track of all annotations that were applied to this mask
        self._annotations = {}

        def mask_property(p):
            if hasattr(p, "_sparse_param"):
                if not SparseParameter.DISABLE_MASK_ACCESS_WARNING:
                    warn(
                        f"Using the mask tensor haphazardly can lead to compile failures "
                        f"and/or be very computationally expensive. Please only use the "
                        f"mask tensor directly if you really know what you are doing."
                    )
                return p._sparse_param.mask
            else:
                return None

        # Add a property to the param so that the mask tensor can
        # be accessed as param.mask
        type(self.param).mask = property(mask_property)

    def initialize(self):
        if self._initialized:
            return

        # Use the CPU device if doing eager initialization on CSX.
        # Otherwise, use the parameter's device.
        # This allows us to trace the mask initialization during
        # lazy initialization.
        device = None
        if (
            self._backend.is_csx
            and not self._backend.device.config.lazy_initialization
        ):
            device = "cpu"

        with self._backend.device:
            mask = self.init_method(self.param, device=device)
            if not isinstance(mask, torch.Tensor):
                raise TypeError(
                    f"Expected init_method to return a Tensor, "
                    f"but got {type(mask)}"
                )
            if mask.device.type != self._backend.torch_device.type:
                mask = mask.to(self._backend.torch_device)

            # overwrite buffer
            setattr(self.module, self.mask_name, mask)

        self._initialized = True

    @property
    def module(self):
        m = self._module_ref()
        if m is None:
            raise ValueError(f"Attempting to access mask after module deleted")
        return m

    @property
    def param(self):
        return self.module._parameters[self.param_name]

    @property
    def data(self):
        return self.param

    @property
    def mask(self):
        return self.module._buffers[self.mask_name]

    @mask.setter
    def mask(self, new_mask):
        self.update(new_mask)

    def annotate(self, name, value):
        if len(self._pruned_tensors) > 0:
            raise RuntimeError(
                f"Detected that annotations are being set after pruning tensors: "
                f"{sorted(self._pruned_tensors.values())}"
            )

        self._annotations[name] = value

    @torch.no_grad()
    def prune(self, tensor=None, tensor_name=None):
        """
        Prunes the tensor using the sparse parameter's mask.
        If no tensor is provided, re-prune all tensors in the
        pruned tensor registry.
        """
        if tensor is None:
            for tensor, name in self._pruned_tensors.items():
                self.prune(tensor, name)
            return

        # annotate the tensor before pruning
        for name, value in self._annotations.items():
            self._backend.set_attribute(tensor, name, value)

        tensor.mul_(self.mask)

        self._pruned_tensors[tensor] = tensor_name

    def update(self, new_mask):
        if not self._initialized:
            raise RuntimeError(
                "Detected that mask is being updated before it was initialized"
            )
        if len(self._pruned_tensors) == 0:
            raise RuntimeError(
                "Detected that mask is being updated before it was used"
            )

        self.module._buffers[self.mask_name].copy_(new_mask)

        # Need to re-prune all tensors that depend on this mask
        # to let the compiler know that the sparsity pattern
        # has changed for all tensors sparsified by this mask
        self.prune()

    def clear(self):
        """Clear pruned tensors and annotations for the next iteration."""
        self._pruned_tensors.clear()
        self._annotations.clear()

    def __str__(self):
        return f"SparseParameter({self.name})"
