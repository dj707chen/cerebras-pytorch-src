# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Contains the abstract base backend class. """
import contextlib
from abc import ABC
from collections import OrderedDict, defaultdict
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Union
from warnings import warn
from weakref import WeakValueDictionary

import torch
from torch.utils.weak import WeakIdKeyDictionary
from tqdm import tqdm

import cerebras.pytorch as cstorch
import cerebras.pytorch.metrics as metrics
from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.appliance.utils.tracker import Tracker
from cerebras.pytorch.backend import BackendType
from cerebras.pytorch.core.device import Device
from cerebras.pytorch.core.modes import EVAL, TRAIN
from cerebras.pytorch.core.name_scope import (
    ScopeName,
    add_debug_name,
    get_debug_name,
)
from cerebras.pytorch.utils.nest import visit_torch_tensors

COMPILE_ONLY_MSG = "Compiling the model. This may take a few minutes."
COMPILE_SUCCESS_MSG = "Compile was successful!"
PROGRAMMING_CS_MSG = (
    "Programming Cerebras Wafer Scale Cluster for execution. "
    "This may take a few minutes."
)


@named_class_logger
class BaseBackend(ABC, ClassLogger):
    """
    The abstract base backend.
    Contains the logic common to all backends.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        backend_type: BackendType,
        device: Device,
        artifact_dir: Optional[str] = None,
    ):
        self.backend_type = backend_type
        self.device = device

        if artifact_dir is None:
            artifact_dir = "./"

        self.artifact_dir = Path(artifact_dir)

        self._compile_only = False
        self._validate_only = False

        self.data_executor_stack = []

        self._compiled_models: Dict[torch.nn.Module, int] = (
            WeakIdKeyDictionary()
        )

        self.optimizer_registry = WeakIdKeyDictionary()

        self.sparsity_registry = WeakIdKeyDictionary()

        self.grad_scaler_registry = []

        self._dataloaders = WeakValueDictionary()

        # detached here means that the metric is not a part of any module
        # and needs to be handled separately
        self.detached_metrics: Optional[torch.nn.Module] = None

        # Dictionary of weak references to stateful tensors that
        # are not attached to a module/optimizer
        self.detached_stateful = WeakValueDictionary()

        # Each executor corresponds to a single DataExecutor that's been entered.
        self.executor_counter = 0

        self.appliance_tracker: Optional[Tracker] = None

        self.reset()

    def reset(self):
        """Resets the backend variables to its initial state."""
        self.mode = None

        self.model: Optional[torch.nn.Module] = None

        # The list if model ids that were used by the current session.
        # This list is being updated every time we trace a step function.
        self._active_model_id: Optional[int] = None

        # queue of step closures
        self.step_closures = []

        # progress tracker
        self._progress_tracker = None

        # Create a new tracker and start tracking "initialization"
        self.appliance_tracker = Tracker()
        self.appliance_tracker.start("Initialization")

        # flag to indicate if we're in tracing mode
        self._is_tracing = False

        self.current_scope_name = ScopeName()

        # For debug_names that are invoked multiple times in the model's
        # forward(), this tracks the call number and is reset each batch.
        self._debug_name_call_counters = defaultdict(int)
        self._pre_fwd_scope_names = defaultdict(list)

        if getattr(self, "module_forward_hook", None):
            self.module_forward_hook.remove()

        self.module_forward_hook = None

        if getattr(self, "module_forward_pre_hook", None):
            self.module_forward_pre_hook.remove()

        self.module_forward_pre_hook = None

    # alias properties from backend type
    is_cpu = property(lambda self: self.backend_type.is_cpu)
    is_gpu = property(lambda self: self.backend_type.is_gpu)
    is_csx = property(lambda self: self.backend_type.is_csx)

    @property
    def is_tracing(self) -> bool:
        """Returns True if the backend is currently tracing the model."""
        return self._is_tracing

    @property
    def is_e2e_execution(self) -> bool:
        """Returns True if the backend is configured for end-to-end execution."""
        return True

    @property
    def torch_device(self) -> torch.device:
        """Returns the corresponding torch device."""
        return self.device.torch_device

    @property
    def in_run_context(self):
        return len(self.data_executor_stack) > 0

    @property
    def data_executor(self):
        """
        Get the current data executor which will be used to configure the
        appliance run.
        """
        if len(self.data_executor_stack) == 0:
            raise RuntimeError(
                "Detected that a data executor was not used.\n"
                "Please wrap your dataloader in a Cerebras DataExecutor:\n\n"
                "\texecutor = cstorch.utils.data.DataExecutor(dataloader, ...)\n\n"
                "Which can be used in the execution loop as follows:\n\n"
                "\tfor i, batch in enumerate(executor):\n\t\t...\n\n"
                "For more details, please see the documentation for "
                "cstorch.utils.data.DataExecutor."
            )

        return self.data_executor_stack[-1]

    @property
    def run_context(self):
        """
        Get the current run context which will be used to configure the
        appliance run.
        """
        return self.data_executor.run_context

    @property
    def progress_tracker(self) -> tqdm:
        """Used to update users on the progress of weight initialization."""
        if (
            self._progress_tracker is None
            and cstorch.backends.csx.debug.log_initialization
        ):
            self._progress_tracker = tqdm(
                ncols=0, bar_format="{desc}[{elapsed}{postfix}]"
            )

        return self._progress_tracker

    def move_to_device(self, struct):
        """Moves all tensors in the provided structure to the torch device."""
        return self.device.move_to_device(struct)

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """State dict of the backend that contains all state tensors.

        This method is not to be used with checkpoints and is mainly for marking
        tensors as outputs/aliases of the model.
        """
        s = {}
        if self.model is not None:
            s["model"] = full_state_dict(self.model)

        # pylint: disable=protected-access
        if len(self.optimizer_registry) == 1:
            optimizer = next(iter(self.optimizer_registry))
            s["optimizer"] = optimizer.state_dict()
            s["schedulers"] = [
                scheduler.state_dict()
                for scheduler in optimizer._schedulers_registry
            ]
        else:
            s["optimizers"] = []
            for o in self.optimizer_registry:
                state_dict = o.state_dict()
                state_dict["schedulers"] = [
                    scheduler.state_dict()
                    for scheduler in o._schedulers_registry
                ]
                s["optimizers"].append(state_dict)

        if self.grad_scaler_registry is not None:
            s["grad_scaler"] = [
                gs.state_dict() for gs in self.grad_scaler_registry
            ]

        if self.sparsity_registry:
            s["sparsity"] = [s.state_dict() for s in self.sparsity_registry]

        if self.detached_metrics is not None:
            s["metrics"] = full_state_dict(self.detached_metrics)

        if self.detached_stateful is not None:
            s["detached_stateful"] = {
                str(k): v for k, v in self.detached_stateful.items()
            }

        return s

    def setup_model(self, model) -> int:
        """
        Moves the model to the torch device and tracks the duplicate tensors.
        """
        # Compute a new model id based on the total number of compiled models
        model_id = max(self._compiled_models.values(), default=-1) + 1

        self._compiled_models[model] = model_id

        # Set the backend's mode via calls to model.train and model.eval
        model_train = model.train

        @wraps(model_train)
        def _train(_self, is_training: bool = True):
            self.mode = TRAIN if is_training else EVAL
            self.logger.debug(f"Setting mode to {self.mode}")
            return model_train(is_training)

        model.train = MethodType(_train, model)

        def named_members(model, get_member_fn, prefix=""):
            """
            Helper method which returns a map of param_name -> set of duplicate param names.
            """
            memo = dict()  # dict from tensor -> str name of tensor
            names = defaultdict(
                set
            )  # dict from str name of tensor -> set of duplicates
            modules = model.named_modules(prefix=prefix, remove_duplicate=False)
            for module_prefix, module in modules:
                for k, v in get_member_fn(module):
                    if v is None:
                        continue

                    name = module_prefix + ('.' if module_prefix else '') + k
                    if v in memo:
                        # whenever a duplicate is found
                        # update the existing list of duplicate
                        # names corresponding to the first name
                        names[memo[v]] |= {memo[v], name}
                        # also add a key for new name with
                        # value as the duplicates list
                        names[name] = names[memo[v]]
                    else:
                        memo[v] = name

            return names

        # pylint: disable=protected-access
        # set duplicate params for params and buffers in the model
        model_params_duplicates_map = named_members(
            model, lambda module: module._parameters.items()
        )
        model_params_duplicates_map.update(
            named_members(model, lambda module: module._buffers.items())
        )

        self.move_to_device(model)
        model.device = self.torch_device

        # Add _debug_name attribute to module and its children
        add_debug_name(model)

        def retie_weights(module, scope, duplicates_map):
            # pylint: disable=protected-access
            for tensor_dict in (module._parameters, module._buffers):
                tensor_names = list(tensor_dict.keys())
                for name in tensor_names:
                    tensor_name = ".".join(scope + [name])
                    if tensor_name not in model_params_duplicates_map:
                        continue

                    if tensor_name in duplicates_map:
                        setattr(module, name, duplicates_map.pop(tensor_name))
                        continue

                    for duplicate_name in model_params_duplicates_map[
                        tensor_name
                    ]:
                        duplicates_map[duplicate_name] = tensor_dict[name]

            for name, child in module.named_children():
                retie_weights(child, scope + [name], duplicates_map)

        if model_params_duplicates_map:
            retie_weights(model, [], {})

        return model_id

    def _add_name_scope_hooks(self):
        # Helper for hooks
        def get_name(module, counter_increment=0):
            # TODO: need to reset _num_instances on batch start. Otherwise we will get
            # different names between itertions.
            name = get_debug_name(module)

            counter = self._debug_name_call_counters[name]
            self._debug_name_call_counters[name] += counter_increment
            if counter:
                name = f"{name}.call{counter}"
            return name

        def fwd_pre_name_scope(
            module, inputs
        ):  # pylint: disable=redefined-builtin
            scope_name = ScopeName(get_name(module), "fwd")
            self._pre_fwd_scope_names[module].append(
                self.set_scope_name(scope_name)
            )

        def fwd_post_name_scope(
            module, input, output
        ):  # pylint: disable=redefined-builtin
            # Exit FWD scope
            # Restore name_scope we saved during `fwd_pre_name_scope`
            pre_fwd_scopes = self._pre_fwd_scope_names[module]
            pre_fwd_scope = ScopeName()
            if pre_fwd_scopes:
                pre_fwd_scope = pre_fwd_scopes.pop()
            self.set_scope_name(pre_fwd_scope)

            # Set up the BWD scope.

            # This will actually be the name for the bwd pass entered from
            # the module's output's grad hook.
            # Also, increment the counter for the next fwd_pre to hit.
            bwd_name_scope = ScopeName(get_name(module, 1), "bwd")

            for _, tensor in visit_torch_tensors(output):
                # In case a module returns a tensor unmodifed, don't change its
                # scope.
                has_bwd_name_scope = getattr(
                    tensor, "_has_bwd_name_scope", False
                )
                if tensor.requires_grad and not has_bwd_name_scope:
                    # pylint: disable=protected-access
                    tensor._has_bwd_name_scope = True

                    def hook(x):
                        self.set_scope_name(bwd_name_scope)

                    tensor.register_hook(lambda x: hook(x))

        from torch.nn.modules import module

        if self.module_forward_pre_hook is not None:
            self.module_forward_pre_hook.remove()

        self.module_forward_pre_hook = module.register_module_forward_pre_hook(
            fwd_pre_name_scope
        )

        if self.module_forward_hook is not None:
            self.module_forward_hook.remove()

        self.module_forward_hook = module.register_module_forward_hook(
            fwd_post_name_scope
        )

    def set_scope_name(self, scope_name: ScopeName) -> ScopeName:
        """Set new scope name and return the old one."""
        old_name = self.current_scope_name
        self.current_scope_name = scope_name or ScopeName()
        return old_name

    def register_dataloader(self, dataloader: cstorch.utils.DataLoader):
        """Registers a dataloader to be used for a run."""
        self._dataloaders[dataloader.id] = dataloader

    def register_active_model(self, model, model_id):
        if self._active_model_id is not None:
            if self._active_model_id != model_id:
                raise RuntimeError(
                    "Detected that more than one compiled model was used in a step function. "
                    "Only a single compiled model may be used in a given forward pass."
                )
            else:
                return

        self.model = model
        self._active_model_id = model_id

    def on_run_start(self):  # pylint: disable=no-self-use
        """Runs once at the beginning of the run.

        Used by cstorch.utils.data.DataLoader
        """
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self._setup_optimizers()
        self._setup_detached_metrics()

        # TODO: fix move_to_device for metrics buffers in _setup_detached_metrics.
        for model in self._compiled_models:
            self.move_to_device(model)

        # Initialize sparse parameters if they haven't already been initialized
        for sparsity in self.sparsity_registry:
            sparsity.initialize()

        # Clean up the progress bar if it exists
        if self._progress_tracker is not None:
            self._progress_tracker.close()
            self._progress_tracker = None

        self._add_name_scope_hooks()

    def on_run_end(
        self, exec_type=None, exec_value=None, traceback=None
    ):  # pylint: disable=no-self-use
        """Runs once at the end of the run."""
        self.appliance_tracker.flush(
            self.data_executor.artifact_dir / "track.json"
        )

        self.reset()

    def on_batch_start(self, batch):
        """Used by cstorch.utils.data.DataExecutor."""

        # Clear debug_name call counters.
        self._debug_name_call_counters = defaultdict(int)
        self._pre_fwd_scope_names = defaultdict(list)

        # Clear cached dataloader state
        if self.run_context.dataloader.is_restartable:
            self.run_context.dataloader.cached_state = (
                cstorch.utils.data.DataLoader.STATE_UNKNOWN
            )

        batch_on_device = self.move_to_device(batch)
        self._is_tracing = True
        return batch_on_device

    def on_batch_end(self):
        """Used by cstorch.utils.data.DataExecutor."""
        self._is_tracing = False

        # Update the profiler as we have processed a batch. Note that this is
        # done after mark_step so that we don't jump the gun and updated samples
        # processed before compile/execute is actually done.
        if self.run_context.profiler is not None:
            self.run_context.profiler.step(
                self.run_context.dataloader.batch_size
            )

        self._update_dataloader_state()

        self.run_step_closures()

    def shutdown(self):
        """Shutdown the backend."""

    def mark_output(self, struct, force=False):  # pylint: disable=no-self-use
        """Marks the tensors in the struct as outputs of the model."""

    def forward(self, model, *args, **kwargs):  # pylint: disable=no-self-use
        """Runs the forward pass for the model."""
        return model(*args, **kwargs)

    @contextlib.contextmanager
    def name_scope(self, name: str):
        """Context manager for setting the name scope for the current context."""
        old_name = self.set_scope_name(ScopeName(name))
        yield
        self.set_scope_name(old_name)

    def register_optimizer(self, optimizer):
        """
        Adds the optimizer to the registry to be wrapped when a run starts.
        """
        # Need to keep track of optimizers for amp loss scaling
        self.optimizer_registry[optimizer] = None

        # pylint: disable=protected-access
        optimizer._schedulers_registry = WeakIdKeyDictionary()

    def unregister_optimizer(self, optimizer):
        """
        Removes a previously registered optimizer.
        """
        self.optimizer_registry.pop(optimizer, None)

    def _setup_optimizers(self):
        for optimizer in self.optimizer_registry:
            if getattr(optimizer, "_cstorch_setup", False):
                # Don't double setup.
                continue
            # pylint: disable=protected-access
            optimizer._cstorch_setup = True

            self.setup_optimizer(optimizer)

    def setup_optimizer(self, optimizer):
        def check_set_to_none(set_to_none: bool = True):
            if not set_to_none:
                warn(
                    "Calling optimizer.zero_grad(set_to_none=False) can prevent "
                    "the construction of a static graph which can cause multiple "
                    "compiles"
                )

        optimizer.register_zero_grad_pre_hook(
            lambda optimizer, args, kwargs: check_set_to_none(*args, **kwargs)
        )

        def check_mode(optimizer, args, kwargs):
            """Action to perform just after optimizer step is called."""
            # The fact that we performed an optimizer step must mean that we are training
            if self.mode == EVAL:
                warn(
                    "Detected a call to model.eval() as well as a call to "
                    "optimizer.step(). If you are intending to train the model, "
                    "please call model.train() instead of model.eval(). If you "
                    "are not intending to train the model, please remove the call "
                    "to optimizer.step()."
                )
            self.mode = TRAIN
            self.logger.debug(
                "Setting mode to train as optimizer.step() was called."
            )

        optimizer.register_step_post_hook(check_mode)

    def setup_scheduler(self, scheduler):  # pylint: disable=no-self-use
        """Set up the scheduler."""
        # pylint: disable=protected-access
        optimizer = scheduler.optimizer
        optimizer._schedulers_registry[scheduler] = None

        scheduler.device = self.torch_device
        with self.device:
            if not isinstance(scheduler.last_epoch, torch.Tensor):
                # The tensor representation of last_epoch
                scheduler.last_epoch = torch.tensor(
                    scheduler.last_epoch, dtype=torch.int64
                )

            scheduler.last_epoch = scheduler.last_epoch.to(self.torch_device)

    def setup_grad_scaler(self, grad_scaler):
        """Set up the grad scaler."""
        self.grad_scaler_registry.append(grad_scaler)

    def setup_sparsity(self, sparsity):
        self.sparsity_registry[sparsity] = None

    def _setup_detached_metrics(self):
        """Find all detached metrics."""

        attached_metrics = set()
        for model in self._compiled_models:
            for submodule in model.modules():
                if isinstance(submodule, metrics.Metric):
                    attached_metrics.add(id(submodule))

        # Compile replaces "/" with "_" in parameters, so we need to do the same
        # here to avoid mismatches
        self.detached_metrics = torch.nn.ModuleDict(
            {
                metric_name.replace("/", "_"): torch.nn.ModuleList(
                    filter(
                        lambda metric: id(metric) not in attached_metrics,
                        metric_list,
                    )
                )
                for metric_name, metric_list in metrics.Metric.registry.items()
            }
        )

        self.move_to_device(self.detached_metrics)

    def set_attribute(
        self,
        tensor: torch.Tensor,
        attribute: str,
        value: Union[bool, int, float, str, list, dict],
    ):
        """
        On supported backends, adds an attribute to the traced tensor at
        compile time to communicating with the Cerebras Compiler Stack.
        """

    def add_step_closure(
        self,
        closure,
        args,
        kwargs,
        run_async: bool = False,
        repeat: bool = False,
    ):
        """
        Adds the provided function to a queue of closures to be run at the end
        of the step.
        """
        if run_async:
            self.logger.warning(
                f"Asynchronous step closures not supported by "
                f"{self.backend_type} backend. "
                f"Will run `{closure.__name__}` synchronously"
            )

        # There is no guarantee that the tensors in args and kwargs aren't
        # mutated in place after being added to a step closure. To avoid reading
        # future values when the step closure runs, we pass a copy of the tensor
        # to the closure.
        args, kwargs = torch.utils._pytree.tree_map_only(
            torch.Tensor, lambda t: t.detach().clone(), (args, kwargs)
        )

        self.step_closures.append((closure, args, kwargs, repeat))

    def run_step_closures(self):  # pylint: disable=no-self-use
        """Run all the queued closures."""
        step_closures = self.step_closures
        self.step_closures = []

        for closure, args, kwargs, repeat in step_closures:
            closure(*args, **kwargs)

            if repeat:
                self.step_closures.append((closure, args, kwargs, repeat))

    def start_implict_loop(
        self,
        input_tensor: torch.IntTensor,
        loop_dim: int,
    ) -> torch.IntTensor:
        """
        Return an index tensor signaling an implicit loop over the given tensor
        along the given dimension, used for autoregressive inference.

        Args:
            input_tensor: This tensor will be updated before re-running the model
            loop_dim: The dimension of ``input_tensor`` to loop over.
        """
        raise NotImplementedError(
            "Implicit autoregressive loop is not supported on "
            f"{self.backend_type} backend"
        )

    def update_implicit_loop(
        self,
        input_tensor: torch.IntTensor,
        index_tensor: torch.IntTensor,
        update_tensor: torch.IntTensor,
        stop_sequences_tensor: torch.IntTensor,
        start_token: Union[int, List[int]],
        max_tokens: Optional[int] = None,
    ) -> torch.IntTensor:
        """Experimental implcit autoregressive loop."""
        raise NotImplementedError(
            "Implicit autoregressive loop is not supported on "
            f"{self.backend_type} backend"
        )

    # Distributed Data Parallel
    def spawn(self, func):  # pylint: disable=no-self-use
        """Spawns a process on each GPU.

        Raises:
            A RuntimeError if called on a non-GPU backend
        """
        raise RuntimeError("Spawning is only supported on GPU backends")

    @cstorch.step_closure
    def _update_dataloader_state(self):
        """Update the cached dataloader state at every step.

        The dataloader state is only available at checkpointing steps. At all other steps,
        the state is unavailable. So if the same dataloader is used in another data
        executor, we check the state to ensure that we are "resuming" from a correct
        state.
        """
        if self.run_context.dataloader.is_restartable:
            # We always save the final step's state, even if checkpointing was disabled
            if (
                self.run_context.is_checkpoint_step
                or self.run_context.is_final_step
            ):
                # This caches the state in the dataloader to be used in subsequent steps
                _ = self.run_context.dataloader.state_dict()
            else:
                self.run_context.dataloader.cached_state = (
                    cstorch.utils.data.DataLoader.STATE_UNAVAILABLE
                )


def full_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Returns the full state dict of a module, including persistent buffers.

    This helper method is used to collect all buffers, parameters, and other
    extra states of a module, mostly for marking them as outputs/aliases in
    the graph. It is not to be used for checkpointing.
    """
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Expected a torch.nn.Module, got {type(module)}")

    # Persistent buffers are not included in the state dict but we need
    # to mark them as outputs/aliases, otherwise they won't show up in
    # the graph. `named_buffers` returns all buffers (including
    # non-persistent ones). Updating with `state_dict` will override
    # non-persistent buffers again with the same name, which is ok.
    state_dict = OrderedDict(module.named_buffers())
    state_dict.update(module.state_dict(keep_vars=True))
    return state_dict
