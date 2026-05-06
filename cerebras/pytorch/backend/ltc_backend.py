# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" The PyTorch/LTC backend implementation. """
import atexit
import inspect
import os
import sys
import time
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from functools import cached_property, lru_cache
from pathlib import Path
from pprint import pformat
from tempfile import TemporaryDirectory
from threading import Event
from typing import Callable, Dict, List, Optional, Set, Union
from weakref import finalize

import grpc
import torch
import torch._lazy  # pylint: disable=import-error
from torch.utils.hooks import RemovableHandle

import cerebras.pytorch as cstorch
from cerebras.appliance import DEFAULT_COMPILE_DIR
from cerebras.appliance.appliance_client import fw_user_serialize
from cerebras.appliance.cluster.client import ClusterManagementClient
from cerebras.appliance.cluster.config import ClusterConfigError
from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.appliance.pb.framework.appliance_service_pb2 import RunRequest
from cerebras.appliance.pb.ws.common_pb2 import WS_RT_JOB_INCOMPATIBLE
from cerebras.appliance.pb.ws.common_pb2 import Schedule as PbSchedule
from cerebras.appliance.pb.ws.cross_compile_state_pb2 import CrossCompileState
from cerebras.appliance.storage.s3_storage import S3Interface
from cerebras.appliance.utils.file import create_symlink
from cerebras.appliance.utils.global_perf_view import GlobalPerfView
from cerebras.appliance.utils.traceback import get_lowering_error_from_json
from cerebras.pytorch.amp import init as amp_init
from cerebras.pytorch.amp._amp_state import _amp_state, is_cbfloat16_tensor
from cerebras.pytorch.backend.base_backend import (
    COMPILE_ONLY_MSG,
    COMPILE_SUCCESS_MSG,
    PROGRAMMING_CS_MSG,
    BaseBackend,
)
from cerebras.pytorch.checks import add_checks
from cerebras.pytorch.core.appliance import ApplianceMode
from cerebras.pytorch.core.constants import INPUT_NAME_PREFIX, STATE_NAME_PREFIX
from cerebras.pytorch.core.device import LazyDevice
from cerebras.pytorch.core.modes import EVAL
from cerebras.pytorch.core.name_scope import ScopeName
from cerebras.pytorch.cs_op_lib.utils.validate_torch_mlir import check_mlir_text
from cerebras.pytorch.lib import cerebras_pytorch_lib
from cerebras.pytorch.storage.serializers import (
    DeferredGraphTensor,
    DeferredTorchTensor,
)
from cerebras.pytorch.storage.utils import lazy_tensor_data_wrapper
from cerebras.pytorch.utils.nest import visit_device_tensors


@named_class_logger("LtcBackend")
class PyTorchLtcBackendImpl(BaseBackend, ClassLogger):
    """The backend subclass for PyTorch/LTC runs."""

    def __init__(
        self,
        backend_type,
        artifact_dir: Optional[str] = None,
        compile_dir: str = DEFAULT_COMPILE_DIR,
        compile_only: bool = False,
        validate_only: bool = False,
        cluster_config: Optional[cstorch.distributed.ClusterConfig] = None,
    ):
        if artifact_dir is None:
            artifact_dir = Path.cwd().joinpath("cerebras_logs")

        super().__init__(backend_type, LazyDevice(), artifact_dir)

        self.compile_dir = compile_dir or DEFAULT_COMPILE_DIR

        self.compile_only = compile_only
        self.validate_only = validate_only

        if cluster_config is None:
            cluster_config = cstorch.distributed.ClusterConfig()
        if cluster_config.num_csx <= 0:
            raise ValueError(
                f"num_csx in the cluster config must be a positive integer, got {cluster_config.num_csx}"
            )

        self.cluster = ClusterManager(cluster_config)

        if (
            compile_only
            or validate_only
            or cstorch.backends.csx.debug.drop_data
        ):
            # We can drop any tensor data that already exists as soon as it's
            # moved to the device
            # That is, trace the initialization, but don't actually initialize
            # the weights
            cstorch.backends.csx.debug.drop_data = True

        self.appliance = None
        self._appliance_execute_cleanup_stack = None

        self._compile_appliance = None

        self._has_worker_image = False

        self._flags = None

        # A dictionary to store the compile cache. The key is the hash of the
        # compile that includes cross_compile_state, cirh and debug_args. The value is the
        # compile response.
        self._compile_cache = {}

        self.appliance_gpv = GlobalPerfView()

        debug = bool(int(os.environ.get("CSTORCH_DEBUG", "1")))

        cerebras_pytorch_lib.initialize(ir_debug=debug)
        atexit.register(self.shutdown)

        if debug:
            os.environ["LTC_IR_DEBUG_ROOT_PATH"] = ":".join(
                # sys.path entries in order from longest to shortest
                sorted(
                    (path + "/" for path in sys.path if path),
                    key=lambda x: -len(x),
                )
            )

        num_threads_str = os.environ.get("TORCH_NUM_THREADS", "1")
        try:
            num_threads = int(num_threads_str)
            if num_threads < 1:
                raise ValueError(
                    "Number of threads must be a positive integer."
                )
        except ValueError as e:
            e.add_note("TORCH_NUM_THREADS must be a positive integer.")
            raise

        # Set the number of OMP threads to 1 by default to avoid resource
        # contention when multiple processes are running on the same node
        os.environ.setdefault("OMP_NUM_THREADS", num_threads_str)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", num_threads_str)
        os.environ.setdefault("MKL_NUM_THREADS", num_threads_str)
        torch.set_num_threads(num_threads)

        self.logger.debug(torch.__config__.parallel_info())

        # Always enable mixed precision for CSX runs
        cstorch.amp.enable_mixed_precision()

        self.logger.verbose("Running using LTC backend")

        # add prehook checks
        add_checks(self.device.type)

        # setup pytorch seed and manual_seed hooks
        self.setup_seed_hooks()

    @property
    def is_e2e_execution(self) -> bool:
        """Returns True if the backend is configured for end-to-end execution."""
        return not (self.compile_only or self.validate_only)

    @property
    def cluster_config(self) -> cstorch.distributed.ClusterConfig:
        return self.cluster.config

    def _generate_tensor_names(
        self, prefix: str, tensors: list, delimiter: str
    ):
        for scope, tensor in visit_device_tensors(
            data_structure=tensors,
            device_type=self.torch_device.type,
            scope=[prefix] if prefix else None,
        ):
            yield delimiter.join(scope), tensor

    def _generate_state_names(self, tensors: list):
        yield from self._generate_tensor_names(
            STATE_NAME_PREFIX,
            tensors,
            '.',
        )

    def _generate_input_names(self, tensors: list):
        yield from self._generate_tensor_names(
            INPUT_NAME_PREFIX,
            tensors,
            '_',
        )

    def _generate_output_names(self, tensors: list):
        yield from self._generate_tensor_names(
            "output",
            tensors,
            '_',
        )

    def mark_output(self, struct, force=False):
        name_mapping = {}
        for name, tensor in self._generate_output_names(struct):
            name_mapping[id(tensor)] = name

        def map_fn(arg):
            if isinstance(arg, torch.Tensor) and (
                arg.device.type == self.torch_device.type
            ):
                name = name_mapping[id(arg)]

                # a reshape to ensure that the mark output gets sent to function mode
                if cstorch.backends.csx.debug.retrace_every_iteration:
                    arg.reshape(arg.size())

                # Remove after SW-120798.
                if is_cbfloat16_tensor(arg):
                    if cstorch.backends.csx.debug.implicit_cast_cb16_to_fp32:
                        arg = arg.float()
                    else:
                        raise RuntimeError(
                            "Tensor has an underlying dtype of `CBFloat16`. This tensor is"
                            " currently not representable in PyTorch and therefore cannot"
                            " be converted to a CPU tensor. To represent this tensor "
                            "without loss of precision or range, either explicitly cast to"
                            " float32 using `tensor.to(torch.float32)` or set "
                            "`cstorch.backends.csx.debug.implicit_cast_cb16_to_fp32 = True`"
                            " for the cast to be automatically done for you."
                        )

                # This might return a new tensor
                # pylint: disable=c-extension-no-member
                return cerebras_pytorch_lib.mark_output_tensor(
                    arg, name=name, force=force
                )

            return arg

        return torch.utils._pytree.tree_map(map_fn, struct)

    ################################################
    #               DataLoader hooks               #
    ################################################

    def initial_mark_step(self):
        """Run the initial mark step."""

        if self.device.config.drop_data or self.device.config.skip_weight_init:
            msg = "Skipping weight initialization"
            if not self.is_e2e_execution:
                msg += " as the backend was configured for compile/validation only."
            self.logger.info(msg)

        # Sync all functional tensors so that if any of their views
        # were updated inplace, the updates are visible to the original tensor
        # After syncing the tensors, reset the functional storage so that
        # the base tensor is the same as the latest value tensor
        for tensor in self.device.functional_tensors:
            cerebras_pytorch_lib.sync_functional_tensor(tensor)
            cerebras_pytorch_lib.reset_functional_tensor(tensor)

        self.logger.trace("Calling initial mark step")

        # Cleanup any cached computation and outputs that might
        # have been left over from previous runs
        cerebras_pytorch_lib.clear_ltc_cached_computation()
        cerebras_pytorch_lib.clear_cached_computation()
        cerebras_pytorch_lib.clear_cached_outputs()

        # Call initial mark_step to trigger asynchronous lazy initialization
        # pylint: disable=protected-access
        with self.device:
            torch._lazy.mark_step()

        self.logger.trace("Finished initial mark step")

    def set_tensor_name(self, tensors: list, names_generator, is_param):
        for name, tensor in names_generator(tensors):
            # pylint: disable=protected-access,c-extension-no-member
            self.logger.debug(
                f"Setting name {name} for tensor {cerebras_pytorch_lib.get_tensor_info(tensor)}"
            )

            if not cerebras_pytorch_lib.set_parameter_name(tensor, name):
                raise RuntimeError(
                    f"Failed to set name \"{name}\" for tensor: "
                    f"{cerebras_pytorch_lib.get_tensor_info(tensor)}"
                )

            if is_param:
                self._param_names.add(name)

    def register_active_model(self, model, model_id):
        super().register_active_model(model, model_id)
        state_dict = self.state_dict()
        self.set_tensor_name(
            {"model": state_dict["model"]},
            self._generate_state_names,
            is_param=True,
        )

    def on_run_start(self):
        self.appliance_tracker.stop("Initialization")

        super().on_run_start()

        # From debug_args get whether to run in strict soundness or not.
        cerebras_pytorch_lib.strict_soundness_mode(
            cstorch.backends.csx.debug.debug_args.debug_usr.strict_soundness_mode
        )
        if (
            cstorch.backends.csx.debug.debug_args.debug_usr.strict_soundness_mode
        ):
            self.logger.info(f"Running with strict_soundness_mode ON")

        # Reset flags to ensure that we aren't comparing flags
        # from a previous run
        self._flags = None

        # pylint: disable=protected-access,c-extension-no-member
        cerebras_pytorch_lib.get_appliance().artifact_dir = str(
            self.data_executor.artifact_dir
        )

        # initialize automatic mixed precision
        amp_init(verbose=(_amp_state.verbosity == 2))

        # Create the run request to be used in execute
        run_request = RunRequest(
            num_iterations=self.run_context.num_steps,
            activation_freq=self.run_context.activation_steps,
            live_dataloaders=list(self._dataloaders.keys()),
            dataloader=RunRequest.DataLoaderConfig(
                id=self.run_context.dataloader.id,
                builder=fw_user_serialize(
                    self.run_context.dataloader.input_fn,
                    name="DataLoader function",
                    from_usr=True,
                    recurse=True,
                ),
                builder_inputs=fw_user_serialize(
                    self.run_context.dataloader.input_fn_params,
                    name="DataLoader input arguments",
                    from_usr=True,
                    recurse=True,
                ),
            ),
        )

        for interval in self.run_context.checkpoint_schedule.intervals:
            run_request.checkpoint_schedule.intervals.append(
                PbSchedule.Range(
                    start=interval.start,
                    end=interval.end,
                    step=interval.step,
                    include_last=interval.include_last,
                )
            )

        if self.run_context.dataloader.is_restartable:
            state = self.run_context.dataloader.cached_state
            if state == cstorch.utils.data.DataLoader.STATE_UNAVAILABLE:
                raise RuntimeError(
                    "DataLoader state is not available. This can happen when the "
                    "DataExecutor using this DataLoader instance was not fully iterated "
                    "in a previous execution or it was stopped at a step other than a "
                    "checkpoint step. To avoid this, please fully iterate the DataExecutor "
                    "from previous execution."
                )
            elif state != cstorch.utils.data.DataLoader.STATE_UNKNOWN:
                run_request.dataloader.initial_state = fw_user_serialize(
                    state,
                    name="DataLoader state",
                    from_usr=True,
                )

        if self.appliance is None:
            self.appliance = self.create_appliance()

        cerebras_pytorch_lib.set_fp16_type(cstorch.amp.get_half_dtype_str())

        terminate = Event()

        # Dictionary of all materialized tensors that needs to be send to the appliance.
        initial_state_dict = {}
        # List of all weights that needs to be carried over from previous session to the new one.
        appliance_weights = {}

        # pylint: disable=redefined-builtin
        def compile(
            batch_size: int,
            torch_str: str,
            cirh_str: str,
            weights,
        ) -> bool:
            if batch_size < self.cluster_config.num_csx:
                raise ValueError(
                    f"The number of CS devices ({self.cluster_config.num_csx}) must be "
                    f"less than or equal to the model batch size ({batch_size}). Please "
                    "update your model batch size or reduce number of CSX systems to use."
                )

            if cstorch.backends.csx.debug.check_constraints:
                constraint_check_results = check_mlir_text(
                    torch_str,
                    cstorch.backends.csx.debug.op_definition_module,
                )
                if constraint_check_results:
                    error_lines = [
                        "The following Cerebras constraints were violated:"
                    ]
                    for op, reason in constraint_check_results.items():
                        error_lines.append(f"Op: {op}, Reason: {reason}")
                    raise RuntimeError("\n".join(error_lines))

            def is_s3(t):
                return isinstance(
                    t, DeferredTorchTensor
                ) and S3Interface.is_valid_path(t.deferred._path)

            if self.is_e2e_execution and any(
                is_s3(t)
                or (
                    isinstance(t, DeferredGraphTensor)
                    and any(is_s3(lazy_tensor_data_wrapper(a)) for a in t._args)
                )
                for t in (
                    lazy_tensor_data_wrapper(w)
                    for w in weights
                    if w.get_appliance_info() is None
                )
            ):
                self.cluster.check_storage_connectivity()

            if self.is_e2e_execution:
                for weight in weights:
                    # We need to carry over only the weights that were not changed beween sessions.
                    # If a weight has appliance info, it means that tensor is available in appliance
                    # and it was not changed between sessions. Otherwise, modified tensor will replace
                    # appliance info with graph info.
                    # In case of the first session run, all tensors will be materialized and will be
                    # sent to the appliance within initial ckpt.
                    appliance_info = weight.get_appliance_info()
                    if not appliance_info:
                        self.logger.debug(
                            f"Weight {weight.name} has materialized tensor: {weight}"
                        )
                        initial_state_dict[weight.name] = weight
                        continue

                    if not weight.is_weight:
                        raise RuntimeError(
                            f"Cerebras backend has detected that activation tensor \'{weight.name}\' "
                            f"was reused between executions which is currently unsupported."
                        )

                    if (
                        appliance_info.state
                        != cerebras_pytorch_lib.ApplianceInfoState.InRepo
                    ):
                        raise RuntimeError(
                            f"Weight tensor \"{weight.name}\" with tensor_id={appliance_info.uid},"
                            f"state={appliance_info.state} is not available in the repository, so "
                            f"it can not be carried over to the new execution."
                        )

                    appliance_weights[weight.name] = weight

            self.logger.info(COMPILE_ONLY_MSG)

            with self.appliance.build_worker_image(
                should_skip=not self.is_e2e_execution or self._has_worker_image,
            ):
                try:
                    self._has_worker_image = True

                    cross_compile_state = CrossCompileState()
                    # Propagate cross compile state iff the active model
                    # is unchanged. Otherwise, we will restart the execute
                    # job below which shouldn't be bound by any previous
                    # cross compile states.
                    # TODO: if active model changed, stop the previous job
                    # here instead of after compile and before execute.
                    if (
                        self.appliance.compile_resp is not None
                        and not self._active_model_changed()
                    ):
                        cross_compile_state = (
                            self.appliance.compile_resp.cross_compile_state
                        )

                    compile_hash = self.appliance.compute_compile_hash(
                        cirh_str, batch_size, cross_compile_state
                    )

                    if compile_hash not in self._compile_cache:
                        # Instantiate new ApplianceMode for compile job only.
                        self._compile_appliance = ApplianceMode(
                            self.data_executor.artifact_dir,
                            self.compile_dir,
                            self.cluster_config,
                            op_profiler_config=self.data_executor.op_profiler_config,
                        )
                        # Set the sidecar image in the compile appliance to facilitate
                        # more robust image rotation in registry.
                        sidecar_image = self.appliance.get_sidecar_image()
                        self._compile_appliance.set_sidecar_image(sidecar_image)
                        self.appliance.compile_resp = (
                            self._compile_appliance.compile(
                                batch_size,
                                cirh_str,
                                cross_compile_state,
                                self.validate_only,
                            )
                        )
                        if self.appliance.compile_resp is not None:
                            # Save compile hash which includes previous CrossCompileState,
                            # current model and appliance artifacts.
                            self._compile_cache[compile_hash] = (
                                self.appliance.compile_resp
                            )
                            # Save compile hash which includes current model with appliance
                            # artifacts and generated CrossCompileState produced by compile.
                            self._compile_cache[
                                self.appliance.compute_compile_hash(
                                    cirh_str,
                                    batch_size,
                                    self.appliance.compile_resp.cross_compile_state,
                                )
                            ] = self.appliance.compile_resp
                    else:
                        self.logger.info(f"Found existing cached compile.")
                        self.appliance.compile_resp = self._compile_cache[
                            compile_hash
                        ]

                    # Check if exists as it won't for system runs
                    if self.appliance.compile_resp and os.path.exists(
                        self.appliance.compile_resp.cache_compile_dir
                    ):
                        create_symlink(
                            self.data_executor.artifact_dir.joinpath(
                                "remote_compile_dir"
                            ),
                            self.appliance.compile_resp.cache_compile_dir,
                        )

                    # Reset the compile appliance so we don't accidentally
                    # use it for anything else.
                    self._compile_appliance = None
                except Exception:
                    terminate.set()
                    raise

            self.logger.info(COMPILE_SUCCESS_MSG)
            return True

        def execute(batch_size: int) -> Set[str]:
            if not self.is_e2e_execution:
                return set()

            def restart():
                # Since we restart the appliance, we no longer have appliance weights, so
                # `appliance_weights` needs to be moved to `initial_state_dict`.
                nonlocal appliance_weights
                initial_state_dict.update(appliance_weights)
                appliance_weights = {}

                self.restart_appliance()

            # Check if model has changed or resource check failed, so we restart the appliance.
            if self._active_model_changed():
                self.logger.info(
                    f"Active model has changed from {self.appliance._active_model_id} to "
                    f"{self._active_model_id}. Restarting the appliance."
                )
                restart()
            elif self._appliance_execute_cleanup_stack is not None:
                resp = self.appliance.check_compile_compatibility()
                if resp.code == WS_RT_JOB_INCOMPATIBLE:
                    self.logger.info(
                        f"Restarting the appliance due to: \"{resp.message}\"."
                    )
                    restart()

            if self.mode is None:
                # This means that the user did not call optimizer.step()
                # So, assume that the user wants to run eval
                self.mode = EVAL

                if self.model and self.model.training:
                    self.logger.warning(
                        "Model is in training mode but no optimizer.step() "
                        "call was detected. The model will be compiled for "
                        "eval mode but numerics may be affected if ops "
                        "like dropout are present in the model."
                    )

            if self._appliance_execute_cleanup_stack is None:
                self._appliance_execute_cleanup_stack = ExitStack()
                self._appliance_execute_cleanup_stack.__enter__()

            self.logger.debug(
                f"Initialized weights {initial_state_dict.keys()}"
            )
            self.logger.debug(f"Carried over weights {appliance_weights}")

            self.logger.info(PROGRAMMING_CS_MSG)

            if cstorch.backends.csx.debug.save_gpv:
                self.appliance_gpv.start_session()
            self.appliance.execute(
                run_request,
                self._appliance_execute_cleanup_stack,
                initial_state_dict,
                appliance_weights,
                has_modified_seed=self._has_modified_seed,
            )
            self._has_modified_seed = False

            self.appliance._active_model_id = self._active_model_id

            # Manually update the skipped weights
            self.appliance.skipped_weights.update(
                self._param_names
                - set(initial_state_dict.keys())
                - set(appliance_weights.keys())
            )
            self.logger.debug(
                f"Assigning skipped weights: {self.appliance.skipped_weights}"
            )

            return self.appliance.skipped_weights

        def get_tensor(name, iteration, appliance_info):
            if self.appliance is None:
                raise RuntimeError(
                    "Trying to fetch tensor from appliance before it is initialized"
                )

            self.logger.debug(
                f"Fetching tensor {name} from the appliance on {iteration=}"
            )

            if (
                appliance_info.state
                == cerebras_pytorch_lib.ApplianceInfoState.InRepo
            ):
                tensor = self.appliance.get_from_ptr(
                    name, appliance_info.uid, keep_in_repo=True
                )
            elif (
                appliance_info.state
                == cerebras_pytorch_lib.ApplianceInfoState.InBuffer
            ):
                tensor = self.appliance.receive_output(iteration, name)
            else:
                raise RuntimeError(
                    f"The tensor \"{name}\" was dropped, so it can "
                    f"not be fetched from the appliance."
                )

            try:
                # Make the tensor writable so that we don't have to copy it
                # in `cstorch.from_numpy()`. Some arrays cannot be modified
                # so we ignore the error and copy the array instead.
                tensor.flags.writeable = True
            except Exception:  # pylint: disable=broad-except
                pass

            tensor = cstorch.from_numpy(tensor)

            if list(tensor.shape) != appliance_info.shape:
                raise RuntimeError(
                    f"The shape of the tensor from the appliance {list(tensor.shape)} "
                    f"does not match the original shape of the tensor {appliance_info.shape}. "
                    f"This indicates an internal bug. Please contact Cerebras Support for help."
                )

            # Update ApplianceInfo storage with materialized tensor.
            # This part specifically moved to the python, so we can
            # use device context to avoid dropping the tensor. This
            # may happen when lazy initialization is enabled, so the
            # get_tensor call may outlive device context used for init
            # mark_step.
            with self.device:
                appliance_info.storage.share_storage(
                    cerebras_pytorch_lib.ApplianceDataInfo.create(tensor=tensor)
                )

        def delete_tensor(
            appliance_info: cerebras_pytorch_lib.ApplianceDataInfo,
        ):
            if (
                self.appliance
                and appliance_info.state
                == cerebras_pytorch_lib.ApplianceInfoState.InRepo
                and self.is_e2e_execution
            ):
                self.appliance.delete_from_ptr(appliance_info.uid)

        # pylint: disable=c-extension-no-member
        cerebras_pytorch_lib.get_appliance().set_callbacks(
            compile_callback=compile,
            execute_callback=execute,
            get_tensor_callback=get_tensor,
            release_tensor_callback=delete_tensor,
        )

        self.initial_mark_step()

        self.run_step_closures()

        self._param_names = set()
        self.set_tensor_name(
            self.state_dict(), self._generate_state_names, is_param=True
        )

    def on_run_end(self, exec_type=None, exec_value=None, traceback=None):
        # If user breaks out of data_executor loop, `on_batch_end` is not called,
        # so we need to set tracing to False here just in case.
        cerebras_pytorch_lib.is_tracing_step(False)

        if (
            self._appliance_execute_cleanup_stack is not None
            and exec_type is None
        ):
            response = self.appliance.finalize()
            if cstorch.backends.csx.debug.save_gpv:
                wsjob = self.cluster.client.get_latest_execute_job()
                wsjob_id = wsjob.get("id", None) if wsjob else None

                wsjob_dashboard = None
                try:
                    wsjob_dashboard = self.cluster.client.get_job(
                        wsjob_id
                    ).dashboard
                except grpc.RpcError:
                    pass

                self.appliance_gpv.end_session(
                    response.perf,
                    wsjob_dashboard,
                    self.data_executor.artifact_dir / "global_perf_summary.txt",
                )

            if response.trace_events and len(response.trace_events) > 0:
                from cerebras.pytorch.profiler import ProfilerRegistry

                prof = ProfilerRegistry.get_profiler()
                prof.appliance_response = response
                prof._trace_ready()

        if exec_type:
            self.shutdown_appliance(
                exec_type=exec_type, exec_value=exec_value, traceback=traceback
            )

        # pylint: disable=import-error
        from torch._lazy.closure import AsyncClosureHandler

        async_closure_handler = AsyncClosureHandler()
        if async_closure_handler._closure_queue.qsize() > 0:
            self.logger.info("Waiting for async closures to finish running")
            async_closure_handler._closure_queue.join()

        self.step_closures = []

        # Save all availabe weights so we can later move them to the repo.
        if (
            not exec_type
            and self.appliance is not None
            and self.is_e2e_execution
        ):
            # Move all weight tensors to the persistance tensor repository, so they can be
            # accessed in the next session.
            for appliance_data in cerebras_pytorch_lib.get_cached_output_data():
                appliance_info = appliance_data.get_appliance_info()
                if appliance_data.is_weight and appliance_info is not None:
                    if (
                        appliance_info.state
                        != cerebras_pytorch_lib.ApplianceInfoState.InBuffer
                    ):
                        raise RuntimeError(
                            f"Tensor \"{appliance_data.name}\" with tensor_id={appliance_info.uid},"
                            f"state={appliance_info.state} is not available in runtime, so it can "
                            f"not be moved to tensor repository."
                        )

                    # Runtime doesn't support `move_to_ptr` for autoregressive runs.
                    # We also expect that in autoregressive run we don't have weights
                    # with updates, or if they present, they should be fetched at the
                    # runtime within a `cstorch.step_closure`.
                    if (
                        self.appliance.compile_resp.cross_compile_state.is_autoregressive
                    ):
                        if not appliance_data.is_tensor_available:
                            raise RuntimeError(
                                f"Autoregressive runs require all tensors to be fetched from the "
                                f"appliance. Please make sure that \'cstorch.step_closure\' is "
                                f"used for \"{appliance_data.name}\" tensor."
                            )

                        # Update storage from ApplianceInfo to Memory/File info since this tensor
                        # won't be available in the appliance and can not be carried over.
                        appliance_data.share_storage(appliance_info.storage)
                        appliance_info.state = (
                            cerebras_pytorch_lib.ApplianceInfoState.Dropped
                        )
                        continue

                    self.appliance.move_to_ptr(
                        appliance_data.name, appliance_info.uid
                    )

                    appliance_info.state = (
                        cerebras_pytorch_lib.ApplianceInfoState.InRepo
                    )

        # Cleanup all cached computation and outputs.
        cerebras_pytorch_lib.clear_ltc_cached_computation()
        cerebras_pytorch_lib.clear_cached_computation()
        cerebras_pytorch_lib.clear_cached_outputs()

        if not exec_type:
            # In the multisession run we may have a state between sessions where
            # some of the LTC tensors are still alive from previous session, so they
            # can interfere with following session. This happens when autograd is enabled
            # or if some of the python tensors (like loss) hold the reference to the
            # LTC tensors. To avoid this we need to cleanup all tensors before the next session.
            with cerebras_pytorch_lib.MarkStepContextManager(
                cerebras_pytorch_lib.MarkStepType.DUMMY
            ):
                torch._lazy.mark_step()

        if cstorch.backends.csx.debug.save_gpv:
            # Note: gpv needs to get TTFL stats from appliance_tracker before it
            # gets reset in BaseBackend.on_run_end
            self.appliance_gpv.add_session_init_stats(
                self.appliance_tracker.get_timestamps()
            )
        super().on_run_end(exec_type, exec_value, traceback)

    def on_batch_start(self, batch):
        batch = super().on_batch_start(batch)

        cerebras_pytorch_lib.is_tracing_step(True)

        # Clear amp cache for the next iteration
        # pylint: disable=protected-access
        _amp_state.handle._clear_cache()

        self.set_tensor_name(batch, self._generate_input_names, is_param=False)

        for optimizer in self.optimizer_registry:
            if hasattr(optimizer, "_amp_stash"):
                optimizer._amp_stash.dls_update_manager.__enter__()

        @cstorch.step_closure
        def gpv_add_first_step():
            # call this only at first step
            if self.run_context.user_iteration == 1:
                self.appliance_gpv.start_first_step()

        if cstorch.backends.csx.debug.save_gpv:
            gpv_add_first_step()

        return batch

    def mark_step(self, mark_step_execution_type):
        self.logger.trace("Calling Mark Step")
        # In case we had no tensors to sync during the initial mark_step,
        # we need to force regular mark_step here.
        try:
            with cerebras_pytorch_lib.MarkStepContextManager(
                mark_step_execution_type
            ):
                torch._lazy.mark_step()
        except RuntimeError as error:
            # check if it's the right type of error, if not just re raise the error
            if "MLIR Compile Lowering Exception" not in str(error):
                raise

            # try creating the diagnostic from the JSON, and if it errors out
            # report the original error instead
            try:
                _, error_json_str = str(error).split("\n", 1)
                diagnostic_error = get_lowering_error_from_json(error_json_str)
            except:
                pass
            else:
                # raise from none so we avoid showing the huge traceback json
                raise diagnostic_error from None

            raise

    def on_batch_end(self):
        @cstorch.trace
        def _finalize_optimizer():
            for optimizer in self.optimizer_registry:
                for scheduler in optimizer._schedulers_registry:
                    # The scheduler step should always be one greater than the optimizer
                    # step if the scheduler was stepped.
                    # If the scheduler was not stepped, we still need to update the group
                    # with the scalar values.
                    # If an scheduler was not stepped, its probably a user error.
                    # But we should still support this behaviour anyways as its
                    # supported in eager mode
                    if optimizer._step_count >= scheduler._step_count:
                        scheduler.update_groups(scheduler._last_value)
                if hasattr(optimizer, "_amp_stash") and (
                    cstorch.backends.csx.debug.retrace_every_iteration
                    or self.run_context.is_initial_step
                ):
                    optimizer._amp_stash.dls_update_manager.__exit__()

        # Run the operations in a traced context
        _finalize_optimizer()

        for name, tensor in self._generate_state_names(self.state_dict()):
            if name not in self._param_names:
                continue
            # The following set_alias call also marks the tensor as an output.
            # pylint: disable=protected-access,c-extension-no-member
            assert cerebras_pytorch_lib.set_alias(tensor, name), (
                f"failed to set alias {name} for tensor: "
                f"{cerebras_pytorch_lib.get_tensor_info(tensor)}"
            )

        self._update_dataloader_state()

        self._is_tracing = False
        cerebras_pytorch_lib.is_tracing_step(False)

        # pylint: disable=import-error
        # Seed the ltc backend for e.g. dropout. This doesn't influence model
        # initialization or dataloader shuffling.
        # Use the initial seed value set via torch.manual_seed()
        cerebras_pytorch_lib.set_rng_state(torch.initial_seed())

        cerebras_pytorch_lib.get_appliance().set_iteration(
            self.run_context.iteration
        )

        if self._flags is None:
            self._flags = dict(cstorch.backends)
        elif flag_diff := list(
            cstorch.utils.nest.diff(self._flags, dict(cstorch.backends))
        ):
            flag_diff = "\n\n".join(flag_diff)
            raise RuntimeError(
                f"Detected changes in the backend flags during the execution run. "
                f"This is unsupported behaviour. "
                f"The flags before the execution run were:\n{pformat(self._flags)}\n"
                f"and the current flags are:\n{pformat(dict(cstorch.backends))}\n"
                f"The differences are as follows:\n{flag_diff}"
            )

        # pylint: disable=protected-access
        if (
            cstorch.backends.csx.debug.retrace_every_iteration
            or self.run_context.is_initial_step
        ):
            self.mark_step(cerebras_pytorch_lib.MarkStepType.EXECUTION)

        # Update the profiler as we have processed a batch. Note that this is
        # done after mark_step so that we don't jump the gun and updated samples
        # processed before compile/execute is actually done.
        batch_size = self.run_context.dataloader.batch_size
        if self.run_context.profiler is not None:
            self.run_context.profiler.step(batch_size)

        # Workaround to handle the case when retracing is disabled and appliance info
        # insdie the tensor points to 0 iteration. So we make an assumption that the
        # iteration of the tensor is the same as the current iteration.
        if not cstorch.backends.csx.debug.retrace_every_iteration:
            for appliance_data in cerebras_pytorch_lib.get_cached_output_data():
                appliance_info = appliance_data.get_appliance_info()
                if not appliance_info or appliance_info.allows_updates:
                    # In case if current iteration is different from the iteration of the tensor,
                    # it will reset materialized tensor underneath, so the new tensor will be
                    # fetched from the appliance.
                    appliance_data.iteration = self.run_context.iteration

        @cstorch.step_closure
        def gpv_add_step(batch_size):
            # Log step to GPV at the end of step closures
            self.appliance_gpv.end_step(
                self.run_context.user_iteration,
                batch_size,
                self.run_context.is_checkpoint_step,
            )

        if cstorch.backends.csx.debug.save_gpv:
            gpv_add_step(batch_size)

            # This is a hack to make sure that gpv adds its step after we receive
            # the first activation, but before we take a checkpoint.
            if pos := next(
                (
                    i + 1
                    for i, (
                        closure,
                        args,
                        kwargs,
                        run_async,
                        repeat,
                    ) in enumerate(self.step_closures)
                    if any(
                        a.device.type == self.torch_device.type
                        for a in args
                        if isinstance(a, torch.Tensor)
                    )
                    or any(
                        v.device.type == self.torch_device.type
                        for v in kwargs.values()
                        if isinstance(v, torch.Tensor)
                    )
                ),
                None,
            ):
                closure = self.step_closures.pop()
                self.step_closures.insert(pos, closure)

        self.run_step_closures()

    def forward(self, model, *args, **kwargs):  # pylint: disable=no-self-use
        """Runs the forward pass for the model."""
        return model(*args, **kwargs)

    def set_scope_name(self, scope_name):
        old_scope = super().set_scope_name(scope_name)
        if scope_name is None:
            scope_name = ScopeName()
        cerebras_pytorch_lib.set_scope_name(str(scope_name))
        return old_scope

    def create_appliance(self):
        return ApplianceMode(
            self.data_executor.artifact_dir,
            self.compile_dir,
            self.cluster_config,
            op_profiler_config=self.data_executor.op_profiler_config,
        )

    def restart_appliance(self):
        """
        Fetch all weights that are available on the client side from
        PTR and restart the appliance.
        """
        for (
            appliance_data
        ) in cerebras_pytorch_lib.get_live_appliance_data_info():
            if not appliance_data.is_weight:
                continue

            appliance_info = appliance_data.get_appliance_info()
            if not appliance_info or (
                appliance_info.state
                != cerebras_pytorch_lib.ApplianceInfoState.InRepo
                and not appliance_data.is_tensor_available
            ):
                continue

            with self.device:
                # Fetch weight from appliance.
                if (
                    appliance_info.state
                    == cerebras_pytorch_lib.ApplianceInfoState.InRepo
                ):
                    appliance_data.tensor  # noqa

                # Update storage from ApplianceInfo to Memory/File info since this tensor
                # won't be available in the appliance and can not be carried over.
                appliance_data.share_storage(appliance_info.storage)

                # This appliance info no longer available in the appliance since we
                # restarting the execute job. So, we set `Dropped` state, so when the
                # `ApplianceInfo` is being destructed, we won't try to delete it from
                # the tensor repo.
                appliance_info.state = (
                    cerebras_pytorch_lib.ApplianceInfoState.Dropped
                )

        appliance = self.create_appliance()
        self.appliance.copy_to(appliance)
        self.shutdown_appliance()
        self.appliance = appliance

        # TODO: reuse the same worker image between appliances SW-121619.
        with self.appliance.build_worker_image(
            should_skip=not self.is_e2e_execution
        ):
            return True

    def shutdown_appliance(
        self, exec_type=None, exec_value=None, traceback=None
    ):
        """Shutdown the appliance."""
        if self._appliance_execute_cleanup_stack is not None:
            self._appliance_execute_cleanup_stack.__exit__(
                exec_type, exec_value, traceback
            )
            self._appliance_execute_cleanup_stack = None

        self.appliance = None

    def shutdown(self):
        """Shutdown the backend."""
        if cstorch.backends.csx.debug.save_gpv:
            self.appliance_gpv.end_run(self.artifact_dir / "gpv.pb")
        self.shutdown_appliance()
        cerebras_pytorch_lib.shutdown()

        if self.cluster.workflow_id is not None:
            self.cluster.stop_workflow()

    #######################################################
    #               Optimizer related hooks               #
    #######################################################

    def setup_optimizer(self, optimizer):
        super().setup_optimizer(optimizer)
        self.post_optimizer_load_state_dict(optimizer)

        optimizer.register_load_state_dict_post_hook(
            self.post_optimizer_load_state_dict
        )

        def set_value(optimizer, args, kwargs):
            # Set the value to be the tensor
            for scheduler in optimizer._schedulers_registry:
                scheduler.update_last_value()

        optimizer.register_step_pre_hook(set_value)

    def post_optimizer_load_state_dict(self, optimizer):
        """
        Post-process the optimizer param groups and state
        after loading the state dict.
        """

        def tensor_cast(value):
            if isinstance(value, torch.Tensor):
                # When we load the optimizer state dict, tensors may be moved to
                # device. But we don't want to trace param groups. So we move
                # them back to CPU here. We also ensure these tensors are explicitly
                # materialized.
                value = lazy_tensor_data_wrapper(value).to("cpu").detach()
            elif isinstance(value, int):
                value = torch.tensor(value, dtype=torch.int32, device="cpu")
            elif isinstance(value, float):
                value = torch.tensor(value, dtype=torch.float32, device="cpu")
            elif isinstance(value, (list, tuple)):
                value = type(value)(map(tensor_cast, value))
            return value

        # Convert all python scalars in the param groups to 32 bit torch tensors
        # This is because python int/float are represented as 64-bit scalars,
        # whereas compile can only handle 32-bit scalars.
        for param_group in optimizer.param_groups:
            keys = list(param_group.keys())
            for key in keys:
                if key == "params":
                    continue
                value = param_group.pop(key)
                param_group[key] = tensor_cast(value)

        # Make optimizer state tensors into appliance tensors. When we load a
        # normal torch checkpoint, it's loaded onto CPU. But optimizer state
        # needs to be on the device. Note that loading an optimizer state dict
        # replaces the state variables. This is in constrast to loading a model
        # state dict, which updates the state variables using `param.copy_()`.
        def make_appliance(value):
            if isinstance(value, torch.Tensor) and value.device.type != "lazy":
                return value.to(self.device.torch_device)
            return None

        with self.device:
            optimizer.visit_state(make_appliance)

    def setup_grad_scaler(self, grad_scaler):
        super().setup_grad_scaler(grad_scaler)

        with self.device:
            state_dict = {
                name: (
                    tensor.to(self.torch_device)
                    if isinstance(tensor, torch.Tensor)
                    else tensor
                )
                for name, tensor in grad_scaler.state_dict().items()
            }
        grad_scaler.load_state_dict(state_dict)

    def _get_cpu_tensor(self, arg: torch.Tensor):
        """Get a CPU tensor from the appliance."""
        # pylint: disable=c-extension-no-member
        name = cerebras_pytorch_lib.get_tensor_name(arg)

        if cerebras_pytorch_lib.is_weight_tensor(arg):
            raise RuntimeError(
                f"Attempting to get weight tensor \"{name}\" with info "
                f"{cerebras_pytorch_lib.get_tensor_info(arg)} in a step "
                f"closure but this is not supported yet. Please use "
                f"\"cstorch.save()\" API to save model weights."
            )

        for hook in _before_tensor_to_cpu_hooks.values():
            hook(arg, name)

        # Getting tensor from the appliance may create underlying memory or file backed tensor,
        # so we need to make the following call using device context not to drop the tensor.
        with self.device:
            tensor = cerebras_pytorch_lib.get_appliance_data(arg).tensor

        for hook in _after_tensor_to_cpu_hooks.values():
            hook(tensor, name)

        return tensor

    def set_attribute(
        self,
        tensor: torch.Tensor,
        attribute: str,
        value: Union[bool, int, float, str, list, dict],
    ):
        """
        Adds an attribute to the traced tensor at compile time to communicating
        with the Cerebras Compiler Stack.

        Args:
            tensor: A tensor on the backend device.
            attribute: Name of the attribute to set
            value: Value of the attribute to set.
        """

        # These attributes eventally land in MLIR attributes, potentially on
        # the arguments to the main function. MLIR requires such attributes be
        # scoped to a dialect, so ensure the attribute name is prefixed with
        # `cs.`
        name = "cs." + attribute

        from cerebras.pytorch.lib import cerebras_pytorch_lib

        cerebras_pytorch_lib.set_attribute(tensor, name, value)

    #################################################
    #               Appliance related               #
    #################################################

    def add_step_closure(
        self,
        closure,
        args,
        kwargs,
        run_async: bool = False,
        repeat: bool = False,
    ):
        if hasattr(closure, "__wrapped__"):
            pos_arg_names = inspect.getfullargspec(closure.__wrapped__).args
        else:
            pos_arg_names = inspect.getfullargspec(closure).args

        if len(pos_arg_names) == len(args) and not any(
            pos_arg_name in kwargs for pos_arg_name in pos_arg_names
        ):
            # Use the names of the positional arguments in the step closure as
            # the output name.
            kwargs.update(dict(zip(pos_arg_names, args)))
            kwargs = self.mark_output(kwargs, force=True)
            # Strip positional arguments back out
            args = type(args)(
                kwargs.pop(arg_name) for arg_name in pos_arg_names
            )
        else:
            # Use anonymous positional arguments
            args, kwargs = self.mark_output((args, kwargs), force=True)

        self.step_closures.append((closure, args, kwargs, run_async, repeat))

    def run_step_closures(self):
        step_closures = self.step_closures
        self.step_closures = []

        if self.compile_only or self.validate_only:
            self.logger.debug(
                f"Skipping running step closures since backend is configured "
                f"for {'compile' if self.compile_only else 'validate'}_only "
                f"mode."
            )
            return

        # pylint: disable=import-error
        from torch._lazy.closure import AsyncClosureHandler

        async_closure_handler = AsyncClosureHandler()

        for closure, args, kwargs, run_async, repeat in step_closures:
            if self.run_context.is_activation_step:
                # fetching tensors from appliance here
                # pylint: disable=protected-access
                cpu_args, cpu_kwargs = torch.utils._pytree.tree_map(
                    lambda arg: (
                        self._get_cpu_tensor(arg)
                        if isinstance(arg, torch.Tensor)
                        and arg.device.type == self.torch_device.type
                        else arg
                    ),
                    (args, kwargs),
                )

                if run_async:
                    async_closure_handler.run(
                        lambda c=closure, a=cpu_args, k=cpu_kwargs: c(*a, **k)
                    )
                else:
                    closure(*cpu_args, **cpu_kwargs)
            else:
                self.logger.trace(
                    f"Skipping step closure at iteration {self.run_context.user_iteration} as it "
                    f"is not an activation step."
                )

            if repeat:
                self.step_closures.append(
                    (closure, args, kwargs, run_async, repeat)
                )

    def to_cpu(self, tensor):
        if tensor.device.type == "cpu":
            return tensor

        if not self.data_executor_stack or self.run_context.is_pre_initial_step:
            # If we are on the first step, we don't need to fetch the
            # tensor from the appliance since it is already the initial
            # tensor value (initial weights).
            # pylint: disable=protected-access,c-extension-no-member
            from cerebras.pytorch.storage.utils import lazy_tensor_data_wrapper

            return lazy_tensor_data_wrapper(tensor)

        # recv_weight returns a numpy array, so we need to convert it to a tensor
        array = self.appliance.recv_weight(
            tensor,
            self.run_context.iteration,
        )
        return cstorch.from_numpy(array)

    def save_tensors(self, tensors, path):
        if not self.data_executor_stack or self.run_context.is_pre_initial_step:
            if not self.data_executor_stack:
                # If we are not within a data executor, we need to call
                # mark step before saving any tensors so that we can
                # actualize any tensors that are not yet available.
                self.initial_mark_step()

            # If we are on the first step, we don't need to fetch the
            # tensor from the appliance since it is already the initial
            # tensor value (initial weights).
            yield from tensors
            return

        from cerebras.appliance.storage.s3_storage import S3Writer
        from cerebras.appliance.storage.serializers import DeferredObject
        from cerebras.pytorch.storage.serializers import DeferredTorchTensor

        if S3Writer.is_valid_path(path):
            self.cluster.check_storage_connectivity()

            lazy_tensors = {}
            for key, tensor in tensors:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.device.type != "lazy"
                ):
                    yield key, tensor
                else:
                    lazy_tensors[key] = tensor

            parsed = S3Writer.parse_path(path)

            for k, weight in self.appliance.save_weights_to_s3(
                lazy_tensors,
                path,
                parsed["bucket"],
                parsed["key"],
                self.run_context.iteration,
            ):
                t = lazy_tensors[k]
                yield k, (
                    DeferredTorchTensor(weight, t.shape, t.dtype)
                    if isinstance(weight, DeferredObject)
                    else weight
                )

        else:
            yield from tensors

    def setup_seed_hooks(self):
        """
        Hook torch.manual_seed and torch.seed so we can detect if seed
        was changed between runs, so we do not carry over the CMD state.
        """
        self._has_modified_seed = False

        _orig_manual_seed = torch.manual_seed

        def _manual_seed_hook(*args, **kwargs):
            self._has_modified_seed = True
            return _orig_manual_seed(*args, **kwargs)

        torch.manual_seed = _manual_seed_hook

        _orig_seed = torch.seed

        def _seed_hook(*args, **kwargs):
            self._has_modified_seed = True
            return _orig_seed(*args, **kwargs)

        torch.seed = _seed_hook

    def start_implicit_loop(
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

        shape = list(input_tensor.shape)
        rank = len(shape)
        if not isinstance(loop_dim, int):
            raise TypeError(
                f"loop_dim must be an integer. Got: {type(loop_dim)}"
            )
        elif not ((1 - rank) <= loop_dim < rank):
            raise ValueError(
                f"Expected {1 - rank} <= loop_dim < {rank}. Got: {loop_dim}"
            )
        if loop_dim < 0:
            loop_dim = loop_dim + rank
            # This is a sanity check
            assert loop_dim >= 0

        # e.g. for loop_dim = 1
        # [B][S] -> [B][1] or
        # [B][S][N] -> [B][1]
        del shape[loop_dim:]
        shape.append(1)

        # Configure attributes marking this as a non-input tensor that will be
        # synthesized by the runtime itself.
        input_name = cerebras_pytorch_lib.get_tensor_name(input_tensor)

        # Initialize index to zeros.
        idx = torch.zeros(shape, dtype=torch.int32).to(input_tensor.device)

        # TorchMlir hardcodes .startswith("input")...
        index_name = input_name.replace(
            "input", "inputautoregressive_index_over", 1
        )
        cerebras_pytorch_lib.set_parameter_name(idx, index_name)
        self.set_attribute(idx, "autoregressive_index", True)
        return idx

    def update_implicit_loop(
        self,
        input_tensor: torch.IntTensor,
        index_tensor: torch.IntTensor,
        update_tensor: torch.IntTensor,
        stop_sequences_tensor: torch.IntTensor,
        start_token: Union[int, List[int]],
        max_tokens: Optional[int] = None,
    ) -> torch.IntTensor:
        """
        Experimental implcit autoregressive loop. Configures the runtime inner
        loop via attributes.

        Args:
            input_tensor: Each step, the ``update_tensor`` will populate the
                         ``loop_dim`` slice of this input tensor at position
                         ``index_tensor + 1``. The final value is returned.
            index_tensor: The tensor returned from start_implict_loop.
            update_tensor: This tensor will be inserted into input_tensor at
                           the subsequent position along ``loop_dim`` in
                           ``input_tensor`` each inner-step. It should be the
                           same shape and type as ``input_tensor`` except the
                           extend of the ``loop_dim`` should be  1.
            stop_sequences_tensor: For LM autoregessive use, this tensor holds the list of
                            stop token sequences that, if seen in the output, marks
                            the end of generation; i.e. the inner autoregressive loop
                            is exited.
            start_token: For LM autoregessive use, this token in the input
                         marks the beginning of generation. All tokens before
                         it are left unmodified.
            max_tokens: If given, only this many tokens will be generated
                        before stopping.

        Returns:
            The final "modified" version of ``input_tensor`` with all updates
            made.
        """

        def validate_tensor(tensor, name_prefix):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Expected {name_prefix}_tensor to be a torch Tensor. "
                    f"Got: {type(tensor)}"
                )
            elif (dtype := tensor.dtype) != torch.int32:
                raise TypeError(
                    f"Expected data type of {name_prefix}_tensor to be "
                    f"`torch.int32`, got {dtype}"
                )

        validate_tensor(input_tensor, "input")
        validate_tensor(index_tensor, "index")
        validate_tensor(update_tensor, "update")
        validate_tensor(stop_sequences_tensor, "stop_sequences")

        loop_dim = int(index_tensor.dim() - 1)

        expected_update_shape = list(input_tensor.shape)
        expected_update_shape[loop_dim] = 1
        if list(update_tensor.shape) != expected_update_shape:
            raise ValueError(
                f"update_tensor must be the shape of one slice of "
                f"input_tensor: {expected_update_shape}. Got: "
                f"{update_tensor.shape}"
            )

        start_tok_err = "start_token must be non-negative integer or list of non-negative integers."
        if isinstance(start_token, list):
            if len(start_token) == 0:
                raise ValueError(f"{start_tok_err} Got empty list")
            for t in start_token:
                if not isinstance(t, int) or t < 0:
                    raise ValueError(f"{start_tok_err} One element was {t}")
        elif not isinstance(start_token, int) or start_token < 0:
            raise ValueError(f"{start_tok_err} Got: {start_token}")
        else:
            start_token = [start_token]

        if not (
            max_tokens is None
            or (isinstance(max_tokens, int) and max_tokens >= 0)
        ):
            raise ValueError(
                f"max_tokens must be a non-negative integer. Got: {max_tokens}"
            )

        input_name = cerebras_pytorch_lib.get_tensor_name(input_tensor)
        index_name = cerebras_pytorch_lib.get_tensor_name(index_tensor)
        update_name = input_name.replace(
            "input", "autoregressive_update_for", 1
        )
        update_tensor = cerebras_pytorch_lib.mark_output_tensor(
            update_tensor, update_name, force=True
        )
        self.set_attribute(update_tensor, "autoregressive_unused_output", True)

        # Explicitly mark the stop sequences tensor as an output so it shows up in the compute graph
        stop_sequences_name = cerebras_pytorch_lib.get_tensor_name(
            stop_sequences_tensor
        )
        stop_sequences_tensor = cerebras_pytorch_lib.mark_output_tensor(
            stop_sequences_tensor,
            stop_sequences_name.replace("input", "output", 1),
            force=True,
        )
        self.set_attribute(
            stop_sequences_tensor, "autoregressive_unused_output", True
        )

        # The RT will perform the actual update.
        output_tensor = input_tensor + 0

        # Now, add the "output" for loop-carrying the input, but also holding
        # the final result.
        output_name = input_name.replace("input", "output", 1)

        # And finally, configure the autoregressive loop of this tensor.
        autoregressive = {}
        for name, value in (
            ("input_name", input_name),
            ("index_name", index_name),
            ("update_name", update_name),
            ("stop_sequences_name", stop_sequences_name),
            ("output_name", output_name),
            ("loop_dim", loop_dim),
            ("start_token", start_token),
            ("max_tokens", max_tokens),
        ):
            if value is not None:
                autoregressive[name] = value
            elif name != "max_tokens":
                raise ValueError(
                    f"Expected {name} to be an integer but got None"
                )

        # mark_output_tensor returns a new tensor that's used a surrogate
        # whenever it's added to a step closure. To avoid overwriting the
        # attribute, we return the original tensor (not the surrogate) so
        # that if another method adds output_tensor to a step closure, we
        # return the already surrogate_output_tensor instead and not create
        # a new output.
        surrogate_output_tensor = cerebras_pytorch_lib.mark_output_tensor(
            output_tensor, output_name, force=True
        )
        self.set_attribute(
            surrogate_output_tensor, "autoregressive", autoregressive
        )

        return output_tensor

    def _active_model_changed(self) -> bool:
        """Returns true if the current model is different from the previously active model."""
        return (
            self.appliance is not None
            and getattr(self.appliance, "_active_model_id", None) is not None
            and self.appliance._active_model_id != self._active_model_id
        )


TensorToCpuHook = Callable[[torch.Tensor, str], None]
_before_tensor_to_cpu_hooks: Dict[int, TensorToCpuHook] = OrderedDict()
_after_tensor_to_cpu_hooks: Dict[int, TensorToCpuHook] = OrderedDict()


def register_before_tensor_to_cpu(hook: TensorToCpuHook) -> RemovableHandle:
    """
    Registers a hook to be called before retrieving a cpu tensor from the appliance.

    Args:
        hook: The function to be called prior to retrieving a cpu tensor.
            The function should have the following signature:
                "hook(arg: torch.Tensor, name: str) -> None"
    Returns:
        handle: A handle that can be used to delete the registered hook.
    """
    handle = RemovableHandle(_before_tensor_to_cpu_hooks)
    _before_tensor_to_cpu_hooks[handle.id] = hook
    return handle


def register_after_tensor_to_cpu(hook: TensorToCpuHook) -> RemovableHandle:
    """
    Registers a hook to be called after we retrieve a cpu tensor from the appliance.

    Args:
        hook: The function to be called after we retrieve a cpu tensor.
            The function should have the following signature:
                "hook(tensor: torch.Tensor, name: str) -> None"
    Returns:
        handle: A handle that can be used to delete the registered hook.
    """
    handle = RemovableHandle(_after_tensor_to_cpu_hooks)
    _after_tensor_to_cpu_hooks[handle.id] = hook
    return handle


@named_class_logger("ClusterManager")
class ClusterManager(ClassLogger):
    def __init__(self, config: cstorch.distributed.ClusterConfig):
        self._config = config
        self._cleanup_stack = None
        self._f = None

    @staticmethod
    def cleanup(exit_stack):
        exit_stack.__exit__(*sys.exc_info())

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        if not isinstance(value, cstorch.distributed.ClusterConfig):
            raise TypeError(
                f"Expected value to be of type ClusterConfig. Got: {type(value)}"
            )
        # If a workflow is already running and the config is updated,
        # we need to ensure that the workflow_id is set in the new config.
        if value.workflow_id is None:
            value.workflow_id = self.workflow_id
        self._config = value

    @property
    def workflow_id(self):
        return self.config.workflow_id

    @cached_property
    def client(self):
        self._cleanup_stack = ExitStack()
        self._cleanup_stack.__enter__()

        self._f = finalize(self, self.cleanup, self._cleanup_stack)

        tmp_artifact_dir = self._cleanup_stack.enter_context(
            TemporaryDirectory()
        )
        try:
            return self._cleanup_stack.enter_context(
                ClusterManagementClient(
                    server=self.config.mgmt_address,
                    crt_file=self.config.credentials_path,
                    namespace=self.config.mgmt_namespace,
                    job_timer=self.config.job_timer,
                    cbcore_image=self.config.cbcore_image,
                    workdir=tmp_artifact_dir,
                    fabric_type_blacklist=(
                        cstorch.backends.csx.debug.fabric_type_blacklist
                    ),
                    workflow_id=self.workflow_id,
                )
            )
        except ClusterConfigError as e:
            # If we encounter a cluster config error, then it means there
            # is no cluster to connect to.
            self.logger.warning(f"ClusterConfigError: {e}")
            self._f()
            self._cleanup_stack = None
            self._f = None
            return None

    def start_workflow(self, lock_resources=True) -> bool:
        """Start cluster workflow for the CSX backend.

        Args:
            lock_resources: If True, reserve the first compile and execute jobs' resources.

        Returns:
            True only if a new workflow was started.
            False if a workflow already exists.
        """
        if self.workflow_id is None:
            if self.client is None:
                raise RuntimeError(
                    "Failed to connect to cluster. Unable to start workflow."
                )

            @contextmanager
            def client_context():
                try:
                    self.client.init_workflow(resource_reserve=lock_resources)
                    self.config.workflow_id = self.client.workflow_id

                    self.logger.info(
                        f"Successfully initialized workflow with id: {self.workflow_id}"
                    )
                    yield
                finally:
                    self.config.workflow_id = None
                    if lock_resources:
                        self.client.release_workflow_resources()

            self._cleanup_stack.enter_context(client_context())

            return True
        else:
            self.logger.info(
                f"Workflow is already initialized with id: {self.workflow_id}. "
                f"Stop the current workflow before starting a new one."
            )
            return False

    def stop_workflow(self):
        if self._f is not None:
            self._f()

            self._cleanup_stack = None
            self._f = None
            self.__dict__.pop("client", None)

    @lru_cache
    def _check_s3_storage_connectivity(self, endpoint_url, region_name):
        # Cache S3 storage connectivity check so that we only
        # check once per unique endpoint url

        if self.client is None:
            # Count not connect to cluster, so we can't check storage connectivity.
            return False

        # Attempt 3 times to check storage connectivity.
        num_attempts = 3
        for attempt in range(num_attempts):
            try:
                self.client.check_storage_connectivity(
                    endpoint_url=endpoint_url,
                    region_name=region_name,
                )
                return True
            except Exception as e:
                if attempt == num_attempts - 1:
                    raise

                self.logger.warning(
                    f"Storage connectivity check failed due to: {str(e)}\n"
                    f"Retrying after {attempt + 1} second(s)"
                )
                # Retry after `attempt + 1` seconds.
                time.sleep(attempt + 1)

    def check_storage_connectivity(self) -> bool:
        if self.client is None:
            # Count not connect to cluster, so we can't check storage connectivity.
            return False

        if cstorch.backends.csx.debug.skip_connectivity_check:
            self.logger.debug("Skipping S3 storage connectivity check")
            return True

        from cerebras.appliance.storage.s3_storage import get_s3_client

        client_meta = get_s3_client().meta
        endpoint_url = client_meta.endpoint_url

        if endpoint_url is not None:
            self._check_s3_storage_connectivity(
                endpoint_url,
                client_meta.region_name,
            )

        return True

    def active_resources(
        self,
    ) -> Optional[int]:
        """Fetch the number of healthy systems in the cluster for the initialized workflow."""
        if self.workflow_id is None:
            raise RuntimeError(
                "No workflow has been initialized. Use the `start_workflow` "
                "method to initialize a new workflow first."
            )

        _, execute_jobs = self.client.list_workflow_jobs()
        if not execute_jobs:
            self.logger.debug(
                "No active jobs found in the current workflow. "
                "Unable to fetch the healthy system count."
            )
            return None

        return self.client.get_reservation_status(job_id=execute_jobs[0])
