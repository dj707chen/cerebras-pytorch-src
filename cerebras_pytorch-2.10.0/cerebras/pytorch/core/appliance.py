# Copyright 2016-2025 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Appliance Support For Pytorch."""
import json
import logging
import sys
import uuid
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import reduce
from math import prod
from pathlib import Path
from threading import Event, Lock
from traceback import format_exception
from typing import Any, Dict, List, Optional, Set, Union

import dill
import numpy as np
import torch
from tqdm import tqdm

import cerebras.pytorch as cstorch
from cerebras.appliance.appliance_manager import ApplianceManager, PowerProfile
from cerebras.appliance.cluster.client import ClusterManagementClient
from cerebras.appliance.cluster.cluster_details import ClusterDetailsParser
from cerebras.appliance.cluster_config import ClusterConfig
from cerebras.appliance.data.conversions import rtfx_dtype_from_np_dtype
from cerebras.appliance.errors import (
    ApplianceClientException,
    ApplianceDropModeComplete,
    ApplianceResourceExhausted,
    ApplianceTensorDropped,
)
from cerebras.appliance.pb.framework.appliance_service_pb2 import (
    CompileRequest,
    CompileResponse,
    InitRequest,
    LoadRequest,
    RunInferenceRequest,
    RunRequest,
    SendCheckRequest,
    SendDeferredInputRequest,
    TokenizerInfo,
)
from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ClusterDetails,
    DebugArgs,
    FrameworkType,
)
from cerebras.appliance.pb.workflow.appliance.common.message_queue_pb2 import (
    ValidTopics,
)
from cerebras.appliance.pb.ws.cross_compile_state_pb2 import CrossCompileState
from cerebras.appliance.utils.signal import on_sigint
from cerebras.pytorch import cerebras_pytorch_lib
from cerebras.pytorch.backend import current_backend_impl
from cerebras.pytorch.core.appliance_utils import (
    assign_tensor_ids_to_shards,
    np_array_to_rtfx_proto,
    np_array_to_rtfx_scalar,
)
from cerebras.pytorch.storage.utils import (
    torch_to_np_dtype,
    torch_to_rtfx_dtype,
)
from cerebras.pytorch.utils.nest import serialize_visit_fns


class ApplianceMode(ApplianceManager):
    """Manage pytorch interactions with the appliance."""

    # pylint: disable=signature-differs
    def __init__(
        self,
        artifact_dir: str,
        compile_dir: str,
        cluster_config: ClusterConfig,
        op_profiler_config: Optional[LoadRequest.OpProfilerConfig] = None,
    ):
        super().__init__(
            config=deepcopy(cluster_config),
            debug_args=deepcopy(cstorch.backends.csx.debug.debug_args),
            compile_dir=Path(compile_dir),
            artifact_dir=Path(artifact_dir),
            framework_type=FrameworkType.PYTORCH,
            op_profiler_config=op_profiler_config,
        )

        self._last_fetched_iteration = 0
        self.output_names = []
        self.weight_names = {}
        self._output_name_to_idx = {}
        self.auxilliary_state_names = {}
        self._async_grpc_client = None

        # Overwrite the number of transfer threads to be used for data transfer
        # using the global configuration flag
        self._transfer_threads = (
            cstorch.backends.csx.performance.transfer_processes
        )
        # Add fabric type blacklist to the management client arguments
        self._mgmt_client_args["fabric_type_blacklist"] = (
            cstorch.backends.csx.debug.fabric_type_blacklist
        )

        self.tracker_execute = current_backend_impl().appliance_tracker

        # A set of rt tensor names that are sparsity masks.
        self._sparsity_mask_names = set()

        # Counter to track the initial checkpoint id.
        self._initial_ckpt_id = 0

        self._exec_prep_is_done = False

        self.is_sidecar_available = False
        self.is_act_lmr_supported = False

        self._transferred_weights = set()
        self._sent_weights = set()

    def receive_activations(self, iteration: int):
        """Get activations from appliance.

        Args:
            iteration: Iteration to receive activations for. Note that
                activations are received sequentially. So, if this number is
                a few steps ahead of the last iteration that was received, then
                it counts from last fetched iteration up to this iteration, but
                only returns activations for the requested iteration.
        """
        for i in range(iteration, iteration + 1):
            activations = super().receive_activations(i)

            activation_map = dict()
            received_activations = dict()
            for name, tensor in activations.items():
                if name not in self._output_name_to_idx:
                    raise RuntimeError(
                        f"Received activation with name {name} at iteration {i}"
                        f", but no such name has been registered. Expected "
                        f"activation names are: "
                        f"{self._output_name_to_idx.values()}"
                    )
                activation = cstorch.from_numpy(tensor)
                activation_map[self._output_name_to_idx[name]] = activation
                received_activations[name] = activation

        self.tracker_execute.stop("recv_first_activation")
        self.tracker_execute.stop("execute_till_recv_loss")

        return received_activations

    def receive_output(self, iteration, name):
        out = super().receive_output(iteration, name)

        self.tracker_execute.stop("recv_first_activation")
        self.tracker_execute.stop("execute_till_recv_loss")

        return out

    # pylint: disable=arguments-differ
    def compile(
        self,
        batch_size: int,
        cirh_str: str,
        cross_compile_state: Optional[CrossCompileState] = None,
        validate_only: bool = False,
        expected_prompt_table_entries: Optional[int] = None,
        power_profile: PowerProfile = PowerProfile.NONE,
    ):
        """
        Send a compile request to the coordinator.
        """
        with self.tracker_execute.entry("compile"):
            if validate_only:
                logging.info("Compile validated locally")
                return None

            with self.tracker_execute.entry("pt_cirh"):
                compile_request = CompileRequest(
                    cirh_content=cirh_str,
                    compile_dir=str(self._compile_dir),
                    batch_size=batch_size,
                    num_csx=self.num_csx,
                    max_wgt_servers=self.max_wgt_servers,
                    num_workers_per_csx=self.num_workers_per_csx,
                    max_act_per_csx=self.max_act_per_csx,
                    # Each CompileRequest is associated with a UUID, which
                    # is used to identify the request in the appliance during
                    # retries.
                    compile_id=str(uuid.uuid4()),
                )
                if expected_prompt_table_entries is not None:
                    compile_request.expected_prompt_table_entries = (
                        expected_prompt_table_entries
                    )
                if cross_compile_state is not None:
                    compile_request.cross_compile_state.CopyFrom(
                        cross_compile_state
                    )

            return super().compile(
                compile_request,
                power_profile=power_profile,
            )

    def execute_prep(
        self,
        cleanup_stack: ExitStack,
        init_request: Optional[InitRequest] = None,
    ) -> dict:
        """Prepare for execution and return the exe job details."""
        with self.tracker_execute.entry("execute_prep_details"):
            mgmt_client = cleanup_stack.enter_context(
                ClusterManagementClient(
                    tracker_execute=self.tracker_execute,
                    **self._mgmt_client_args,
                )
            )
            # Check if it is supported, AND if we actually have a sidecar
            # container image built.
            self.is_sidecar_available = mgmt_client.is_sidecar_available()
            if self.is_sidecar_available and self._inference_mode:
                # Actually require a sidecar image, not venv mounting.
                self.is_sidecar_available = bool(
                    self._debug_args.debug_mgr.activation_sidecar.image
                ) and bool(self._debug_args.debug_mgr.swdriver_sidecar.image)

            response = self.request_execute_job(mgmt_client, self._compile_resp)
            response["mgmt_client"] = mgmt_client
            self.stage_execute_coordinator(response)

            cleanup_stack.enter_context(self.subscribe(ValidTopics.STALL))

            # From this point on, if we error out, we try to do a clean exit
            cleanup_stack.enter_context(
                self.clean_shutdown(mgmt_client, response['job_id'])
            )

        with self.tracker_execute.entry("execute_init_transfer_setup"):
            self.initialize_servers(init_request)
        return response

    def execute_session(
        self,
        initial_state_dict: Dict[str, Any] = None,
        appliance_weights: Dict[str, Any] = None,
        has_modified_seed: Optional[bool] = None,
    ):
        """Prepares weights and starts execution.

        Args:
            initial_state_dict: Weights that needs to be sent to the appliance.
            appliance_weights: Weights that need to be carried over from PTR.
        """
        tensor_rt_groups = self.grpc_client.send_check(
            0, info_type=SendCheckRequest.InfoType.GROUP
        )

        rt_initial_state_dict = (
            {
                self._map_wgt_name_fw_to_rt(fw_name): (fw_name, weight)
                for fw_name, weight in initial_state_dict.items()
            }
            if initial_state_dict
            else {}
        )

        rt_appliance_weights = (
            {
                self._map_wgt_name_fw_to_rt(fw_name): (fw_name, info)
                for fw_name, info in appliance_weights.items()
            }
            if appliance_weights
            else {}
        )

        self._sparsity_mask_names = set()

        skipped_weights = set()

        for group in tensor_rt_groups:
            tensor_names = group.tensor_names
            if len(tensor_names) <= 1:
                continue

            # Make an assumption that first tensor in the group is a sparsity mask
            sparsity_mask, _weights = tensor_names[0], tensor_names[1:]
            self._sparsity_mask_names.add(sparsity_mask)

            # Check if all tensors from the group are either in initial checkpoint or in the appliance,
            # otherwise, we need to send all tensors from the group to the appliance.
            if all(name in rt_initial_state_dict for name in tensor_names):
                continue

            if all(name in rt_appliance_weights for name in tensor_names):
                # We do not carry over sparsity mask if it wasn't changed.
                fw_name, _ = rt_appliance_weights[sparsity_mask]
                skipped_weights.add(fw_name)
                del rt_appliance_weights[sparsity_mask]
                continue

            for name in tensor_names:
                if name in rt_appliance_weights:
                    rt_initial_state_dict[name] = rt_appliance_weights[name]
                    del rt_appliance_weights[name]

        # Bake remaining apliances weights into initial_state_dict, so they can be
        # carried over to the next session.
        for rt_name, (fw_name, data) in rt_appliance_weights.items():
            rt_initial_state_dict[rt_name] = (
                fw_name,
                data.get_appliance_info(),
            )

        # The final dictionary that contains the weights that will be sent to the appliance
        # or carried over from the previous session.
        initial_weights = {
            fw_name: weight
            for fw_name, weight in rt_initial_state_dict.values()
        }

        out = super().execute_session(
            initial_weights,
            skipped_weights,
            has_modified_seed,
        )

        # Update ApplianceInfo with InBuffer state after we sent/carried over
        # all the weight to the appliance.
        for data in (appliance_weights or {}).values():
            data.get_appliance_info().state = (
                cerebras_pytorch_lib.ApplianceInfoState.InBuffer
            )

        self._skipped_weights -= self._sparsity_mask_names
        return out

    def execute(
        self,
        run_request: RunRequest,
        cleanup_stack: ExitStack,
        initial_state_dict: Dict[str, Any] = None,
        appliance_weights: Dict[str, Any] = None,
        has_modified_seed: Optional[bool] = None,
    ):
        """Run a model on the appliance."""
        self.tracker_execute.start("execute_till_recv_loss")

        if not self._exec_prep_is_done:
            from cerebras.appliance.storage.s3_storage import get_credentials

            init_request = None
            if credentials := get_credentials():
                init_request = InitRequest()
                s3_credentials = init_request.credentials.add().s3_credentials

                for k, v in credentials.items():
                    setattr(s3_credentials, k, v)

            self.execute_prep(cleanup_stack, init_request)
            self._exec_prep_is_done = True

        with self.tracker_execute.entry("execute_init_transfer_setup"):
            self.initialize_session(
                run_request,
                self._compile_resp,
                has_modified_seed,
            )
            self.execute_session(
                initial_state_dict,
                appliance_weights,
                has_modified_seed,
            )

        self.tracker_execute.start("recv_first_activation")

    def send_weights(
        self,
        initial_weights: Optional[dict] = None,
        skipped_weights: Optional[Set[str]] = None,
    ):
        self._sent_weights.clear()
        self._transferred_weights.clear()

        tensor_rt_groups = self.grpc_client.send_check(
            0, info_type=SendCheckRequest.InfoType.GROUP
        )

        tensor_rt_name_group_map = {}
        tensor_id_group_map = {}
        tensor_id_rt_name_map = {}
        group_map = {}

        @dataclass
        class Group:
            group_id: int
            group_type: str
            tensor_names: List[str]

            @dataclass
            class NameID:
                tensor_name: str
                tensor_id: Union[int, str]

            tensor_name_id_pairs: List[NameID]
            send_all_weights: bool = False

        for i, group in enumerate(tensor_rt_groups):
            if not group.HasField("group_id"):
                # This must be the mock model case
                # We treat the name as the id in this case
                group = Group(
                    group_id=i,
                    group_type=group.group_type,
                    tensor_names=list(group.tensor_names),
                    tensor_name_id_pairs=[
                        Group.NameID(name, name) for name in group.tensor_names
                    ],
                    send_all_weights=group.send_all_weights,
                )
            else:
                group = Group(
                    group_id=group.group_id,
                    group_type=group.group_type,
                    tensor_names=list(group.tensor_names),
                    tensor_name_id_pairs=[
                        Group.NameID(p.tensor_name, p.tensor_id)
                        for p in group.tensor_name_id_pairs
                    ],
                    send_all_weights=group.send_all_weights,
                )

            group_map[group.group_id] = group

            # This is a sanity check, we should always have at least one pair,
            # unless send_all_weights is True
            assert (
                len(group.tensor_name_id_pairs) != 0 or group.send_all_weights
            )

            for p in group.tensor_name_id_pairs:
                tensor_rt_name_group_map[p.tensor_name] = group
                tensor_id_group_map[p.tensor_id] = group
                tensor_id_rt_name_map[p.tensor_id] = p.tensor_name

        self.logger.trace(
            f"Tensor Group Map:\n"
            f"{json.dumps({group_id: asdict(group) for group_id, group in group_map.items()}, indent=4)}"
        )
        self.logger.trace(f"Initial Weights: {sorted(initial_weights)}")

        tensor_rt_names = set(tensor_rt_name_group_map)
        send_all = any(group.send_all_weights for group in group_map.values())
        if send_all:
            tensor_rt_names = set()

        from cerebras.appliance.storage import S3Reader
        from cerebras.pytorch.storage.serializers import (
            DeferredFullTensor,
            DeferredGraphTensor,
            DeferredTorchTensor,
        )
        from cerebras.pytorch.storage.utils import lazy_tensor_data_wrapper

        skipped_weights = skipped_weights or set()

        weight_map = {}
        for key, weight in initial_weights.items():
            rt_name = self._map_wgt_name_fw_to_rt(key)

            if rt_name in tensor_rt_names or send_all:
                weight_map[rt_name] = weight
            else:
                skipped_weights.add(key)

        self._skipped_weights = skipped_weights

        def is_s3(t):
            return isinstance(
                t, DeferredTorchTensor
            ) and S3Reader.is_valid_path(t.deferred._path)

        def is_graph_tensor(t):
            return isinstance(t, DeferredGraphTensor)

        def is_full_or_s3(t):
            return isinstance(t, DeferredFullTensor) or is_s3(t)

        def should_send_deferred(tensors):
            total_args_size = 0
            for _, t in tensors:
                if is_graph_tensor(t):
                    if not self.is_sidecar_available:
                        # If the sidecar is not available, we can't send the graph
                        # tensor, so it has to be sent using the legacy path
                        return False

                    def get_args_size(arg):
                        if is_full_or_s3(arg):
                            return 0
                        if is_graph_tensor(arg):
                            return sum(map(get_args_size, arg._args))
                        if isinstance(arg, torch.Tensor):
                            return arg.nbytes

                        return 0

                    total_args_size += sum(map(get_args_size, t._args))

                elif is_full_or_s3(t):
                    if (
                        t._post_materialize_hooks
                        and not self.is_sidecar_available
                    ):
                        # If the sidecar is not available, we can't run any post
                        # materialize hooks on the appliance
                        return False
                    # If the tensor is a full tensor or an S3 tensor, its size
                    # is negligible, so we don't have to account for it here.
                else:
                    # If any one of the tensors in the group is none of the above,
                    # then we have to send the entire group using the legacy path
                    return False

            # If the sum of the argument sizes exceeeds the max transfer bytes,
            # we have to do some tensor chunking and thus cannot use the faster
            # wgt init on memx path.
            # Note, we add about 5MB to the total arg size to account for the
            # other overhead and reduce the chance of going over the message
            # size limit
            return (
                total_args_size + 5 * 1024 * 1024
            ) < cstorch.backends.csx.performance.max_transfer_bytes

        futures = {}
        cancel_grpc = Event()

        # Callback to cancel futures on SIGINT
        def cancel_futures(*args, **kwargs):
            for future in futures.values():
                future.cancel()
            cancel_grpc.set()

        with ExitStack() as exit_stack:
            progress = None
            # Only enable progress bar on TTY
            if sys.stdout.isatty():
                progress = exit_stack.enter_context(
                    tqdm(
                        total=len(weight_map),
                        desc="Sending initial weights",
                        dynamic_ncols=True,  # Match console width
                        unit=" tensors",
                        file=sys.stdout,
                    )
                )
                # Used to safely update progress bar from multiple threads
                progress_lock = Lock()

            num_workers = cstorch.backends.csx.performance.transfer_processes

            eager_executor = exit_stack.enter_context(
                ThreadPoolExecutor(max_workers=num_workers)
            )
            exit_stack.enter_context(on_sigint(cancel_futures))
            # Even if no sigint encountered, cancel all futures on exit
            exit_stack.callback(cancel_futures)

            def carry_over_from_ptr(rt_name, tensor_id):
                self.grpc_client.carry_over_from_ptr(
                    iteration=0,
                    tensor_name=rt_name,
                    tensor_id=tensor_id,
                    keep_in_repo=False,
                )
                if progress is not None:
                    with progress_lock:
                        progress.update(1)

            # Send tensors to the appliance using the legacy gRPC path
            def send_group(tensors):
                for rt_name, t in tensors:
                    scalar_broadcast = isinstance(t, DeferredFullTensor)
                    if scalar_broadcast:
                        t = np.array(t._value, dtype=torch_to_np_dtype(t.dtype))
                    else:
                        self.logger.trace(
                            f"About to materialize tensor {rt_name}"
                        )
                        t = cstorch.to_numpy(t)
                        self.logger.trace(
                            f"Finished materializing tensor {rt_name}"
                        )

                    self.logger.debug(
                        f"Sending {rt_name} "
                        f"(shape={list(t.shape)}, "
                        f"dtype={t.dtype}, "
                        f"bytes={t.nbytes}) "
                        f"to appliance via grpc"
                    )

                    self.grpc_client.send_weight(
                        0, rt_name, t, scalar_broadcast, cancel_grpc
                    )

                    if progress is not None:
                        with progress_lock:
                            progress.update(1)

            @contextmanager
            def ingest_weights(tensor_id_shard_id):
                responses = self.grpc_client.start_weight_ingestion(
                    tensor_id_shard_id
                )
                responses_iterator = iter(responses)
                # Make sure to get a response back first before sending the tensors
                next(responses_iterator)

                yield

                for _ in responses_iterator:
                    if progress is not None:
                        with progress_lock:
                            progress.update(1)

            deferred_tensors = defaultdict(dict)
            for group_id, group in group_map.items():
                if send_all:
                    weights = list(weight_map.items())
                else:
                    weights = [
                        (rt_name, weight_map.get(rt_name))
                        for rt_name in group.tensor_names
                    ]

                # Keep track of all weights that are sent to the appliance
                self._sent_weights.update(
                    rt_name for rt_name, w in weights if w is not None
                )

                if any(
                    isinstance(w, cerebras_pytorch_lib.ApplianceInfo)
                    for _, w in weights
                ):
                    if not all(
                        isinstance(w, cerebras_pytorch_lib.ApplianceInfo)
                        for _, w in weights
                        if w is not None
                    ):
                        raise RuntimeError(
                            f"Not all weights in group {group_id} are being carried over"
                        )

                    for rt_name, weight in weights:
                        if weight is not None:
                            futures[rt_name] = eager_executor.submit(
                                carry_over_from_ptr, rt_name, weight.uid
                            )

                    continue

                if any(w is None for _, w in weights):
                    raise RuntimeError(
                        f"Group {group_id} has missing weights:\n\t"
                        f"{[rt_name for rt_name, w in weights if w is None]}"
                    )

                self._transferred_weights.update(
                    rt_name for rt_name, _ in weights
                )

                tensors = [
                    (rt_name, lazy_tensor_data_wrapper(w))
                    for rt_name, w in weights
                ]

                if (
                    not self._debug_args.debug_usr.force_send_weights
                    and should_send_deferred(tensors)
                ):
                    deferred_tensors[group_id] = {
                        name: t for name, t in tensors
                    }
                else:
                    futures[tuple(group.tensor_names)] = eager_executor.submit(
                        send_group, tensors
                    )

            if deferred_tensors:
                cluster_details = ClusterDetailsParser(
                    self._compile_resp.cluster_details
                )
                try:
                    # TODO: This should really be number of wgt coordinators
                    num_wgt_servers = cluster_details.extract_num_wgt_srvs()
                except Exception:
                    num_wgt_servers = 1

                def get_weight_ids(group_id):
                    # All (non-mask) tensors belonging to the same group need to
                    # be assigned to the same shard
                    weight_ids = tuple(
                        p.tensor_id
                        for p in group_map[group_id].tensor_name_id_pairs
                    )
                    if (
                        len(weight_ids) > 1
                        and group_map[group_id].group_type == "sparsity"
                    ):
                        # This is the sparsity case. We remove the mask id
                        # from the weight id list as it is not a weight
                        weight_ids = weight_ids[1:]
                    return weight_ids

                if num_wgt_servers == 1:
                    tensor_id_shard_id = [
                        (weight_id, 0)
                        for group_id, _ in deferred_tensors.items()
                        for weight_id in get_weight_ids(group_id)
                    ]
                else:
                    tensor_id_shard_id = assign_tensor_ids_to_shards(
                        {
                            get_weight_ids(group_id): reduce(
                                lambda cost1, cost2: {
                                    k: cost1[k] + cost2[k] for k in cost1
                                },
                                (
                                    (
                                        cerebras_pytorch_lib.estimate_compute_memory_cost(
                                            t._jit_graph
                                        )
                                        if isinstance(t, DeferredGraphTensor)
                                        else {
                                            "total_cpu_time": 0,
                                            "total_memory": prod(t.shape),
                                            "max_memory": prod(t.shape),
                                        }
                                    )
                                    for t in tensors.values()
                                ),
                            )
                            for group_id, tensors in deferred_tensors.items()
                        },
                        num_wgt_servers,
                    )

                def send_tensor_ids(shards):
                    seen = set()
                    for tensor_id, shard_id in tensor_id_shard_id:
                        if cancel_grpc.is_set():
                            return

                        if shard_id not in shards or tensor_id in seen:
                            continue

                        group = tensor_id_group_map[tensor_id]

                        # NOTE: Its important for all tensors in the same group
                        # to be sent to the same shard
                        # NOTE: Also important to send the tensors in the same
                        # order as in the tensor mapping
                        tensors = []
                        tensor_names = []
                        for p in group.tensor_name_id_pairs:
                            deferred_tensor = deferred_tensors[group.group_id][
                                p.tensor_name
                            ]
                            self.logger.debug(
                                f"Sending deferred tensor {p.tensor_name} "
                                f"(shape={list(deferred_tensor.shape)}, "
                                f"dtype={deferred_tensor.dtype}, "
                                f"nbytes={deferred_tensor.nbytes})"
                            )

                            tensor = SendDeferredInputRequest.Tensor(
                                shape=tuple(deferred_tensor.shape),
                                dtype=rtfx_dtype_from_np_dtype(
                                    torch_to_np_dtype(deferred_tensor.dtype)
                                ),
                            )

                            if isinstance(p.tensor_id, str):
                                tensor.tensor_name = p.tensor_id
                            else:
                                tensor.tensor_id = p.tensor_id

                            def set_tensor_impl(deferred_tensor, tensor):
                                if is_graph_tensor(deferred_tensor):
                                    tensor.graph_tensor.graph = (
                                        deferred_tensor._jit_graph
                                    )
                                    for arg in deferred_tensor._args:
                                        graph_arg = (
                                            tensor.graph_tensor.args.add()
                                        )
                                        graph_arg.shape.extend(tuple(arg.shape))
                                        graph_arg.dtype = torch_to_rtfx_dtype(
                                            arg.dtype
                                        )

                                        set_tensor_impl(arg, graph_arg)

                                elif isinstance(
                                    deferred_tensor, DeferredFullTensor
                                ):
                                    tensor.scalar_broadcast_tensor.value.CopyFrom(
                                        np_array_to_rtfx_scalar(
                                            np.array(
                                                deferred_tensor._value,
                                                dtype=torch_to_np_dtype(
                                                    deferred_tensor.dtype
                                                ),
                                            )
                                        )
                                    )
                                elif is_s3(deferred_tensor):
                                    parsed = S3Reader.parse_path(
                                        deferred_tensor.deferred._path
                                    )
                                    bucket = parsed["bucket"]
                                    key = "/".join(
                                        (
                                            parsed["key"],
                                            deferred_tensor.deferred._key,
                                        )
                                    )

                                    tensor.s3_tensor.bucket = bucket
                                    tensor.s3_tensor.key = key
                                    tensor.s3_tensor.index = 0
                                elif isinstance(
                                    deferred_tensor, torch.Tensor
                                ) and hasattr(tensor, "rtfx_tensor"):
                                    tensor.rtfx_tensor.CopyFrom(
                                        np_array_to_rtfx_proto(
                                            cstorch.to_numpy(deferred_tensor)
                                        )
                                    )
                                else:
                                    raise TypeError(
                                        f"Found unsupported deferred tensor type "
                                        f"for {p.tensor_name}: {type(deferred_tensor)}"
                                    )

                            set_tensor_impl(deferred_tensor, tensor)

                            if deferred_tensor._post_materialize_hooks:
                                tensor.post_materialize_hooks = dill.dumps(
                                    list(
                                        deferred_tensor._post_materialize_hooks.values()
                                    )
                                ).hex()

                            tensors.append(tensor)
                            tensor_names.append(p.tensor_name)

                            seen.add(p.tensor_id)

                        self.grpc_client.send_deferred_tensor_group(
                            0, tensors, tensor_names, shard_id
                        )

                buckets = [[] for i in range(num_workers)]
                for shard_id in range(num_wgt_servers):
                    buckets[shard_id % len(buckets)].append(shard_id)

                tensor_names = [[] for i in range(num_workers)]
                for tensor_id, shard_id in tensor_id_shard_id:
                    tensor_names[shard_id % len(tensor_names)].append(
                        tensor_id_rt_name_map[tensor_id]
                    )

                exit_stack.enter_context(ingest_weights(tensor_id_shard_id))

                deferred_executor = exit_stack.enter_context(
                    ThreadPoolExecutor(
                        max_workers=min(num_workers, num_wgt_servers)
                    )
                )
                for bucket, names in zip(buckets, tensor_names):
                    futures[tuple(names)] = deferred_executor.submit(
                        send_tensor_ids, bucket
                    )

            for k, f in futures.items():
                if not isinstance(k, tuple):
                    k = (k,)
                f.tensor_names = k

            success = []
            dropped = []
            exceptions = []

            pending = list(futures.values())
            while pending:
                # Wait for all futures to complete
                done, pending = wait(pending, return_when=FIRST_EXCEPTION)

                for f in done:
                    if (e := f.exception()) is None:
                        success.extend(f.tensor_names)
                        continue
                    elif isinstance(e, ApplianceTensorDropped):
                        dropped.extend(f.tensor_names)
                        continue  # Track but don't raise early for dropped tensors

                    k = ", ".join(map(str, f.tensor_names))
                    if isinstance(e, ApplianceResourceExhausted):
                        self.logger.error(
                            f"Resource exhausted when transferring '{k}':\n{''.join(format_exception(e))}"
                        )
                    else:
                        self.logger.error(
                            f"Ran into error when transferring '{k}':\n{''.join(format_exception(e))}"
                        )

                    exceptions.append(e)

                    if pending:
                        # Cancel all pending futures if any one of them failed.
                        # There shouldn't be any pending futures if all weights
                        # were sent successfully.
                        for p in pending:
                            p.cancel()

                        pending = []

            if exceptions:
                if len(exceptions) > 1:
                    exc = ExceptionGroup(
                        "Errors encountered while transferring weights",
                        exceptions,
                    )
                else:
                    exc = exceptions[0]
                raise exc

            if success and dropped:
                raise ApplianceClientException(
                    f"Some weights were successfully transferred, while some were "
                    f"dropped at the coordinator. This indicates an "
                    f"internal error.\n"
                    f"Transferred Tensors: {success}\n"
                    f"Dropped Tenors: {dropped}"
                )
            if dropped:
                raise ApplianceDropModeComplete(
                    f"All {len(dropped)} weight tensors were dropped."
                )

    @dataclass
    class InferenceModelParams:
        compile_response: CompileResponse
        model_ini: dict
        num_instances: int = 1
        initial_weights: Optional[dict] = None

    @staticmethod
    def _get_task_map(compile_response, task_type):
        for task in compile_response.cluster_details.tasks:
            if task.task_type == task_type:
                return task.task_map
        raise RuntimeError(f"Unable to find task map for task type {task_type}")

    def merge_compile_responses_and_inis(
        self,
        orig_ini: dict,
        main_model_params: InferenceModelParams,
        draft_model_params: Optional[InferenceModelParams] = None,
        image_encoder_model_params: Optional[InferenceModelParams] = None,
    ) -> CompileResponse:
        """
        Merge the compile responses and INIs from multiple models into
        self.compile_resp and self._debug_args respectively.

        Args:
        InferenceModelParams.initial_weights is not needed by this function
        """
        # First, copy the main model since we're going to mutate it.
        self.compile_resp.CopyFrom(main_model_params.compile_response)

        TaskType = ClusterDetails.TaskInfo.TaskType

        def add_task_ini(task_type, task_id, ini):
            for task_ini in self._debug_args.task_ini:
                if (
                    task_ini.task_type == task_type
                    and task_ini.task_id == task_id
                ):
                    # Replace existing task ini
                    task_ini.ini.CopyFrom(ini)
                    break
            else:
                # No existing, so add a new one.
                task_ini = DebugArgs.TaskDebugINI()
                task_ini.task_type = task_type
                task_ini.task_id = task_id
                task_ini.ini.CopyFrom(ini)

                self._debug_args.task_ini.append(task_ini)

        def update_cluster_details(
            num_csx_offset, num_models, compile_response, ini=None
        ):
            # Modify compile_resp.cluster_details for target model
            # with cluster details for imahe encoder / draft models
            task_types_to_populate = (
                TaskType.WSE,
                TaskType.ACT,
                TaskType.CHF,
                TaskType.KVSS,
            )

            try:
                swd_tasks_count = len(
                    self._get_task_map(self.compile_resp, TaskType.SWD)
                )
            except:
                swd_tasks_count = 0

            for extra_task in compile_response.cluster_details.tasks:
                if extra_task.task_type in task_types_to_populate:
                    # find a target task to populate
                    target_task = None
                    for task in self.compile_resp.cluster_details.tasks:
                        if task.task_type == extra_task.task_type:
                            target_task = task
                            break

                    # if target does not have such type of task, create explicitly
                    if target_task is None:
                        target_task = (
                            self.compile_resp.cluster_details.tasks.add()
                        )
                        # copy all fields other than 'task_map'
                        target_task.CopyFrom(extra_task)
                        # actual 'task_map' will be added in the for loop below
                        target_task.ClearField("task_map")

                    for _ in range(num_models):
                        task_id_offset = len(target_task.task_map)

                        for extra_task in extra_task.task_map:
                            new_task = target_task.task_map.add()
                            # Copy from extra model task_map entirely.
                            new_task.CopyFrom(extra_task)
                            # Offset each of task_id
                            new_task.task_id.task_id += task_id_offset

                            # Offset all the wse_id from the original target
                            new_task.task_id.wse_id += num_csx_offset
                            for i, wse_id in enumerate(
                                new_task.task_id.wse_ids
                            ):
                                new_task.task_id.wse_ids[i] = (
                                    wse_id + num_csx_offset
                                )

                            if ini:
                                add_task_ini(
                                    target_task.task_type,
                                    new_task.task_id.task_id,
                                    ini,
                                )

        num_csx_for_target = len(
            self._get_task_map(self.compile_resp, TaskType.WSE)
        )
        num_csx_for_draft_inference = 0

        if draft_model_params:
            # First, get the number of WSE in the target compile, we'll offset
            # all wse_id by this number in the draft compile.
            num_csx_for_draft_inference = len(
                self._get_task_map(
                    draft_model_params.compile_response, TaskType.WSE
                )
            )
            update_cluster_details(
                num_csx_for_target,
                draft_model_params.num_instances,
                draft_model_params.compile_response,
            )

        if image_encoder_model_params:
            if image_encoder_model_params.num_instances != 1:
                raise ValueError(
                    f"Image encoder model must have num_instances=1, got {image_encoder_model_params.num_instances}"
                )

            # We need to set a set of INIs for the image encoder model servers.
            # At this point all INIs are baked into debug_args including INIs that
            # ApplianceMode sets in c-tor (they are not available in debug.ini).
            # To get the right set of INIs for image encoder we need to combine the
            # following INIs:
            # 1. Image encoder INIs from the compile.
            # 2. Original INIs passed to the merge_compile_responses_and_inis().
            # 3. All other INIs that are set in the appliance mode.

            model_ini = image_encoder_model_params.model_ini
            model_ini.update(orig_ini)

            # To get INIs that were set in the appliance mode we need to get them from debug_args
            # and remove main model INIs from them.
            from cerebras.appliance.utils.ini import clear_ini, set_ini

            debug_args = DebugArgs()
            debug_args.CopyFrom(self._debug_args)
            clear_ini(debug_args, **main_model_params.model_ini)
            set_ini(debug_args, **model_ini)

            num_csx_offset = num_csx_for_target + num_csx_for_draft_inference

            update_cluster_details(
                num_csx_offset,
                image_encoder_model_params.num_instances,
                image_encoder_model_params.compile_response,
                debug_args.ini,
            )

            # Update ClusterDetails with CMD task for image encoder.
            for (
                task
            ) in (
                image_encoder_model_params.compile_response.cluster_details.tasks
            ):
                if task.task_type == TaskType.CMD:
                    new_task = self.compile_resp.cluster_details.tasks.add()
                    new_task.CopyFrom(task)

                    for tm in new_task.task_map:
                        tm.task_id.wse_id = (
                            max(0, tm.task_id.wse_id) + num_csx_offset
                        )
                        for i, wse_id in enumerate(tm.task_id.wse_ids):
                            tm.task_id.wse_ids[i] = (
                                max(0, wse_id) + num_csx_offset
                            )

                    for task in new_task.task_map:
                        add_task_ini(
                            new_task.task_type,
                            task.task_id.task_id,
                            debug_args.ini,
                        )

        return self.compile_resp

    def host_inference(
        self,
        cleanup_stack: ExitStack,
        main_model_params: InferenceModelParams,
        tokenizer_info: Optional[TokenizerInfo] = None,
        draft_model_params: Optional[InferenceModelParams] = None,
        image_encoder_model_params: Optional[InferenceModelParams] = None,
    ):
        """Host an inference model in the appliance and return the wsjobid."""

        num_csx_for_image_encoder = 0
        if image_encoder_model_params:
            num_csx_for_image_encoder = len(
                self._get_task_map(
                    image_encoder_model_params.compile_response,
                    ClusterDetails.TaskInfo.TaskType.WSE,
                )
            )
        num_csx_for_draft_inference = 0
        if draft_model_params:
            num_csx_for_draft_inference = len(
                self._get_task_map(
                    draft_model_params.compile_response,
                    ClusterDetails.TaskInfo.TaskType.WSE,
                )
            )

        init_request = InitRequest(
            inference=True,
            num_csx_for_draft_inference=num_csx_for_draft_inference,
            num_csx_for_image_encoder=num_csx_for_image_encoder,
        )

        from cerebras.appliance.storage.s3_storage import get_credentials

        if credentials := get_credentials():
            s3_credentials = init_request.credentials.add().s3_credentials
            for k, v in credentials.items():
                setattr(s3_credentials, k, v)

        details = self.execute_prep(cleanup_stack, init_request=init_request)

        with self.tracker_execute.entry("execute_load_request"):
            target_load_request = LoadRequest(
                cache_compile_dir=self.compile_resp.cache_compile_dir,
                model_type=LoadRequest.INF_TARGET,
                drop_cmd_state=True,
            )

            self.grpc_client.load_rtir(target_load_request)

            if num_csx_for_draft_inference:
                cache_compile_dir = (
                    draft_model_params.compile_response.cache_compile_dir
                )
                self.grpc_client.load_rtir(
                    LoadRequest(
                        cache_compile_dir=cache_compile_dir,
                        model_type=LoadRequest.INF_DRAFT,
                        num_instances=draft_model_params.num_instances,
                        drop_cmd_state=True,
                    )
                )

        # Ensure that any traced ops are properly handled
        current_backend_impl().initial_mark_step()

        if main_model_params.initial_weights:
            with self.tracker_execute.entry("execute_send_weights"):
                self.logger.info(f"About to send weights")
                self.send_weights(main_model_params.initial_weights)
                if (
                    draft_model_params
                    and draft_model_params.initial_weights is not None
                ):
                    self.logger.info(f"About to send draft model weights")
                    self.send_weights(draft_model_params.initial_weights)

        if num_csx_for_image_encoder:
            self.grpc_client.load_rtir(
                LoadRequest(
                    cache_compile_dir=image_encoder_model_params.compile_response.cache_compile_dir,
                    drop_cmd_state=True,
                    model_type=LoadRequest.INF_IMAGE_ENCODER,
                    num_instances=1,
                )
            )

            run_request = RunRequest(
                num_iterations=2**31 - 1,
                activation_freq=1,
            )
            self.grpc_client.run_deferred(run_request)

            self.logger.debug("Waiting for runtime to be initialized")
            self.grpc_client.check_runtime_initialized()

            if image_encoder_model_params.initial_weights:
                with self.tracker_execute.entry("execute_send_weights"):
                    self.logger.info(
                        "About to send image encoder model weights"
                    )
                    self.send_weights(
                        image_encoder_model_params.initial_weights
                    )

        self.logger.info("Finished sending weights")

        with self.tracker_execute.entry("execute_program"):
            # Wait for wafers to program
            self.logger.info("Waiting for device programming to complete")
            self.grpc_client.wait_for_programming()
        with self.tracker_execute.entry("execute_start_streaming"):
            # Wait for weights to load
            self.start_streaming()
        with self.tracker_execute.entry("execute_run_inference"):
            # Wait for iteration 0 to run
            self.grpc_client.run_inference(
                RunInferenceRequest(tokenizer_info=tokenizer_info)
            )

        self.logger.info("Appliance staging is complete")

        return details

    def move_to_ptr(self, tensor_name: str, tensor_id: int) -> None:
        """Move a tensor to PTR which makes is available for the next session."""
        rt_name = self._map_wgt_name_fw_to_rt(tensor_name)
        return super().move_to_ptr(rt_name, tensor_id)

    def get_from_ptr(
        self, name: str, tensor_id: int, keep_in_repo: bool = False
    ):
        """Get a tensor from PTR."""
        return super().get_from_ptr(tensor_id, keep_in_repo)

    def construct_debug_args(self) -> DebugArgs:
        debug_args = super().construct_debug_args()

        if self.is_sidecar_available:
            if self._inference_mode:
                if hasattr(debug_args.debug_mgr, "activation_sidecar"):
                    debug_args.debug_mgr.activation_sidecar.strategy = (
                        DebugArgs.DebugMGR.UserSidecar.STRATEGY_ENABLED
                    )
                if hasattr(debug_args.debug_mgr, "swdriver_sidecar"):
                    debug_args.debug_mgr.swdriver_sidecar.strategy = (
                        DebugArgs.DebugMGR.UserSidecar.STRATEGY_ENABLED
                    )

            # TODO: Conditionlize this on whether we actually need to send any
            #       deferred tensors
            else:
                debug_args.debug_mgr.weight_sidecar.strategy = (
                    DebugArgs.DebugMGR.UserSidecar.STRATEGY_ENABLED
                )

        # Populate the visit_fn_map in the debug_args so that they
        # can be propagated to the workers
        for (
            serialized_types,
            serialized_visit_fn,
        ) in serialize_visit_fns().items():
            debug_args.debug_wrk.visit_fn_map[serialized_types] = (
                serialized_visit_fn
            )

        return debug_args

    def compute_compile_hash(
        self,
        cirh_str: str,
        batch_size: int,
        cross_compile_state: CrossCompileState,
    ):
        """
        Compute the hash for the compile request. When modifying this function,
        please make sure that the hashing logic is in sync with logic in CRD.
        Note that the purpose of this function is to define a compile hash
        which is going to be used only locally (within this process), so we
        doesn't include static information (like fabric info, release id, etc.)
        to compute the hash.
        """
        checksum = [
            cirh_str,
            self._debug_args.debug_crd.stop_compile_at,
            self._debug_args.debug_crd.autogen_policy,
            self._debug_args.debug_crd.numeric_config,
            self._debug_args.debug_crd.disable_ws_incremental_compile,
            self._debug_args.debug_crd.disable_autogen_leftover_conversion,
            self.num_csx,
            batch_size,
            self.max_wgt_servers,
            self.max_act_per_csx,
            cross_compile_state,
        ]
        return hash("".join([str(item) for item in checksum]))

    def _map_wgt_name_fw_to_rt(self, tensor_name):
        if tensor_name in self.weight_names:
            return self.weight_names[tensor_name]
        if tensor_name in self.auxilliary_state_names:
            return self.auxilliary_state_names[tensor_name]
        return tensor_name

    def save_weights_to_s3(
        self,
        tensors,
        path: str,
        bucket: str,
        key_prefix: str,
        iteration: int,
    ):
        from cerebras.appliance.storage.serializers import DeferredObject
        from cerebras.pytorch.storage.utils import lazy_tensor_data_wrapper

        tensors = {
            cerebras_pytorch_lib.get_tensor_name(t): (k, t)
            for k, t in tensors.items()
        }

        weight_infos = []
        for rt_name, (name, t) in tensors.items():
            if rt_name in self.skipped_weights:
                if t.device.type == "lazy":
                    t = lazy_tensor_data_wrapper(t)
                yield name, t
            elif rt_name not in self.skipped_weights:
                weight_infos.append(
                    (rt_name, tuple(t.shape), torch_to_np_dtype(t.dtype))
                )

        with self.tracker_execute.entry("save_weights", overwrite=True):
            # Send gRPC request to have tensors saved to S3 by weight host
            responses = self.grpc_client.save_weights(
                iteration,
                weight_infos,
                bucket,
                key_prefix,
                compress_data=cstorch.backends.csx.performance.compress_weights,
            )

            for response in responses:
                if response is None:
                    raise RuntimeError(
                        f"Unable to save {name}. Failed to save all weights."
                    )

                k, t = tensors[response.name]

                yield k, DeferredObject(
                    path,
                    response.s3_location.key,
                    response.s3_location.index,
                    metadata={"__TYPE__": "TorchTensorSerializer"},
                    # We don't want to save a copy, we want to save this exact deferred object
                    force_external_link=True,
                )

    def recv_weight(self, tensor, iteration):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "lazy":
            # Nothing to receive if tensor is not a lazy torch tensor
            return tensor

        # pylint: disable=c-extension-no-member
        # pylint: disable=no-member
        weight_name = cerebras_pytorch_lib.get_tensor_name(tensor)

        # If there is no weight name, then it is not a weight tensor
        if weight_name is None or weight_name in self.skipped_weights:
            logging.debug(f"Not fetching skipped: {weight_name}")

            from cerebras.pytorch.storage.utils import lazy_tensor_data_wrapper

            return lazy_tensor_data_wrapper(tensor)

        # Get the runtime name of the tensor
        tensor_name = self._map_wgt_name_fw_to_rt(weight_name)

        logging.debug(f"Fetching {tensor_name} at iteration {iteration}")
        return self.grpc_client.recv_output(iteration, tensor_name)
