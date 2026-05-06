#!/usr/bin/env python3
# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""GRPC Server For weight init/loading."""
import argparse
import atexit
import logging
import os
import sys
import tempfile
import threading
import urllib.parse
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from queue import Empty as EmptyQueue
from queue import Queue
from threading import Thread
from traceback import format_exception
from typing import Any, Optional

import dill
import numpy as np

import cerebras.pytorch as cstorch
from cerebras.appliance import log
from cerebras.appliance.cluster.cluster_details import ClusterDetailsParser
from cerebras.appliance.pb.framework import appliance_service_pb2_grpc
from cerebras.appliance.pb.framework.appliance_service_pb2 import (
    DoneResponse,
    FinalizeResponse,
    GetOutputResponse,
    InitRequest,
    InitResponse,
    NotifyCrashResponse,
    NotifyStallResponse,
    SendDeferredInputRequest,
    SendDeferredInputResponse,
)
from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ClusterDetails,
    DebugArgs,
    SendToEveryTask,
)
from cerebras.appliance.pb.ws.common_pb2 import (
    WS_RT_INTERNAL_ERROR,
    ClockSyncResponse,
)
from cerebras.appliance.utils import resolve_command_line_arg
from cerebras.pytorch.core.appliance_utils import rtfx_to_np_array
from cerebras.pytorch.storage.utils import rtfx_to_torch_dtype
from cerebras.pytorch.utils._app.utils import (
    _proto_msg_from_jsonfile,
    _serve,
    _service_api,
    _setup_logging,
)


@dataclass
class TensorPayload(ABC):
    tensor_id: int
    shape: tuple
    dtype: Any
    post_materialize_hooks: Optional[str] = None

    @abstractmethod
    def compute(self):
        pass

    def postprocess(self, tensor):
        if self.post_materialize_hooks:
            for hook in dill.loads(bytes.fromhex(self.post_materialize_hooks)):
                out = hook(tensor)
                if out is not None:
                    tensor = out
        return tensor


@dataclass
class GraphTensorPayload(TensorPayload):
    tensor: SendDeferredInputRequest.Tensor.Graph = ...

    def compute(self):
        args = []
        for i, arg in enumerate(self.tensor.args):
            if arg.HasField("rtfx_tensor"):
                rtfx_proto = arg.rtfx_tensor
                args.append(
                    cstorch.from_numpy(
                        rtfx_to_np_array(
                            rtfx_proto.tensor.data,
                            rtfx_proto.tensor.shape,
                            rtfx_proto.dtype,
                        )
                    )
                )
            elif arg.HasField("scalar_broadcast_tensor"):
                args.append(
                    ScalarBroadcastTensorPayload(
                        tensor_id=None,
                        shape=arg.shape,
                        dtype=arg.dtype,
                        tensor=arg.scalar_broadcast_tensor,
                    ).compute()
                )
            elif arg.HasField("s3_tensor"):
                args.append(
                    S3TensorPayload(
                        tensor_id=None,
                        shape=arg.shape,
                        dtype=arg.dtype,
                        tensor=arg.s3_tensor,
                    ).compute()
                )
            elif arg.HasField("graph_tensor"):
                args.append(
                    GraphTensorPayload(
                        tensor_id=None,
                        shape=arg.shape,
                        dtype=arg.dtype,
                        tensor=arg.graph_tensor,
                    ).compute()
                )

        args_msg = ""
        if args:
            args_msg = "\n".join(
                f"\t{self.tensor.args[i].WhichOneof('arg_impl')} arg {i}: "
                f"shape={arg.shape}, dtype={arg.dtype}"
                for i, arg in enumerate(args)
            )
            args_msg = f"\nwith args:\n{args_msg}\n"

        logging.info(
            f"Tensor {self.tensor_id} has graph:\n{self.tensor.graph}{args_msg}"
        )

        from cerebras.pytorch.storage.serializers import DeferredGraphTensor

        return DeferredGraphTensor(
            self.tensor.graph,
            args,
            tuple(self.shape),
            rtfx_to_torch_dtype(self.dtype),
        )._to_cpu()


@dataclass
class ScalarBroadcastTensorPayload(TensorPayload):
    tensor: SendDeferredInputRequest.Tensor.ScalarBroadcast = ...

    def compute(self):
        from cerebras.pytorch.storage.serializers import DeferredFullTensor

        return DeferredFullTensor(
            tuple(self.shape),
            rtfx_to_torch_dtype(self.dtype),
            rtfx_to_np_array(
                self.tensor.value.scalar.data, (), self.dtype
            ).item(),
        )._to_cpu()


@dataclass
class S3TensorPayload(TensorPayload):
    tensor: SendDeferredInputRequest.Tensor.S3 = ...

    def compute(self):
        from cerebras.appliance.storage.s3_storage import S3Reader
        from cerebras.appliance.storage.serializers import DeferredObject
        from cerebras.pytorch.storage.serializers import DeferredTorchTensor

        path, key = os.path.split(self.tensor.key)

        return DeferredTorchTensor(
            DeferredObject(
                S3Reader.construct(
                    urllib.parse.urlunparse(
                        ("s3", self.tensor.bucket, path, "", "", "")
                    )
                ),
                key,
                self.tensor.index,
            ),
            tuple(self.shape),
            rtfx_to_torch_dtype(self.dtype),
        )._to_cpu()


@log.named_class_logger("weight.WeightSidecarServicer")
class WeightSidecarServicer(
    appliance_service_pb2_grpc.ApplianceServicer, log.ClassLogger
):
    """Service for initializing and/or loading weights."""

    def __init__(
        self,
        cluster_details: ClusterDetails,
        debug_args: DebugArgs,
        role_id: int,
    ):
        super().__init__()

        self._done_event = threading.Event()
        self._tensor_queue = Queue()
        # We set the maxsize to be 3 to avoid the queue from growing too large
        # If runtime consumes the tensors at a slower rate than the producer
        # produces them, we block to avoid computing and having to store too
        # many tensors in memory at once.
        # If runtime consumes tensors at a faster rate than the producer can
        # produce them then, the queue will never reach maximum capacity.
        self._results_queue = Queue(maxsize=3)

        # Use a ThreadPoolExecutor for computation and allow this many workers
        # for "concurrent" execution, really for overlapping I/O and compute.
        self._max_workers = debug_args.debug_wgt.sidecar_compute_threads or 1

        self.thread = None

    def wait(self):
        """Blocks until the server is done."""
        self._done_event.wait()
        if self.thread is not None:
            self.thread.join()

    def __del__(self):
        self.logger.info(f"{self.__class__.__name__} is being destructed.")

    @_service_api()
    def UnaryInit(self, request: InitRequest, context):
        for credentials in request.credentials:
            if credentials.HasField("s3_credentials"):
                s3_credentials = credentials.s3_credentials
                for key in (
                    "endpoint_url",
                    "access_key_id",
                    "secret_access_key",
                    "session_token",
                ):
                    if s3_credentials.HasField(key):
                        os.environ[f"AWS_{key.upper()}"] = getattr(
                            s3_credentials, key
                        )
                # region_name doesn't fit the pattern.
                if region := s3_credentials.region_name:
                    os.environ["AWS_DEFAULT_REGION"] = region

                # TLS validation is all done via AWS_CA_BUNDLE:
                #   If an empty string: disable verification
                #   If a string, the file to the CA bundle PEM
                #   Else, the default CA is used.

                # The protobuf message encode this tri-state using 2 values:
                if not s3_credentials.verify_tls:
                    # Exporting an empty string as AWS_CA_BUNDLE disables TLS
                    # verification. This takes precedence over ca_bundle
                    os.environ["AWS_CA_BUNDLE"] = ""
                elif ca_bundle := s3_credentials.ca_bundle:
                    with tempfile.NamedTemporaryFile(delete=False) as f:
                        atexit.register(os.unlink, f.name)
                        os.environ["AWS_CA_BUNDLE"] = f.name
                        f.write(ca_bundle)

        def compute(payload):
            try:
                self.logger.info(f"Starting to compute {payload.tensor_id}")
                result = payload.compute()
                expected_dtype = rtfx_to_torch_dtype(payload.dtype)
                # verify payload before postprocessing, as that can change
                # the shape.
                if (
                    list(result.shape) != list(payload.shape)
                    or result.dtype != expected_dtype
                ):
                    raise ValueError(
                        f"Expected shape {payload.shape} and dtype {expected_dtype} "
                        f"for tensor {payload.tensor_id}, "
                        f"but got shape {list(result.shape)} and dtype {result.dtype}"
                    )

                result = payload.postprocess(result)
                result = replace(payload, tensor=cstorch.to_numpy(result))

                self.logger.info(
                    f"Computed {result.tensor_id}, "
                    f"shape={result.tensor.shape}, "
                    f"dtype={result.tensor.dtype}"
                )
            except Exception as e:
                result = replace(payload, tensor=e)
            return result

        def compute_tensors():
            logging.info(f"Computing tensors with {self._max_workers} workers")
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                while not self._done_event.is_set():
                    try:
                        payload = self._tensor_queue.get(timeout=1)
                    except EmptyQueue:
                        continue

                    # Submit request for processing.
                    # Blocks if the queue is full.
                    self._results_queue.put(executor.submit(compute, payload))

        self.thread = Thread(target=compute_tensors)
        self.thread.start()

        return InitResponse()

    @_service_api()
    def UnarySendDeferredInput(
        self, request: SendDeferredInputRequest, context
    ):
        for tensor in request.tensors:
            kwargs = dict(
                tensor_id=tensor.tensor_id,
                shape=tensor.shape,
                dtype=tensor.dtype,
                post_materialize_hooks=tensor.post_materialize_hooks,
            )
            if tensor.HasField("graph_tensor"):
                t = GraphTensorPayload(tensor=tensor.graph_tensor, **kwargs)

            elif tensor.post_materialize_hooks:
                if tensor.HasField("scalar_broadcast_tensor"):
                    t = ScalarBroadcastTensorPayload(
                        tensor=tensor.scalar_broadcast_tensor, **kwargs
                    )
                elif tensor.HasField("s3_tensor"):
                    t = S3TensorPayload(tensor=tensor.s3_tensor, **kwargs)

            else:
                # No post materialize hooks means non-graph tensors
                # don't need to be handled by the sidecar
                continue

            self.logger.info(
                f"Received a {tensor.WhichOneof('tensor_impl')} request for tensor: {tensor.tensor_id}"
            )
            self._tensor_queue.put(t)

        return SendDeferredInputResponse()

    @_service_api()
    def UnaryGetOutput(self, request, context):
        self.logger.info(
            f"Received UnaryGetOutput request for tensor: {request.tensor_id}"
        )
        # We always receive results and requests in the same order
        # So, there's no need to cache results. Only if we move
        # to using multiple threads to compute results would we need to
        # check the ordering
        payload = self._results_queue.get().result()
        result = payload.tensor

        if isinstance(result, Exception):
            self.logger.error(
                "".join(
                    format_exception(type(result), result, result.__traceback__)
                )
            )
            return GetOutputResponse(
                code=WS_RT_INTERNAL_ERROR, message=str(result)
            )

        if payload.tensor_id != request.tensor_id:
            return GetOutputResponse(
                code=WS_RT_INTERNAL_ERROR,
                message=f"Requested tensor {request.tensor_id} but got {payload.tensor_id}",
            )

        try:
            shm = SharedMemory(name=request.shm_name)

            # Disable mp resource tracking for each shared memory block in order
            # to prevent memory leak warnings when the resource tracker tries
            # to clean up. The weight host is responsible for creating this shared memory
            # region and cleaning it up, whereas in the weight sidecar we only consume it
            # and therefore don't need to track it. (ref: https://bugs.python.org/issue39959)
            # pylint: disable=protected-access
            resource_tracker.unregister(shm._name, 'shared_memory')

            # Runtime expects booleans as T_I1 which are with 16-bit tensors
            shared_array = np.ndarray(
                result.shape,
                result.dtype if result.dtype != bool else np.uint16,
                buffer=shm.buf,
            )
            np.copyto(shared_array, result, casting="same_kind")

            self.logger.info(
                f"Copied tensor {request.tensor_id} into shared memory"
            )
        except Exception as e:
            self.logger.error(
                "".join(format_exception(type(e), e, e.__traceback__))
            )
            return GetOutputResponse(code=WS_RT_INTERNAL_ERROR, message=str(e))
        finally:
            shm.close()

        return GetOutputResponse()

    @_service_api()
    def UnaryFinalize(self, request, context):
        return FinalizeResponse()

    @_service_api()
    def UnaryNotifyStall(self, request, context):
        return NotifyStallResponse()

    @_service_api()
    def UnaryNotifyCrash(self, request, context):
        return NotifyCrashResponse()

    @_service_api()
    def UnaryDone(self, request, context):
        self.logger.info(f"Received UnaryDone request")
        self._done_event.set()
        return DoneResponse()

    @_service_api()
    def UnaryClockSync(self, request, context):
        return ClockSyncResponse()


def main():
    """Start the weight init/load server."""
    parser = argparse.ArgumentParser(
        "Wafer-Scale Cluster weight init/load service."
    )
    parser.add_argument(
        '-a',
        '--all_details',
        required=False,
        type=str,
        help="Path to file containing json protobuf of SendToEveryTask",
    )
    default_task_type = ClusterDetails.TaskInfo.TaskType.Name(
        ClusterDetails.TaskInfo.TaskType.WGT
    )
    parser.add_argument(
        '-t',
        '--task_type',
        required=False,
        default=default_task_type,
        type=str,
        help=f"Task type. Default is '{default_task_type}'.",
    )
    args = parser.parse_args()

    all_details = resolve_command_line_arg(
        args.all_details,
        "CEREBRAS_CLUSTER_ALL_DETAILS_FP",
        "Cluster details json file path",
        "cs_weight_app",
        "-a/--all_details",
    )

    task_type = args.task_type
    if (
        task_type == default_task_type
        and "--task_type" not in sys.argv
        and "-t" not in sys.argv
    ):
        task_type = resolve_command_line_arg(
            "",
            "CEREBRAS_CLUSTER_SHORT_TASK_TYPE",
            "Task type",
            "cs_weight_app",
            "-t/--task_type",
            required=False,
            default=default_task_type,
        )

    sent_data = _proto_msg_from_jsonfile(
        SendToEveryTask,
        all_details,
        ignore_unknown=True,  # Ignore the `_comment` field.
    )

    _setup_logging(sent_data.debug_args.debug_wgt.log_settings)

    cluster_details_parser = ClusterDetailsParser(sent_data.cluster_details)
    task_type_str = task_type.strip().upper()
    task_type = ClusterDetails.TaskInfo.TaskType.Value(task_type_str)

    wse_id, _ = cluster_details_parser.extract_wse_details(
        task_type, sent_data.id
    )[0]

    logging.info(f"Task details:")
    logging.info(f"  Task type: {task_type_str}")
    logging.info(f"  Task ID:   {sent_data.id}")
    logging.info(f"  WSE ID:    {wse_id}")

    _serve(
        cluster_details_parser.extract_sidecar_address(task_type, sent_data.id),
        WeightSidecarServicer(
            sent_data.cluster_details,
            sent_data.debug_args,
            sent_data.id,
        ),
    )


if __name__ == "__main__":
    # PyTorch DataLoader in multi-processing mode will try to create sockets
    # with temporary file descriptors when using fork. If the directory name
    # is too long, it causes "AF_UNIX path too long". So we reset TMPDIR to
    # avoid these issues.
    tempfile.tempdir = None  # Clear the cached tempdir if it already exists
    os.environ["TMPDIR"] = ""

    main()
