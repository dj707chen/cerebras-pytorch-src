#!/usr/bin/env python3
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""GRPC Server For streamer role."""
import argparse
import atexit
import itertools
import json
import logging
import os
import pprint
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Generator, List, Optional

import numpy as np

from cerebras.appliance import log
from cerebras.appliance.appliance_client import (
    fw_user_deserialize,
    fw_user_serialize,
)
from cerebras.appliance.cluster.cluster_details import ClusterDetailsParser
from cerebras.appliance.pb.framework import appliance_service_pb2_grpc
from cerebras.appliance.pb.framework.appliance_service_pb2 import (
    DataCheckpointResponse,
    DoneResponse,
    FinalizeRequest,
    FinalizeResponse,
    GetOutputResponse,
    LoadRequest,
    LoadResponse,
    NotifyCrashResponse,
    NotifyStallResponse,
    RunRequest,
    RunResponse,
    StartStreamingResponse,
)
from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ClusterDetails,
    DebugArgs,
    SendToEveryTask,
)
from cerebras.appliance.pb.ws.common_pb2 import ClockSyncResponse, PingResponse
from cerebras.appliance.utils import resolve_command_line_arg
from cerebras.pytorch.distributed.cluster_resolver import (
    ApplianceWorkerClusterResolver,
)
from cerebras.pytorch.utils._app.utils import (
    _format_stack_traces,
    _proto_msg_from_jsonfile,
    _serve,
    _service_api,
    _setup_logging,
)
from cerebras.pytorch.utils.data.streamer.data_pipe import SampleSpec
from cerebras.pytorch.utils.data.streamer.pytorch_generator import (
    InputGeneratorPyTorch,
)
from cerebras.pytorch.utils.nest import deserialize_visit_fns


@log.named_class_logger("streamer.StreamerServicer")
class StreamerServicer(
    appliance_service_pb2_grpc.ApplianceServicer, log.ClassLogger
):
    """Streamer service for running the user-defined input pipeline and returning data."""

    def __init__(
        self,
        cluster_details: ClusterDetails,
        debug_args: DebugArgs,
        role_id: int,
    ):
        super().__init__()

        self._done_event = threading.Event()
        self._debug_args = debug_args
        self._shm_arrays: Dict[int, SharedMemory] = {}
        self._sample_spec: Optional[SampleSpec] = None
        self._subbatch_sizes: Optional[List[List[int]]] = None
        self._async_generators: Dict[int, _AsyncInputGenerator] = {}
        self._curr_generator_id: Optional[int] = None

        # Register any and all visit functions that we might need to recursively
        # traverse the data structures used by the dataloader
        deserialize_visit_fns(self._debug_args.debug_wrk.visit_fn_map)

        # Inject user-defined env variables so dataloader has access to them.
        os.environ.update(dict(self._debug_args.debug_wrk.env_vars))

        # Register a handler to clean up shared memory at exit
        atexit.register(self._close_shm)

        # Configure the cluster resolver
        ApplianceWorkerClusterResolver.configure_from_cluster_details(
            role_id, ClusterDetailsParser(cluster_details)
        )

    def wait(self):
        """Blocks until the server is done."""
        self._done_event.wait()

    def _curr_generator(
        self, raise_if_missing: bool = True
    ) -> Optional["_AsyncInputGenerator"]:
        if self._curr_generator_id is None:
            if raise_if_missing:
                raise RuntimeError(
                    "No input generator has been instantiated yet."
                )
            return None

        generator = self._async_generators.get(self._curr_generator_id)
        if generator is None and raise_if_missing:
            raise RuntimeError(
                f"Invalid input generator ID {self._curr_generator_id}. "
                f"Available IDs are: {list(self._async_generators.keys())}"
            )

        return generator

    def __del__(self):
        self.logger.info(f"{self.__class__.__name__} is being destructed.")

    @_service_api()
    def UnaryLoad(self, request: LoadRequest, context):
        self.logger.info(f"Loading a new IR")

        # Clear artifacts from previous session
        self._close_shm()

        # Load spec from the new session
        sample_spec_json = json.loads(request.rtir.content)
        self._subbatch_sizes = sample_spec_json.pop("subbatch_sizes")
        self._sample_spec = SampleSpec.from_dict(sample_spec_json)

        self.logger.info(f"Subbatch sizes: {self._subbatch_sizes}")
        self.logger.info(
            f"Sample spec for the new IR is:\n"
            f"{pprint.pformat(self._sample_spec.tensors)}"
        )

        # For live dataloaders that support checkpointing, we expect the caller to use
        # initial state for resuming between sessions. So if the current dataloader
        # supports checkpointing, we destruct it here.
        if self._curr_generator_id is not None and (
            self._async_generators[
                self._curr_generator_id
            ].input_generator.supports_checkpointing()
        ):
            self.logger.info(
                f"Removing previous dataloader with ID {self._curr_generator_id} "
                f"since it supports checkpointing."
            )
            self._async_generators.pop(self._curr_generator_id)
        self._curr_generator_id = None

        return LoadResponse()

    @_service_api()
    def UnaryRun(self, request: RunRequest, context):
        for tensor_spec in self._sample_spec.tensors:
            self._shm_arrays[tensor_spec.tensor_id] = SharedMemory(
                name=tensor_spec.shm_name
            )
            # Disable mp resource tracking for each shared memory block in order
            # to prevent memory leak warnings when the resource tracker tries
            # to clean up. The worker is responsible for creating this shared memory
            # region and cleaning it up, whereas in the streamer we only consume it
            # and therefore don't need to track it. (ref: https://bugs.python.org/issue39959)
            # pylint: disable=protected-access
            resource_tracker.unregister(
                self._shm_arrays[tensor_spec.tensor_id]._name, 'shared_memory'
            )

        self.logger.info(
            f"Received a run request:"
            f"\n\tnum_iterations={request.num_iterations}"
            f"\n\tcheckpoint_schedule={request.checkpoint_schedule}"
            f"\n\tactivation_freq={request.activation_freq}"
            f"\n\tlive_dataloaders={request.live_dataloaders}"
            f"\n\texisting_dataloaders={list(self._async_generators.keys())}"
            f"\n\tdataloader_id={request.dataloader.id}"
            f"\n\thas_initial_state={request.dataloader.initial_state != ''}"
        )

        if to_delete := set(self._async_generators) - set(
            request.live_dataloaders
        ):
            self.logger.info(
                f"Removing the following inactive dataloader IDs "
                f"from previous sessions: {to_delete}"
            )
            for dataloader_id in to_delete:
                self._async_generators.pop(dataloader_id)

        if (
            request.dataloader.id in self._async_generators
            and not request.dataloader.initial_state
        ):
            self.logger.info(
                f"Resuming existing generator with id {request.dataloader.id}"
            )
            self._curr_generator_id = request.dataloader.id
            self._curr_generator().input_generator.update_spec(
                self._sample_spec, self._subbatch_sizes
            )
        elif request.dataloader.builder:
            self.logger.info(
                f"Creating new generator with id {request.dataloader.id}"
            )

            dataloader = fw_user_deserialize(
                request.dataloader.builder,
                name="DataLoader function",
                from_usr=True,
            )
            args, kwargs = fw_user_deserialize(
                request.dataloader.builder_inputs,
                name="DataLoader args/kwargs",
                from_usr=True,
            )

            state_dict = None
            if request.dataloader.initial_state:
                state_dict = fw_user_deserialize(
                    request.dataloader.initial_state,
                    name="DataLoader state_dict",
                    from_usr=True,
                )

            dataloader_fn = lambda: dataloader(*args, **kwargs)

            total_steps = request.num_iterations
            assert total_steps > 0

            iterables = [[0]]
            for interval in request.checkpoint_schedule.intervals:
                # checkpoint_schedule is zero-indexed, but requests for checkpoints from client
                # are 1-indexed, hence the +1.
                iterables.append(
                    range(interval.start + 1, interval.end + 1, interval.step)
                )
                if interval.include_last:
                    iterables.append([interval.end])
            # Always request checkpointing at the final step to handle resumption
            # across session without having to keep the dataloader alive.
            iterables.append([total_steps])
            checkpoint_schedule = itertools.chain(*iterables)

            # Prime the data pipeline asynchronously
            self._curr_generator_id = request.dataloader.id
            self._async_generators[self._curr_generator_id] = (
                _AsyncInputGenerator()
            )
            self._async_generators[self._curr_generator_id].start(
                dataloader_fn=dataloader_fn,
                input_spec=self._sample_spec,
                subbatch_sizes=self._subbatch_sizes,
                checkpoint_schedule=checkpoint_schedule,
                state_dict=state_dict,
                debug_args=self._debug_args,
            )
        else:
            raise RuntimeError(
                f"Expected a run request to provide an existing dataloader ID or a new builder "
                f"function. Requested dataloader ID is {request.dataloader.id} but available "
                f"dataloader IDs are: {list(self._async_generators.keys())}"
            )

        return RunResponse()

    @_service_api()
    def UnaryPing(self, request, context):
        if (
            generator := self._curr_generator(raise_if_missing=False)
        ) is not None:
            generator.wait()
        return PingResponse()

    @_service_api()
    def UnaryStartStreaming(self, request, context):
        self._curr_generator().wait()
        return StartStreamingResponse()

    @_service_api()
    def UnaryGetOutput(self, request, context):
        for tensor_id, tensor in next(self._curr_generator()).items():
            # Runtime expects booleans as T_I1 which are with 16-bit tensors
            shared_array = np.ndarray(
                tensor.shape,
                tensor.dtype if tensor.dtype != bool else np.uint16,
                buffer=self._shm_arrays[tensor_id].buf,
            )
            np.copyto(shared_array, tensor, casting="same_kind")

        return GetOutputResponse()

    @_service_api()
    def UnaryDataCheckpoint(self, request, context):
        self.logger.info(
            f"Fetching dataloader state at step: {request.iteration}"
        )

        state_dict = self._curr_generator().input_generator.state_dict(
            request.iteration
        )
        # Return empty state_dict to let caller know that checkpoint is not ready yet
        return DataCheckpointResponse(
            state_dict_serialized=[
                (
                    ""
                    if state_dict is None
                    else fw_user_serialize(
                        state_dict, name="DataLoader state", from_usr=False
                    )
                )
            ]
        )

    @_service_api()
    def UnaryFinalize(self, request, context):
        if request.state == FinalizeRequest.FS_STALL:
            self.logger.info(
                "###########################################################"
            )
            self.logger.info(
                "Stall detected. Dumping stack traces of all active threads"
            )
            self.logger.info(
                "###########################################################"
            )
            self.logger.info(_format_stack_traces())
            self.logger.info(
                "###########################################################"
            )
        return FinalizeResponse()

    @_service_api()
    def UnaryNotifyStall(self, request, context):
        return NotifyStallResponse()

    @_service_api()
    def UnaryNotifyCrash(self, request, context):
        return NotifyCrashResponse()

    @_service_api()
    def UnaryDone(self, request, context):
        self._done_event.set()
        self._close_shm()
        if (
            generator := self._curr_generator(raise_if_missing=False)
        ) is not None:
            generator.wait()
        return DoneResponse()

    @_service_api()
    def UnaryClockSync(self, request, context):
        return ClockSyncResponse()

    def _close_shm(self) -> None:
        for shm in self._shm_arrays.values():
            shm.close()
        self._shm_arrays.clear()


@log.named_class_logger
class _AsyncInputGenerator(log.ClassLogger):
    """A shallow wrapper around input generator that primes the pipeline asynchronously."""

    def __init__(self):
        self._input_generator: Optional[InputGeneratorPyTorch] = None
        self._iterator: Optional[
            Generator[Dict[int, np.ndarray], None, None]
        ] = None

        self._threadpool: Optional[ThreadPoolExecutor] = None
        self._first_batch_future: Optional[Future] = None

    @property
    def input_generator(self) -> InputGeneratorPyTorch:
        """Returns the underlying input generator."""
        self.wait()
        if self._input_generator is None:
            raise RuntimeError(f"Streamer has not started yet.")
        return self._input_generator

    def __next__(self) -> Dict[int, np.ndarray]:
        self.wait()
        if self._iterator is None:
            raise RuntimeError(f"Streamer has not started yet.")
        return next(self._iterator)

    def start(self, *args, **kwargs):
        """Starts priming the data pipeline asynchronously."""
        self._threadpool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="streamer-primer"
        )
        self._first_batch_future = self._threadpool.submit(
            self._prime_streamer, *args, **kwargs
        )
        # Shutdown the primer once the single task above is complete.
        self._threadpool.shutdown(wait=False)

    def wait(self):
        """Waits for the streamer to prime the data pipeline.

        Raises:
            Exception if the primer thread ran into an error.
        """
        if self._threadpool is not None:
            self.logger.info("Waiting for streamer to be primed")
            # This blocks and raises an exception if the task ran into error
            self._first_batch_future.result()
            self._first_batch_future = None
            self._threadpool = None
            self.logger.info("Streamer has been primed")

    def _prime_streamer(self, *args, **kwargs):
        """Primes the streamer and returns the first batch for later retreival.

        Priming refers to creating the input pipeline and iterating it for one step.

        Args:
            *args, **kwargs: Forwarded args for creating `InputGeneratorPyTorch`.
        """
        try:
            self.logger.info("Creating the input generator")
            self._input_generator = InputGeneratorPyTorch(*args, **kwargs)
            self.logger.info("Creating iterator over the inputs")
            self._iterator = iter(self._input_generator)
            self.logger.info("Creating the first batch of data")
            data = next(self._iterator)
            self.logger.info("Created the first batch of data")
            self._iterator = itertools.chain([data], self._iterator)
        except Exception as e:
            self.logger.exception(
                f"Streamer ran into error during priming: {e}"
            )
            raise


def main():
    """Start the streamer server."""
    parser = argparse.ArgumentParser("Wafer-Scale Cluster streamer service.")
    parser.add_argument(
        '-a',
        '--all_details',
        required=False,
        type=str,
        help="Path to file containing json protobuf of SendToEveryTask",
    )
    args = parser.parse_args()

    all_details = resolve_command_line_arg(
        args.all_details,
        "CEREBRAS_CLUSTER_ALL_DETAILS_FP",
        "Cluster details json file path",
        "cs_streamer_app",
        "-a/--all_details",
    )

    sent_data = _proto_msg_from_jsonfile(
        SendToEveryTask,
        all_details,
        ignore_unknown=True,  # Ignore the `_comment` field.
    )

    _setup_logging(sent_data.debug_args.debug_wrk.log_settings)

    cluster_details_parser = ClusterDetailsParser(sent_data.cluster_details)
    wse_id, _ = cluster_details_parser.extract_wse_details(
        ClusterDetails.TaskInfo.TaskType.WRK, sent_data.id
    )[0]

    logging.info("Task details:")
    logging.info(f" Task ID: {sent_data.id}")
    logging.info(f" WSE ID: {wse_id}")

    _serve(
        cluster_details_parser.extract_stm_address(sent_data.id),
        StreamerServicer(
            sent_data.cluster_details,
            sent_data.debug_args,
            sent_data.id,
        ),
    )


if __name__ == "__main__":
    # PyTorch DataLoader in multi-processing mode will try to create sockets
    # with temporary file descriptors when using fork. If the direcotry name
    # is too long, it causes "AF_UNIX path too long". So we reset TMPDIR to
    # avoid these issues.
    tempfile.tempdir = None  # Clear the cached tempdir if it already exists
    os.environ["TMPDIR"] = ""

    main()
