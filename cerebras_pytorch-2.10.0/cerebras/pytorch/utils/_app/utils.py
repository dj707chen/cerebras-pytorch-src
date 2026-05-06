#!/usr/bin/env python3
# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import functools
import inspect
import logging
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol, runtime_checkable

import grpc
from google.protobuf import json_format

from cerebras.appliance import log
from cerebras.appliance.appliance_client import MAX_MESSAGE_LENGTH
from cerebras.appliance.pb.framework import appliance_service_pb2_grpc
from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    LogSettings,
)


def _service_api(log_entry: bool = True, log_exit: bool = True) -> Callable:
    """Decorator for gRPC API methods.

    Args:
        log_entry: Whether to log entry to this API.
        log_exit: Whether to log exit from this API.
    """

    def method_decorator(orig_method: Callable) -> Callable:
        # pylint: disable=protected-access
        @functools.wraps(orig_method)
        def wrapped_method(self, request, context: grpc.ServicerContext):
            """Replacement method for the original method."""

            api_name = f"{self.__class__.__name__}::{orig_method.__name__}"

            if log_entry:
                self.logger.debug(f"Entering service API method: {api_name}")

            exc_info = None
            try:
                response = orig_method(self, request, context)
                if log_exit:
                    if inspect.isgenerator(response):

                        def response_with_logging():
                            yield from response
                            self.logger.debug(
                                f"Exiting service API method: {api_name}"
                            )

                        return response_with_logging()
                    else:
                        self.logger.debug(
                            f"Exiting service API method: {api_name}"
                        )
                return response
            except Exception as e:
                if log_exit:
                    self.logger.exception(
                        f"API call {api_name} failed due to: {e}"
                    )

                exc_info = "".join(
                    traceback.format_exception(
                        type(e), value=e, tb=e.__traceback__
                    )
                )

            if exc_info is not None:
                context.abort(grpc.StatusCode.INTERNAL, exc_info)

        return wrapped_method

    return method_decorator


@runtime_checkable
class ApplianceServicer(Protocol):
    def wait(self) -> None: ...


def _serve(
    address: str, servicer: appliance_service_pb2_grpc.ApplianceServicer
) -> None:
    """Simple GRPC server management."""
    if not isinstance(servicer, appliance_service_pb2_grpc.ApplianceServicer):
        raise TypeError("servicer must be an ApplianceServicer")
    if not isinstance(servicer, ApplianceServicer):
        raise TypeError("servicer must implement a wait method")

    channel_options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_metadata_size', MAX_MESSAGE_LENGTH),
    ]
    server = grpc.server(
        ThreadPoolExecutor(max_workers=4),
        options=channel_options,
    )
    appliance_service_pb2_grpc.add_ApplianceServicer_to_server(servicer, server)
    server.add_insecure_port(address)
    server.start()
    logging.info(
        f"{servicer.__class__.__name__} Server started successfully. "
        f"Listening on: {address}"
    )
    servicer.wait()

    # 90s buffer to enable any pending calls to complete before
    # a forced server shut down. This is particularly important
    # for relaying large error messages over gRPC.
    grace = 90
    logging.info(
        f"{servicer.__class__.__name__} has finished. "
        f"Shutting down gracefully with a grace period of {grace} seconds."
    )
    server.stop(grace)
    logging.info(f"{servicer.__class__.__name__} has been shut down")


def _setup_logging(log_settings: LogSettings) -> None:
    """Sets up logging for the worker.

    Args:
        log_settings: Log settings for the worker.
    """

    def _pb_level_to_py_level(pb_level: int) -> int:
        name = LogSettings.LogLevel.Name(pb_level)
        levelno = log.get_level_name(name)
        if not isinstance(levelno, int):
            raise ValueError(f"Unhandled log level: {name}")
        return levelno

    default_level = _pb_level_to_py_level(log_settings.global_level)
    if default_level == logging.NOTSET:
        default_level = logging.INFO

    for message_tag in log_settings.message_tags:
        logging.getLogger(message_tag.tag).setLevel(
            _pb_level_to_py_level(message_tag.level)
        )

    logging.basicConfig(
        level=default_level,
        format=("%(asctime)s %(levelname)8s: [pid(%(process)5d)] %(message)s"),
        handlers=[logging.StreamHandler()],
    )


def _proto_msg_from_jsonfile(proto_cls, json_file, ignore_unknown=False):
    try:
        with open(json_file, 'r') as filehandle:
            json_text = filehandle.read()
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid JSON file") from exc

    try:
        proto_msg = json_format.Parse(
            json_text, proto_cls(), ignore_unknown_fields=ignore_unknown
        )
    except (UnicodeDecodeError, json_format.ParseError) as exc:
        raise ValueError("Invalid JSON text") from exc
    return proto_msg


def _format_stack_traces() -> str:
    """Returns stack traces of all currently active threads."""
    frames = sys._current_frames()

    result = ""
    for thread in threading.enumerate():
        result += f"Stack trace for thread with id={thread.ident} name={thread.name}\n"
        try:
            result += "".join(
                str(x) for x in traceback.format_stack(frames[thread.ident])
            )
        except:
            result += "\tCould not get stack trace for thread\n"
    return result
