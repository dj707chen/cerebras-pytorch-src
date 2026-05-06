# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Any, Callable, List, Optional

from cerebras.appliance.pb.framework.appliance_service_pb2 import LoadRequest

__all__ = [
    "ProfilerRegistry",
    "profile",
    "schedule",
    "tensorboard_trace_handler",
]


class ProfilerRegistry:
    """Registry which stores the cstorch op profiler instance"""

    _profiler = None

    @classmethod
    def set_profiler(cls, profiler):
        """Stores the profiler instance

        Args:
            profiler: The profiler instance which needs to be stored.
        """
        if cls._profiler is not None:
            raise RuntimeError(
                "A profiler has already been set. Only one profiler instance is allowed."
            )
        cls._profiler = profiler

    @classmethod
    def get_profiler(cls):
        """Returns the profiler instance"""
        return cls._profiler

    @classmethod
    def unset_profiler(cls):
        """Deletes the profiler instance stored with it"""
        cls._profiler = None


def schedule(*, start_step: int, end_step: int) -> Callable:
    """
    This function is used to set the range of profiling iterations
    for the cstorch op profiler.

    Args:
        start_step (int): The starting step from where the profiling starts.
        end_step (int): The end step (including) after which the profiling ends.
    """
    if start_step > end_step:
        raise ValueError(
            "The value of start step should be less than or equal to end_step"
        )

    def schedule_fn() -> List[int]:
        return [start_step, end_step]

    return schedule_fn


def tensorboard_trace_handler(
    dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False
):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.

    Modified from https://pytorch.org/docs/stable/_modules/torch/profiler/profiler.html#profile
    """
    import os
    import socket
    import time

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        prof.export_chrome_trace(os.path.join(dir_name, file_name))

    handler_fn.origin = tensorboard_trace_handler
    return handler_fn


class profile:
    """Class to encapsulate all the profiling info
    that user requests.
    """

    def __init__(
        self,
        *,
        schedule: Callable[[], List[int]],
        host_activities: Optional[List[str]] = None,
        on_trace_ready: Optional[Callable[..., Any]] = None,
    ):
        """Initialize the profiler with schedule and profilerActivity

        Args:
            schedule: The input schedule which encapsulates the range of
                      profiling interation.
            host_activities: List of activities which stores different devices
                        where the profiling needs to be done.
        """
        self.host_activities = host_activities
        self.schedule = schedule
        self.appliance_response = None

        if (
            hasattr(on_trace_ready, 'origin')
            and on_trace_ready.origin is tensorboard_trace_handler
        ):
            import pkg_resources

            try:
                pkg_resources.get_distribution('torch_tb_profiler')
            except pkg_resources.DistributionNotFound:
                raise ImportError(
                    "TensorBoard trace viewing requires pip package `torch_tb_profiler`. Please install `torch_tb_profiler` for TensorBoard support"
                )
        self.on_trace_ready = on_trace_ready

    def __enter__(self):
        # Start the profiler
        # Register the profiler instance
        ProfilerRegistry.set_profiler(self)
        return self

    def __exit__(self, *args):
        # Unregister the profiler instance
        ProfilerRegistry.unset_profiler()

    def get_summary(self):
        if self.appliance_response:
            import pandas as pd
            from google.protobuf.json_format import MessageToDict
            from tabulate import tabulate

            chrome_trace_obj = MessageToDict(
                self.appliance_response, preserving_proto_field_name=True
            )
            chrome_trace_obj.pop("code", None)
            chrome_trace_obj.pop("message", None)
            trace_events = chrome_trace_obj["trace_events"]

            for item in trace_events:
                item.setdefault('ts', 0.0)

            total_exec_time = trace_events[-1]["dur"] + trace_events[-1]["ts"]
            df = pd.DataFrame(data=trace_events, columns=['name', 'dur'])
            df = (
                df[~((df["name"].str.contains(" = ")))]
                .groupby('name', as_index=False)['dur']
                .sum()
                .sort_values(by=['dur'], ascending=False)
                .reset_index(drop=True)
                .head(10)
            )
            df["% CSX TIME"] = (df["dur"] * 100) / total_exec_time
            df["dur"] = df["dur"].div(100).round(2)  # Change the time to ms
            df.rename(
                columns={
                    'name': 'PYTORCH MODULE NAME',
                    'dur': 'CSX TIME (in ms)',
                },
                inplace=True,
            )
            return tabulate(df, headers='keys', tablefmt='psql')

    def export_chrome_trace(self, path: str):
        if self.appliance_response:
            import json

            from google.protobuf.json_format import MessageToDict

            with open(path, "w") as out_file:
                chrome_trace_obj = MessageToDict(
                    self.appliance_response, preserving_proto_field_name=True
                )
                chrome_trace_obj.pop("code", None)
                chrome_trace_obj.pop("message", None)
                chrome_trace_obj['traceEvents'] = chrome_trace_obj.pop(
                    'trace_events'
                )

                for item in chrome_trace_obj['traceEvents']:
                    item.update({'ph': 'X', 'pid': 0, 'tid': 0})
                    item.setdefault('ts', 0.0)
                    item.setdefault('dur', 0.0)
                    p_ops = item.pop("pytorch_ops", None)
                    if p_ops is not None:
                        item["args"] = {}
                        item["args"]["pytorch_ops"] = p_ops
                        item["args"]["rtir_op_info"] = item.pop(
                            "rtir_op_info", None
                        )

                chrome_trace_obj['traceEvents'] = [
                    item
                    for item in chrome_trace_obj['traceEvents']
                    if item["dur"] != 0.0
                ]

                chrome_trace_obj["traceEvents"] = [
                    {
                        key: value
                        for key, value in dictionary.items()
                        if key != 'event_type'
                    }
                    for dictionary in chrome_trace_obj["traceEvents"]
                ]

                chrome_trace_obj["displayTimeUnit"] = "ns"
                logging.info(f"chrome trace object: {chrome_trace_obj}")
                json.dump(chrome_trace_obj, out_file, indent=6)

    def _trace_ready(self):
        if self.on_trace_ready:
            self.on_trace_ready(self)

    def configure_profiler(self, is_csx: bool, num_steps: int):
        [start_step, end_step] = self.schedule()
        # For modelzoo models, the default value of -1 means that the
        # profiling was not asked for.
        op_profiler_config = None
        if start_step != -1:
            if is_csx:
                if num_steps >= end_step >= start_step >= 1:
                    op_profiler_config = LoadRequest.OpProfilerConfig()
                    op_profiler_config.start_step = start_step
                    op_profiler_config.end_step = end_step
                else:
                    raise ValueError(
                        f"profiling step range should be between 1(inclusive) and {num_steps}(exclusive), "
                        f"instead it is {start_step}:{end_step}"
                    )

                if not self.host_activities:
                    op_profiler_config.csx_list.append(0)
                else:
                    import re

                    pattern = r'([A-Z]+)-(\d+|\*)'
                    for item in self.host_activities:
                        match = re.match(pattern, item)
                        if match:
                            # Extract the hostname and number from the match groups
                            host_name = match.group(1)
                            star = False
                            host_number = -1
                            if match.group(2) == '*':
                                star = True
                            else:
                                host_number = int(match.group(2))

                            if host_name == "ACT":
                                if star:
                                    op_profiler_config.act_host_list.append(-1)
                                else:
                                    op_profiler_config.act_host_list.append(
                                        host_number
                                    )
                            elif host_name == "WGT":
                                if star:
                                    op_profiler_config.wgt_host_list.append(-1)
                                else:
                                    op_profiler_config.wgt_host_list.append(
                                        host_number
                                    )
                            elif host_name == "CSX":
                                if star:
                                    op_profiler_config.csx_list.append(-1)
                                else:
                                    op_profiler_config.csx_list.append(
                                        host_number
                                    )
                            else:
                                raise ValueError(
                                    "Incorrect host name format. Please use either ACTHOST or WGTHOST or CSX"
                                )
                        else:
                            raise ValueError(
                                "Incorrect host. Please input the hostname first followed by the server number. For ex - ACTHOST0, WGTHOST0, CSX0, CSX1, etc"
                            )
            elif start_step >= 0:
                raise RuntimeError(f"Only CSX backend can be profiled")

        return op_profiler_config
