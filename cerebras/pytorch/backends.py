# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
""" Backends configuration flags. """

import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import cerebras.appliance
from cerebras.appliance.appliance_client import (
    MAX_MESSAGE_LENGTH,
    MAX_TRANSFER_BYTES,
)
from cerebras.appliance.cluster.config import FABRIC_TYPE_CS2, FABRIC_TYPE_CS3
from cerebras.appliance.utils.debug_args import DebugArgs, DebugArgsDescriptor
from cerebras.appliance.utils.descriptor import Descriptor
from cerebras.appliance.utils.ini import set_ini
from cerebras.pytorch.core.compile import RetraceEveryIteration
from cerebras.pytorch.core.device import Config as DeviceConfig
from cerebras.pytorch.utils._flags import DEFAULT, _flags
from cerebras.pytorch.utils._micro_batch_size import MicroBatchSize
from cerebras.pytorch.utils.pol import POL


@dataclass
class _backends(_flags):
    """
    Backends configuration flags.

    Attributes:
        csx: CSX configuration flags
        DEFAULT: The sentinel value used to reset a flag back to its default value
    """

    DEFAULT = DEFAULT

    @dataclass
    class _csx(_flags):
        """
        CSX configuration flags.

        Attributes:
            precision: Precision configuration flags
            performance: Performance configuration flags
            debug: Debug configuration flags
        """

        @dataclass
        class _precision(_flags):
            """
            CSX precision configuration flags.

            Attributes:
                optimization_level: The precision optimization level (POL) to use
                    when compiling the model. The POL determines the level of
                    precision to use for the model's weights and activations and
                    can thus affect the model's accuracy and performance.
            """

            optimization_level: int = POL(default=1)

        @dataclass
        class _performance(_flags):
            """
            CSX performance configuration flags.

            Attributes:
                micro_batch_size: The micro-batch size to use when compiling the
                    model. The micro-batch size can affect the model's performance
                    and memory usage.

                    Valid values include:

                    * "auto": Automatically choose an optimal micro batch size.

                    * "explore": Search for an optimal micro batch size and return.

                    *
                      .. code:: python

                          {
                              "explore": {
                                  "min": Optional[<positive_int>],
                                  "max": Optional[<positive_int>]
                              }
                          }

                      Search for an optimal micro batch size within the min and
                      max bounds and return.

                    * <positive_int>: Use this micro batch size.

                    * None: Disable micro batch tiling.

                transfer_processes: The number of processes to use for transferring data
                    to and from the Wafer Scale Cluster.

                max_transfer_bytes: The maximum number of bytes to transfer at
                    once in a single transfer process.

                compress_weights: Whether to compress weights when saving them to storage.

                use_speculative_optimizers: Whether to enable speculative optimizers.
            """

            micro_batch_size: MicroBatchSize.Type = MicroBatchSize(
                default="auto"
            )

            class _TransferProcessesDescriptor(Descriptor):
                def set_attr(self, obj, value):
                    if not isinstance(value, int) or value <= 0:
                        raise ValueError(
                            f"transfer_processes must be a positive integer, got {value}"
                        )
                    super().set_attr(obj, value)

            transfer_processes: int = _TransferProcessesDescriptor(default=5)

            class _MaxTransferBytesDescriptor(Descriptor):
                def set_attr(self, obj, value):
                    if (
                        not isinstance(value, int)
                        or value > MAX_MESSAGE_LENGTH
                        or value < 0
                    ):
                        raise ValueError(
                            f"max_transfer_bytes must be a positive integer "
                            f"less than {MAX_MESSAGE_LENGTH}, got {value}"
                        )
                    super().set_attr(obj, value)

            max_transfer_bytes: int = _MaxTransferBytesDescriptor(
                default=MAX_TRANSFER_BYTES
            )

            compress_weights: bool = False

            class _SpecOptDescriptor(Descriptor):
                def set_attr(self, obj, value):
                    if hasattr(obj, self._attr_name):
                        super().set_attr(obj, value)
                        set_ini(
                            backends.csx.debug.debug_args,
                            ws_opt_disable_speculate_optimizer=not value,
                        )
                    else:
                        super().set_attr(obj, value)

            use_speculative_optimizers: bool = _SpecOptDescriptor(default=True)

        @dataclass
        class _cluster(_flags):
            """
            CSX cluster configuration flags.

            Attributes:
                auto_taint: Whether to automatically taint systems when a
                    taint action is found.
            """

            class _AutoTaintDescriptor(Descriptor):
                def set_attr(self, obj, value):
                    if hasattr(obj, self._attr_name):
                        super().set_attr(obj, value)
                        set_ini(
                            backends.csx.debug.debug_args,
                            ws_fwk_auto_taint=value,
                        )
                    else:
                        super().set_attr(obj, value)

            auto_taint: bool = _AutoTaintDescriptor(default=False)

        @dataclass
        class _storage(_flags):
            """
            CSX storage configuration flags.
            """

            @dataclass
            class _s3(_flags):
                """
                S3 configuration flags.
                """

                class _S3ProfileDescriptor(Descriptor):
                    def __get__(self, obj, _type):
                        return os.environ.get("AWS_PROFILE")

                    def __set__(self, obj, value):
                        if isinstance(value, str):
                            os.environ["AWS_PROFILE"] = value
                        else:
                            os.environ.pop("AWS_PROFILE", None)

                profile: Optional[str] = _S3ProfileDescriptor(default=None)

            s3: _s3 = field(default_factory=_s3)

        @dataclass
        class _debug(_flags):
            """
            CSX debug configuration flags.

            Attributes:
                ini: INI configuration flags
                debug_args: Debug arguments to pass to the cluster
                drop_data: Whether to drop weight data if its not needed
                lazy_initialization: Whether to lazily initialize weights
                skip_weight_init: Whether to skip weight initialization and simply
                    initialize everything to zeros
                optimize_initialization: Whether to optimize the weight initialization
                    graph. This should only be set to False if the optimization failed
                    for whatever reason.
                log_initialization: Whether to log initialization
                retrace_every_iteration: Whether to retrace the
                    training/validation graph every iteration
                implicit_cast_cb16_to_fp32: Experimental. Autocast cb16 tensor
                    to fp32 in step closures.
                save_gpv: Whether to save global perf view
                check_constraints: Whether to check constraints on Torch ops
                    in the MLIR file.
                op_definition_module: The module containing the definitions of ops to register
                    for constraint checking. If not provided, uses the default definitions module.
                skip_connectivity_check: Whether to skip the connectivity check
            """

            class INI(cerebras.appliance.utils.ini.INI):
                @property
                def _debug_args(self):
                    return backends.csx.debug.debug_args

            class INIDescriptor(Descriptor):
                def sanitize(self, value):
                    if isinstance(value, dict):
                        return self.default.__class__(**value)
                    return value

            ini: Union[INI, dict] = INIDescriptor(default=INI())

            class _DebugArgsDescriptor(DebugArgsDescriptor):
                def set_attr(self, obj, value):
                    if hasattr(obj, self._attr_name):
                        super().set_attr(obj, value)
                        # Only copy if backends has already been fully initialized
                        backends.csx.debug.ini.copy_from(
                            backends.csx.debug.debug_args
                        )
                    else:
                        super().set_attr(obj, value)

            debug_args: Union[DebugArgs, dict] = _DebugArgsDescriptor()

            drop_data: bool = DeviceConfig(default=False)
            lazy_initialization: bool = DeviceConfig(default=True)
            skip_weight_init: bool = DeviceConfig(default=False)
            optimize_initialization: bool = DeviceConfig(default=True)
            log_initialization: bool = True
            retrace_every_iteration: bool = RetraceEveryIteration(default=False)
            implicit_cast_cb16_to_fp32: bool = False
            save_gpv: bool = False
            check_constraints: bool = False
            op_definition_module: Optional[str] = None
            skip_connectivity_check: bool = False

            fabric_type_blacklist: Optional[
                List[Literal[FABRIC_TYPE_CS2, FABRIC_TYPE_CS3]]
            ] = None

            class MemoryLimit(Descriptor):
                """Descriptor for DebugArgs memory_limit."""

                def __init__(self, debug_arg_property):
                    super().__init__(default=None)
                    self.debug_arg_property = debug_arg_property

                def sanitize(self, value):
                    if value is not None:
                        getattr(
                            backends.csx.debug.debug_args.debug_usr,
                            self.debug_arg_property,
                        ).memory_bytes = (
                            value << 30
                        )  # gb to bytes conversion

                    return value

            compile_crd_memory_gi: Optional[int] = MemoryLimit(
                "compile_coord_resource"
            )
            execute_crd_memory_gi: Optional[int] = MemoryLimit(
                "execute_coord_resource"
            )
            wrk_memory_gi: Optional[int] = MemoryLimit("worker_resource")
            act_memory_gi: Optional[int] = MemoryLimit("activation_resource")
            cmd_memory_gi: Optional[int] = MemoryLimit("command_resource")
            wgt_memory_gi: Optional[int] = MemoryLimit("weight_resource")
            chf_memory_gi: Optional[int] = MemoryLimit("chief_resource")

        precision: _precision = field(default_factory=_precision)
        performance: _performance = field(default_factory=_performance)
        cluster: _cluster = field(default_factory=_cluster)
        storage: _storage = field(default_factory=_storage)
        debug: _debug = field(default_factory=_debug)

    csx: _csx = field(default_factory=_csx)


backends: _backends = _backends()
