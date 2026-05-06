# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Literal, Union

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.descriptor import Descriptor
from cerebras.appliance.utils.ini import INI


class MicroBatchSize(Descriptor):
    """
    Descriptor class for micro_batch_size setting.
    """

    Type = Union[
        None, int, Literal["explore", "auto"], Dict[str, Dict[str, int]]
    ]

    def set_attr(self, obj, micro_batch_size):
        if not hasattr(obj, self._attr_name):
            # If this is the first time we're setting the attribute
            # we will not be able to set the INI due to circular import
            # So, just set it for now and then allow a future "reset" to handle
            # setting the INI
            super().set_attr(obj, micro_batch_size)
            return

        error_msg = (
            f'Invalid value "{micro_batch_size}" for "micro_batch_size". Expected one of:'
            '\n\t"auto": Automatically choose an optimal micro batch size.'
            '\n\t"explore": Search for an optimal micro batch size and return.'
            '\n\t{"explore": {"min": Optional[<positive_int>], "max": Optional[<positive_int>]}}: '
            'Search for an optimal micro batch size within the min and max bounds and return.'
            "\n\t<positive_int>: Use this micro batch size."
            "\n\tNone: Disable micro batch tiling."
        )

        # Clear the following INI flags from the debug args
        ini = INI(
            ws_opt_explore_batch_sizes=0,
            ws_opt_batch_exploration_min_micro_batch=0,
            ws_opt_batch_exploration_max_micro_batch=0,
            ws_opt_force_min_grad_accum_batch=0,
            ws_opt_force_max_grad_accum_batch=0,
        )
        ini.clear(cstorch.backends.csx.debug.debug_args)

        # Clear these flags from the global INI
        cstorch.backends.csx.debug.ini -= ini

        ini = INI()
        ini.ws_opt_disable_grad_accum = micro_batch_size is None

        if micro_batch_size is None:
            # Already handled above, nothing to do here
            pass
        elif isinstance(micro_batch_size, str):
            if micro_batch_size == "auto":
                # Already handled above, nothing to do here
                pass
            elif micro_batch_size == "explore":
                ini.ws_opt_explore_batch_sizes = True
            else:
                raise ValueError(error_msg)
        elif isinstance(micro_batch_size, dict):
            if isinstance(micro_batch_size.get("explore"), dict):
                ini.ws_opt_explore_batch_sizes = True
                if "min" in micro_batch_size["explore"]:
                    min_micro_batch = micro_batch_size["explore"]["min"]
                    ini.ws_opt_batch_exploration_min_micro_batch = (
                        min_micro_batch
                    )
                if "max" in micro_batch_size["explore"]:
                    max_micro_batch = micro_batch_size["explore"]["max"]
                    ini.ws_opt_batch_exploration_max_micro_batch = (
                        max_micro_batch
                    )
                micro_batch_size = "explore"
            else:
                raise ValueError(error_msg)
        elif isinstance(micro_batch_size, int):
            if micro_batch_size > 0:
                ini.ws_opt_force_max_grad_accum_batch = micro_batch_size
                ini.ws_opt_force_min_grad_accum_batch = micro_batch_size
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)

        ini.propagate(cstorch.backends.csx.debug.debug_args, override=True)

        # merge the INI into the global INI
        cstorch.backends.csx.debug.ini |= ini

        super().set_attr(obj, micro_batch_size)
