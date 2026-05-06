"""Cerebras Automatic Mixed Precision module."""

# ###############################################################
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#
# From original Apex:
# Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright notice,
#        this list of conditions and the following disclaimer in the documentation
#        and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its contributors
#        may be used to endorse or promote products derived from this software without
#        specific prior written permission.
#
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#        WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#        IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#        INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
#        NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#        PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#        WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#        ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#        POSSIBILITY OF SUCH DAMAGE.
#
#################################################################

from ._amp_state import (
    _amp_state,
    enable_mixed_precision,
    get_floating_point_dtype,
    get_floating_point_dtype_str,
    get_half_dtype,
    get_half_dtype_str,
    is_cbfloat16_tensor,
    mixed_precision,
    set_half_dtype,
)
from ._process_optimizer import _process_optimizer as setup_optimizer
from .amp import (
    float_function,
    half_function,
    init,
    promote_function,
    register_float_function,
    register_half_function,
    register_promote_function,
)
from .autocast import autocast
from .conditional_update import isfinite, update_if_finite
from .grad_scaler import GradScaler
from .handle import disable_casts
from .optimizer_step import optimizer_step

__all__ = ["GradScaler", "optimizer_step"]
