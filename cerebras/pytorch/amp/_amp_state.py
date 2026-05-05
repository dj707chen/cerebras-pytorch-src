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

import warnings
from typing import Literal, Union

import torch

import cerebras.pytorch.distributed as dist
from cerebras.appliance.environment import appliance_environ

_DTYPE_ENV_VAR = "CEREBRAS_FP16_DTYPE"
HalfDtypeLiteral = Literal["float16", "bfloat16", "cbfloat16"]


class AmpState:
    def __init__(self):
        self.hard_override = False
        self.allow_incoming_model_not_fp32 = False
        self.verbosity = 1
        self._real_dtype_str = "float16"

    @property
    def half_dtype(self) -> torch.dtype:
        # TODO: Temporarily read the value in workers through an env variable. Once RT IR has the
        # value in the module, we should read it from there instead.
        if not dist.is_master_ordinal():
            dtype_str = appliance_environ.get(
                _DTYPE_ENV_VAR, self._real_dtype_str
            )
        else:
            dtype_str = self._real_dtype_str

        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "cbfloat16":
            return torch.float16  # proxy dtype
        else:
            assert False, f"Invalid dtype str: {dtype_str}"

    @half_dtype.setter
    def half_dtype(self, value: Union[HalfDtypeLiteral, torch.dtype]):
        if not dist.is_master_ordinal():
            raise RuntimeError(
                "Setting half dtype in the dataloader is not allowed as it might conflict with "
                "what the model was compiled with. Please ensure to set the half dtype outside "
                "of the dataloader before constructing the model."
            )

        if value == torch.float16:
            self._real_dtype_str = "float16"
        elif value == torch.bfloat16:
            self._real_dtype_str = "bfloat16"
        elif isinstance(value, str) and value in [
            "float16",
            "bfloat16",
            "cbfloat16",
        ]:
            self._real_dtype_str = value
        else:
            raise ValueError(
                f"Invalid half dtype: {value}. Accepted values are: "
                f"\"float16\", \"bfloat16\", \"cbfloat16\", {torch.float16}, {torch.bfloat16}."
            )

        appliance_environ[_DTYPE_ENV_VAR] = self._real_dtype_str

    @property
    def half_dtype_str(self) -> HalfDtypeLiteral:
        if not dist.is_master_ordinal():
            return appliance_environ.get(_DTYPE_ENV_VAR, self._real_dtype_str)
        else:
            return self._real_dtype_str


# Attribute stash.  Could also just stash things as global module attributes.
_amp_state = AmpState()


def warn_or_err(msg):
    if _amp_state.hard_override:
        print("Warning:  " + msg)
    else:
        raise RuntimeError(msg)


def maybe_print(msg):
    if _amp_state.verbosity > 0:
        print(msg)


def use_bfloat16(value: bool) -> None:
    warnings.warn(
        f"`use_bfloat16()` method is deprecated and will be removed in a future release. "
        f"Use `set_half_dtype()` instead."
    )
    set_half_dtype(torch.bfloat16 if value else torch.float16)


def set_half_dtype(value: Union[HalfDtypeLiteral, torch.dtype]) -> torch.dtype:
    """Sets the underlying 16-bit floating point dtype to use.

    Args:
        value: Either a 16-bit floating point torch dtype or one of "float16", "bfloat16", or
            "cbfloat16" string.

    Returns:
        The proxy torch dtype to use for the model. For dtypes that have a torch representation,
        this returns the same as `value` passed in. Otherwise, it returns a proxy dtype to use in
        the model. On CSX, these proxy dtypes are automatically and transparently converted to the
        real dtype during compilation.
    """
    _amp_state.half_dtype = value
    return _amp_state.half_dtype


def get_half_dtype() -> torch.dtype:
    """Gets the 16-bit floating point dtype to use in the model.

    This returns the value set through `set_half_dtype()`.
    """
    return _amp_state.half_dtype


def get_half_dtype_str() -> str:
    """
    Gets the string representation of the 16-bit floating point dtype to use in
    the model.
    """
    return _amp_state.half_dtype_str
