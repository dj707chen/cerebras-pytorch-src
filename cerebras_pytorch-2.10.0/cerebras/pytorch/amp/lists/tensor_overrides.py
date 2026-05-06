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

import importlib

import torch

from .. import compat
from . import torch_overrides

MODULE = torch.Tensor

FP16_FUNCS = compat.filter_attrs(
    MODULE,
    [
        '__matmul__',
    ],
)

FP32_FUNCS = compat.filter_attrs(
    MODULE,
    [
        '__ipow__',
        '__pow__',
        '__rpow__',
        # Cast to fp32 before transfer to CPU
        'cpu',
    ],
)

CASTS = compat.filter_attrs(
    MODULE,
    [
        '__add__',
        '__div__',
        '__eq__',
        '__ge__',
        '__gt__',
        '__iadd__',
        '__idiv__',
        '__imul__',
        '__isub__',
        '__itruediv__',
        '__le__',
        '__lt__',
        '__mul__',
        '__ne__',
        '__radd__',
        '__rdiv__',
        '__rmul__',
        '__rsub__',
        '__rtruediv__',
        '__sub__',
        '__truediv__',
    ],
)

# None of these, but here to make code cleaner.
SEQUENCE_CASTS = []

# We need to grab all the methods from torch_overrides and add them to
# the Tensor lists as well, as almost all methods are duplicated
# between `torch` and `torch.Tensor` (and check with `hasattr`,
# because a few random ones aren't defined on Tensor)
_self_mod = importlib.import_module(__name__)
for attrname in ['FP16_FUNCS', 'FP32_FUNCS', 'CASTS', 'SEQUENCE_CASTS']:
    lst = getattr(_self_mod, attrname)
    for fn in getattr(torch_overrides, attrname):
        if hasattr(MODULE, fn):
            lst.append(fn)
