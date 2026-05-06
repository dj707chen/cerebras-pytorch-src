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

import torch


# True for post-0.4, when Variables/Tensors merged.
def variable_is_tensor():
    v = torch.autograd.Variable()
    return isinstance(v, torch.Tensor)


def tensor_is_variable():
    x = torch.Tensor()
    return type(x) == torch.autograd.Variable


# False for post-0.4
def tensor_is_float_tensor():
    x = torch.Tensor()
    return type(x) == torch.FloatTensor


# Akin to `torch.is_tensor`, but returns True for Variable
# objects in pre-0.4.
def is_tensor_like(x):
    return torch.is_tensor(x) or isinstance(x, torch.autograd.Variable)


# Wraps `torch.is_floating_point` if present, otherwise checks
# the suffix of `x.type()`.
def is_floating_point(x):
    if hasattr(torch, 'is_floating_point'):
        return torch.is_floating_point(x)
    try:
        torch_type = x.type()
        return (
            torch_type.endswith('FloatTensor')
            or torch_type.endswith('HalfTensor')
            or torch_type.endswith('DoubleTensor')
        )
    except AttributeError:
        return False


def scalar_python_val(x):
    if hasattr(x, 'item'):
        return x.item()
    else:
        if isinstance(x, torch.autograd.Variable):
            return x.data[0]
        else:
            return x[0]


# Accounts for the possibility that some ops may be removed from a namespace.
def filter_attrs(module, attrs):
    return list(attrname for attrname in attrs if hasattr(module, attrname))
