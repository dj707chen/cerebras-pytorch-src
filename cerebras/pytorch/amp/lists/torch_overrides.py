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

MODULE = torch

FP16_FUNCS = [
    # Low level functions wrapped by torch.nn layers.
    # The wrapper layers contain the weights which are then passed in as a parameter
    # to these functions.
    'conv1d',
    'conv2d',
    'conv3d',
    'conv_transpose1d',
    'conv_transpose2d',
    'conv_transpose3d',
    'conv_tbc',
    'prelu',
    # BLAS
    'addmm',
    'addmv',
    'addr',
    'matmul',
    'mm',
    'mv',
]

FP32_FUNCS = [
    # Pointwise
    'acos',
    'asin',
    'cosh',
    'erfinv',
    'exp',
    'expm1',
    'log',
    'log10',
    'log2',
    'reciprocal',
    'rsqrt',
    'sinh',
    'tan',
    # Other math
    'pow',
    # Reduction
    'cumprod',
    'cumsum',
    'dist',
    # 'mean',
    'norm',
    'prod',
    'std',
    'sum',
    'var',
    # Misc
    'renorm',
]


# Multi-tensor fns that may need type promotion
CASTS = [
    # Multi-tensor math
    'addcdiv',
    'addcmul',
    'atan2',
    'cross',
    'bilinear',
    'dot',
    # Element-wise _or_ tensor-wise math
    'add',
    'div',
    'mul',
    # Comparison
    'eq',
    'equal',
    'ge',
    'gt',
    'le',
    'lt',
    'ne',
]

# Functions that take sequence arguments. We need to inspect the whole
# sequence and cast to the widest type.
SEQUENCE_CASTS = ['cat', 'stack']
