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

# Notes:
# F.instance_norm uses batch_norm internally. Which correctly handles
#   fp16 in/out with fp32 weights. So we shouldn't do anything for
#   either of these.
# F.normalize calls `input.norm()` internally, so it's redundant, but
#   kept here in case impl. changes.
# F.cosine_similarity is same: calls `x.norm()` internally.

import torch.nn.functional

MODULE = torch.nn.functional

FP16_FUNCS = [
    'avg_pool1d',
    'avg_pool2d',
    'avg_pool3d',
    'batch_norm',
    'conv1d',
    'conv2d',
    'conv3d',
    'conv_transpose1d',
    'conv_transpose2d',
    'conv_transpose3d',
    'conv_tbc',  # Undocumented / maybe new?
    'dropout',
    'group_norm',
    'linear',
    'max_pool1d',
    'max_pool2d',
    'max_pool3d',
]

# Some functions represent kernel boundaries where the output must be FP16
# regardless of the inner computation
FP32_OUT_FP16_FUNCS = [
    # Pointwise
    #'gelu',
    # Normalization
    'layer_norm',
    # inputs are integer so won't be cast, but output must be FP16
    'embedding',
]

FP32_FUNCS = [
    # Interpolation/Upsampling TODO:  Remove for 1.2
    'interpolate',
    'grid_sample',
    # Pointwise
    'softplus',
    'softmin',
    'log_softmax',
    #'softmax',
    # Normalization
    'local_response_norm',
    'normalize',
    'cosine_similarity',
    # Loss functions
    # TODO: which of these can be fp16?
    'poisson_nll_loss',
    'cosine_embedding_loss',
    'cross_entropy',
    'hinge_embedding_loss',
    'kl_div',
    'l1_loss',
    'mse_loss',
    'margin_ranking_loss',
    'multilabel_margin_loss',
    'multilabel_soft_margin_loss',
    'multi_margin_loss',
    'nll_loss',
    'binary_cross_entropy_with_logits',
    'smooth_l1_loss',
    'soft_margin_loss',
    'triplet_margin_loss',
    'ctc_loss',
]

BANNED_FUNCS = [
    (
        'binary_cross_entropy',
        (
            "\namp does not work out-of-the-box with `F.binary_cross_entropy` or `torch.nn.BCELoss.` "
            "It requires that the output of the previous function be already a FloatTensor. \n\n"
            "Most models have a Sigmoid right before BCELoss. In that case, you can use\n"
            "    torch.nn.BCEWithLogitsLoss\nto combine Sigmoid+BCELoss into a single layer "
            "that is compatible with amp.\nAnother option is to add\n"
            "    amp.register_float_function(torch, 'sigmoid')\nbefore calling `amp.init()`.\n"
            "If you _really_ know what you are doing, you can disable this warning by passing "
            "allow_banned=True to `amp.init()`."
        ),
    )
]
