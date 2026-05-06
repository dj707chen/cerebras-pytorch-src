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

import types

import torch

from .conditional_update import ConditionalUpdateManager
from .utils import type_string


class AmpOptimizerState(object):
    def __init__(self):
        pass


def post_backward_models_are_masters(
    scaler, params, stashed_grads, scale_override=None
):
    grads_have_scale, stashed_have_scale, out_scale = (
        scaler.get_scale(),
        1.0,
        1.0,
    )

    # not much to do if static scaling and scale == 1.0
    if not scaler.dynamic and scaler.get_scale() == 1.0:
        # Clear the stash.
        for i in range(len(stashed_grads)):
            stashed_grads[i] = None
        return

    if scale_override is not None:
        grads_have_scale, stashed_have_scale, out_scale = scale_override

    # This is a lot of python overhead...
    grads_needing_unscale = []
    grads_needing_unscale_with_stash = []
    stashed = []
    for param, stashed_grad in zip(params, stashed_grads):
        if param.grad is None and stashed_grad is not None:
            param.grad = stashed_grad
        elif param.grad is not None and stashed_grad is None:
            grads_needing_unscale.append(param.grad)
        elif param.grad is not None and stashed_grad is not None:
            grads_needing_unscale_with_stash.append(param.grad)
            stashed.append(stashed_grad)
        else:  # param.grad is None and stashed_grad is None
            continue

    # unscale() implements grads*(1/scale), so "scale" should be grads_have_scale/out_scale.
    if len(grads_needing_unscale) > 0:
        scaler._unscale(
            grads_needing_unscale,
            grads_needing_unscale,
            None,  # unused_scale, currently present to avoid API breakage elsewhere
            models_are_masters=True,
            scale_override=grads_have_scale / out_scale,
        )

    if len(grads_needing_unscale_with_stash) > 0:
        scaler._unscale_with_stashed(
            grads_needing_unscale_with_stash,
            stashed,
            grads_needing_unscale_with_stash,
            scale_override=(grads_have_scale, stashed_have_scale, out_scale),
        )

    # Clear the stash.
    for i in range(len(stashed_grads)):
        stashed_grads[i] = None


def lazy_init_no_master_weights(self):
    stash = self._amp_stash
    stash.all_fp16_params = []
    stash.all_fp32_params = []
    for i, param_group in enumerate(self.param_groups):
        for i, param in enumerate(param_group['params']):
            if type_string(param) == 'HalfTensor':
                stash.all_fp16_params.append(param)
            elif type_string(param) == 'FloatTensor':
                stash.all_fp32_params.append(param)
            else:
                raise TypeError(
                    "Optimizer's parameters must be either "
                    "torch.FloatTensor or torch.HalfTensor. "
                )

    stash.all_fp16_grad_stash = [None for _ in stash.all_fp16_params]
    stash.all_fp32_grad_stash = [None for _ in stash.all_fp32_params]


def prepare_backward_no_master_weights(self):
    stash = self._amp_stash

    self._amp_lazy_init()

    for i, param in enumerate(stash.all_fp16_params):
        stash.all_fp16_grad_stash[i] = param.grad
        # Set up to leverage grad copy elision:
        param.grad = None

    for i, param in enumerate(stash.all_fp32_params):
        stash.all_fp32_grad_stash[i] = param.grad
        # Set up to leverage grad copy elision:
        param.grad = None


def post_backward_no_master_weights(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()

    split_types = (
        (stash.all_fp16_params, stash.all_fp16_grad_stash),
        (stash.all_fp32_params, stash.all_fp32_grad_stash),
    )

    for params, stashed_grads in split_types:
        post_backward_models_are_masters(scaler, params, stashed_grads)


def _amp_lazy_init(self):
    stash = self._amp_stash

    if not stash.lazy_init_called:
        self._lazy_init_maybe_master_weights()
        stash.lazy_init_called = True


def _process_optimizer(optimizer):
    if hasattr(optimizer, "_amp_stash"):
        raise RuntimeError("A given optimizer should only be processed once.")
    else:
        optimizer._amp_stash = AmpOptimizerState()

    optimizer._amp_stash.dls_update_manager = ConditionalUpdateManager()
    optimizer._amp_stash.lazy_init_called = False
    optimizer._amp_stash.already_patched = False
    optimizer._amp_stash.params_have_scaled_gradients = False

    for name in (
        "_prepare_amp_backward",
        "_post_amp_backward",
        "_amp_lazy_init",
    ):
        if hasattr(optimizer, name):
            raise RuntimeError(
                "Incoming optimizer already has {} defined.".format(name)
            )

    optimizer._lazy_init_maybe_master_weights = types.MethodType(
        lazy_init_no_master_weights, optimizer
    )

    optimizer._prepare_amp_backward = types.MethodType(
        prepare_backward_no_master_weights, optimizer
    )
    optimizer._post_amp_backward = types.MethodType(
        post_backward_no_master_weights, optimizer
    )

    optimizer._amp_lazy_init = types.MethodType(_amp_lazy_init, optimizer)

    old_add_param_group = optimizer.add_param_group

    def new_add_param_group(self, new_group):
        stash = self._amp_stash

        if not stash.lazy_init_called:
            self._lazy_init_maybe_master_weights()
            stash.lazy_init_called = True

        assert isinstance(new_group, dict), "param group must be a dict"

        new_params = new_group['params']
        if isinstance(new_params, torch.Tensor):
            new_group['params'] = [new_params]
        elif isinstance(new_params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            new_group['params'] = list(new_params)

        for param in new_group['params']:
            if type_string(param) == 'HalfTensor':
                stash.all_fp16_params.append(param)
                stash.all_fp16_grad_stash.append(None)
            elif type_string(param) == 'FloatTensor':
                stash.all_fp32_params.append(param)
                stash.all_fp32_grad_stash.append(None)
            else:
                raise TypeError(
                    "Optimizer's parameters must be either "
                    "torch.FloatTensor or torch.HalfTensor. "
                )

        old_add_param_group(new_group)

    optimizer.add_param_group = types.MethodType(new_add_param_group, optimizer)

    return optimizer
