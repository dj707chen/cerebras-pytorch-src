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

import functools
import itertools
from warnings import warn

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from cerebras.pytorch.utils.call_once import call_once

from . import compat, rnn_compat, utils, wrap
from ._amp_state import _amp_state, enable_mixed_precision
from .handle import AmpHandle, NoOpHandle
from .lists import functional_overrides, tensor_overrides, torch_overrides

_DECORATOR_HANDLE = None
_USER_CAST_REGISTRY = set()
_USER_PROMOTE_REGISTRY = set()


def _decorator_helper(orig_fn, cast_fn, wrap_fn):
    def wrapper(*args, **kwargs):
        handle = _DECORATOR_HANDLE
        if handle is None or not handle.is_active():
            return orig_fn(*args, **kwargs)
        inner_cast_fn = utils.verbosify(
            cast_fn, orig_fn.__name__, handle.verbose
        )
        return wrap_fn(orig_fn, inner_cast_fn, handle)(*args, **kwargs)

    return wrapper


# Decorator form
def half_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=True)
    return _decorator_helper(fn, utils.maybe_half, wrap_fn)


def float_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=False)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)


def promote_function(fn):
    wrap_fn = functools.partial(wrap.make_promote_wrapper)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)


# Registry form
def register_half_function(module, name):
    if not hasattr(module, name):
        raise ValueError(
            'No function named {} in module {}.'.format(name, module)
        )
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_half))


def register_float_function(module, name):
    if not hasattr(module, name):
        raise ValueError(
            'No function named {} in module {}.'.format(name, module)
        )
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_float))


def register_promote_function(module, name):
    if not hasattr(module, name):
        raise ValueError(
            'No function named {} in module {}.'.format(name, module)
        )
    _USER_PROMOTE_REGISTRY.add((module, name))


# Top-level function to insert _all_ the hooks.
@call_once()
def init(
    enabled=True,
    enable_caching=True,
    verbose=False,
    allow_banned=False,
):
    global _DECORATOR_HANDLE

    if not enabled:
        handle = NoOpHandle()
        _DECORATOR_HANDLE = handle
        return handle

    # Initializing AMP implies that mixed precision is being used
    enable_mixed_precision()

    handle = AmpHandle(enable_caching, verbose)

    # 0) Force-{fp16, fp32} for user-annotated functions
    for mod, fn, cast_fn in _USER_CAST_REGISTRY:
        try_caching = cast_fn == utils.maybe_half
        wrap.cached_cast(mod, fn, cast_fn, handle, try_caching, verbose)
    _USER_CAST_REGISTRY.clear()

    # 0.5) Force-promote for user-annotated functions
    for mod, fn in _USER_PROMOTE_REGISTRY:
        wrap.promote(mod, fn, handle, verbose)
    _USER_PROMOTE_REGISTRY.clear()

    # 1) Force-{fp16, fp32} on white- / black-list functions
    override_modules = [functional_overrides, torch_overrides, tensor_overrides]
    cast_table = [
        ('FP16_FUNCS', utils.maybe_half, False),
        ('FP32_FUNCS', utils.maybe_float, False),
        ('FP32_OUT_FP16_FUNCS', utils.maybe_float, True),
    ]
    for module, (list_name, cast_fn, cast_out) in itertools.product(
        override_modules, cast_table
    ):
        for fn in getattr(module, list_name, []):
            try_caching = cast_fn == utils.maybe_half
            wrap.cached_cast(
                module.MODULE,
                fn,
                cast_fn,
                handle,
                try_caching,
                verbose,
                cast_out,
            )

    # 1.5) Pre-0.4, put the blacklist methods on HalfTensor and whitelist
    #      methods on FloatTensor, since they're distinct types.
    if compat.tensor_is_float_tensor():
        for fn in tensor_overrides.FP16_FUNCS:
            wrap.cached_cast(
                torch.FloatTensor,
                fn,
                utils.maybe_half,
                handle,
                try_caching=True,
                verbose=verbose,
            )
        for fn in tensor_overrides.FP32_FUNCS:
            wrap.cached_cast(
                torch.HalfTensor,
                fn,
                utils.maybe_float,
                handle,
                try_caching=False,
                verbose=verbose,
            )

    # 2) Enable type-promotion on multi-arg functions and methods.
    #    NB: special handling for sequence fns (e.g. `torch.cat`).
    promote_modules = [torch_overrides, tensor_overrides]
    promote_table = [
        ('CASTS', wrap.promote),
        ('SEQUENCE_CASTS', wrap.sequence_promote),
    ]
    for promote_mod, (list_name, promote_fn) in itertools.product(
        promote_modules, promote_table
    ):
        for fn in getattr(promote_mod, list_name):
            promote_fn(promote_mod.MODULE, fn, handle, verbose)

    # 2.5) Pre-0.4, add blacklist methods directly to HalfTensor and FloatTensor types
    if compat.tensor_is_float_tensor():
        for cls, (list_name, promote_fn) in itertools.product(
            [torch.FloatTensor, torch.HalfTensor], promote_table
        ):
            for fn in getattr(tensor_overrides, list_name):
                promote_fn(cls, fn, handle, verbose)

    # 3) For any in-place version of a blacklist function, error if any input is fp16.
    #    NB: this is overly conservative.
    for fn in utils.as_inplace(torch_overrides.FP32_FUNCS):
        wrap.err_if_any_half(torch_overrides.MODULE, fn, handle)

    # 3.5) For any in-place blacklist method, error if called on fp16 tensor
    for fn in utils.as_inplace(tensor_overrides.FP32_FUNCS):
        wrap.err_if_arg0_half(tensor_overrides.MODULE, fn, handle, verbose)
        if compat.tensor_is_float_tensor():
            wrap.err_if_arg0_half(torch.HalfTensor, fn, handle, verbose)

    # 4) For other in-place methods, match the type of self tensor
    for fn in utils.as_inplace(
        itertools.chain(tensor_overrides.FP16_FUNCS, tensor_overrides.CASTS)
    ):
        wrap.promote_match_arg0(tensor_overrides.MODULE, fn, handle, verbose)
        if compat.tensor_is_float_tensor():
            wrap.promote_match_arg0(torch.HalfTensor, fn, handle, verbose)
            wrap.promote_match_arg0(torch.FloatTensor, fn, handle, verbose)

    # 5) RNNs + RNN cells are whitelisted specially
    if rnn_compat.has_old_rnns():
        wrap.rnn_cast(torch.nn.backends.thnn.backend, 'RNN', handle, verbose)
    if not rnn_compat.has_old_rnns():
        # Patch in our own indirection of `_VF` in modules/rnn s.t. it is mutable.
        torch.nn.modules.rnn._VF = rnn_compat.VariableFunctionsShim()
        # Wrap all the rnns
        for x in rnn_compat.RNN_NAMES:
            wrap.new_rnn_cast(x.upper(), handle, verbose)

    # Wrap all the RNN cells
    rnn_compat.whitelist_rnn_cells(handle, verbose)

    # 6) Place error+print message on banned functions.
    #    Or, if allow_banned, then cast to FP32.
    for fn, err_msg in functional_overrides.BANNED_FUNCS:
        if allow_banned:
            wrap.cached_cast(
                functional_overrides.MODULE,
                fn,
                utils.maybe_float,
                handle,
                try_caching=True,
                verbose=verbose,
            )
        else:
            wrap.err_if_any_half(
                functional_overrides.MODULE, fn, handle, err_msg
            )

    def check_grad_scaler(optimizer, args, kwargs):
        if _amp_state.half_dtype_str == "float16" and not hasattr(
            optimizer, "_amp_stash"
        ):
            warn(
                "Detected that the half dtype is set to float16 but no "
                "grad scaler is instantiated. Please use a grad scaler to "
                "prevent gradient underflow.\n\n"
                "\tgrad_scaler = cstorch.amp.GradScaler(...)"
            )

    register_optimizer_step_pre_hook(check_grad_scaler)

    _DECORATOR_HANDLE = handle

    _amp_state.handle = handle

    return handle
