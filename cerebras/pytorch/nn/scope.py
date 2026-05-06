# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch

import cerebras.pytorch.nn.functional as F


class Scope(torch.nn.Module):
    """
    This module is to set a boundary of the scope after the `input` in the CIRH graph.
    The name of scope is specified by argument `scope_name`.

    The 'enter_scope' and 'exit_scope' are wrapper functions of Torch AutoGrad functions, and
    it can set the "BEGIN" or "END" boundaries in the CIRH graph.

    In the foward pass, it will set the "BEGIN" boundary for the scope, and an "END" boundary
    will be set automatically by the AutoGrad function in the backward pass.

    The `exit()` method is syntax suger for setting for `END` boundary of the scope without specifying
    the `scope_name` again. In the corresponding backward pass,a "BEGIN" boundary
    will be set automatically by the AutoGrad function.

    See cerebras.pytorch.nn.functional for more details.

    Args:
        scope_name (str): The name of the scope.
    """

    def __init__(self, scope_name):
        super().__init__()
        self._scope_name = scope_name

    def forward(self, input):
        return F.enter_scope(input, self._scope_name)

    def exit(self, input):
        return F.exit_scope(input, self._scope_name)
