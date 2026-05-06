# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
This module includes features that are currently under development,
and we're actively working to enhance them. Please be aware that
backward compatibility is not guaranteed at this stage.

Users are advised to proceed with caution and acknowledge all
associated risks when using these features.
"""

from typing import List, Optional, Union

import torch

from .compression import Compression

from . import listener  # noqa


def start_implicit_loop(
    input_tensor: torch.IntTensor, loop_dim: int
) -> torch.IntTensor:
    """
    Return an index tensor signaling an implicit loop over the given tensor
    along the given dimension, used for autoregressive inference.

    Args:
        input_tensor: This tensor will be updated before re-running the model
        loop_dim: The dimension of ``input_tensor`` to loop over.
    """
    from cerebras.pytorch.backend import current_backend_impl

    return current_backend_impl().start_implicit_loop(input_tensor, loop_dim)


def get_loop_iteration_slice(
    input_tensor: torch.IntTensor,
    index_tensor: torch.IntTensor,
) -> torch.IntTensor:
    """
    Helper for use with the implicit loops. Gathers one slice of the
    input_tensor at index_tensor.


    Args:
        input_tensor: The tensor to slice the active token from.
        index_tensor: The tensor returned from start_implict_loop.
    """
    input_shape = list(input_tensor.shape)
    index_shape = list(index_tensor.shape)
    loop_dim = len(index_shape) - 1
    return input_tensor.gather(
        loop_dim,
        index_tensor.view(
            index_shape + [1] * (len(input_shape) - len(index_shape))
        )
        .broadcast_to(index_shape + input_shape[len(index_shape) :])
        .long(),
    )


def update_implicit_loop(
    input_tensor: torch.IntTensor,
    index_tensor: torch.IntTensor,
    update_tensor: torch.IntTensor,
    stop_sequences_tensor: torch.IntTensor,
    start_token: Union[int, List[int]],
    max_tokens: Optional[int] = None,
) -> torch.IntTensor:
    """
    Configure the update step of an active implicit loop.

    Conceptually, the input tensor slice at the next index is set to the update
    tensor and the loop is re-run until the stop_sequence appears in the
    input_tensor.

    At each step, input_tensor[..., index_tensor+1] = update_tensor
    along ``loop_dim``.

    Only one such loop at a time is supported.

    This is equivalent to the following pseudocode:

        def inner_loop(
            input_tensor,
            loop_dim,
            start_token,
            stop_sequences,
            model
        ):

            shape = list(input_tensor.shape)
            extent = shape[loop_dim]
            del shape[loop_dim:]
            started = torch.zeros(shape, dtype=torch.bool)
            stopped = torch.zeros(shape, dtype=torch.bool)
            index_tensor = torch.zeros(shape, dtype=torch.int32)
            output_tensor = torch.zeros_like(input_tensor)

            def update_output(update_tensor):
                # Update one whole slice of output_tensor with the update
                input_shape = list(input_tensor.shape)
                index_shape = list(index_tensor.shape)
                index = index_tensor.view(
                    index_shape + [1] * (len(input_shape) - len(index_shape))
                ).broadcast_to(
                    index_shape + input_shape[len(index_shape):]
                ).long()

                output_tensor.scatter_add_(
                    loop_dim,
                    update_tensor,
                    index
                )

            output_token= model(input_tensor, index_tensor)
            update_output(output_token)
            for i in range(extent-1):
                started |= (
                    input_tensor.index_select(
                        loop_dim, torch.tensor(i)
                    ) == start_token
                ).view(shape+[-1]).all(dim=-1)

                if not started.any():
                    # optimization to skip re-running model when no input would
                    # be updated.
                    continue

                for stop_sequence in stop_sequences:
                    # TODO: update this logic for multi-token sequence check
                    stopped |= (
                        output_token == stop_sequence
                    ).view(shape+[-1]).all(dim=-1)

                if stopped.all()
                    # optimization to stop running updates onces all outputs
                    # have stopped.
                    break

                # autoregress this position and run again
                update = started && ~stopped
                updated_input = input_tensor.index_copy(
                    loop_dim, torch.tensor(i+1), output_token
                )

                input_tensor = torch.where(
                    update,
                    updated_input_tensor,
                    input_tensor,
                )
                index_tensor.add_(1)
                update_output(model(input_tensor, index_tensor))

    Args:
        input_tensor: Each step, the ``update_tensor`` will populate the
                     ``loop_dim`` slice of this input tensor at position
                     ``index_tensor + 1``. The final value is returned.
        index_tensor: The tensor returned from start_implict_loop.
        update_tensor: This tensor will be inserted into input_tensor at the
                       subsequent position along ``loop_dim`` in
                       ``input_tensor`` each inner-step. It should be the same
                       shape and type as ``input_tensor`` except the extend of
                       the ``loop_dim`` should be  1.
        stop_sequences_tensor: For LM autoregessive use, this tensor holds the list of
                            stop token sequences that, if seen in the output, marks
                            the end of generation; i.e. the inner autoregressive loop
                            is exited.
        start_token: For LM autoregessive use, this token in the input marks
                     the beginning of generation. All tokens before it are left
                     unmodified.
        max_tokens: If given, only this many tokens will be generated before
                    stopping.
    Returns:
        The final "modified" version of ``input_tensor`` with all updates made.
    """
    from cerebras.pytorch.backend import current_backend_impl

    return current_backend_impl().update_implicit_loop(
        input_tensor,
        index_tensor,
        update_tensor,
        stop_sequences_tensor,
        start_token,
        max_tokens,
    )
