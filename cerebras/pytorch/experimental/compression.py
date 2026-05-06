# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import typing
from itertools import chain
from typing import List, Literal, Union

import torch

from cerebras.pytorch.backend import current_backend_impl
from cerebras.pytorch.utils.utils import convert_glob_to_regex

FormatLiteral = Literal[
    "mx8", "mx8-e3m4", "mx8-e4m3", "mx9", "mx10", "mx10-e2m7"
]


class Compression:
    """
    Compression class that can be applied to any torch module that will annotate which tensors
    we want to mark with a certain compression format. This will also only be applied during
    training and evaluation, and so compression format information is not saved to checkpoints.
    Users will be required to re-apply compression formatting after checkpoint loading, as in
    needing to run `model.apply(compression)` after `model.load_state_dict(...)`.

    Example usage:

    ..code:: python

        model = Model()
        compression = cstorch.Compression(
            format="mx8-e3m4",
            param_filter=["fc1.weight", "fc2.*"],
        )
        model.apply(compression)

    See https://arxiv.org/abs/2310.10537 for more details. This paper describes what are the "mx8"
    formats as well as how to leverage compression for better performance.
    """

    def __init__(
        self,
        format: FormatLiteral,
        param_filter: Union[str, List[str]],
    ):
        """
        Constructs a `Compression` instance.

        Args:
            format: A string specifying which format to compress with.
            param_filter: A string or list of strings, in glob format, that filter which
                module/submodule parameters/buffers this compression applies to. Any parameters that do
                not match these filters will not be compressed with this format.
        """

        allowed_formats = typing.get_args(FormatLiteral)

        if not isinstance(format, str):
            raise TypeError(
                f'Expected format to be a string, but got type {type(format)}'
            )
        if format not in allowed_formats:
            raise ValueError(
                f'Expected format to be one of {allowed_formats}, but got "{format}"'
            )
        self.format = format

        self.param_filter = list(
            map(
                convert_glob_to_regex,
                (
                    [param_filter]
                    if isinstance(param_filter, str)
                    else param_filter
                ),
            )
        )

        self.compress_tensors = torch.utils.weak.WeakTensorKeyDictionary()

    def apply(self, module: torch.nn.Module):
        """
        Annotates the given torch module with the compression object created before.

        Args:
            module: A torch.nn.Module specifying which model to annotate with compression formats.
                This function will annotate the module's parameters and buffers with the format
                given if a match is found with the filters given.
        """
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module but got {type(obj)}")

        for name, tensor in chain(
            module.named_parameters(), module.named_buffers()
        ):
            # make sure that our filter is a list of strings
            if any(filter.fullmatch(name) for filter in self.param_filter):
                self.compress_tensors[tensor] = name

        module.register_forward_pre_hook(self._forward_pre_hook)

    def _forward_pre_hook(self, module, input):
        """
        Hook the given module such that the compression annotation is applied to the parameters
        before forward()
        """
        for tensor in self.compress_tensors:
            self.annotate(tensor)

    def annotate(self, tensor: torch.Tensor):
        """
        Annotates a specific tensor with the compression format given

        Args:
            tensor: The torch tensor to be annotated
        """
        current_backend_impl().set_attribute(tensor, "compress", self.format)

    def __repr__(self):
        """
        Returns a string representation of a Compression object
        """
        return f"Compression(format={self.format}, param_filter: {self.param_filter})"
