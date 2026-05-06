# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#
import torch


class DefaultWeakIdKeyDictionary(torch.utils.weak.WeakIdKeyDictionary):
    def __init__(self, default_factory=None, dict=None, **kwargs):
        super().__init__(dict, **kwargs)
        self.default_factory = default_factory

    def __getitem__(self, key):
        if key not in self:
            self.__missing__(key)
        return super().__getitem__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = self.default

    @property
    def default(self):
        return self.default_factory()
