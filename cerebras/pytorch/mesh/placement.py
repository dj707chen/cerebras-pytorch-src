# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass


class Placement:
    """
    Base class for placement strategies
    """

    def is_shard(self) -> bool:
        return isinstance(self, Shard)

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)


@dataclass
class Shard(Placement):
    """
    Represents a sharding strategy for a tensor dimension.

    Attributes:
        dim: The mesh dimension over which to shard.
    """

    dim: int


@dataclass
class Replicate(Placement):
    """
    Represents a replication strategy for a tensor dimension.
    """
