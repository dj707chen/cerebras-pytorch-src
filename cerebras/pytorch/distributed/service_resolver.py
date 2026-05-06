# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import abc
from typing import List, Type, TypeVar

from cerebras.pytorch.distributed.cluster_resolver import (
    ClusterResolver,
    ClusterSpec,
    TaskRole,
)

TServiceResolver = TypeVar("TServiceResolver", bound="BaseServiceResolver")


class BaseServiceResolver(metaclass=abc.ABCMeta):
    """Base PyTorch service resolver for a distributed environment."""

    def __init__(self):
        """Constructs a `BaseServiceResolver` instance."""
        self._resolver = ClusterResolver.get_resolver()

    @abc.abstractmethod
    def streamer_ordinals(self) -> List[int]:
        pass

    @abc.abstractmethod
    def receiver_ordinals(self) -> List[int]:
        pass

    @abc.abstractmethod
    def setup(self):
        pass

    @classmethod
    def get_resolver(cls: Type[TServiceResolver]) -> TServiceResolver:
        """Returns the appropriate resolver instance for the environment.

        This method loops over the registered resolvers and checks if any of
        them are able to resolve the environment and returns the first one.
        If none of the resolvers can handle the environment, an error is raised.
        """
        return ApplianceServiceResolver()

    @property
    def cluster_resolver(self) -> ClusterResolver:
        return self._resolver

    @property
    def cluster_spec(self) -> ClusterSpec:
        return self._resolver.cluster_spec()


class ApplianceServiceResolver(BaseServiceResolver):
    """Appliance service resolver for a distributed envionment."""

    def __init__(self):
        self._resolver = ClusterResolver.get_resolver()

    def streamer_ordinals(self) -> List[int]:
        return [
            task.rank
            for task in self.cluster_spec.filter_tasks_by_role(TaskRole.WORKER)
        ]

    def receiver_ordinals(self) -> List[int]:
        return [
            task.rank
            for task in self.cluster_spec.filter_tasks_by_role(TaskRole.MASTER)
        ]

    def setup(self):
        pass
