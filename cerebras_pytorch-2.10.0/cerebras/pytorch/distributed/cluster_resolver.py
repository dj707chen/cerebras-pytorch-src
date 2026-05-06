# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import abc
import copy
import functools
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Type, TypeVar

from cerebras.appliance.cluster.cluster_details import ClusterDetailsParser
from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ClusterDetails,
)

TClusterResolver = TypeVar("TClusterResolver", bound="ClusterResolver")


class TaskRole(Enum):
    """Roles that a task can have."""

    MASTER = 'master'
    WORKER = 'worker'


@dataclass
class TaskSpec:
    """Specification of a single task.

    Args:
        rank: Global rank of the task across tasks on all physical nodes.
        local_rank: Local rank of the task across tasks on this physical node.
        wse_id: ID of the WSE this task is running on.
        node_name: Name of the node that task is running on.
        roles: List of roles that this task assumes.
    """

    rank: int
    local_rank: int
    wse_id: int
    node_name: str
    roles: List[TaskRole] = None

    def __post_init__(self) -> None:
        """Run validation after dataclass is initialized."""
        assert self.rank >= 0, f"Rank must be a positive int."
        assert self.local_rank >= 0, f"Local rank must be a postive integer."
        assert self.wse_id >= 0, f"WSE ID must be a positive integer."
        assert self.node_name, f"Nodename must not be empty."


@dataclass
class ClusterSpec:
    """Specification of a cluster of tasks.

    Args:
        tasks: List of all tasks in the cluster.
        rank: Rank of the current process's task in the cluster.
        num_csx: Number of CSX in the cluster
        num_workers_per_csx: Number of worker tasks per CSX.
    """

    tasks: List[TaskSpec]
    rank: int
    num_csx: int
    num_workers_per_csx: int

    def __post_init__(self) -> None:
        """Run validation after dataclass is initialized."""
        assert self.tasks, f"A cluster must have at least one task."
        assert 0 <= self.rank < len(self.tasks), f"Invalid rank {self.rank}"

        expected_ranks = list(range(len(self.tasks)))
        given_ranks = [task.rank for task in self.tasks]
        assert (
            expected_ranks == given_ranks
        ), f"Expected task ranks to be {expected_ranks}, but got {given_ranks}"
        assert (
            self.num_workers_per_csx >= 0
        ), f"Num of workers per CSX must be a positive integer."

    def num_tasks(self) -> int:
        """Returns the number of tasks in the cluster."""
        return len(self.tasks)

    def task(self, rank: Optional[int] = None) -> TaskSpec:
        """Returns the task spec for the given rank.

        If rank is not given, it returns the task spec for the current process.

        Args:
            rank: Rank of the task for which to return the task spec. If None,
                it returns the task spec for the current process.
                Defaults to None.

        Returns:
            The task spec for the given rank.
        """
        if rank is None:
            rank = self.rank

        if rank < 0 or rank >= self.num_tasks():
            raise ValueError(
                f"Invalid task rank {rank}. "
                f"Available ranks are: {list(range(self.num_tasks()))}."
            )

        return self.tasks[rank]

    def filter_tasks_by_role(self, role: TaskRole) -> List[TaskSpec]:
        """Returns a list of tasks filtered by the given role.

        Args:
            role: The role to filter tasks by.
        Returns:
            List of tasks filtered by the given role.
        """
        return [task for task in self.tasks if role in task.roles]


class ClusterResolver(metaclass=abc.ABCMeta):
    """Abstract class for resolving a cluster."""

    # List of registered resolver implementations that will be checked in order.
    _RESOLVERS: List[TClusterResolver] = []

    @abc.abstractmethod
    def cluster_spec(self) -> ClusterSpec:
        """Returns the cluster spec."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _check(cls: Type[TClusterResolver]) -> bool:
        """Checks whether this is the correct resolver for the environment."""
        raise NotImplementedError()

    @classmethod
    def register_resolver(
        cls: Type[TClusterResolver],
        resolver_cls: Type[TClusterResolver],
    ) -> None:
        """Registers a resolver to be checked for.

        Args:
            resolver_cls: The resolver class to register.
        """
        if not issubclass(resolver_cls, cls):
            raise ValueError(
                f"Cluster resolver `{resolver_cls.__name__}` must be a "
                f"subclass of `{cls.__name__}`."
            )

        if resolver_cls in cls._RESOLVERS:
            raise ValueError(f"Resolver {resolver_cls} is already registered.")

        cls._RESOLVERS.append(resolver_cls)

    @classmethod
    def get_resolver(
        cls: Type[TClusterResolver],
    ) -> Type[TClusterResolver]:
        """Returns the appropriate resolver instance for the environment.

        This method loops over the registered resovlers and checks if any of
        them are able to resolve the environment and returns the first one.
        If none of the resolvers can handle the environment, an error is raised.
        """
        for resolver_cls in cls._RESOLVERS:
            if resolver_cls._check():
                return resolver_cls()

        raise ValueError("Failed find an appropriate cluster resolver.")

    @property
    def num_tasks(self) -> int:
        """Returns the total number of tasks in the cluster."""
        return self.cluster_spec().num_tasks()

    @property
    def rank(self) -> int:
        """Returns the rank of the current task in the cluster."""
        return self.cluster_spec().rank

    @property
    def task(self) -> TaskSpec:
        """Returns the task spec for the current task."""
        return self.cluster_spec().task()

    @property
    def master_task(self) -> TaskSpec:
        """Returns the master task of the cluster."""
        master_tasks = self.get_task(TaskRole.MASTER)
        assert len(master_tasks) == 1
        return master_tasks[0]

    def get_task(self, role: TaskRole) -> TaskSpec:
        """
        Returns the tasks performing the specified role
        in the cluster
        """
        tasks = self.cluster_spec().filter_tasks_by_role(role)
        return tasks

    def assumes_role(self, role: TaskRole, rank: Optional[int] = None) -> bool:
        """Checks whether the task with given rank is has the specified role.

        Args:
            role: TaskRole to check for
            rank: Rank of the task to check. If None, it checks the current
            task. Defaults to None.

        Returns:
            bool: True if task is master, False otherwise.
        """
        return role in self.cluster_spec().task(rank).roles


class ApplianceClientClusterResolver(ClusterResolver):
    """Cluster resolver for the client environment."""

    @functools.lru_cache()
    def cluster_spec(self):
        task_spec = TaskSpec(0, 0, 0, "unknown", roles=[TaskRole.MASTER])
        cluster_spec = ClusterSpec([task_spec], 0, 1, 0)
        return cluster_spec

    @classmethod
    def _check(cls):
        return True


class ApplianceWorkerClusterResolver(ClusterResolver):
    """Cluster resolver for worker roles in appliance mode."""

    _CONFIGURED = None

    @functools.lru_cache()
    def cluster_spec(self):
        return copy.deepcopy(self.__class__._CONFIGURED)

    @classmethod
    def _check(cls):
        return cls._CONFIGURED is not None

    @classmethod
    def configure(
        cls,
        rank: int,
        num_tasks: int,
        num_csx: int,
        num_workers_per_csx: int,
    ):
        assert num_tasks > 0, f"Expected at least one task, got {num_tasks}"
        assert (
            rank >= 0 and rank < num_tasks
        ), f"Rank must be in [0, {num_tasks}) range, got {rank}"
        assert num_csx > 0, f"Num CSX must be > 0, got {num_csx}"
        assert (
            num_workers_per_csx > 0
        ), f"Num workers per CSX must be > 0, got {num_workers_per_csx}"

        tasks = [
            TaskSpec(
                idx,
                idx % num_workers_per_csx,
                idx // num_workers_per_csx,
                "unknown",
                roles=[TaskRole.WORKER],
            )
            for idx in range(num_tasks)
        ]

        cls._CONFIGURED = ClusterSpec(tasks, rank, num_csx, num_workers_per_csx)

    @classmethod
    def configure_from_cluster_details(
        cls, rank: int, cdp: ClusterDetailsParser
    ):
        wse_id, _ = cdp.extract_wse_details(
            ClusterDetails.TaskInfo.TaskType.WRK, rank
        )[0]
        num_workers_per_csx = cdp.extract_num_workers(wse_id)

        tasks = [
            TaskSpec(
                idx,
                idx % num_workers_per_csx,
                cdp.extract_wse_details(
                    ClusterDetails.TaskInfo.TaskType.WRK, idx
                )[0][0],
                "unknown",
                roles=[TaskRole.WORKER],
            )
            for idx in range(cdp.extract_num_workers())
        ]

        cls._CONFIGURED = ClusterSpec(
            tasks,
            rank,
            cdp.extract_num_csx(),
            num_workers_per_csx,
        )


# Register resolvers in order
ClusterResolver.register_resolver(ApplianceWorkerClusterResolver)
ClusterResolver.register_resolver(ApplianceClientClusterResolver)
