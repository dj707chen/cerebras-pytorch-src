# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import logging
from abc import ABC, abstractmethod
from typing import Callable, List

from ..constraints.base import Constraint
from ..data_containers import Operand, Particle
from ..ulp import UlpStrategy

LOGGER = logging.getLogger(__name__)


class CsOpParticleMixin:
    """
    Mixin class for validating a particle against an operation's constraints.
    """

    def __init__(self, particle: Particle):
        self.particle = particle

    def validate(self) -> None:
        """
        Validates a particle against the operation's constraints.
        """
        LOGGER.debug(f"Validating particle: {self.particle}")
        self.constraint.validate(self.particle)


class CsOpMlirMixin(CsOpParticleMixin):
    """
    Mixin class for validating an mlir operation against an operation's constraints.
    """

    def __init__(self, mlir_operands: List["torch_mlir.ir.Value"]):
        self.particle = Particle(
            Operand.from_mlir(name, value)
            for name, value in zip(self.operands, mlir_operands)
        )


class CsOp(ABC):
    """
    Base class for operations.

    This class should not be instantiated directly. Use the factory methods
    `from_particle` or `from_mlir` to create instances.
    """

    constraint: Constraint  # Cerebras-spspecific constraint for the operation
    op_func: Callable  # Operation function associated with the operation
    name: (
        str  # Name of the operation, including the overload name if applicable
    )
    operands: List[str]  # List of operand names for the operation
    ulp_strategy: UlpStrategy

    required_attrs = {
        "constraint",
        "op_func",
        "name",
        "operands",
        "ulp_strategy",
    }

    @abstractmethod
    def validate(self) -> None:
        """
        Validates the operation's inputs.
        This abstract method forces subclasses to implement validation.
        """

    @classmethod
    def from_particle(cls, particle: Particle) -> "CsOpFromParticle":
        """
        Constructs an operation from a particle.
        """

        # Note that the mixin preceeds the base class to ensure
        # that the methods defined in the mixin are resolved first.
        class CsOpFromParticle(CsOpParticleMixin, cls):
            pass

        return CsOpFromParticle(particle=particle)

    @classmethod
    def from_mlir(
        cls, mlir_operands: List["torch_mlir.ir.Value"]
    ) -> "CsOpFromMlir":
        """
        Constructs an operation from mlir operands.
        """

        # Note that the mixin preceeds the base class to ensure
        # that the methods defined in the mixin are resolved first.
        class CsOpFromMlir(CsOpMlirMixin, cls):
            pass

        return CsOpFromMlir(mlir_operands=mlir_operands)

    @classmethod
    def op_name(cls) -> str:
        return cls.name

    def __repr__(self):
        particle_str = str(getattr(self, "particle", "No particle"))
        return f"<{self.__class__.__name__}({self.name}): {particle_str}>"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr in cls.required_attrs:
            if not hasattr(cls, attr):
                raise AttributeError(f"Missing required attribute {attr}")
