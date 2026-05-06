# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from inspect import isclass

from cerebras.appliance.utils.descriptor import Descriptor
from cerebras.appliance.utils.typing import check_type, type_hint_to_string

DEFAULT = object()


@dataclass(repr=False)
class _flags:
    """
    Base class for flag classes.

    All flag classes should be dataclasses that inherit from this class.
    """

    def __post_init__(self):
        self.__class__.__repr__ = _flags.__repr__

    def __setattr__(self, name, value):
        if name.startswith("_"):
            # Don't perform any checks for private members
            return super().__setattr__(name, value)

        field_map = {f.name: f for f in fields(self)}

        if name not in field_map:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no flag '{name}'"
            )

        if (
            isclass(field_map[name].type)
            and issubclass(field_map[name].type, _flags)
            and hasattr(self, name)
            and getattr(self, name) is not value
        ):
            raise AttributeError(
                f"{name} is a nested flag and cannot be set directly"
            )

        for field in fields(self):
            if field.name == name:
                if value is DEFAULT:
                    # If value is DEFAULT, set it to the default value
                    value = field.default
                elif value is Descriptor._DEFAULT:
                    # Don't type check if the value is a descriptor default
                    pass
                elif not check_type(value, field.type):
                    raise TypeError(
                        f"Expected {type_hint_to_string(field.type)} for `{name}`. "
                        f"Got {type(value)}: {value}"
                    )

                break

        return super().__setattr__(name, value)

    def __iter__(self):
        """
        Implement an iterator across the dictionary representation.

        This allows dict(flags) to correctly return a dictionary representation
        """
        yield from self.asdict().items()

    def asdict(self) -> dict:
        """Return a copy of the flags as a dictionary."""
        return asdict(self)

    def load_flags(self, data: dict):
        """Load flags recursively from a dictionary."""
        if isinstance(data, _flags):
            if not isinstance(data, self.__class__):
                raise ValueError(
                    f"Expected a {self.__class__.__name__}. "
                    f"Got {type(data)}"
                )
            else:
                data = data.asdict()

        if not isinstance(data, dict):
            raise ValueError(f"Expected a dictionary, got {type(data)}")

        for key, value in data.items():
            if (flags := getattr(self, key)) and isinstance(flags, _flags):
                flags.load_flags(value)
            else:
                setattr(self, key, deepcopy(value))

    def reset(self, recurse=True):
        """Reset all attributes to their default values."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, _flags) and recurse:
                value.reset(recurse)
            else:
                setattr(self, field.name, field.default)

    def __repr__(self):
        def _repr(obj):
            values = ", ".join(
                (
                    f"{key}={_repr(value)}"
                    if isinstance(value, dict)
                    else f"{key}={value}"
                )
                for key, value in obj.items()
            )
            return f"({values})"

        return f"{self.__class__.__name__.lstrip('_')}{_repr(asdict(self))}"

    def __enter__(self):
        self.__dict__.setdefault("_old_flags", []).append(self.asdict())
        return self

    def __exit__(self, *args):
        old_flags = self.__dict__.get("_old_flags", [])
        if old_flags:
            self.load_flags(old_flags.pop())
