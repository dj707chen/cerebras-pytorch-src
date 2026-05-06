# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from collections import UserDict
from inspect import isclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from warnings import warn

import dill
import torch
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

TScope = List[str]
TNested = Union[Any, list, tuple, dict]
TSelectFn = Callable[[Any], bool]
TVisitFnRet = Generator[Tuple[List[str], Any], None, None]

_recursive_structures = {}


def register_visit_fn(*types):
    """Register a function to visit a specific type of data structure.

    The function should take a single argument, the data structure, and yield
    a tuple of (scope, item) for each item in the data structure. The scope is
    a list of strings that represents the hierarchical path to the item in the
    data structure.
    """

    def decorator(fn):
        nonlocal types

        for t in types:
            if not isclass(t):
                raise TypeError(f"Expected a type, got {t}")

        # convert the types to a frozenset to remove duplicates
        # and enforce a strict order
        types = tuple(frozenset(types))
        if types in _recursive_structures:
            if _recursive_structures[types] != fn:
                warn(
                    f"Visit function for {types} was previously registered to\n"
                    f"{_recursive_structures[types]}\n"
                    f"Overwriting with\n{fn}."
                )

        # Otherwise we're seeing these types and function for the first time
        error = dill.detect.errors(types)
        if error:
            raise TypeError(
                f"Cannot register a visit function for the provided types {types}"
                f"because it contains the following unserializable types."
            ) from error

        error = dill.detect.errors(fn)
        if error:
            raise RuntimeError(
                f"Cannot register visit function {fn} for the provided {types} "
                f"because it is unserializable."
            ) from error

        _recursive_structures[types] = fn
        return fn

    return decorator


def unregister_visit_fn(*types):
    """Unregister a function to visit a specific type of data structure."""
    for t in types:
        _recursive_structures.pop(t, None)


def find_visit_fn(struct: TNested) -> Callable[[TNested], TVisitFnRet]:
    """Find the visit function for a given data structure."""
    for types, visit_fn in _recursive_structures.items():
        if isinstance(struct, types):
            return visit_fn


def serialize_visit_fns() -> Dict[str, str]:
    """
    Serialize the visit functions to a dictionary so that they can be propagated
    to the workers.
    """
    return {
        dill.dumps(types).hex(): dill.dumps(visit_fn).hex()
        for types, visit_fn in _recursive_structures.items()
    }


def deserialize_visit_fns(visit_fn_map: Dict[str, str]) -> None:
    """
    Deserialize the visit functions from a dictionary and register them
    so that they can be used to traverse the data structures.
    """
    for serialized_types, serialized_visit_fn in visit_fn_map.items():
        register_visit_fn(*dill.loads(bytes.fromhex(serialized_types)))(
            dill.loads(bytes.fromhex(serialized_visit_fn))
        )


@register_visit_fn(dict, UserDict)
def visit_dict(struct: Union[dict, UserDict]) -> TVisitFnRet:
    """Visit a dict."""
    for k, v in struct.items():
        yield [str(k)], v


@register_visit_fn(list, tuple)
def visit_list_tuple(struct: Union[tuple, list]) -> TVisitFnRet:
    """Visit a list or tuple."""
    for i, v in enumerate(struct):
        yield [str(i)], v


def visit_structure(
    data_structure: TNested,
    select_fn: TSelectFn,
    strict: bool = False,
    scope: Optional[TScope] = None,
) -> Generator[Tuple[TScope, Any], None, None]:
    """Recursively traverse nested structure and return the items accepted by
    the selector.

    Args:
        data_structure: A nested data structure to traverse recursively.
        select_fn: A callable that returns true if the item passed should be
            selected.
        strict: Strictly checks that an item in the nested structure is either
            a list/dict/tuple or selected by the select_fn. Otherwise, raises
            an error. Defaults to False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Yields:
        A tuples of (scope, item) for each item selected by the select_fn.
    """
    scope = scope or []
    visit_fn = find_visit_fn(data_structure)
    if visit_fn:
        for scope_value in visit_fn(data_structure):
            if not isinstance(scope_value, (list, tuple)):
                raise TypeError(
                    f"Expected visit function to return a list or tuple, "
                    f"got {type(scope_value)}"
                )
            elif len(scope_value) != 2:
                raise ValueError(
                    f"Expected visit function to return a list or tuple of length 2, "
                    f"got {len(scope_value)}"
                )

            s, value = scope_value
            yield from visit_structure(value, select_fn, strict, scope + s)
    elif select_fn(data_structure):
        yield scope, data_structure
    elif strict:
        raise ValueError(f"Unknown data structure: {data_structure}")


def visit_torch_tensors(
    data_structure: TNested,
    strict: bool = False,
    scope: Optional[TScope] = None,
) -> Generator[Tuple[TScope, torch.Tensor], None, None]:
    """Recursively finds all torch tensors in the nested data structure.

    Args:
        data_structure: A nested data structure to traverse recursively.
        strict: Strictly checks that an item in the nested structure is one of
            a list/dict/tuple/tensor. Otherwise, raises an error. Defaults to
            False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Yields:
        A tuple of (scope, tensor) for each tensor in the nested structure.
    """
    yield from visit_structure(
        data_structure,
        select_fn=lambda item: isinstance(item, torch.Tensor),
        strict=strict,
        scope=scope,
    )


def visit_device_tensors(
    data_structure: TNested,
    device_type: str,
    strict: bool = False,
    scope: Optional[TScope] = None,
) -> Generator[Tuple[TScope, torch.Tensor], None, None]:
    """Recursively finds all device tensors in the nested data structure.

    Args:
        data_structure: A nested data structure to traverse recursively.
        strict: Strictly checks that an item in the nested structure is one of
            a list/dict/tuple/device tensor. Otherwise, raises an error.
            Defaults to False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Yields:
        A Tuple of (scope, tensor) for each tensor in the nested structure.
    """
    for s, tensor in visit_torch_tensors(
        data_structure,
        strict=strict,
        scope=scope,
    ):
        if tensor.device.type == device_type:
            yield s, tensor


def visit_lazy_tensors(
    data_structure: TNested,
    strict: bool = False,
    scope: Optional[TScope] = None,
) -> Generator[Tuple[TScope, torch.Tensor], None, None]:
    """Recursively finds all Lazy tensors in the nested data structure.

    Args:
        data_structure: A nested data structure to traverse recursively.
        strict: Strictly checks that an item in the nested structure is one of
            a list/dict/tuple/Lazy tensor. Otherwise, raises an error. Defaults
            to False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Yields:
        A Tuple of (scope, Lazy tensor) for each tensor in the nested structure.
    """
    yield from visit_structure(
        data_structure,
        select_fn=lambda item: isinstance(item, torch.Tensor)
        and item.device.type == "lazy",
        strict=strict,
        scope=scope,
    )


def map_structure(
    map_fn: Callable,
    data_structure: TNested,
    select_fn: TSelectFn,
    scope: Optional[TScope] = None,
):
    objs, spec = tree_flatten(data_structure)

    objs = [
        map_fn(s, t) if select_fn(t) else t
        for s, t in zip(recurse_spec(spec, scope), objs)
    ]
    return tree_unflatten(objs, spec)


def map_torch_tensors(
    map_fn: Callable,
    data_structure: TNested,
    scope: Optional[TScope] = None,
):
    return map_structure(
        map_fn,
        data_structure,
        select_fn=lambda item: isinstance(item, torch.Tensor),
        scope=scope,
    )


def map_lazy_tensors(
    map_fn: Callable,
    data_structure: TNested,
    scope: Optional[TScope] = None,
):
    return map_structure(
        map_fn,
        data_structure,
        select_fn=lambda item: isinstance(item, torch.Tensor)
        and item.device.type == "lazy",
        scope=scope,
    )


def recurse_spec(spec: TreeSpec, scope: Optional[TScope] = None):
    if spec.num_leaves == 0:
        return

    scope = scope or []
    if context := spec.context:
        # If the context is a named tuple
        if (
            isclass(context)
            and issubclass(context, tuple)
            and hasattr(context, "_fields")
        ):
            context = range(len(context._fields))

        for key, val in zip(context, spec.children_specs):
            yield from recurse_spec(val, scope + [str(key)])
    elif not spec.children_specs:
        yield scope
    else:
        for i, val in enumerate(spec.children_specs):
            yield from recurse_spec(val, scope + [str(i)])


def diff(
    data_structure_1: TNested, data_structure_2: TNested
) -> Generator[str, None, None]:
    """
    Compare two nested data structures.

    Yields all the keys and values where the data structures differ.
    """
    value1, spec1 = torch.utils._pytree.tree_flatten(data_structure_1)
    value2, spec2 = torch.utils._pytree.tree_flatten(data_structure_2)

    flattened1 = {
        ".".join(scope): value
        for value, scope in zip(value1, recurse_spec(spec1))
    }
    flattened2 = {
        ".".join(scope): value
        for value, scope in zip(value2, recurse_spec(spec2))
    }

    if set(flattened1) != set(flattened2):
        yield (
            f"Data structures have differing keys.\n"
            f"Keys only in data structure 1: "
            f"{sorted(set(flattened1) - set(flattened2))}\n"
            f"Keys only in data structure 2: "
            f"{sorted(set(flattened2) - set(flattened1))}\n"
        )

    for key, value1 in flattened1.items():
        value2 = flattened2[key]
        if value1 != value2:
            yield f"Found mismatching values for {key}: {value1} != {value2}"
