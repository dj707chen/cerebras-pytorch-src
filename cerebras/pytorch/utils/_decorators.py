# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from functools import lru_cache, wraps

import torch


def lazy_tensor_lru_cache(**lru_kwargs):
    """
    Decorator to cache the result of a function that takes in
    lazy tensors as input.

    Lazy tensors are not hashable. As a workaround, the id(tensor)
    is used as the hash. However, inplace ops change the underlying
    tensor, but the id remains the same. This can lead to incorrect
    caching results. To avoid this, we use the unique_id of the tensor
    as part of the hash.

    Args:
        **kwargs: Keyword arguments to pass to the lru_cache function.
    """
    from cerebras.pytorch.lib import cerebras_pytorch_lib

    # Custom tuple class so that pytree doesn't recurse into the structure
    class _tuple(tuple):
        pass

    def decorator(func):
        @lru_cache(**lru_kwargs)
        def cached_func(*args, **kwargs):
            args, kwargs = torch.utils._pytree.tree_map(
                # unwrap lazy tensors so that the original function
                # gets the original arguments
                lambda a: a[1],
                (args, kwargs),
            )
            return func(*args, **kwargs)

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            args, kwargs = torch.utils._pytree.tree_map(
                # wrap lazy tensors in a tuple that includes its
                # unique id so that inplace changes induce a cache miss
                lambda a: _tuple(
                    (
                        (
                            cerebras_pytorch_lib.get_unique_id(a)
                            if isinstance(a, torch.Tensor)
                            and a.device.type == "lazy"
                            else 0
                        ),
                        a,
                    )
                ),
                (args, kwargs),
            )
            return cached_func(*args, **kwargs)

        wrapped_func.cache_info = cached_func.cache_info
        wrapped_func.cache_clear = cached_func.cache_clear
        wrapped_func.__wrapped__ = func

        return wrapped_func

    return decorator
