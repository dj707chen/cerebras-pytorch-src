# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from functools import wraps
from threading import Lock
from typing import Callable, Optional


def call_once(error_callable: Optional[Callable] = None):
    def wrapper_generator(f):
        """
        Wraps the function and ensures its only called once

        The output is cached and returned on subsequent calls
        if the error callable is not None and returns
        """
        lock = Lock()
        output = []

        @wraps(f)
        def wrapper(*args, **kwargs):
            with lock:
                if output:
                    if error_callable:
                        error_callable()
                    return output[0]
                output.append(f(*args, **kwargs))

            return output[0]

        return wrapper

    return wrapper_generator
