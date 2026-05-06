import functools


def get_available_memory(unit="bytes"):
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return 0


def get_process_memory_full_info():
    try:
        import psutil
        return psutil.Process().memory_full_info()
    except (ImportError, AttributeError):
        return None


def with_memory_info_logged(description="", info=None, logger=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
