from collections import namedtuple


def register_serializer(type_or_func=None):
    def decorator(cls):
        return cls
    if type_or_func is None:
        return decorator
    if isinstance(type_or_func, type):
        return decorator
    return decorator(type_or_func)


class StorageReader:
    Stats = namedtuple("Stats", ["num_objects", "total_bytes"], defaults=[0, 0])

    def __init__(self, *args, **kwargs):
        pass


class StorageWriter:
    def __init__(self, *args, **kwargs):
        pass


class DeferredStorageReader:
    def __init__(self, *args, **kwargs):
        pass


class DeferredObject:
    def __init__(self, *args, **kwargs):
        pass


class S3Reader:
    def __init__(self, *args, **kwargs):
        pass


class SerializationContext:
    def __init__(self, *args, **kwargs):
        self.metadata = {}
