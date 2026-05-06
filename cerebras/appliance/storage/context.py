from typing import NamedTuple, Any


class SerializationContext:
    def __init__(self, *args, **kwargs):
        self.metadata = {}


class DeserializationContext:
    def __init__(self, *args, **kwargs):
        pass


class SerializedObject:
    def __init__(self, *args, **kwargs):
        pass


class SerializedMetadata:
    def __init__(self, *args, **kwargs):
        pass


class NestedObject(NamedTuple):
    key: Any = None
    value: Any = None


class HoistedObject:
    def __init__(self, *args, **kwargs):
        pass
