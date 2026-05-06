import logging

logger = logging.getLogger("cerebras.appliance")

TRACE = 5
VERBOSE = 15


class ClassLogger:
    _logger = logger

    @property
    def logger(self):
        return self._logger


def named_class_logger(cls_or_name=None):
    if isinstance(cls_or_name, str):
        def decorator(cls):
            cls._logger = logging.getLogger(f"cerebras.appliance.{cls_or_name}")
            return cls
        return decorator
    cls = cls_or_name
    cls._logger = logging.getLogger(f"cerebras.appliance.{cls.__name__}")
    return cls
