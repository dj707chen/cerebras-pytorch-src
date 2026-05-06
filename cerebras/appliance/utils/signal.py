from contextlib import contextmanager


@contextmanager
def on_sigint(handler=None):
    yield
