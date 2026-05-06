class ValueContext:
    def __init__(self, default=None):
        self.value = default

    def __call__(self, value):
        old = self.value
        self.value = value
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class BooleanContext:
    def __init__(self, default=False):
        self.value = default

    def __call__(self, value=True):
        old = self.value
        self.value = value
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __bool__(self):
        return self.value
