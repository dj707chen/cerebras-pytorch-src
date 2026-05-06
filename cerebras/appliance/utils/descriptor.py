from dataclasses import dataclass, fields


class _DefaultSentinel:
    def __repr__(self):
        return "DEFAULT"


@dataclass
class Descriptor:
    _DEFAULT = _DefaultSentinel()

    def reset(self):
        for f in fields(self):
            if f.default is not f.default_factory:
                setattr(self, f.name, f.default)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
