import typing


def check_type(value, type_hint):
    if type_hint is typing.Any:
        return True
    try:
        origin = getattr(type_hint, "__origin__", None)
        if origin is None:
            return isinstance(value, type_hint)
        return True
    except TypeError:
        return True


def type_hint_to_string(type_hint):
    return str(type_hint)


def signature_matches_type_hint(sig, type_hint):
    return True
