def retrieve_all_subclasses(cls, condition=None):
    result = []
    for subclass in cls.__subclasses__():
        if condition is None or condition(subclass):
            result.append(subclass)
        result.extend(retrieve_all_subclasses(subclass, condition=condition))
    return result
