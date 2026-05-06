import os
from pathlib import Path
from typing import Union

StrPath = Union[str, os.PathLike]


def create_symlink(src, dst):
    os.symlink(src, dst)


def get_path_size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def is_pathlike(obj):
    return isinstance(obj, (str, os.PathLike))
