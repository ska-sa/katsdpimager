"""Fake version of :mod:`katsdpimager._nvtx` that does nothing."""

import contextlib
from typing import Optional, Generator, Union


def initialize() -> None:
    pass


class RegisteredString:
    def __init__(self, handle: object) -> None:
        pass


class Domain:
    def __init__(self, name: str) -> None:
        pass


def register_string(value: str, domain: Optional[Domain] = None) -> RegisteredString:
    return RegisteredString(object())


@contextlib.contextmanager
def thread_range(name: Union[str, RegisteredString], *,
                 payload: Union[None, int, float] = None,
                 domain: Optional[Domain] = None) -> Generator[None, None, None]:
    yield
