"""Python bindings for NVIDIA Tool Extensions.

This currently only handles a small subset of the API.

If the NVTX v3 headers were not detected at compile time, this module can still
be used but it will consist of no-ops.
"""

import contextlib
from typing import Generator

try:
    from ._nvtx import ffi, lib
except ModuleNotFoundError:
    def initialize() -> None:
        pass

    @contextlib.contextmanager
    def thread_range(name: str) -> Generator[None, None, None]:
        yield
else:
    def initialize() -> None:
        lib.nvtxInitialize(ffi.NULL)

    @contextlib.contextmanager
    def thread_range(name: str) -> Generator[None, None, None]:
        p = ffi.new("wchar_t[]", name)
        lib.nvtxRangePushW(p)
        try:
            yield
        finally:
            lib.nvtxRangePop()
