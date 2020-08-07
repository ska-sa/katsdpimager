"""Python bindings for NVIDIA Tool Extensions.

This currently only handles a small subset of the API.

If the NVTX v3 headers were not detected at compile time, this module can still
be used but it will consist of no-ops.
"""

try:
    from ._nvtx_real import (             # noqa: F401
        initialize, RegisteredString, Domain, register_string, thread_range
    )
except ModuleNotFoundError:
    from ._nvtx_fallback import (         # noqa: F401
        initialize, RegisteredString, Domain, register_string, thread_range
    )
