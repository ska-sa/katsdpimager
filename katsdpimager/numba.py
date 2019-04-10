"""Wrapper around numba that either uses numba, or provides a no-op jit
function."""

try:
    from numba import jit
    have_numba = True
except ImportError:
    def jit(*args, **kwargs):
        return lambda x: x
    have_numba = False
