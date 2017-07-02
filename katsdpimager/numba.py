"""Wrapper around numba that either uses numba, or provides a no-op jit
function."""
from __future__ import division, print_function, absolute_import

try:
    from numba import jit
    have_numba = True
except ImportError:
    def jit(*args, **kwargs):
        return lambda x: x
    have_numba = False
