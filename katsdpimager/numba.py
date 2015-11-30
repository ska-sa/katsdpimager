"""Wrapper around numba that either uses numba, or provides a no-op jit
function."""
from __future__ import absolute_import

try:
    from numba import jit
    real_numba = True
except ImportError:
    import warnings
    warnings.warn('numba not found - expect poor performance')
    def jit(*args, **kwargs):
        return lambda x: x
