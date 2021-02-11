"""Variants of standard maths functions written for speed."""

import numpy as np
import numba


@numba.vectorize([numba.complex64(numba.float32),
                  numba.complex128(numba.float64)])
def expj2pi(x):
    """Equivalent to ``expj(2 * np.pi * x)`` where `x` is real.

    x is reduced to a small value before multiplication, which
    improves precision at a small cost in performance.
    """
    y = 2 * np.pi * (x - np.rint(x))
    return complex(np.cos(y), np.sin(y))


def nansum(x, *args, **kwargs):
    """Like np.nansum, provided `x` is a floating-point type."""
    return np.sum(x, *args, where=~np.isnan(x), **kwargs)
