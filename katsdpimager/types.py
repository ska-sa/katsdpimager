"""Utilities for converting type names between numpy and C"""

import numpy as np


def dtype_to_ctype(dtype):
    """Convert a numpy dtype to a C type suitable for CUDA or OpenCL.

    Only float32/64, complex64/128 and int32/uint32 are supported for now.
    """
    if dtype == np.float32:
        return 'float'
    elif dtype == np.complex64:
        return 'float2'
    elif dtype == np.float64:
        return 'double'
    elif dtype == np.complex128:
        return 'double2'
    elif dtype == np.int32:
        return 'int'
    elif dtype == np.uint32:
        return 'uint'
    else:
        raise ValueError('Unrecognised dtype {}'.format(dtype))


def real_to_complex(dtype):
    """Convert a real type to its complex equivalent"""
    if dtype == np.float32:
        return np.complex64
    elif dtype == np.float64:
        return np.complex128
    else:
        raise ValueError('Unrecognised dtype {}'.format(dtype))


def complex_to_real(dtype):
    """Convert a complex type to its real equivalent"""
    if dtype == np.complex64:
        return np.float32
    elif dtype == np.complex128:
        return np.float64
    else:
        raise ValueError('Unrecognised dtype {}'.format(dtype))
