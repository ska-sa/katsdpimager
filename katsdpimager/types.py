"""Utilities for converting type names between numpy and C"""

from __future__ import division, print_function
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


def cffi_ctype_to_dtype(ffi, ctype, override_fields=None):
    """Convert a CFFI ctype representing a struct to a numpy dtype.

    This does not necessarily handle all possible cases, but should correctly
    account for things like padding.

    Parameters
    ----------
    ffi
        cffi object imported from the library module
    ctype : `CType`
        Type object created from CFFI
    override_fields : dict, optional
        For each key matching a field in `ctype`, the corresponding value is
        used as the dtype instead of a recursive calls. This only applies at
        the top level.
    """
    if ctype.kind == 'primitive':
        if ctype.cname in ['int8_t', 'int16_t', 'int32_t', 'int64_t', 'int', 'ptrdiff_t']:
            return np.dtype('i' + str(ffi.sizeof(ctype)))
        elif ctype.cname in ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'unsigned int', 'size_t']:
            return np.dtype('u' + str(ffi.sizeof(ctype)))
        elif ctype.cname in ['float', 'double', 'long double']:
            return np.dtype('f' + str(ffi.sizeof(ctype)))
        else:
            raise ValueError('Unhandled primitive type {}'.format(ctype.cname))
    elif ctype.kind == 'struct':
        names = []
        formats = []
        offsets = []
        for field in ctype.fields:
            names.append(field[0])
            if override_fields is not None and field[0] in override_fields:
                dtype = override_fields[field[0]]
            else:
                dtype = cffi_ctype_to_dtype(ffi, field[1].type)
            formats.append(dtype)
            offsets.append(field[1].offset)
        return np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=ffi.sizeof(ctype)))
    elif ctype.kind == 'array':
        shape = []
        while ctype.kind == 'array':
            shape.append(ctype.length)
            ctype = ctype.item
        return np.dtype((cffi_ctype_to_dtype(ffi, ctype), tuple(shape)))
    else:
        raise ValueError('Unhandled kind {}'.format(ctype.kind))
