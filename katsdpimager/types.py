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


_FLOAT_TYPES = set(['float', 'double', 'long double'])


def _sub_overrides(overrides, prefix):
    out = {}
    for (key, value) in overrides.iteritems():
        if key.startswith(prefix):
            out[key[len(prefix):]] = value
    return out


def cffi_ctype_to_dtype(ffi, ctype, overrides=None):
    """Convert a CFFI ctype representing a struct to a numpy dtype.

    This does not necessarily handle all possible cases, but should correctly
    account for things like padding.

    Parameters
    ----------
    ffi
        cffi object imported from the library module
    ctype : `CType`
        Type object created from CFFI
    overrides : dict, optional
        Map elements of the type to specified numpy types. The keys are
        strings. To specify an element of a structure, use ``.name``. To
        specify the elements of an array type, use ``[]``. These strings can
        be concatenated (with no whitespace) to select sub-elements.
    """
    if overrides is None:
        overrides = {}
    try:
        return overrides['']
    except KeyError:
        pass

    if ctype.kind == 'primitive':
        if ctype.cname in _FLOAT_TYPES:
            return np.dtype('f' + str(ffi.sizeof(ctype)))
        elif ctype.cname == 'char':
            return np.dtype('c')
        elif ctype.cname == '_Bool':
            return np.dtype(np.bool_)
        else:
            test = int(ffi.cast(ctype, -1))
            if test == -1:
                return np.dtype('i' + str(ffi.sizeof(ctype)))
            else:
                return np.dtype('u' + str(ffi.sizeof(ctype)))
    elif ctype.kind == 'struct':
        names = []
        formats = []
        offsets = []
        for field in ctype.fields:
            if field[1].bitsize != -1:
                raise ValueError('bitfields are not supported')
            names.append(field[0])
            sub_overrides = _sub_overrides(overrides, '.' + field[0])
            formats.append(cffi_ctype_to_dtype(ffi, field[1].type, sub_overrides))
            offsets.append(field[1].offset)
        return np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=ffi.sizeof(ctype)))
    elif ctype.kind == 'array':
        shape = []
        prefix = ''
        while ctype.kind == 'array' and prefix not in overrides:
            shape.append(ctype.length)
            ctype = ctype.item
            prefix += '[]'
        sub_overrides = _sub_overrides(overrides, prefix)
        return np.dtype((cffi_ctype_to_dtype(ffi, ctype, sub_overrides), tuple(shape)))
    else:
        raise ValueError('Unhandled kind {}'.format(ctype.kind))
