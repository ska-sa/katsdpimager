"""Routines for dealing with transformations between polarization bases

Enumeration of polarizations uses the CASA enumeration values, which are
different from those used in FITS:

.. py:data:: STOKES_I

.. py:data:: STOKES_Q

.. py:data:: STOKES_U

.. py:data:: STOKES_V

.. py:data:: STOKES_RR

.. py:data:: STOKES_RL

.. py:data:: STOKES_LR

.. py:data:: STOKES_LL

.. py:data:: STOKES_XX

.. py:data:: STOKES_XY

.. py:data:: STOKES_YX

.. py:data:: STOKES_YY
"""

import numpy as np


STOKES_I = 1
STOKES_Q = 2
STOKES_U = 3
STOKES_V = 4
STOKES_RR = 5
STOKES_RL = 6
STOKES_LR = 7
STOKES_LL = 8
STOKES_XX = 9
STOKES_XY = 10
STOKES_YX = 11
STOKES_YY = 12

STOKES_IQUV = [STOKES_I, STOKES_Q, STOKES_U, STOKES_V]

#: Names for polarizations used in display and command line
STOKES_NAMES = [None, 'I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL', 'XX', 'XY', 'YX', 'YY']

#: Coefficients for each polarization relative to IQUV
STOKES_COEFF = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1j, 0],
    [0, 1, -1j, 0],
    [1, 0, 0, -1],
    [1, 1, 0, 0],
    [0, 0, 1, 1j],
    [0, 0, 1, -1j],
    [1, -1, 0, 0]], np.complex64)


def polarization_matrix(outputs, inputs):
    """Return a Mueller matrix that will map the input polarizations to the outputs.

    The matrix is computed using linear algebra. Let `s` represent the Stokes
    parameters (IQUV). The inputs correspond to :math:`As`, for some matrix
    `A`, and similarly the outputs to :math:`Bs`. We are thus searching for
    the matrix `X` such that :math:`Bs=XAs` or :math:`A^TX^T = B^T`.

    If redundant inputs are given, then the solution is not uniquely
    determined.

    This function is general, and does not require 4 inputs and outputs, or
    even the same number of inputs and outputs.

    Raises
    ------
    ValueError
        if the inputs do not contain the necessary elements to compute the
        outputs.
    """
    A = np.matrix(STOKES_COEFF[inputs, :]).T
    B = np.matrix(STOKES_COEFF[outputs, :]).T
    X, residuals, rank, s = np.linalg.lstsq(A, B, 1e-5)
    # We can't just check for non-zero residuals, because lstsq doesn't
    # return them if A is rank-deficient. Rank-deficiency doesn't
    # necessarily mean there isn't a solution, if B lies in the subspace
    # spanned by A.
    if np.linalg.norm(A * X - B, 'fro') > 1e-5:
        raise ValueError('no solution')
    # In the common cases, the values are all multiples of 0.25, but lstsq has
    # rounding errors. Round off anything that is close enough to a multiple.
    # In particular, tiny values will be flushed to zero, which is important
    # to apply_mueller_matrix.
    Xr = np.round(np.float32(4) * X) * np.float32(0.25)
    np.putmask(X, np.isclose(X, Xr), Xr)
    assert X.dtype == np.complex64
    return X.T


def polarization_matrices(outputs, inputs):
    """Compute a pair of Mueller matrices for conversion via a circular (RL)
    frame.

    Parameters
    ----------
    outputs : list
        Output polarizations
    inputs : list
        Input polarizations

    Returns
    -------
    from_circular : matrix
        Mueller matrix from circular frame to `output`
    to_circular : matrix
        Mueller matrix from `inputs` to circular frame

    Raises
    ------
    ValueError
        If `inputs` are not a full polarization basis
    """
    circular = [STOKES_RR, STOKES_RL, STOKES_LR, STOKES_LL]
    return polarization_matrix(outputs, circular), polarization_matrix(circular, inputs)


def parse_stokes(str_value):
    ans = []
    for p in str_value:
        if p not in 'IQUV':
            raise ValueError('Invalid Stokes parameter {}'.format(p))
    if not str_value:
        raise ValueError('Empty Stokes parameter list')
    for p in 'IQUV':
        cnt = str_value.count(p)
        if cnt > 1:
            raise ValueError('Stokes parameter {} listed multiple times'.format(p))
        elif cnt > 0:
            ans.append(STOKES_NAMES.index(p))
    return sorted(ans)
