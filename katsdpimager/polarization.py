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
from contextlib import contextmanager

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


@contextmanager
def _np_seterr(*args, **kwargs):
    """Context-manager version of :py:func:`np.seterr` that restores the
    previous state on exit from the context.
    """
    old = np.seterr(*args, **kwargs)
    yield
    np.seterr(**old)

def polarization_matrix(outputs, inputs):
    """Return a matrix that will map the input polarizations to the outputs.

    The matrix is computed using linear algebra. Let `s` represent the Stokes
    parameters (IQUV). The inputs correspond to :math:`As`, for some matrix
    `A`, and similarly the outputs to :math:`Bs`. We are thus searching for
    the matrix `X` such that :math:`Bs=XAs` or :math:`A^TX^T = B^T`.

    If redundant inputs are given, then the solution is not uniquely
    determined.

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
    # to apply_polarization_matrix.
    Xr = np.round(np.float32(4) * X) * np.float32(0.25)
    np.putmask(X, np.isclose(X, Xr), Xr)
    assert X.dtype == np.complex64
    return X.T

def apply_polarization_matrix(data, matrix):
    """Convert polarization basis in data using a matrix computed by
    :py:func:`polarization_matrix`. Rather than using a straight
    matrix product, only the non-zero elements of the matrix are taken into
    account. Apart from reducing computation, this makes this function
    suitable when some of the input values are non-finite but do not
    contribute to the output.

    Parameters
    ----------
    data : array-like
        Visibility data, or inverse weights (variances). The last
        dimension corresponds to polarization.
    matrix : array-like
        Matrix returned by :py:func:`polarization_matrix`, or constructed
        otherwise.
    """
    out_shape = data.shape[:-1] + matrix.shape[0:1]
    out = np.zeros(out_shape, dtype=data.dtype)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                out[..., i] += matrix[i, j] * data[..., j]
    return out

def apply_polarization_matrix_weighted(data, weights, matrix):
    """Apply a polarization change to visibilities and weights. It is suitable
    even when some weights are zero, indicating flagged data.

    Parameters
    ----------
    data : array-like
        Visibility data. The last dimension corresponds to polarization.
    weights : array-like
        Real-valued weights, in the same shape as `data`.
    matrix : array-like
        Matrix returned by :py:func:`polarization_matrix`, or constructed
        otherwise.

    Returns
    -------
    out_data : array-like
        Transformed visibilities
    out_weights : array-like
        Transformed weights
    """
    data = apply_polarization_matrix(data, matrix)
    # Transform weights to variance estimates. The abs() is to force
    # negative zeros to positive zeros, so that the reciprocal
    # is +inf.
    with _np_seterr(divide='ignore'):
        variance = np.reciprocal(np.abs(weights))
    weight_matrix = np.multiply(matrix, matrix.conj()).real  # Square of abs, element-wise
    variance = apply_polarization_matrix(variance, weight_matrix)
    weights = np.reciprocal(variance)
    return data, weights
