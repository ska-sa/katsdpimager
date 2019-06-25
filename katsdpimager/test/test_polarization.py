"""Tests for :py:mod:`katsdpimager.polarization`."""

import numpy as np
from nose.tools import assert_equal, assert_raises

from .. import polarization


class TestPolarizationMatrix:
    """Tests for :py:func:`katsdpimager.polarization.polarization_matrix`,
    using standard coordinate systems.
    """
    def setup(self):
        self.IQUV = polarization.STOKES_IQUV
        self.IQ = [polarization.STOKES_I, polarization.STOKES_Q]
        self.XY = [polarization.STOKES_XX,
                   polarization.STOKES_XY,
                   polarization.STOKES_YX,
                   polarization.STOKES_YY]
        self.XY_DIAG = [polarization.STOKES_XX, polarization.STOKES_YY]
        self.RL = [polarization.STOKES_RR,
                   polarization.STOKES_RL,
                   polarization.STOKES_LR,
                   polarization.STOKES_LL]
        self.RL_DIAG = [polarization.STOKES_RR, polarization.STOKES_LL]

    def test_xy_to_iquv(self):
        expected = 0.5 * np.matrix([[1, 0, 0, 1],
                                    [1, 0, 0, -1],
                                    [0, 1, 1, 0],
                                    [0, -1j, 1j, 0]])
        actual = polarization.polarization_matrix(self.IQUV, self.XY)
        np.testing.assert_array_equal(expected, actual)

    def test_stokes_to_rl(self):
        expected = np.matrix([[1, 0, 0, 1],
                              [0, 1, 1j, 0],
                              [0, 1, -1j, 0],
                              [1, 0, 0, -1]])
        actual = polarization.polarization_matrix(self.RL, self.IQUV)
        np.testing.assert_array_equal(expected, actual)

    def test_xy_to_rl(self):
        expected = 0.5 * np.matrix([[1, -1j, 1j, 1],
                                    [1, 1j, 1j, -1],
                                    [1, -1j, -1j, -1],
                                    [1, 1j, -1j, 1]])
        actual = polarization.polarization_matrix(self.RL, self.XY)
        np.testing.assert_array_equal(expected, actual)

    def test_xy_diag_to_iq(self):
        expected = 0.5 * np.matrix([[1, 1], [1, -1]])
        actual = polarization.polarization_matrix(self.IQ, self.XY_DIAG)
        np.testing.assert_array_equal(expected, actual)

    def test_xy_diag_to_i(self):
        expected = 0.5 * np.matrix([[1, 1]])
        actual = polarization.polarization_matrix([polarization.STOKES_I], self.XY_DIAG)
        np.testing.assert_array_equal(expected, actual)

    def test_xy_diag_to_iquv(self):
        with assert_raises(ValueError):
            polarization.polarization_matrix(self.IQUV, self.XY_DIAG)


class TestParseStokes:
    def test_good(self):
        assert_equal(polarization.parse_stokes('QIV'),
                     [polarization.STOKES_I, polarization.STOKES_Q, polarization.STOKES_V])

    def test_empty(self):
        with assert_raises(ValueError):
            polarization.parse_stokes('')

    def test_duplicate(self):
        with assert_raises(ValueError):
            polarization.parse_stokes('QQ')

    def test_invalid(self):
        with assert_raises(ValueError):
            polarization.parse_stokes('Z')
