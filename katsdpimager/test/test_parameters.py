"""Tests for :py:mod:`katsdpimager.parameters`."""

import numpy as np
from nose.tools import *
import katsdpimager.parameters as parameters

class TestPolarizationMatrix(object):
    def setup(self):
        self.IQUV = [parameters.STOKES_I, parameters.STOKES_Q, parameters.STOKES_U, parameters.STOKES_V]
        self.IQ = [parameters.STOKES_I, parameters.STOKES_Q]
        self.XY = [parameters.STOKES_XX, parameters.STOKES_XY, parameters.STOKES_YX, parameters.STOKES_YY]
        self.XY_DIAG = [parameters.STOKES_XX, parameters.STOKES_YY]
        self.RL = [parameters.STOKES_RR, parameters.STOKES_RL, parameters.STOKES_LR, parameters.STOKES_LL]
        self.RL_DIAG = [parameters.STOKES_RR, parameters.STOKES_LL]

    def test_xy_to_iquv(self):
        expected = 0.5 * np.matrix([[1, 0, 0, 1],
                                    [1, 0, 0, -1],
                                    [0, 1, 1, 0],
                                    [0, -1j, 1j, 0]])
        actual = parameters.polarization_matrix(self.IQUV, self.XY)
        np.testing.assert_array_equal(expected, actual)

    def test_stokes_to_rl(self):
        expected = np.matrix([[1, 0, 0, 1],
                              [0, 1, 1j, 0],
                              [0, 1, -1j, 0],
                              [1, 0, 0, -1]])
        actual = parameters.polarization_matrix(self.RL, self.IQUV)
        np.testing.assert_array_equal(expected, actual)

    def test_xy_to_rl(self):
        expected = 0.5 * np.matrix([[1, -1j, 1j, 1],
                                    [1, 1j, 1j, -1],
                                    [1, -1j, -1j, -1],
                                    [1, 1j, -1j, 1]])
        actual = parameters.polarization_matrix(self.RL, self.XY)
        np.testing.assert_array_equal(expected, actual)

    def test_xy_diag_to_iq(self):
        expected = 0.5 * np.matrix([[1, 1], [1, -1]])
        actual = parameters.polarization_matrix(self.IQ, self.XY_DIAG)
        np.testing.assert_array_equal(expected, actual)

    def test_xy_diag_to_i(self):
        expected = 0.5 * np.matrix([[1, 1]])
        actual = parameters.polarization_matrix([parameters.STOKES_I], self.XY_DIAG)
        np.testing.assert_array_equal(expected, actual)

    def test_xy_diag_to_iquv(self):
        assert_raises(ValueError, parameters.polarization_matrix, self.IQUV, self.XY_DIAG)
