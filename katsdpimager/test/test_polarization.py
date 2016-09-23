"""Tests for :py:mod:`katsdpimager.polarization`."""

from __future__ import print_function, division
import numpy as np
from nose.tools import *
import katsdpimager.polarization as polarization


class TestPolarizationMatrix(object):
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
        assert_raises(ValueError, polarization.polarization_matrix, self.IQUV, self.XY_DIAG)


class TestApplyMuellerMatrixWeighted(object):
    def setup(self):
        self.inputs = [
            polarization.STOKES_XX,
            polarization.STOKES_XY,
            polarization.STOKES_YX,
            polarization.STOKES_YY]
        self.outputs = [
            polarization.STOKES_I,
            polarization.STOKES_Q,
            polarization.STOKES_U,
            polarization.STOKES_V]
        self.pm = polarization.polarization_matrix(self.outputs, self.inputs)

    def test_unflagged(self):
        vis = np.array([
            [2 + 4j, 4 - 2j, 4, 8j],
            [0, 2 + 2j, 2j, 6 - 2j]], np.complex64)
        weights = np.array([
            [4, 2, 1, 4],
            [2, 8, 1, 4]], np.float32)
        expected_vis = np.array([
            [1 + 6j, 1 - 2j, 4 - 1j, -1],
            [3 - 1j, -3 + 1j, 1 + 2j, -1j]], np.complex64)
        expected_weights = np.array([
            [8, 8, 8 / 3, 8 / 3],
            [16 / 3, 16 / 3, 32 / 9, 32 / 9]], np.float32)
        actual_vis = polarization.apply_mueller_matrix(vis, self.pm)
        actual_weights = polarization.apply_mueller_matrix_weights(
            weights, self.pm)
        np.testing.assert_allclose(actual_vis, expected_vis)
        np.testing.assert_allclose(actual_weights, expected_weights)

    def test_flagged(self):
        vis = np.array([
            [2 + 4j, 4 - 2j, 4, 8j],
            [0, 2 + 2j, 2j, 6 - 2j]], np.complex64)
        weights = np.array([
            [4, 0, 1, 4],
            [0, 8, 1, 0]], np.float32)
        expected_vis = np.array([
            [1 + 6j, 1 - 2j, 4 - 1j, -1],
            [3 - 1j, -3 + 1j, 1 + 2j, -1j]], np.complex64)
        expected_weights = np.array([
            [8, 8, 0, 0],
            [0, 0, 32 / 9, 32 / 9]], np.float32)
        actual_vis = polarization.apply_mueller_matrix(vis, self.pm)
        actual_weights = polarization.apply_mueller_matrix_weights(
            weights, self.pm)
        np.testing.assert_allclose(actual_vis, expected_vis)
        np.testing.assert_allclose(actual_weights, expected_weights)
