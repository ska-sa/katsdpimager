"""Tests for :py:mod:`katsdpimager.preprocess`."""

import tempfile
import logging
import os
from contextlib import closing

import astropy.units as units
import numpy as np
from nose.tools import assert_equal, assert_true, assert_false

import katsdpimager.parameters as parameters
import katsdpimager.polarization as polarization
import katsdpimager.preprocess as preprocess


def _empty_recarray(dtype):
    return np.rec.array(None, dtype=dtype, shape=(0,))


class BaseTestVisibilityCollector:
    def setup(self):
        self.image_parameters = []
        self.grid_parameters = []
        for wavelength in np.array([0.25, 0.125]) * units.m:
            self.image_parameters.append(parameters.ImageParameters(
                q_fov=1.0,
                image_oversample=5.0,
                frequency=wavelength, array=None,
                polarizations=polarization.STOKES_IQUV,
                dtype=np.float32,
                pixel_size=1.0/(4096.0*wavelength.value),
                pixels=2048))
            self.grid_parameters.append(parameters.GridParameters(
                antialias_width=7.0,
                oversample=8,
                image_oversample=4.0,
                w_slices=1,
                w_planes=128,
                max_w=400 * units.m,
                kernel_width=64))

    def check(self, collector, expected):
        reader = collector.reader()
        assert_equal(reader.num_channels, collector.num_channels)
        for channel, channel_data in enumerate(expected):
            for w_slice, slice_data in enumerate(channel_data):
                assert_equal(len(slice_data), reader.len(channel, w_slice))
                for block_size in [None, 1, 2, 100]:
                    pieces = []
                    for piece in reader.iter_slice(channel, w_slice, block_size):
                        pieces.append(piece.copy())
                    if pieces:
                        actual = np.rec.array(np.hstack(pieces))
                    else:
                        actual = np.rec.recarray(0, collector.store_dtype)
                    # np.testing functions don't seem to handle structured arrays
                    # well, so test one field at a time.
                    np.testing.assert_array_equal(actual.uv, slice_data.uv)
                    np.testing.assert_array_equal(actual.sub_uv, slice_data.sub_uv)
                    np.testing.assert_allclose(actual.weights, slice_data.weights)
                    np.testing.assert_allclose(actual.vis, slice_data.vis, rtol=1e-5)
                    np.testing.assert_array_equal(actual.w_plane, slice_data.w_plane)

    def test_empty(self):
        with closing(self.factory(self.image_parameters, self.grid_parameters, 2)) as collector:
            pass
        self.check(collector, [[np.rec.recarray(0, collector.store_dtype)] for channel in range(2)])

    def _test_impl(self, use_feed_angles):
        uvw = np.array([
            [12.1, 2.3, 4.7],
            [-3.4, 7.6, 2.5],
            [-5.2, -10.6, 7.2],
            [12.102, 2.299, 4.6],   # Should merge with the first visibility in first channel
            [-1.0, 2.0, 3.0]
        ], dtype=np.float32) * units.m
        weights = np.array([
            [
                [1.3, 0.6, 1.2, 0.1],
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.6, 0.7, 0.8],
                [1.1, 1.2, 1.3, 1.4],
                [1.0, 0.0, 1.0, 1.0]    # Has a zero weight, so should be skipped
            ], [
                [0.2, 2.4, 1.2, 2.6],   # Second channel is double and reverse of first
                [2.0, 2.0, 2.0, 2.0],
                [1.6, 1.4, 1.2, 1.0],
                [2.8, 2.6, 2.4, 2.2],
                [2.0, 2.0, 0.0, 2.0]
            ]], dtype=np.float32)
        baselines = np.array([
            0,
            -1,    # Auto-correlation: should be removed
            1,
            0,
            0
        ], dtype=np.int16)
        vis = np.array([
            [
                [0.5 - 2.3j, 0.1 + 4.2j, 0.0 - 3j, 1.5 + 0j],
                [0.5 - 2.3j, 0.1 + 4.2j, 0.0 - 3j, 1.5 + 0j],
                [1.5 + 1.3j, 1.1 + 2.7j, 1.0 - 2j, 2.5 + 1j],
                [1.2 + 3.4j, 5.6 + 7.8j, 9.0 + 1.2j, 3.4 + 5.6j],
                [10.0, 10.0, 10.0, 10.0]
            ], [
                [3.0 + 0j, 0.0 - 6j, 0.2 + 8.4j, 1.0 - 4.6j],
                [3.0 + 0j, 0.0 - 6j, 0.2 + 8.4j, 1.0 - 4.6j],
                [3.0 + 2j, 2.0 - 4j, 2.2 + 5.4j, 3.0 + 2.6j],
                [6.8 + 11.2j, 18.0 + 2.4j, 11.2 + 15.6j, 2.4 + 6.8j],
                [20.0, 20.0, 20.0, 20.0]
            ]], dtype=np.complex64)
        if use_feed_angles:
            # TODO: use non-trivial feed angles and matrices
            feed_angle1 = feed_angle2 = np.zeros(5, np.float32)
            mueller_stokes = mueller_circular = np.matrix(np.identity(4, np.complex64))
        else:
            feed_angle1 = feed_angle2 = mueller_circular = None
            mueller_stokes = np.matrix(np.identity(4, np.complex64))
        with closing(self.factory(self.image_parameters, self.grid_parameters, 64)) as collector:
            collector.add(uvw, weights, baselines, vis,
                          feed_angle1, feed_angle2,
                          mueller_stokes, mueller_circular)
        self.check(collector, [
            [np.rec.fromarrays([
                [[96, 18], [-42, -85]],
                [[6, 3], [3, 1]],
                [[2.4, 1.8, 2.5, 1.5], [0.5, 0.6, 0.7, 0.8]],
                [[1.97 + 0.75j, 6.78 + 11.88j, 11.7 - 2.04j, 4.91 + 7.84j],
                 [0.75 + 0.65j, 0.66 + 1.62j, 0.7 - 1.4j, 2.0 + 0.8j]],
                [64, 65]], dtype=collector.store_dtype)],
            [np.rec.fromarrays([
                [[387, 73], [387, 73], [-167, -340]],
                [[1, 4], [2, 4], [4, 6]],
                [[0.2, 2.4, 1.2, 2.6], [2.8, 2.6, 2.4, 2.2], [1.6, 1.4, 1.2, 1.0]],
                [[0.6 + 0.0j, 0.0 - 14.4j, 0.24 + 10.08j, 2.6 - 11.96j],
                 [19.04 + 31.36j, 46.8 + 6.24j, 26.88 + 37.44j, 5.28 + 14.96j],
                 [4.8 + 3.2j, 2.8 - 5.6j, 2.64 + 6.48j, 3.0 + 2.6j]],
                [64, 64, 65]], dtype=collector.store_dtype)]
        ])

    def test_simple(self):
        self._test_impl(False)

    def test_feed_angles(self):
        self._test_impl(True)


def test_is_prime():
    assert_true(preprocess._is_prime(2))
    assert_true(preprocess._is_prime(3))
    assert_true(preprocess._is_prime(11))
    assert_true(preprocess._is_prime(10007))
    assert_false(preprocess._is_prime(4))
    assert_false(preprocess._is_prime(6))
    assert_false(preprocess._is_prime(18))
    assert_false(preprocess._is_prime(21))


class TestVisibilityCollectorMem(BaseTestVisibilityCollector):
    def factory(self, *args, **kwargs):
        return preprocess.VisibilityCollectorMem(*args, **kwargs)


class TestVisibilityCollectorHDF5(BaseTestVisibilityCollector):
    def setup(self):
        super().setup()
        self._tmpfiles = []

    def factory(self, *args, **kwargs):
        handle, filename = tempfile.mkstemp(suffix='.h5')
        self._tmpfiles.append(filename)
        os.close(handle)
        return preprocess.VisibilityCollectorHDF5(filename, *args, **kwargs)

    def teardown(self):
        for filename in self._tmpfiles:
            try:
                os.remove(filename)
            except OSError as e:
                logging.warning("Failed to remove {}: {}".format(filename, e))
        self._tmpfiles = []
