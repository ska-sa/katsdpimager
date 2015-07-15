"""Tests for :mod:`katsdpimager.grid`."""

from __future__ import print_function, division
import numpy as np
import katsdpimager.parameters as parameters
import katsdpimager.polarization as polarization
import katsdpimager.grid as grid
import astropy.units as units
import katsdpsigproc.accel as accel
from katsdpsigproc.test.test_accel import device_test, cuda_test
from nose.tools import *
import mock


class TestDegridder(object):
    """Tests for :class:`grid.Degridder`"""
    def setup(self):
        self.image_parameters = parameters.ImageParameters(
            q_fov=1.0,
            image_oversample=None,
            frequency=0.01 * units.m,
            array=None,
            polarizations=polarization.STOKES_IQUV,
            dtype=np.float64,
            pixel_size=0.0001,
            pixels=256)
        self.grid_parameters = parameters.GridParameters(
            antialias_width=7.0,
            oversample=8,
            image_oversample=4,
            w_slices=1,
            w_planes=32,
            max_w=5 * units.m,
            kernel_width=32)
        self.array_parameters = mock.Mock()
        self.array_parameters.longest_baseline = 5 * units.m

    @device_test
    @cuda_test
    def test(self, context, command_queue):
        max_vis = 1280
        n_vis = 1000
        pixels = self.image_parameters.pixels
        template = grid.DegridderTemplate(context, self.image_parameters, self.grid_parameters)
        fn = template.instantiate(command_queue, self.array_parameters, max_vis,
            allocator=accel.SVMAllocator(context))
        fn.ensure_all_bound()
        shape = (4, pixels, pixels)
        rs = np.random.RandomState(seed=1)
        grid_data = (rs.uniform(-1, 1, size=shape) + 1j * rs.uniform(-1, 1, size=shape)).astype(np.complex128)
        uv = np.random.random_integers(10, 150, (n_vis, 2))
        sub_uv = np.random.random_integers(0, 7, (n_vis, 2))
        w_plane = np.random.random_integers(0, 31, (n_vis,))
        fn.buffer('grid').set(command_queue, grid_data)
        vis = np.zeros((n_vis, 4), np.complex128)
        fn.degrid(uv, sub_uv, w_plane, vis)
        expected = np.zeros_like(vis)
        for i in range(n_vis):
            kernel = np.outer(template.convolve_kernel.data[w_plane[i], sub_uv[i, 1], :],
                              template.convolve_kernel.data[w_plane[i], sub_uv[i, 0], :])
            for j in range(4):
                footprint = grid_data[j, uv[i, 1] : uv[i, 1] + kernel.shape[0],
                                         uv[i, 0] : uv[i, 0] + kernel.shape[1]]
                expected[i, j] = np.dot(kernel.ravel(), footprint.ravel())
        np.testing.assert_allclose(expected, vis, 1e-5)
