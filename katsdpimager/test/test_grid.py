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


class BaseTest(object):
    def setup(self):
        pixels = 256
        w_planes = 32
        oversample = 8
        n_vis = 1000
        kernel_width = 32
        self.image_parameters = parameters.ImageParameters(
            q_fov=1.0,
            image_oversample=None,
            frequency=0.01 * units.m,
            array=None,
            polarizations=polarization.STOKES_IQUV,
            dtype=np.float64,
            pixel_size=0.0001,
            pixels=pixels)
        self.grid_parameters = parameters.GridParameters(
            antialias_width=7.0,
            oversample=oversample,
            image_oversample=4,
            w_slices=1,
            w_planes=w_planes,
            max_w=5 * units.m,
            kernel_width=kernel_width)
        self.array_parameters = mock.Mock()
        self.array_parameters.longest_baseline = 5 * units.m
        # Create a track in which movement happens in various subsets of the
        # axes, using a random walk
        rs = np.random.RandomState(seed=1)
        self.uv = np.empty((n_vis, 2), dtype=np.int16)
        self.sub_uv = np.empty((n_vis, 2), dtype=np.int16)
        self.w_plane = np.empty((n_vis,), dtype=np.int16)
        for i in range(0, n_vis):
            if i % 73 == 0:
                # Add an occasional total jump
                self.uv[i, :] = rs.randint(0, pixels - kernel_width, (2,))
                self.sub_uv[i, :] = rs.randint(0, oversample, (2,))
                self.w_plane[i] = rs.randint(0, w_planes)
            else:
                for j in range(2):
                    self.uv[i, j] = (self.uv[i - 1, j] + rs.random_integers(-1, 1)) % (pixels - kernel_width)
                    self.sub_uv[i, j] = (self.sub_uv[i - 1, j] + rs.random_integers(-1, 1)) % oversample
                self.w_plane[i] = (self.w_plane[i - 1] + rs.random_integers(-1, 1)) % w_planes


class TestGridder(BaseTest):
    """Tests for :class:`grid.Gridder`"""
    @device_test
    @cuda_test
    def test(self, context, command_queue):
        max_vis = 1280
        n_vis = len(self.w_plane)
        pixels = self.image_parameters.pixels
        template = grid.GridderTemplate(context, self.image_parameters, self.grid_parameters)
        fn = template.instantiate(command_queue, self.array_parameters, max_vis,
            allocator=accel.SVMAllocator(context))
        fn.ensure_all_bound()
        rs = np.random.RandomState(seed=2)
        vis = (rs.uniform(-1, 1, size=(n_vis, 4))
               + 1j * rs.uniform(-1, 1, size=(n_vis, 4))).astype(np.complex128)
        grid_data = fn.buffer('grid')
        grid_data.fill(0)
        fn.grid(self.uv, self.sub_uv, self.w_plane, vis)
        command_queue.finish()
        expected = np.zeros_like(grid_data)
        for i in range(n_vis):
            kernel = np.outer(template.convolve_kernel.data[self.w_plane[i], self.sub_uv[i, 1], :],
                              template.convolve_kernel.data[self.w_plane[i], self.sub_uv[i, 0], :])
            kernel = np.conj(kernel, kernel)
            for j in range(4):
                footprint = expected[j, self.uv[i, 1] : self.uv[i, 1] + kernel.shape[0],
                                        self.uv[i, 0] : self.uv[i, 0] + kernel.shape[1]]
                footprint[:] += vis[i, j].astype(np.complex128) * kernel
        np.testing.assert_allclose(expected, grid_data, 1e-5, 1e-12)


class TestDegridder(BaseTest):
    """Tests for :class:`grid.Degridder`"""
    @device_test
    @cuda_test
    def test(self, context, command_queue):
        max_vis = 1280
        n_vis = len(self.w_plane)
        pixels = self.image_parameters.pixels
        template = grid.DegridderTemplate(context, self.image_parameters, self.grid_parameters)
        fn = template.instantiate(command_queue, self.array_parameters, max_vis,
            allocator=accel.SVMAllocator(context))
        fn.ensure_all_bound()
        shape = (4, pixels, pixels)
        rs = np.random.RandomState(seed=2)
        grid_data = (rs.uniform(-1, 1, size=shape) + 1j * rs.uniform(-1, 1, size=shape)).astype(np.complex128)
        fn.buffer('grid').set(command_queue, grid_data)
        vis = np.zeros((n_vis, 4), np.complex128)
        fn.degrid(self.uv, self.sub_uv, self.w_plane, vis)
        expected = np.zeros_like(vis)
        for i in range(n_vis):
            kernel = np.outer(template.convolve_kernel.data[self.w_plane[i], self.sub_uv[i, 1], :],
                              template.convolve_kernel.data[self.w_plane[i], self.sub_uv[i, 0], :])
            for j in range(4):
                footprint = grid_data[j, self.uv[i, 1] : self.uv[i, 1] + kernel.shape[0],
                                         self.uv[i, 0] : self.uv[i, 0] + kernel.shape[1]]
                expected[i, j] = np.dot(kernel.ravel(), footprint.ravel())
        np.testing.assert_allclose(expected, vis, 1e-5)
