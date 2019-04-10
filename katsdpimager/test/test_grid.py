"""Tests for :mod:`katsdpimager.grid`."""

from unittest import mock

import numpy as np
import astropy.units as units
import katsdpsigproc.accel as accel
from katsdpsigproc.test.test_accel import device_test, force_autotune

from .. import parameters, polarization, grid
from .utils import RandomState


def _middle(array, shape):
    """Returns a view of the central part of `array` with size `shape`."""
    index = []
    for a, s in zip(array.shape, shape):
        assert a >= s
        assert (a - s) % 2 == 0
        pad = (a - s) // 2
        index.append(np.s_[pad : a - pad])
    return array[tuple(index)]


class BaseTest:
    def setup(self):
        pixels = 256
        grid_cover = 180
        w_planes = 32
        oversample = 8
        n_vis = 1000
        kernel_width = 28
        assert grid_cover + kernel_width < pixels
        self.image_parameters = parameters.ImageParameters(
            q_fov=1.0,
            image_oversample=None,
            frequency=0.01 * units.m,
            array=None,
            polarizations=polarization.STOKES_IQUV,
            dtype=np.float64,
            pixel_size=0.0001,
            pixels=pixels)
        # Make a single-precision version for cases that don't need double
        self.image_parameters_sp = parameters.ImageParameters(
            q_fov=1.0,
            image_oversample=None,
            frequency=self.image_parameters.wavelength,
            array=None,
            polarizations=self.image_parameters.polarizations,
            dtype=np.float32,
            pixel_size=self.image_parameters.pixel_size,
            pixels=self.image_parameters.pixels)
        self.grid_parameters = parameters.GridParameters(
            antialias_width=7.0,
            oversample=oversample,
            image_oversample=4,
            w_slices=1,
            w_planes=w_planes,
            max_w=5 * units.m,
            kernel_width=kernel_width)
        self.array_parameters = mock.Mock()
        self.array_parameters.longest_baseline = self.image_parameters.cell_size * (grid_cover // 2)
        # Create a track in which movement happens in various subsets of the
        # axes, using a random walk
        rs = np.random.RandomState(seed=1)
        self.uv = np.empty((n_vis, 2), dtype=np.int16)
        self.sub_uv = np.empty((n_vis, 2), dtype=np.int16)
        self.w_plane = np.empty((n_vis,), dtype=np.int16)
        for i in range(0, n_vis):
            if i % 73 == 0:
                # Add an occasional total jump
                self.uv[i, :] = rs.randint(0, grid_cover, (2,))
                self.sub_uv[i, :] = rs.randint(0, oversample, (2,))
                self.w_plane[i] = rs.randint(0, w_planes)
            else:
                for j in range(2):
                    self.uv[i, j] = \
                        (self.uv[i - 1, j] + rs.random_integers(-1, 1)) % grid_cover
                    self.sub_uv[i, j] = \
                        (self.sub_uv[i - 1, j] + rs.random_integers(-1, 1)) % oversample
                self.w_plane[i] = (self.w_plane[i - 1] + rs.random_integers(-1, 1)) % w_planes
        self.uv -= grid_cover // 2
        self.convolve_kernel = grid.ConvolutionKernel(
            self.image_parameters, self.grid_parameters)
        self.weights_grid = rs.uniform(size=(4, grid_cover, grid_cover)).astype(np.float32)

    def do_grid(self, callback):
        max_vis = 1280
        n_vis = len(self.w_plane)
        rs = RandomState(seed=2)
        vis = rs.complex_uniform(-1, 1, size=(n_vis, 4)).astype(np.complex128)
        actual = callback(max_vis, vis)
        expected = np.zeros_like(actual)
        pixels = actual.shape[-1]
        uv_bias = (self.convolve_kernel.data.shape[-1] - 1) // 2 - pixels // 2
        for i in range(n_vis):
            kernel = np.outer(self.convolve_kernel.data[self.w_plane[i], self.sub_uv[i, 1], :],
                              self.convolve_kernel.data[self.w_plane[i], self.sub_uv[i, 0], :])
            kernel = np.conj(kernel, kernel)
            u = self.uv[i, 0] - uv_bias
            v = self.uv[i, 1] - uv_bias
            weights_u = self.uv[i, 0] + self.weights_grid.shape[2] // 2
            weights_v = self.uv[i, 1] + self.weights_grid.shape[1] // 2
            for j in range(4):
                footprint = expected[j, v : v + kernel.shape[0], u : u + kernel.shape[1]]
                weight = self.weights_grid[j, weights_v, weights_u]
                footprint[:] += vis[i, j].astype(np.complex128) * weight * kernel
        np.testing.assert_allclose(expected, actual, 1e-5, 1e-8)

    def do_degrid(self, callback):
        max_vis = 1280
        n_vis = len(self.w_plane)
        pixels = self.image_parameters.pixels
        shape = (4, pixels, pixels)
        rs = RandomState(seed=2)
        grid_data = rs.complex_uniform(-1, 1, size=shape).astype(np.complex128)
        vis = rs.complex_uniform(-1, 1, size=(n_vis, 4)).astype(np.complex128)
        weights = rs.uniform(0.5, 1.5, size=(n_vis, 4)).astype(np.float64)
        expected = np.zeros_like(vis.copy())
        uv_bias = (self.convolve_kernel.data.shape[-1] - 1) // 2 - pixels // 2
        for i in range(n_vis):
            kernel = np.outer(self.convolve_kernel.data[self.w_plane[i], self.sub_uv[i, 1], :],
                              self.convolve_kernel.data[self.w_plane[i], self.sub_uv[i, 0], :])
            u = self.uv[i, 0] - uv_bias
            v = self.uv[i, 1] - uv_bias
            for j in range(4):
                footprint = grid_data[j, v : v + kernel.shape[0], u : u + kernel.shape[1]]
                expected[i, j] = (vis[i, j]
                                  - weights[i, j] * np.dot(kernel.ravel(), footprint.ravel()))
        residuals = callback(max_vis, grid_data, weights, vis)
        np.testing.assert_allclose(expected, residuals, 1e-5)


class TestGridder(BaseTest):
    """Tests for :class:`grid.Gridder`"""
    @device_test
    def test(self, context, command_queue):
        def callback(max_vis, vis):
            template = grid.GridderTemplate(context, self.image_parameters, self.grid_parameters)
            fn = template.instantiate(command_queue, self.array_parameters, max_vis,
                                      allocator=accel.SVMAllocator(context))
            fn.ensure_all_bound()
            grid_data = fn.buffer('grid')
            grid_data.fill(0)
            weights_grid = fn.buffer('weights_grid')
            weights_grid_host = weights_grid.empty_like()
            _middle(weights_grid_host, self.weights_grid.shape)[:] = self.weights_grid
            weights_grid.set_async(command_queue, weights_grid_host)
            fn.num_vis = len(self.uv)
            fn.set_coordinates(self.uv, self.sub_uv, self.w_plane)
            fn.set_vis(vis)
            fn()
            command_queue.finish()
            return grid_data
        self.do_grid(callback)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Check that the autotuner runs successfully"""
        grid.GridderTemplate(context, self.image_parameters_sp, self.grid_parameters)


class TestGridderHost(BaseTest):
    """Tests for :Class:`grid.GridderHost`"""
    def test(self):
        def callback(max_vis, vis):
            gridder = grid.GridderHost(self.image_parameters, self.grid_parameters)
            gridder.clear()
            _middle(gridder.weights_grid, self.weights_grid.shape)[:] = self.weights_grid
            gridder.num_vis = len(self.uv)
            gridder.set_coordinates(self.uv, self.sub_uv, self.w_plane)
            gridder.set_vis(vis)
            gridder()
            return gridder.values
        self.do_grid(callback)


class TestDegridder(BaseTest):
    """Tests for :class:`grid.Degridder`"""
    @device_test
    def test(self, context, command_queue):
        def callback(max_vis, grid_data, weights, vis):
            n_vis = len(self.w_plane)
            template = grid.DegridderTemplate(context, self.image_parameters, self.grid_parameters)
            fn = template.instantiate(command_queue, self.array_parameters, max_vis,
                                      allocator=accel.SVMAllocator(context))
            fn.ensure_all_bound()
            grid_buffer = fn.buffer('grid')
            grid_buffer.set(command_queue, _middle(grid_data, grid_buffer.shape))
            fn.num_vis = n_vis
            fn.set_coordinates(self.uv, self.sub_uv, self.w_plane)
            fn.set_weights(weights)
            fn.set_vis(vis)
            fn()
            command_queue.finish()
            return fn.buffer('vis')[:n_vis]
        self.do_degrid(callback)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Check that the autotuner runs successfully"""
        grid.DegridderTemplate(context, self.image_parameters_sp, self.grid_parameters)


class TestDegridderHost(BaseTest):
    """Tests for :class:`grid.DegridderHost`"""
    def test(self):
        def callback(max_vis, grid_data, weights, vis):
            n_vis = len(self.w_plane)
            degridder = grid.DegridderHost(self.image_parameters, self.grid_parameters)
            degridder.values[:] = grid_data
            degridder.num_vis = n_vis
            degridder.set_coordinates(self.uv, self.sub_uv, self.w_plane)
            degridder.set_weights(weights)
            degridder.set_vis(vis)
            degridder()
            return vis
        self.do_degrid(callback)
