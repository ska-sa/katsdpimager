"""Tests for :mod:`katsdpimager.fft`"""

import numpy as np
import math
import katsdpimager.fft as fft
import katsdpsigproc.accel as accel
from katsdpsigproc.test.test_accel import device_test, cuda_test


class TestFftshift(object):
    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def test_2d(self, context, command_queue):
        template = fft.FftshiftTemplate(context, np.int32)
        fn = template.instantiate(command_queue, (10, 6))
        fn.ensure_all_bound()
        data = fn.buffer('data')
        host_data = (np.arange(6).reshape(1, 6) + np.arange(10).reshape(10, 1)).astype(np.int32)
        data.set(command_queue, host_data)
        expected = np.fft.fftshift(host_data)
        fn()
        actual = data.get(command_queue)
        np.testing.assert_equal(expected, actual)

    @device_test
    def test_3d(self, context, command_queue):
        template = fft.FftshiftTemplate(context, np.int32)
        fn = template.instantiate(command_queue, (3, 10, 6))
        # Uses padded data to detect padding bugs
        self.pad_dimension(fn.slots['data'].dimensions[0], 1)
        self.pad_dimension(fn.slots['data'].dimensions[1], 3)
        self.pad_dimension(fn.slots['data'].dimensions[2], 5)
        fn.ensure_all_bound()
        data = fn.buffer('data')
        host_data = np.arange(3 * 10 * 6).reshape(3, 10, 6).astype(np.int32)
        data.set(command_queue, host_data)
        expected = np.fft.fftshift(host_data, axes=(1, 2))
        fn()
        actual = data.get(command_queue)
        np.testing.assert_equal(expected, actual)


class TestFft(object):
    @device_test
    @cuda_test
    def test_forward(self, context, command_queue):
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            command_queue, 2, (3, 2, 16, 48), np.complex64, (4, 5, 24, 64), (4, 5, 20, 48))
        fn = template.instantiate(fft.FFT_FORWARD, allocator=accel.SVMAllocator(context))
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src[:] = (rs.standard_normal(src.shape) +
                  1j * rs.standard_normal(src.shape)).astype(np.complex64)
        fn()
        command_queue.finish()
        expected = np.fft.fftn(src, axes=(2, 3))
        np.testing.assert_allclose(expected, dest, rtol=1e-4)

    @device_test
    @cuda_test
    def test_inverse(self, context, command_queue):
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            command_queue, 2, (3, 2, 16, 48), np.complex64, (4, 5, 24, 64), (4, 5, 20, 48))
        fn = template.instantiate(fft.FFT_INVERSE, allocator=accel.SVMAllocator(context))
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src[:] = (rs.standard_normal(src.shape) +
                  1j * rs.standard_normal(src.shape)).astype(np.complex64)
        fn()
        command_queue.finish()
        expected = np.fft.ifftn(src, axes=(2, 3)) * (src.shape[2] * src.shape[3])
        np.testing.assert_allclose(expected, dest, rtol=1e-4)
