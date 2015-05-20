"""Tests for katsdpimager.fft"""

import numpy as np
import katsdpimager.fft as fft
import katsdpsigproc.accel as accel
from katsdpsigproc.test.test_accel import device_test, cuda_test
from nose.tools import *

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
        fn = template.instantiate(command_queue, (10, 6, 3))
        # Uses padded data to detect padding bugs
        self.pad_dimension(fn.slots['data'].dimensions[0], 3)
        self.pad_dimension(fn.slots['data'].dimensions[1], 5)
        self.pad_dimension(fn.slots['data'].dimensions[2], 1)
        fn.ensure_all_bound()
        data = fn.buffer('data')
        host_data = np.arange(10 * 6 * 3).reshape(10, 6, 3).astype(np.int32)
        data.set(command_queue, host_data)
        expected = np.fft.fftshift(host_data, axes=(0, 1))
        fn()
        actual = data.get(command_queue)
        np.testing.assert_equal(expected, actual)


class TestComplexToReal(object):
    @device_test
    def test_complex_to_real(self, context, command_queue):
        shape = (69, 123)
        rs = np.random.RandomState(1)
        template = fft.ComplexToRealTemplate(context, np.float32)
        fn = template.instantiate(command_queue, shape)
        fn.ensure_all_bound()
        src = (rs.standard_normal(shape) + 1j * rs.standard_normal(shape)).astype(np.complex64)
        fn.buffer('src').set(command_queue, src)
        fn.buffer('dest').set(command_queue, np.zeros(shape, np.float32))
        fn()
        dest = fn.buffer('dest').get(command_queue)
        np.testing.assert_array_equal(src.real, dest)


class TestFft(object):
    @device_test
    @cuda_test
    def test_forward(self, context, command_queue):
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(command_queue, 2, (16, 48, 3, 2), np.complex64, (24, 64, 4, 5), (20, 48, 4, 5))
        fn = template.instantiate(fft.FFT_FORWARD, allocator=accel.SVMAllocator(context))
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src[:] = (rs.standard_normal(src.shape) + 1j * rs.standard_normal(src.shape)).astype(np.complex64)
        fn()
        command_queue.finish()
        expected = np.fft.fftn(src, axes=(0, 1))
        np.testing.assert_allclose(expected, dest, rtol=1e-4)

    @device_test
    @cuda_test
    def test_inverse(self, context, command_queue):
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(command_queue, 2, (16, 48, 3, 2), np.complex64, (24, 64, 4, 5), (20, 48, 4, 5))
        fn = template.instantiate(fft.FFT_INVERSE, allocator=accel.SVMAllocator(context))
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src[:] = (rs.standard_normal(src.shape) + 1j * rs.standard_normal(src.shape)).astype(np.complex64)
        fn()
        command_queue.finish()
        expected = np.fft.ifftn(src, axes=(0, 1)) * (src.shape[0] * src.shape[1])
        np.testing.assert_allclose(expected, dest, rtol=1e-4)
