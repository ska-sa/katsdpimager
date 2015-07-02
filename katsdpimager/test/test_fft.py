"""Tests for katsdpimager.fft"""

import numpy as np
import math
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


class TestTaperDivide(object):
    @device_test
    def test_2d(self, context, command_queue):
        size = 102
        lm_scale = 0.1 / size
        lm_bias = -lm_scale * size / 3   # Off-centre, to check that it's working
        w = 12.3
        template = fft.TaperDivideTemplate(context, np.float32)
        fn = template.instantiate(command_queue, (size, size), lm_scale, lm_bias)
        fn.set_w(w)
        fn.ensure_all_bound()
        # Create random input data
        rs = np.random.RandomState(1)
        src = (rs.uniform(10.0, 100.0, (size, size)) + 1j * rs.uniform(10.0, 100.0, (size, size))).astype(np.complex64)
        kernel1d = rs.uniform(1.0, 2.0, size).astype(np.float32)
        fn.buffer('kernel1d').set(command_queue, kernel1d)
        fn.buffer('src').set(command_queue, src)
        fn.buffer('dest').set(command_queue, np.zeros((size, size), np.float32))
        # Compute expected value
        lm = np.arange(size) * lm_scale + lm_bias
        lm2 = lm * lm
        n = np.sqrt(1 - lm2[:, np.newaxis] - lm2[np.newaxis, :])
        w_correction = np.exp(2j * math.pi * w * (n - 1))
        corrected = np.fft.fftshift(src) * w_correction
        expected = corrected.real * n / np.outer(kernel1d, kernel1d)
        # Check it
        fn()
        actual = fn.buffer('dest').get(command_queue)
        np.testing.assert_allclose(expected, actual, 1e-4)

    @device_test
    def test_3d(self, context, command_queue):
        slices = 3
        size = 102
        shape = (slices, size, size)
        lm_scale = 0.1 / size
        lm_bias = -lm_scale * size / 3   # Off-centre, to check that it's working
        w = 12.3
        template = fft.TaperDivideTemplate(context, np.float32)
        fn = template.instantiate(command_queue, shape, lm_scale, lm_bias)
        fn.set_w(w)
        fn.ensure_all_bound()
        # Create random input data
        rs = np.random.RandomState(1)
        src = (rs.uniform(10.0, 100.0, shape) + 1j * rs.uniform(10.0, 100.0, shape)).astype(np.complex64)
        kernel1d = rs.uniform(1.0, 2.0, size).astype(np.float32)
        fn.buffer('kernel1d').set(command_queue, kernel1d)
        fn.buffer('src').set(command_queue, src)
        fn.buffer('dest').set(command_queue, np.zeros(shape, np.float32))
        # Compute expected value
        lm = np.arange(size) * lm_scale + lm_bias
        lm2 = lm * lm
        n = np.sqrt(1 - lm2[np.newaxis, :, np.newaxis] - lm2[np.newaxis, np.newaxis, :])
        w_correction = np.exp(2j * math.pi * w * (n - 1))
        corrected = np.fft.fftshift(src, axes=(1, 2)) * w_correction
        expected = corrected.real * n / np.outer(kernel1d, kernel1d)[np.newaxis, ...]
        # Check it
        fn()
        actual = fn.buffer('dest').get(command_queue)
        np.testing.assert_allclose(expected, actual, 1e-4)


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
