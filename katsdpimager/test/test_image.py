"""Tests for :mod:`katsdpimager.image`"""

import numpy as np
import math
import katsdpimager.image as image
import katsdpsigproc.accel as accel
from katsdpsigproc.test.test_accel import device_test


class TestLayerToImage(object):
    @device_test
    def test_2d(self, context, command_queue):
        size = 102
        lm_scale = 0.1 / size
        lm_bias = -lm_scale * size / 3   # Off-centre, to check that it's working
        w = 12.3
        template = image.LayerToImageTemplate(context, np.float32)
        fn = template.instantiate(command_queue, (size, size), lm_scale, lm_bias)
        fn.set_w(w)
        fn.ensure_all_bound()
        # Create random input data
        rs = np.random.RandomState(1)
        src = (rs.uniform(10.0, 100.0, (size, size)) + 1j * rs.uniform(10.0, 100.0, (size, size))).astype(np.complex64)
        kernel1d = rs.uniform(1.0, 2.0, size).astype(np.float32)
        fn.buffer('kernel1d').set(command_queue, kernel1d)
        fn.buffer('layer').set(command_queue, src)
        fn.buffer('image').set(command_queue, np.zeros((size, size), np.float32))
        # Compute expected value
        lm = np.arange(size) * lm_scale + lm_bias
        lm2 = lm * lm
        n = np.sqrt(1 - lm2[:, np.newaxis] - lm2[np.newaxis, :])
        w_correction = np.exp(2j * math.pi * w * (n - 1))
        corrected = np.fft.fftshift(src) * w_correction
        expected = corrected.real * n / np.outer(kernel1d, kernel1d)
        # Check it
        fn()
        actual = fn.buffer('image').get(command_queue)
        np.testing.assert_allclose(expected, actual, 1e-4)

    @device_test
    def test_3d(self, context, command_queue):
        slices = 3
        size = 102
        shape = (slices, size, size)
        lm_scale = 0.1 / size
        lm_bias = -lm_scale * size / 3   # Off-centre, to check that it's working
        w = 12.3
        template = image.LayerToImageTemplate(context, np.float32)
        fn = template.instantiate(command_queue, shape, lm_scale, lm_bias)
        fn.set_w(w)
        fn.ensure_all_bound()
        # Create random input data
        rs = np.random.RandomState(1)
        src = (rs.uniform(10.0, 100.0, shape) + 1j * rs.uniform(10.0, 100.0, shape)).astype(np.complex64)
        kernel1d = rs.uniform(1.0, 2.0, size).astype(np.float32)
        fn.buffer('kernel1d').set(command_queue, kernel1d)
        fn.buffer('layer').set(command_queue, src)
        fn.buffer('image').set(command_queue, np.zeros(shape, np.float32))
        # Compute expected value
        lm = np.arange(size) * lm_scale + lm_bias
        lm2 = lm * lm
        n = np.sqrt(1 - lm2[np.newaxis, :, np.newaxis] - lm2[np.newaxis, np.newaxis, :])
        w_correction = np.exp(2j * math.pi * w * (n - 1))
        corrected = np.fft.fftshift(src, axes=(1, 2)) * w_correction
        expected = corrected.real * n / np.outer(kernel1d, kernel1d)[np.newaxis, ...]
        # Check it
        fn()
        actual = fn.buffer('image').get(command_queue)
        np.testing.assert_allclose(expected, actual, 1e-4)


class TestScale(object):
    @device_test
    def test(self, context, command_queue):
        shape = (4, 123, 234)
        dtype = np.float32
        rs = np.random.RandomState(1)
        template = image.ScaleTemplate(context, dtype, shape[0])
        fn = template.instantiate(command_queue, shape)
        fn.ensure_all_bound()
        src = rs.uniform(size=shape).astype(np.float32)
        scale_factor = np.array([1.2, 2.3, 3.4, -4.5], dtype)
        data = fn.buffer('data')
        data.set(command_queue, src)
        fn.set_scale_factor(scale_factor)
        fn()
        actual = data.get(command_queue)
        expected = src * scale_factor[:, np.newaxis, np.newaxis]
        np.testing.assert_allclose(expected, actual)
