"""Tests for :mod:`katsdpimager.image`"""

import math

import numpy as np
from katsdpsigproc.test.test_accel import device_test

from .. import image
from .utils import RandomState


class TestLayerToImage:
    @device_test
    def test(self, context, command_queue):
        slices = 3
        size = 102
        shape = (slices, size, size)
        lm_scale = 0.1 / size
        lm_bias = -lm_scale * size / 3   # Off-centre, to check that it's working
        w = 12.3
        template = image.LayerToImageTemplate(context, np.float32)
        fn = template.instantiate(command_queue, shape, lm_scale, lm_bias)
        fn.set_w(w)
        fn.set_polarization(1)
        fn.ensure_all_bound()
        # Create random input data
        rs = RandomState(1)
        src = rs.complex_uniform(10.0, 100.0, shape[1:]).astype(np.complex64)
        kernel1d = rs.uniform(1.0, 2.0, size).astype(np.float32)
        fn.buffer('kernel1d').set(command_queue, kernel1d)
        fn.buffer('layer').set(command_queue, src)
        fn.buffer('image').zero(command_queue)
        # Compute expected value
        lm = np.arange(size) * lm_scale + lm_bias
        lm2 = lm * lm
        n = np.sqrt(1 - lm2[np.newaxis, :, np.newaxis] - lm2[np.newaxis, np.newaxis, :])
        w_correction = np.exp(2j * math.pi * w * (n - 1))
        corrected = np.fft.fftshift(src) * w_correction
        expected = np.zeros(shape, np.float32)
        expected[1] = corrected.real * n / np.outer(kernel1d, kernel1d)[np.newaxis, ...]
        # Check it
        fn()
        actual = fn.buffer('image').get(command_queue)
        np.testing.assert_allclose(expected, actual, 1e-4)


class TestImageToLayer:
    @device_test
    def test(self, context, command_queue):
        # Since LayerToImage is tested from first principles, we just check
        # that ImageToLayer correctly inverts it
        slices = 3
        size = 102
        shape = (slices, size, size)
        lm_scale = 0.1 / size
        lm_bias = -lm_scale * size / 3   # Off-centre, to check that it's working
        w = 12.3
        image_to_layer_template = image.ImageToLayerTemplate(context, np.float32)
        image_to_layer = image_to_layer_template.instantiate(
            command_queue, shape, lm_scale, lm_bias)
        image_to_layer.ensure_all_bound()
        image_to_layer.set_w(w)
        layer_to_image_template = image.LayerToImageTemplate(context, np.float32)
        layer_to_image = layer_to_image_template.instantiate(
            command_queue, shape, lm_scale, lm_bias)
        layer_to_image.set_w(w)
        layer_to_image.bind(
            image=image_to_layer.buffer('image'),
            layer=image_to_layer.buffer('layer'),
            kernel1d=image_to_layer.buffer('kernel1d'))
        # Create random input data
        rs = np.random.RandomState(1)
        expected = rs.uniform(10.0, 100.0, shape).astype(np.float32)
        kernel1d = rs.uniform(1.0, 2.0, size).astype(np.float32)
        kernel = np.outer(kernel1d, kernel1d)[np.newaxis, ...]
        # Run test function
        image_to_layer.buffer('image').set(command_queue, expected * kernel)
        image_to_layer.buffer('kernel1d').set(command_queue, kernel1d)
        image_to_layer.set_polarization(1)
        image_to_layer()
        # Wipe out the image, to make sure it gets written properly
        image_to_layer.buffer('image').zero(command_queue)
        # Convert back again
        layer_to_image.set_polarization(1)
        layer_to_image()
        actual = layer_to_image.buffer('image').get(command_queue) * kernel
        np.testing.assert_allclose(expected[1], actual[1], 1e-4)


class TestScale:
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
