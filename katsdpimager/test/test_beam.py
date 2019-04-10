"""Tests for :py:mod:`katsdpimager.beam`"""

import numpy as np
import scipy.signal
from astropy.modeling import models
from katsdpsigproc.test.test_accel import device_test, cuda_test

from katsdpimager import beam


def convolve_beam_reference(model, beam):
    """Reference implementation of
    :py:func:`~katsdpimager.beam.convolve_beam`, for comparison.

    It will not produce exactly the same result, because it uses a sampled
    form of the beam, and does not wrap around.
    """
    assert model.shape[1] % 2 == 0
    assert model.shape[2] % 2 == 0
    hh = model.shape[1] // 2
    hw = model.shape[2] // 2
    m = np.arange(-hh + 1, hh)
    l = np.arange(-hw + 1, hw)
    beam_pixels = beam.model(*np.meshgrid(m, l, indexing='ij'))
    out = np.empty_like(model)
    for pol in range(model.shape[0]):
        out[pol, ...] = scipy.signal.fftconvolve(model[pol], beam_pixels, 'same')
    return out


class TestConvolveBeam:
    """Test both host and device beam convolution functions"""
    def setup(self):
        self.beam = beam.Beam(models.Gaussian2D(amplitude=3.5, x_mean=0.0, y_mean=0.0,
                                                x_stddev=2.0, y_stddev=5.0, theta=1))
        self.model = np.zeros((4, 128, 128), np.float32)
        self.model[0, 32, 80] = 1.0
        self.model[0, 100, 40] = 2.0
        self.model[1, 50, 60] = 3.0
        self.model[2, 64, 64] = 4.0
        self.model[2, 80, 64] = 3.0
        self.expected = convolve_beam_reference(self.model, self.beam)

    def test_host(self):
        actual = beam.convolve_beam(self.model, self.beam)
        np.testing.assert_allclose(self.expected, actual, rtol=1e-5, atol=1e-5)

    @device_test
    @cuda_test
    def test_device(self, context, command_queue):
        template = beam.ConvolveBeamTemplate(command_queue, self.model.shape[1:], self.model.dtype)
        fn = template.instantiate()
        fn.ensure_all_bound()
        for pol in range(self.model.shape[0]):
            fn.buffer('image').set(command_queue, self.model[pol])
            fn.beam = self.beam
            fn()
            actual = fn.buffer('image').get(command_queue)
            np.testing.assert_allclose(self.expected[pol], actual, rtol=1e-5, atol=1e-5)
