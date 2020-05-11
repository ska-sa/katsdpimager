"""Tests for :py:mod:`katsdpimager.frontend`"""

import numpy as np

from .. import frontend


def test_find_peak():
    # Create something very vaguely like a primary beam.
    size = 4096
    noise = 15.0
    peak = 200.0

    x = np.linspace(-np.pi / 2, np.pi / 2, size)
    y = np.cos(x)
    pbeam = y[np.newaxis, :] * y[:, np.newaxis]
    pbeam[pbeam < 0.01] = np.nan

    rs = np.random.RandomState(seed=1)
    image = rs.normal(scale=noise, size=(4, size, size)) / pbeam

    np.testing.assert_equal(frontend.find_peak(image, pbeam, noise), np.nan)
    image[1, size // 2 + 5, size // 2 - 10] = peak
    np.testing.assert_equal(frontend.find_peak(image, pbeam, noise), peak)
    image *= -1    # Test negative peaks
    np.testing.assert_equal(frontend.find_peak(image, pbeam, noise), peak)
