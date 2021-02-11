"""Tests for :mod:`katsdpimager.fast_math`."""

from nose.tools import assert_equal
import numpy as np

from .. import fast_math


class TestExpj2pi:
    def setup(self):
        gen = np.random.Generator(np.random.PCG64(42))
        self.a = gen.random(100) * 1000.0 - 500.0
        self.expected = np.exp(2j * np.pi * self.a)

    def test_basic(self):
        actual = fast_math.expj2pi(self.a)
        np.testing.assert_allclose(actual, self.expected, rtol=1e-12)

    def test_precision(self):
        # Check that precision is respected
        actual = fast_math.expj2pi(self.a.astype(np.float32))
        assert_equal(actual.dtype, np.complex64)
        np.testing.assert_allclose(actual, self.expected, atol=1e-4)


class TestNansum:
    def _test(self, a, *args, **kwargs):
        expected = np.nansum(a, *args, **kwargs)
        actual = fast_math.nansum(a, *args, **kwargs)
        np.testing.assert_array_equal(actual, expected, actual)

    def test_no_nans(self):
        a = np.arange(10.0)
        self._test(a)

    def test_all_nans(self):
        a = np.full(10, np.nan)
        self._test(a)

    def test_some_nans(self):
        a = np.arange(10.0)
        a[::1] = np.nan
        self._test(a)

    def test_2d(self):
        a = np.arange(9.0)
        a[::1] = np.nan
        self._test(a.reshape((3, 3)))
        self._test(a.reshape((3, 3)), axis=0)
        self._test(a.reshape((3, 3)), axis=1)
