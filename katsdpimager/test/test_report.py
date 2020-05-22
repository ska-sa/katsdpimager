"""Tests for :py:mod:`katsdpimager.report`"""

import astropy.units as u
import numpy as np

from .. import report


class TestPolynomialSEFDModel:
    def setup(self):
        self.model_simple = report.PolynomialSEFDModel(
            2 * u.MHz, 10 * u.MHz,
            [[3, 1, 2]] * u.Jy, u.MHz, 0.5)
        self.model_mixed_units = report.PolynomialSEFDModel(
            2000 * u.kHz, 1e7 * u.Hz,
            [[3000, 1000, 2000]] * u.mJy, u.MHz, 0.5)
        self.freq = [1, 2, 3, 10, 11] * u.MHz

    def test_simple(self):
        sefd = self.model_simple(self.freq).to_value(u.Jy)
        np.testing.assert_allclose(sefd, [np.nan, 13.0, 24.0, 213.0, np.nan])

    def test_mixed_units(self):
        sefd = self.model_mixed_units(self.freq.to(u.GHz)).to_value(u.Jy)
        np.testing.assert_allclose(sefd, [np.nan, 13.0, 24.0, 213.0, np.nan])

    def test_effective(self):
        sefd = self.model_simple(self.freq).to_value(u.Jy)
        sefd_effective = self.model_simple(self.freq, effective=True).to_value(u.Jy)
        np.testing.assert_allclose(sefd_effective, sefd * 2)

    def test_multi_pol(self):
        model = report.PolynomialSEFDModel(
            2 * u.MHz, 4 * u.MHz,
            [[3, 1], [2, 2]] * u.Jy, u.MHz, 0.5)   # 3 + x and 2 + 2x
        sefd = model(self.freq).to_value(u.Jy)
        # Pols should give (5, 6) and (6, 8) Jy
        np.testing.assert_allclose(sefd, [np.nan, np.sqrt(30.5), np.sqrt(50.0), np.nan, np.nan])
