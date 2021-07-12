"""Tests for :py:mod:`katsdpimager.primary_beam`."""

import hashlib

import numpy as np
import astropy.units as u
from nose.tools import assert_raises, assert_equal, assert_true
from katsdpmodels.primary_beam import OutputType, AltAzFrame

from .. import primary_beam


class TestTrivialPrimaryBeam:
    def setup(self) -> None:
        self.model = primary_beam.meerkat_v1_beam('L')
        N = 101
        STEP = 0.001
        self.coords = (np.arange(N) - N // 2) * STEP

    def test_out_of_range(self) -> None:
        freqs = u.Quantity([1.4], u.GHz)
        # Azimuth out of range
        beam = self.model.sample_grid(
            [-0.5], [0.0], freqs, AltAzFrame(), OutputType.UNPOLARIZED_POWER
        )
        assert_true(np.isnan(beam[0, 0, 0]))
        # Elevation out of range
        beam = self.model.sample_grid(
            [0.0], [-0.5], freqs, AltAzFrame(), OutputType.UNPOLARIZED_POWER
        )
        assert_true(np.isnan(beam[0, 0, 0]))
        # Frequency too low
        beam = self.model.sample_grid(
            [0.0], [0.0], 0.5 * u.GHz, AltAzFrame(), OutputType.UNPOLARIZED_POWER
        )
        assert_true(np.all(np.isnan(beam)))
        # Frequency too high
        beam = self.model.sample_grid(
            [0.0], [0.0], 3 * u.GHz, AltAzFrame(), OutputType.UNPOLARIZED_POWER
        )
        assert_true(np.all(np.isnan(beam)))

    def _show_beam(self, beam: np.ndarray) -> None:
        """Debug function to plot a beam (should not be any calls checked in)."""
        import matplotlib.pyplot as plt
        if beam.ndim == 2:
            beam = beam[np.newaxis]
        rows = 1 if beam.shape[0] == 1 else 2
        fig, axs = plt.subplots(rows, beam.shape[0], sharex=True, sharey=True, squeeze=False)
        for channel in range(beam.shape[0]):
            axs[0][channel].imshow(beam[channel], vmin=0, vmax=1.01)
            if channel > 0:
                delta = beam[channel] - beam[0]
                axs[1, channel].imshow(delta, vmin=-0.01, vmax=0.01)
        plt.show()

    def test_given_frequency(self) -> None:
        """Sample at frequency already present in the model."""
        beam = self.model.sample_grid(
            self.coords, self.coords, 1284 * u.MHz, AltAzFrame(), OutputType.UNPOLARIZED_POWER
        )
        N = len(self.coords)
        assert_equal(beam.shape, (N, N))
        # Should be unity at the centre of the beam
        np.testing.assert_allclose(beam[N // 2, N // 2], 1.0, rtol=1e-2)
        # If it fails due to code/model changes, uncomment to inspect visually:
        # self._show_beam(beam)
        assert_equal(hashlib.md5(beam).hexdigest(), '36a37a6a08f071ed0e26fd523dc7b8a8')

    def test_interpolated_frequency(self) -> None:
        """Sample at frequency not present in the model."""
        # Outer two are frequencies in the model
        freqs = [856, 856.41796875, 856.8359375] * u.MHz
        beam = self.model.sample_grid(
            self.coords, self.coords, freqs, AltAzFrame(), OutputType.UNPOLARIZED_POWER
        )
        assert_equal(beam.shape, (3, len(self.coords), len(self.coords)))
        # If it fails due to code/model changes, uncomment to inspect visually:
        # self._show_beam(beam)
        assert_equal(hashlib.md5(beam[1]).hexdigest(), 'a6a288680257d59ccebdec14e28a029c')


class TestMeerkatV1Beam:
    def test_load(self):
        """Smoke test to check that the files can be found."""
        for band in primary_beam.BANDS:
            primary_beam.meerkat_v1_beam(band)

    def test_bad_band(self):
        with assert_raises(ValueError):
            primary_beam.meerkat_v1_beam('Z')
