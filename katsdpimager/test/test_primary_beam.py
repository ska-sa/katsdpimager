"""Tests for :py:mod:`katsdpimager.primary_beam`."""

import hashlib

import pkg_resources
import numpy as np
import astropy.units as units
from nose.tools import assert_raises, assert_equal, assert_true

from .. import primary_beam


class TestTrivialBeamModel:
    def setup(self) -> None:
        filename = pkg_resources.resource_filename(
            'katsdpimager',
            'models/beams/meerkat/v1/beam_L.h5')
        self.model = primary_beam.TrivialBeamModel(filename)

    def test_out_of_range(self) -> None:
        freqs = units.Quantity([1.4], units.GHz)
        # Azimuth out of range
        beam = self.model.sample(-0.5, 0.01, 100, 0.0, 0.01, 1, freqs)
        assert_true(np.isnan(beam[0, 0, 0, 0, 0]))
        # Elevation out of range
        beam = self.model.sample(0.0, 0.01, 1, -0.5, 0.01, 100, freqs)
        assert_true(np.isnan(beam[0, 0, 0, 0, 0]))
        # Frequency too low
        beam = self.model.sample(0.0, 0.01, 1, 0.0, 0.01, 1, units.Quantity([0.5], units.GHz))
        assert_true(np.all(np.isnan(beam[0, 0])))
        assert_true(np.all(np.isnan(beam[1, 1])))
        # Frequency too high
        beam = self.model.sample(0.0, 0.01, 1, 0.0, 0.01, 1, units.Quantity([3], units.GHz))
        assert_true(np.all(np.isnan(beam[0, 0])))
        assert_true(np.all(np.isnan(beam[1, 1])))

    def test_parameter_values(self) -> None:
        assert_equal(dict(self.model.parameter_values), {})

    def _check_scalar(self, beam: np.ndarray) -> None:
        """Validate that the beam has no leakage and same H and V response."""
        np.testing.assert_array_equal(beam[1, 0], beam[0, 1])
        np.testing.assert_array_equal(beam[0, 0], beam[1, 1])
        np.testing.assert_array_equal(beam[0, 1], np.zeros_like(beam[0, 1]))

    def _show_beam(self, beam: np.ndarray) -> None:
        """Debug function to plot a beam (should not be any calls checked in)."""
        import matplotlib.pyplot as plt
        rows = 1 if beam.shape[2] == 1 else 2
        fig, axs = plt.subplots(rows, beam.shape[2], sharex=True, sharey=True, squeeze=False)
        for channel in range(beam.shape[2]):
            axs[0][channel].imshow(beam[0, 0, channel], vmin=0, vmax=1.01)
            if channel > 0:
                delta = beam[0, 0, channel] - beam[0, 0, 0]
                axs[1, channel].imshow(delta, vmin=-0.01, vmax=0.01)
        plt.show()

    def test_given_frequency(self) -> None:
        """Sample at frequency already present in the model."""
        N = 101
        STEP = 0.001
        beam = self.model.sample(-STEP * (N // 2), STEP, N,
                                 -STEP * (N // 2), STEP, N,
                                 [1284] * units.MHz)
        assert_equal(beam.shape, (2, 2, 1, N, N))
        self._check_scalar(beam)
        # Should be unity at the centre of the beam
        np.testing.assert_allclose(beam[0, 0, 0, N // 2, N // 2], 1.0, rtol=1e-2)
        # If it fails due to code/model changes, uncomment to inspect visually:
        # self._show_beam(beam)
        assert_equal(hashlib.md5(beam[0, 0, 0]).hexdigest(), '8c3ff23e1b8fb76e8b3efd5a88cbf632')

    def test_interpolated_frequency(self) -> None:
        """Sample at frequency not present in the model."""
        N = 101
        STEP = 0.001
        # Outer two are frequencies in the model
        freqs = [856, 856.41796875, 856.8359375] * units.MHz
        beam = self.model.sample(-STEP * (N // 2), STEP, N,
                                 -STEP * (N // 2), STEP, N,
                                 freqs)
        assert_equal(beam.shape, (2, 2, 3, N, N))
        self._check_scalar(beam)
        # If it fails due to code/model changes, uncomment to inspect visually:
        # self._show_beam(beam)
        assert_equal(hashlib.md5(beam[0, 0, 1]).hexdigest(), '2d000becd61f51fda0f38a0163ac99a7')

    def test_offset(self) -> None:
        N = 128
        STEP = 1 / 2048
        freqs = [1284] * units.MHz
        full = self.model.sample(-STEP * (N // 2), STEP, N,
                                 -STEP * (N // 2), STEP, N,
                                 freqs)
        part = self.model.sample(-STEP * (N // 2), STEP, N // 2,
                                 0, STEP, N // 4,
                                 freqs)
        np.testing.assert_array_equal(part, full[..., 64:96, 0:64])


class TestMeerkatBeamModelSet1:
    def test_load(self):
        """Smoke test to check that the files can be found."""
        for band in primary_beam.MeerkatBeamModelSet1.BANDS:
            models = primary_beam.MeerkatBeamModelSet1(band)
            models.sample()
            assert_equal(models.parameters, [])

    def test_bad_band(self):
        with assert_raises(ValueError):
            primary_beam.MeerkatBeamModelSet1('Z')
