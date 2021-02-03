"""Tests for :py:mod:`katsdpimager.loader_katdal`."""

from collections import defaultdict
import tempfile
from unittest import mock

import astropy.units as u
from astropy.coordinates import Angle
import numpy as np
import katpoint
import katsdptelstate

import katdal
from katdal import VisibilityDataV4
from katdal.datasources import TelstateDataSource
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.test.test_datasources import make_fake_data_source

from nose.tools import assert_equal, assert_true, assert_is_instance

from ..loader_katdal import LoaderKatdal
from .. import polarization


ANTENNAS = [
    katpoint.Antenna('m000, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, -8.264 -207.29 8.5965 212.6695 212.6695 1.0, 0:03:13.3 0 0:00:29.9 0:00:45.1 0:00:08.9 -0:00:12.5 0:13:33.4 0:01:18.0, 1.22'),      # noqa: E501
    katpoint.Antenna('m001, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, 1.1205 -171.762 8.4705 209.996 209.996 1.0, -0:40:31.1 0 0:00:05.8 -0:00:18.3 -0:00:02.8 -0:00:42.2 -0:35:36.2 0:01:13.0, 1.22'),   # noqa: E501
    katpoint.Antenna('m002, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, -32.113 -224.236 8.6445 211.7145 211.7145 1.0, 0:43:31.2 0 0:00:42.4 0:05:00.9 0:00:07.4 -0:00:02.1 0:02:16.3 0:01:01.6, 1.22'),    # noqa: E501
    katpoint.Antenna('m003, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, -66.518 -202.276 8.285 213.4215 213.4215 1.0, 1:35:27.0 0 -0:00:53.3 -0:06:05.9 -0:00:00.5 -0:00:07.1 0:33:51.5 0:00:51.3, 1.22')   # noqa: E501
]
TARGET = katpoint.Target('PKS 1934-63, radec, 19:39:25.03, -63:42:45.7, (200.0 12000.0 -11.11 7.777 -1.231)')   # noqa: E501


class TestLoaderKatdal:
    def setup(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = NpyFileChunkStore(self.tempdir.name)
        telstate = katsdptelstate.TelescopeState()
        self.shape = (12, 96, len(ANTENNAS) * (len(ANTENNAS) + 1) * 2)
        self.view, self.cbid, self.stream_name, _, _ = make_fake_data_source(
            telstate, self.store,
            l0_shape=self.shape
        )
        # Put in the minimum necessary for katdal to open the data set. This
        # may need to be updated in future as katdal evolves.
        telstate['sub_pool_resources'] = 'cbf_1,sdp_1,m000,m001,m002,m003'
        telstate['sub_product'] = 'c856M1k'
        telstate['sub_band'] = 'l'
        telstate['obs_params'] = {}
        for ant in ANTENNAS:
            telstate[f'{ant.name}_observer'] = ant.description
            telstate.add(f'{ant.name}_target', TARGET.description, 123456789.0)
            telstate.add(f'{ant.name}_activity', 'track', 1.0)
            telstate.add(f'{ant.name}_pos_actual_scan_azim', 0.0)
            telstate.add(f'{ant.name}_pos_actual_scan_elev', np.deg2rad(30.0))
        telstate.add('obs_activity', 'track', 123456789.0)
        telstate.add('cbf_target', TARGET.description, 123456789.0)

        self._open_patcher = mock.patch('katdal.open', autospec=True, side_effect=self._katdal_open)
        self._open_patcher.start()

    def _katdal_open(self, filename, **kwargs):
        """Mock implementation of katdal.open."""
        data_source = TelstateDataSource(
            self.view, self.cbid, self.stream_name, chunk_store=self.store, **kwargs)
        return VisibilityDataV4(data_source, **kwargs)

    def teardown(self):
        self._open_patcher.stop()
        self.tempdir.cleanup()

    def test_properties(self):
        # Pass a non-zero start channel to check that frequency-related
        # properties are not affected (it is only supposed to be a hint
        # rather than affecting indexing).
        loader = LoaderKatdal('file:///fake_filename', {}, 24, None)
        np.testing.assert_array_equal(
            loader.antenna_diameters().to_value(u.m), [13.5] * len(ANTENNAS))
        assert_equal(loader.num_channels(), self.shape[1])
        assert_equal(loader.frequency(48), 1284 * u.MHz)
        assert_equal(loader.frequency(72), 1498 * u.MHz)
        assert_equal(loader.band(), 'L')
        assert_equal(loader.phase_centre()[0], Angle('19:39:25.03 hours').to(u.rad))
        assert_equal(loader.phase_centre()[1], Angle('-63:42:45.7 degrees').to(u.rad))
        assert_true(loader.has_feed_angles())
        assert_equal(
            loader.polarizations(),
            [
                polarization.STOKES_XX,
                polarization.STOKES_XY,
                polarization.STOKES_YX,
                polarization.STOKES_YY
            ]
        )
        assert_equal(loader.weight_scale(), np.sqrt(0.5))
        for i in range(self.shape[1]):
            assert_true(loader.channel_enabled(i))
        np.testing.assert_allclose(loader.longest_baseline(), 74.2 * u.m, rtol=0.001)
        assert_is_instance(loader.raw_data, VisibilityDataV4)

    def test_sky_model(self):
        pass  # TODO

    def test_extra_fits_headers(self):
        pass  # TODO

    def test_data(self):
        loader = LoaderKatdal('file:///fake_filename', {}, 24, None)
        progress = 0
        bls = len(ANTENNAS) * (len(ANTENNAS) - 1) // 2    # Excludes auto-correlations
        # Collect all the data together
        data = defaultdict(list)
        for chunk in loader.data_iter(36, 72):
            assert_equal(chunk['total'], self.shape[0])
            assert_equal(chunk['progress'], progress + chunk['uvw'].shape[0] // bls)
            progress = chunk['progress']
            for field in ['uvw', 'weights', 'vis', 'feed_angle1', 'feed_angle2']:
                data[field].append(chunk[field])
        assert_equal(progress, self.shape[0])

        # Concatenate the chunks back together
        vis = np.concatenate(data['vis'], axis=1)
        weights = np.concatenate(data['weights'], axis=1)
        uvw = np.concatenate(data['uvw'], axis=0)
        feed_angle1 = np.concatenate(data['feed_angle1'], axis=0)
        feed_angle2 = np.concatenate(data['feed_angle2'], axis=0)

        dataset = katdal.open('file:///fake_filename')
        dataset.select(corrprods='cross', channels=range(36, 72), scans='track')
        expected_vis = dataset.vis[:]
        expected_weights = dataset.weights[:] * np.logical_not(dataset.flags[:])
        # negate because katdal convention is ant2 - ant1, while katsdpimager
        # convention is the opposite.
        expected_uvw = -np.stack([dataset.u, dataset.v, dataset.w], axis=2) * u.m
        pol_map = {
            'hh': polarization.STOKES_XX,
            'hv': polarization.STOKES_XY,
            'vh': polarization.STOKES_YX,
            'vv': polarization.STOKES_YY
        }
        # Expected polarisation product per baseline
        expected_polarizations = [
            pol_map[prod[0][-1:] + prod[1][-1:]]
            for prod in dataset.corr_products
        ]

        # Check the shapes before trying to shuffle data.
        N = self.shape[0] * bls
        C = expected_vis.shape[1]
        P = 4
        assert_equal(vis.shape, (C, N, P))
        assert_equal(weights.shape, (C, N, P))
        assert_equal(uvw.shape, (N, 3))
        assert_equal(feed_angle1.shape, (N,))
        assert_equal(feed_angle2.shape, (N,))

        # The expected visibilities are all unique, so we can use them to
        # determine the permutation applied. We map each complex visibility to
        # its time-frequency-baseline coordinates.
        vis_map = {value: index for index, value in np.ndenumerate(expected_vis)}
        for channel in range(C):
            for i in range(N):
                for j in range(P):
                    # pop ensures that we don't see the same visibility twice
                    idx = vis_map.pop(vis[channel, i, j])
                    assert_equal(idx[1], channel)
                    assert_equal(loader.polarizations()[j], expected_polarizations[idx[2]])
                    assert_equal(vis[channel, i, j], expected_vis[idx])
                    assert_equal(weights[channel, i, j], expected_weights[idx])
                    np.testing.assert_allclose(
                        uvw[i].to_value(u.m),
                        expected_uvw[idx[0], idx[2]].to_value(u.m), rtol=1e-7
                    )

        # TODO: check dtypes
        # TODO: check feed angles

    # TODO: channel mask
    # TODO: missing correlation product
    # TODO: ensure that this is a non-trivial baseline permutation
    # TODO: multiple chunks
    # TODO: chunk alignment
    # TODO: n_times less than chunk alignment (round up and warn)
