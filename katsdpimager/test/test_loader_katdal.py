"""Tests for :py:mod:`katsdpimager.loader_katdal`."""

from collections import defaultdict
import logging
import tempfile
from unittest import mock

import astropy.units as u
from astropy.coordinates import Angle
import astropy.table

import numpy as np
import katpoint
import katsdptelstate

import katsdpmodels.rfi_mask
import katsdpmodels.band_mask

import katdal
from katdal import VisibilityDataV4
from katdal.datasources import TelstateDataSource
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.test.test_datasources import make_fake_data_source

from nose.tools import (
    assert_equal, assert_true, assert_false, assert_is_instance, assert_in,
    assert_logs, assert_raises
)

from ..loader_katdal import LoaderKatdal
from .. import polarization


ANTENNAS = [
    katpoint.Antenna('m000, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, -8.264 -207.29 8.5965 212.6695 212.6695 1.0, 0:03:13.3 0 0:00:29.9 0:00:45.1 0:00:08.9 -0:00:12.5 0:13:33.4 0:01:18.0, 1.22'),      # noqa: E501
    katpoint.Antenna('m001, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, 1.1205 -171.762 8.4705 209.996 209.996 1.0, -0:40:31.1 0 0:00:05.8 -0:00:18.3 -0:00:02.8 -0:00:42.2 -0:35:36.2 0:01:13.0, 1.22'),   # noqa: E501
    katpoint.Antenna('m002, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, -32.113 -224.236 8.6445 211.7145 211.7145 1.0, 0:43:31.2 0 0:00:42.4 0:05:00.9 0:00:07.4 -0:00:02.1 0:02:16.3 0:01:01.6, 1.22'),    # noqa: E501
    katpoint.Antenna('m003, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, -66.518 -202.276 8.285 213.4215 213.4215 1.0, 1:35:27.0 0 -0:00:53.3 -0:06:05.9 -0:00:00.5 -0:00:07.1 0:33:51.5 0:00:51.3, 1.22')   # noqa: E501
]
TARGET = katpoint.Target('PKS 1934-63, radec, 19:39:25.03, -63:42:45.7, (200.0 12000.0 -11.11 7.777 -1.231)')   # noqa: E501
CONTINUUM_COMPONENTS = katpoint.Catalogue([
    '3C286, radec, 13:31:08.3, 30:30:33, (400 50e3 0.1823 1.4757 -0.4739 0.0336 0 0 1 0.039 0.087 0)',          # noqa: E501
    '3C123, radec, 04:37:04.3753, 29:40:13.819, (400 50e3 3.1718 -0.1076 -0.1157)'
])


class TestLoaderKatdal:
    def setup(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = NpyFileChunkStore(self.tempdir.name)
        self.shape = (12, 96, len(ANTENNAS) * (len(ANTENNAS) + 1) * 2)
        self.telstate = katsdptelstate.TelescopeState()
        self._populate_telstate(self.telstate)

        self._open_patcher = mock.patch('katdal.open', autospec=True, side_effect=self._katdal_open)
        self._open_patcher.start()

    def _katdal_open(self, filename, **kwargs):
        """Mock implementation of katdal.open."""
        data_source = TelstateDataSource(
            self.view, self.cbid, self.stream_name, chunk_store=self.store, **kwargs)
        return VisibilityDataV4(data_source, **kwargs)

    @staticmethod
    def _populate_telstate(telstate):
        # Put in the minimum necessary for katdal to open the data set. This
        # may need to be updated in future as katdal evolves.
        telstate['sub_pool_resources'] = 'cbf_1,sdp_1,m000,m001,m002,m003'
        telstate['sub_product'] = 'c856M1k'
        telstate['sub_band'] = 'l'
        telstate['obs_params'] = {'observer': 'John Smith'}
        for i, ant in enumerate(ANTENNAS):
            telstate[f'{ant.name}_observer'] = ant.description
            telstate.add(f'{ant.name}_target', TARGET.description, 0.0)
            telstate.add(f'{ant.name}_activity', 'track', 1.0)
            # Point antennas in substantially different directions to test
            # parallactic angles.
            telstate.add(f'{ant.name}_pos_actual_scan_azim', np.deg2rad(i * 30.0))
            telstate.add(f'{ant.name}_pos_actual_scan_elev', np.deg2rad(30.0))
        telstate.add('obs_activity', 'track', 0.0)
        telstate.add('cbf_target', TARGET.description, 0.0)

    def _make_fake_dataset(self, **kwargs):
        chunks = (4, 8, self.shape[2])
        chunk_overrides = {
            'correlator_data': chunks,
            'flags': chunks,
            'weights': chunks,
            'weights_channel': chunks[:2]
        }
        default_kwargs = {
            'l0_shape': self.shape,
            'l0_chunk_overrides': chunk_overrides,
            'l1_flags_chunk_overrides': chunk_overrides
        }
        default_kwargs.update(kwargs)
        self.view, self.cbid, self.stream_name, _, _ = make_fake_data_source(
            self.telstate, self.store, **default_kwargs)

        # Link to RFI mask and band mask
        s_view = self.telstate.view(self.stream_name)
        bcp_view = self.telstate.view('bcp')
        acv_view = self.telstate.view('acv')
        self.telstate['sdp_model_base_url'] = 'http://models.invalid/'
        rfi_mask_key = self.telstate.join('model', 'rfi_mask', 'config')
        self.telstate[rfi_mask_key] = 'rfi_mask/config/unit_test.alias'
        s_view['src_streams'] = ['bcp']
        bcp_view['src_streams'] = ['acv']
        band_mask_key = self.telstate.join('model', 'band_mask', 'config')
        acv_view[band_mask_key] = 'band_mask/config/unit_test.alias'

        # Create a sky model
        sdp_archived_streams = self.telstate['sdp_archived_streams']
        sdp_archived_streams.append('continuum_image')
        self.telstate.delete('sdp_archived_streams')
        self.telstate['sdp_archived_streams'] = sdp_archived_streams
        continuum_view = self.telstate.view('continuum_image')
        continuum_view['stream_type'] = 'sdp.continuum_image'
        cs_view = self.telstate.view(self.telstate.join(self.cbid, 'continuum_image'))
        cs_view['targets'] = {TARGET.description: 'PKS_1934_63'}
        key = self.telstate.join('PKS_1934_63', 'target0', 'clean_components')
        cs_view[key] = {
            'description': TARGET.description,
            'components': [target.description for target in CONTINUUM_COMPONENTS]
        }

    def teardown(self):
        self._open_patcher.stop()
        self.tempdir.cleanup()

    def test_properties(self):
        # Pass a non-zero start channel to check that frequency-related
        # properties are not affected (it is only supposed to be a hint
        # rather than affecting indexing).
        self._make_fake_dataset()
        loader = LoaderKatdal('file:///fake_filename.rdb', {}, 24, None)
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
        assert_equal(loader.raw_target, TARGET)
        # Just a basic smoke test - doesn't test all the extra headers
        headers = loader.extra_fits_headers()
        assert_equal(headers['OBJECT'], 'PKS 1934-63')
        assert_equal(headers['INSTRUME'], 'c856M1k')
        assert_equal(headers['ONTIME'], 24.0)

    def test_sky_model(self):
        self._make_fake_dataset()
        loader = LoaderKatdal('file:///fake_filename.rdb', {}, 24, None)
        sky_model = loader.sky_model()
        assert_equal(sky_model._catalogue, CONTINUUM_COMPONENTS)

    def _collect(self, iterator):
        """Collect all the data from the ``data_iter`` iterator.

        Returns
        -------
        vis, weights, uvw, feed_angle1, feed_angle2 : np.ndarray
            Arrays of data
        chunks : Tuple[int]
            Number of rows in each loaded chunk
        """
        progress = 0
        chunks = []
        bls = len(ANTENNAS) * (len(ANTENNAS) - 1) // 2    # Excludes auto-correlations
        data = defaultdict(list)
        for chunk in iterator:
            assert_equal(chunk['total'], self.shape[0])
            assert_equal(chunk['progress'], progress + chunk['uvw'].shape[0] // bls)
            progress = chunk['progress']
            for field in ['uvw', 'weights', 'vis', 'feed_angle1', 'feed_angle2']:
                data[field].append(chunk[field])
            N = chunk['vis'].shape[1]
            # Check that nothing got accidentally upgraded to double precision
            assert_equal(chunk['vis'].dtype, np.complex64)
            assert_equal(chunk['weights'].dtype, np.float32)
            assert_equal(chunk['uvw'].dtype, np.float32)
            assert_equal(chunk['feed_angle1'].dtype, np.float32)
            assert_equal(chunk['feed_angle2'].dtype, np.float32)
            # Check that each array has the same number of rows
            assert_equal(chunk['uvw'].shape, (N, 3))
            assert_equal(chunk['weights'].shape, chunk['vis'].shape)
            assert_equal(chunk['feed_angle1'].shape, (N,))
            assert_equal(chunk['feed_angle2'].shape, (N,))
            chunks.append(N)
        assert_equal(progress, self.shape[0])

        # Concatenate the chunks back together
        vis = np.concatenate(data['vis'], axis=1)
        weights = np.concatenate(data['weights'], axis=1)
        uvw = np.concatenate(data['uvw'], axis=0)
        feed_angle1 = np.concatenate(data['feed_angle1'], axis=0)
        feed_angle2 = np.concatenate(data['feed_angle2'], axis=0)
        return vis, weights, uvw, feed_angle1, feed_angle2, tuple(chunks)

    def _get_expected(self, dataset, channel_mask=None):
        """Get values from a dataset.

        It must already have had the appropriate selection applied.
        """
        vis = dataset.vis[:]
        flags = dataset.flags[:]
        if channel_mask is not None:
            flags |= channel_mask[np.newaxis, :, np.newaxis]
        weights = dataset.weights[:] * np.logical_not(flags)
        # negate because katdal convention is ant2 - ant1, while katsdpimager
        # convention is the opposite.
        uvw = -np.stack([dataset.u, dataset.v, dataset.w], axis=2) * u.m
        pol_map = {
            'hh': polarization.STOKES_XX,
            'hv': polarization.STOKES_XY,
            'vh': polarization.STOKES_YX,
            'vv': polarization.STOKES_YY
        }
        # See comment in loader_katdal.py for explanation of pi/2 shift
        pa = np.deg2rad(dataset.parangle) - np.pi / 2
        pa = {ant.name: pa[:, i] for i, ant in enumerate(dataset.ants)}
        feed_angle1 = np.stack(
            [pa[prod[0][:-1]] for prod in dataset.corr_products],
            axis=1
        )
        feed_angle2 = np.stack(
            [pa[prod[1][:-1]] for prod in dataset.corr_products],
            axis=1
        )
        # Expected polarisation product per baseline
        polarizations = [
            pol_map[prod[0][-1:] + prod[1][-1:]]
            for prod in dataset.corr_products
        ]
        return vis, weights, uvw, feed_angle1, feed_angle2, polarizations

    def _test_data(self, *, options=(), channel0=36, channel1=72, max_chunk_vis=None,
                   missing=set(), channel_mask=None):
        loader = LoaderKatdal('file:///fake_filename.rdb', options, 24, None)
        bls = len(ANTENNAS) * (len(ANTENNAS) - 1) // 2    # Excludes auto-correlations
        vis, weights, uvw, feed_angle1, feed_angle2, chunks = \
            self._collect(loader.data_iter(channel0, channel1, max_chunk_vis=max_chunk_vis))

        dataset = katdal.open('file:///fake_filename.rdb')
        dataset.select(corrprods='cross', channels=np.s_[36:72], scans='track')
        if channel_mask is not None:
            channel_mask = channel_mask[channel0:channel1]
        e_vis, e_weights, e_uvw, e_feed_angle1, e_feed_angle2, e_polarizations = \
            self._get_expected(dataset, channel_mask)

        # Check the shapes before trying to shuffle data.
        N = self.shape[0] * bls
        C = e_vis.shape[1]
        P = 4
        assert_equal(vis.shape, (C, N, P))
        assert_equal(weights.shape, (C, N, P))
        assert_equal(uvw.shape, (N, 3))
        assert_equal(feed_angle1.shape, (N,))
        assert_equal(feed_angle2.shape, (N,))

        # The expected visibilities are all unique, so we can use them to
        # determine the permutation applied. We map each complex visibility to
        # its time-frequency-baseline coordinates.
        vis_map = {value: index for index, value in np.ndenumerate(e_vis)}
        for channel in range(C):
            for i in range(N):
                # Handling for missing correlation products assumes that the
                # first polarization product of a baseline will always be
                # present.
                baseline = dataset.corr_products[vis_map[vis[channel, i, 0]][2]]
                ant1 = baseline[0][:-1]
                ant2 = baseline[1][:-1]
                for j in range(P):
                    key = (ant1, ant2, loader.polarizations()[j])
                    if key in missing:
                        # Doesn't matter what the values are; it just has to be flagged
                        assert_equal(weights[channel, i, j], 0.0)
                        continue
                    # pop ensures that we don't see the same visibility twice
                    idx = vis_map.pop(vis[channel, i, j])
                    assert_equal(idx[1], channel)
                    assert_equal(loader.polarizations()[j], e_polarizations[idx[2]])
                    assert_equal(vis[channel, i, j], e_vis[idx])
                    assert_equal(weights[channel, i, j], e_weights[idx])
                    np.testing.assert_allclose(
                        uvw[i].to_value(u.m),
                        e_uvw[idx[0], idx[2]].to_value(u.m), rtol=1e-7
                    )
                    # Won't match exactly because the loader converts to float32
                    np.testing.assert_allclose(
                        feed_angle1[i], e_feed_angle1[idx[0], idx[2]], rtol=1e-6)
                    np.testing.assert_allclose(
                        feed_angle2[i], e_feed_angle2[idx[0], idx[2]], rtol=1e-6)
        assert_equal(len(vis_map), 0)
        loader.close()
        return chunks

    def test_basic(self):
        """Basic smoke test - can load everything in one chunk."""
        self._make_fake_dataset()
        self._test_data()

    def test_chunked(self):
        """Load via multiple chunks."""
        self._make_fake_dataset()
        bls = len(ANTENNAS) * (len(ANTENNAS) - 1) // 2    # Excludes auto-correlations
        chunks = self._test_data(max_chunk_vis=7 * 36 * 4 * bls)
        # It should round down to 4 times per chunk to align with data chunking
        assert_equal(chunks, (4 * bls, 4 * bls, 4 * bls))

    def test_max_chunk_vis_small(self):
        self._make_fake_dataset()
        with assert_logs(logger='katsdpimager.loader_katdal', level=logging.WARNING) as cm:
            self._test_data(max_chunk_vis=1)
        assert_in('Chunk size is 4 dumps but only 1 loaded at a time', cm.records[0].message)

    def test_missing_corr_product(self):
        corr_products = []
        # Apart from deleting one, also use a weird ordering to test the permutation
        for pol1 in 'hv':
            for pol2 in 'vh':
                for i in range(len(ANTENNAS)):
                    for j in range(i + 1):
                        if (i, j, pol1, pol2) != (2, 0, 'h', 'v'):
                            corr_products.append((f'{ANTENNAS[i].name}{pol1}',
                                                  f'{ANTENNAS[j].name}{pol2}'))
        assert_equal(len(corr_products), self.shape[2] - 1)
        self.shape = (self.shape[0], self.shape[1], len(corr_products))
        self._make_fake_dataset(bls_ordering_override=corr_products)
        self._test_data(missing={('m002', 'm000', polarization.STOKES_XY)})

    def _get_model(self, url, model_class, *, lazy=False):
        """Mock implementation of Fetcher.get."""
        if url == 'http://models.invalid/rfi_mask/config/unit_test.alias':
            ranges = astropy.table.Table(
                [[1200e6], [1300e6], [1000.0]],
                names=('min_frequency', 'max_frequency', 'max_baseline'),
                dtype=(np.float64, np.float64, np.float64)
            )
            ranges['min_frequency'].unit = u.Hz
            ranges['max_frequency'].unit = u.Hz
            ranges['max_baseline'].unit = u.m
            return katsdpmodels.rfi_mask.RFIMaskRanges(ranges, False)
        elif url == 'http://models.invalid/band_mask/config/unit_test.alias':
            ranges = astropy.table.Table(
                [[0.624], [0.626]],
                names=('min_fraction', 'max_fraction'),
                dtype=(np.float64, np.float64)
            )
            return katsdpmodels.band_mask.BandMaskRanges(ranges)
        else:
            raise RuntimeError(f'Unexpected URL {url}')

    @mock.patch('katsdpmodels.fetch.requests.Fetcher', autospec=True)
    def test_rfi_band_mask(self, fetcher_mock):
        fetcher_mock.return_value.get.side_effect = self._get_model
        self._make_fake_dataset()
        mask = np.zeros(self.shape[1], np.bool_)
        mask[39:51] = True      # RFI mask
        mask[60] = True         # Bank mask
        self._test_data(options=['--rfi-mask=config'], channel_mask=mask)

    def test_match(self):
        assert_false(LoaderKatdal.match('foo.h5'))
        assert_true(LoaderKatdal.match('foo.rdb'))
        assert_false(LoaderKatdal.match('foo.ms'))
        assert_true(LoaderKatdal.match('http://example.invalid/foo/bar.rdb?query=params#fragment'))
        assert_false(LoaderKatdal.match('http://example.invalid/foo/bar.ms?query=params#fragment'))
        assert_false(LoaderKatdal.match('http://[1.2.3/unmatch-bracket.rdb?query=params'))

    def test_missing_credentials(self):
        self._make_fake_dataset()
        with assert_raises(ValueError):
            self._test_data(options=['--access-key=foo'])
        with assert_raises(ValueError):
            self._test_data(options=['--secret-key=foo'])

    def test_credentials(self):
        self._make_fake_dataset()
        options = ['--access-key=foo', '--secret-key=bar']
        loader = LoaderKatdal('file:///fake_filename.rdb', options, 24, None)
        assert_equal(katdal.open.mock_calls[0][2].get('credentials'), ('foo', 'bar'))
        assert_equal(loader.command_line_options(), [], 'Credentials not redacted')
