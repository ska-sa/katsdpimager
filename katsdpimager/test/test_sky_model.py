#!/usr/bin/env python3

import tempfile
from unittest import mock

from nose.tools import assert_equal, assert_raises
import fakeredis
from astropy import units
import numpy as np
import katpoint
import katsdptelstate.redis

from katsdpimager.sky_model import (KatpointSkyModel, catalogue_from_telstate, open_sky_model,
                                    NoSkyModelError)


_TRG_A = 'A, radec, 20:00:00.00, -60:00:00.0, (200.0 12000.0 1.0 0.5)'
_TRG_B = 'B, radec, 8:00:00.00, 60:00:00.0, (200.0 12000.0 2.0)'
_TRG_C = 'C, radec, 21:00:00.00, -60:00:00.0, (800.0 43200.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.8 -0.7 0.6)'    # noqa: E501


class TestKatpointSkyModel:
    def setUp(self):
        self.catalogue = katpoint.Catalogue([_TRG_A, _TRG_B, _TRG_C])
        self.sky_model = KatpointSkyModel(self.catalogue)

    def test_len(self):
        assert_equal(len(self.sky_model), 3)

    def test_lmn(self):
        phase_centre = np.array([300, -60]) * units.deg   # RA 20.0
        lmn = self.sky_model.lmn(phase_centre)
        # The LMN coordinates for C were determined experimentally
        np.testing.assert_allclose(lmn, [
            [0, 0, 1],
            [0, 0, -1],
            [1.294095e-01, -1.475455e-02, 9.914815e-01]], atol=1e-5)

    def test_flux_density(self):
        flux = self.sky_model.flux_density(1e10 * units.Hz)
        np.testing.assert_allclose(flux, [
            [1000, 0, 0, 0],
            [100, 0, 0, 0],
            [10, 8, -7, 6]])

        # Check that frequency outside the range of the flux density model returns zeros
        flux = self.sky_model.flux_density(500e6 * units.Hz)
        np.testing.assert_allclose(flux, [
            [223.606798, 0, 0, 0],
            [100, 0, 0, 0],
            [0, 0, 0, 0]])


def _put_models(telstate, cbid, continuum, models):
    """Populate telstate with continuum models.

    Each element of `models` is a tuple containing a phase centre and a list of
    components. The component list may also be ``None`` to omit that part of
    the value (to test failure handling).
    """
    # Uses bytes strings to emulate katsdpcontim (which uses Python 2 native strings)
    targets = {}
    for i, (phase_centre, components) in enumerate(models):
        normalised_name = 'test{}'.format(i)
        d = {b'description': phase_centre.encode('utf-8')}
        if components is not None:
            d[b'components'] = [component.encode('utf-8') for component in components]
        namespace = telstate.join(cbid, continuum, normalised_name, 'target0')
        telstate.view(namespace)['clean_components'] = d
        targets[phase_centre] = normalised_name
    telstate.view(telstate.join(cbid, continuum))['targets'] = targets


class TestCatalogueFromTelstate:
    def setup(self):
        self.cbid = '1234567890'
        self.continuum = 'continuum'
        self.telstate = katsdptelstate.TelescopeState()
        self.telstate['sdp_archived_streams'] = [self.continuum]
        self.telstate[self.telstate.join(self.continuum, 'stream_type')] = 'sdp.continuum_image'
        _put_models(self.telstate, self.cbid, self.continuum, [
            (_TRG_A, [_TRG_A, _TRG_C]),
            (_TRG_B, [_TRG_B]),
            ('Nothing, special', None)
        ])

    def test_present(self):
        catalogue = catalogue_from_telstate(self.telstate, self.cbid, self.continuum,
                                            katpoint.Target(_TRG_A))
        assert_equal(len(catalogue), 2)
        assert_equal(catalogue.targets[0], katpoint.Target(_TRG_A))
        assert_equal(catalogue.targets[1], katpoint.Target(_TRG_C))

        catalogue = catalogue_from_telstate(self.telstate, self.cbid, self.continuum,
                                            katpoint.Target(_TRG_B))
        assert_equal(len(catalogue), 1)
        assert_equal(catalogue.targets[0], katpoint.Target(_TRG_B))

    def test_auto_continuum(self):
        self.continuum = None
        self.test_present()

    def test_absent(self):
        with assert_raises(NoSkyModelError):
            catalogue_from_telstate(self.telstate, self.cbid, self.continuum,
                                    katpoint.Target(_TRG_C))

    def test_missing_key(self):
        with assert_raises(NoSkyModelError):
            catalogue_from_telstate(self.telstate, self.cbid, self.continuum,
                                    katpoint.Target('Nothing, special'))


class TestOpenSkyModel:
    def test_bad_format(self):
        with assert_raises(ValueError):
            open_sky_model('file:///does_not_exist?format=sir_not_appearing_in_this_codebase')

    def test_bad_scheme(self):
        with assert_raises(ValueError):
            open_sky_model('ftp://invalid/')

    def test_missing_params(self):
        with assert_raises(ValueError):
            open_sky_model('redis://invalid/?capture_block_id=1234567890'
                           '&continuum=continuum&format=katdal')

    def test_file(self):
        orig = katpoint.Catalogue([_TRG_A, _TRG_B, _TRG_C])
        with tempfile.NamedTemporaryFile('w', suffix='.csv') as f:
            orig.save(f.name)
            test1 = open_sky_model(f.name)
            test2 = open_sky_model('file://' + f.name + '?format=katpoint')
        assert_equal(orig, test1._catalogue)
        assert_equal(orig, test2._catalogue)

    def test_telstate(self):
        client = fakeredis.FakeRedis()
        telstate = katsdptelstate.TelescopeState(katsdptelstate.redis.RedisBackend(client))
        # Fake just enough of telstate to keep katdal happy. This isn't all in the right
        # namespace, but that doesn't really matter.
        telstate['stream_name'] = 'sdp_l0'
        telstate_l0 = telstate.view('sdp_l0')
        telstate_l0['stream_type'] = 'sdp.vis'
        telstate_l0['chunk_info'] = {
            'correlator_data': {
                'prefix': '1234567890-sdp-l0',
                'dtype': '<c8',
                'shape': (0, 0, 0)
            }
        }
        telstate_l0['sync_time'] = 1234567890.0
        telstate_l0['first_timestamp'] = 0.0
        telstate_l0['int_time'] = 1.0

        _put_models(telstate, '1234567890', 'continuum', [(_TRG_A, [_TRG_A, _TRG_C])])
        expected = katpoint.Catalogue([_TRG_A, _TRG_C])

        with mock.patch('redis.Redis', return_value=client) as mock_redis:
            test = open_sky_model(
                'redis://invalid:6379/?format=katdal&db=1&capture_block_id=1234567890'
                '&continuum=continuum'
                '&target=A,+radec,+20:00:00.00,+-60:00:00.0,+(200.0+12000.0+1.0+0.5+0.0)')
            mock_redis.assert_called_with(host='invalid', port=6379, db=1,
                                          socket_timeout=mock.ANY,
                                          health_check_interval=mock.ANY)
            assert_equal(expected, test._catalogue)
