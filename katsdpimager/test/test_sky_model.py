#!/usr/bin/env python3

import tempfile

from nose.tools import assert_equal, assert_raises
import mock
import fakeredis
from astropy import units
import numpy as np
import katpoint
import katsdptelstate.redis

from katsdpimager.sky_model import KatpointSkyModel, catalogue_from_telstate, open_sky_model


_TRG_A = 'A, radec, 20:00:00.00, -60:00:00.0, (200.0 12000.0 1.0 0.5 0.0)'
_TRG_B = 'B, radec, 8:00:00.00, 60:00:00, (200.0 12000.0 2.0 0.0 0.0)'
_TRG_C = 'C, radec, 21:00:00.00, -60:00:00, (800.0 43200.0 1 0 0 0 0 0  1 0.8 -0.7 0.6)'


class TestKatpointSkyModel(object):
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


def _put_target(view, idx, description, targets):
    # Uses bytes strings to emulate katsdpcontim (which uses Python 2 native strings)
    d = {b'description': description.encode('utf-8')}
    if targets is not None:
        d[b'components'] = [target.encode('utf-8') for target in targets]
    view['target{}_clean_components'.format(idx)] = d


class TestCatalogueFromTelstate(object):
    def setUp(self):
        self.cbid = '1234567890'
        self.continuum = 'continuum'
        self.telstate = katsdptelstate.TelescopeState()
        view = self.telstate.view('1234567890_continuum')
        _put_target(view, 0, _TRG_A, [_TRG_A, _TRG_C])
        _put_target(view, 1, _TRG_B, [_TRG_B])
        _put_target(view, 2, 'Nothing, special', None)

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

    def test_absent(self):
        catalogue = catalogue_from_telstate(self.telstate, self.cbid, self.continuum,
                                            katpoint.Target(_TRG_C))
        assert_equal(len(catalogue), 0)

    def test_missing_key(self):
        catalogue = catalogue_from_telstate(self.telstate, self.cbid, self.continuum,
                                            katpoint.Target('Nothing, special'))
        assert_equal(len(catalogue), 0)


class TestOpenSkyModel(object):
    def test_bad_format(self):
        with assert_raises(ValueError):
            open_sky_model('file:///does_not_exist?format=sir_not_appearing_in_this_codebase')

    def test_bad_scheme(self):
        with assert_raises(ValueError):
            open_sky_model('ftp://invalid/')

    def test_missing_params(self):
        with assert_raises(ValueError):
            open_sky_model('redis://invalid/?capture_block_id=1234567890&continuum=continuum')

    def test_file(self):
        orig = katpoint.Catalogue([_TRG_A, _TRG_B, _TRG_C])
        with tempfile.NamedTemporaryFile('w', suffix='.csv') as f:
            orig.save(f.name)
            test1 = open_sky_model(f.name)
            test2 = open_sky_model('file://' + f.name + '?format=katpoint')
        assert_equal(orig, test1._catalogue)
        assert_equal(orig, test2._catalogue)

    def test_telstate(self):
        client = fakeredis.FakeStrictRedis()
        telstate = katsdptelstate.TelescopeState(katsdptelstate.redis.RedisBackend(client))
        view = telstate.view('1234567890_continuum')
        _put_target(view, 0, _TRG_A, [_TRG_A, _TRG_C])
        expected = katpoint.Catalogue([_TRG_A, _TRG_C])

        with mock.patch('redis.StrictRedis.from_url', return_value=client) as from_url:
            test = open_sky_model(
                'redis://invalid:6379/?format=katpoint&db=1&capture_block_id=1234567890'
                '&continuum=continuum'
                '&target=A,+radec,+20:00:00.00,+-60:00:00.0,+(200.0+12000.0+1.0+0.5+0.0)')
            from_url.assert_called_with('redis://invalid:6379/?db=1')
            assert_equal(expected, test._catalogue)
