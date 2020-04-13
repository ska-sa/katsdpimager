"""Data loading backend for CASA Measurement Sets."""

import argparse
import logging
from contextlib import closing

import casacore.tables
import casacore.quanta
import numpy as np
from astropy import units
import astropy.time
import astropy.coordinates
import astropy.io.fits

import katsdpimager.loader_core


_logger = logging.getLogger(__name__)


# Map casacore MFrequency::Types enum (used in MEAS_FREQ_REF column) to FITS reference frames
_MEAS_FREQ_REF_MAP = {
    0: 'SOURCE',        # REST in casacore
    1: 'LSRK',
    2: 'LSRD',
    3: 'BARYCENT',
    4: 'GEOCENTR',
    5: 'TOPOCENT',
    6: 'GALACTOC',
    7: 'LOCALGRP',
    8: 'CMBDIPOL'
}


def _get(table, name, data, casacore_units=None, astropy_units=None,
         measinfo_type=None, measinfo_ref=None):
    """Apply scaling to data loaded from a column. Scaling is done manually
    rather than with :mod:`casacore.quanta`, because the latter only operates
    (slowly) on scalars while a direct scaling is vectorised.

    Because not all writers specify the units for all quantities, units are
    allowed to be missing.

    Parameters
    ----------
    table : `casacore.tables.table`
        Table from which to load
    name : str
        Column name
    data : array-like
        Raw data loaded from the column. The leading dimension must index
        rows.
    casacore_units : str, optional
        Expected unit. A column with compatible units will be converted.
        Use ``None`` for unitless quantities.
    astropy_units : :class:`astropy.units.Unit`, optional
        Astropy unit equivalent to `casacore_units`, used for the return
        value. Use ``None`` for unitless quantities (the return value will
        then be a plain numpy array).
    measinfo_type : str, optional
        If specified, requires the ``MEASINFO`` field keyword to have this
        type.
    measinfo_ref : str, optional
        If specified, requires the ``MEASINFO`` field keyword to have this
        reference.
    """
    keywords = table.getcolkeywords(name)
    quantum_units = keywords.get('QuantumUnits')
    if quantum_units is not None:
        if casacore_units is None:
            raise ValueError('Found unexpected QuantumUnits for column {}'.format(name))
        quantum_units = np.array(quantum_units)
        if len(data.shape) == 1:
            # Scalar column. Convert quantum_units from a 1D array of 1 element
            # to a numpy scalar
            if quantum_units.shape != (1,):
                raise ValueError('Column {} is scalar, but QuantumUnits has shape {}'
                                 .format(name, quantum_units.shape))
            quantum_units = quantum_units[0]
        # data[0] is included in the iteration so that broadcasting rules are
        # triggered if quantum_units has a smaller shape (e.g. one unit for both
        # receptors on a feed).
        it = np.nditer([quantum_units, data[0]], flags=['multi_index'])
        output_quantity = casacore.quanta.quantity(1, casacore_units)
        while not it.finished:
            input_quantity = casacore.quanta.quantity(1, it[0][()])
            if not input_quantity.conforms(output_quantity):
                raise ValueError('Expected {} in column {} but found {} instead'.format(
                                 casacore_units, name, input_quantity.get_unit()))
            ratio = input_quantity.get_value(casacore_units)
            if ratio != 1:
                # Special-case ratio=1 because it is the common case, and the scaling
                # is potentially expensive.
                data[np.index_exp[:] + it.multi_index] *= ratio
            it.iternext()
    if astropy_units is not None:
        data = units.Quantity(data, astropy_units, copy=False)
    measinfo = keywords.get('MEASINFO')
    if measinfo is not None:
        if (measinfo_ref is not None and measinfo['Ref'] != measinfo_ref) or \
                (measinfo_type is not None and measinfo['type'] != measinfo_type):
            raise ValueError('Unsupported MEASINFO for {}: {}'.format(name, measinfo))
    return data


def _getcol(table, name, start=0, count=-1,
            casacore_units=None, astropy_units=None,
            measinfo_type=None, measinfo_ref=None):
    """Wrap the getcol method to fetch a batch of data from a column, applying
    scaling to the expected units. See :func:`_get` for details.

    Parameters
    ----------
    table : `casacore.tables.table`
        Table from which to load
    name : str
        Column name
    start : int
        First row to load
    count : int
        Number of rows to load
    casacore_units : str, optional
        Expected unit. A column with compatible units will be converted.
        Use ``None`` for unitless quantities.
    astropy_units : :class:`astropy.units.Unit`, optional
        Astropy unit equivalent to `casacore_units`, used for the return
        value. Use ``None`` for unitless quantities (the return value will
        then be a plain numpy array).
    measinfo_type : str, optional
        If specified, requires the ``MEASINFO`` field keyword to have this
        type.
    measinfo_ref : str, optional
        If specified, requires the ``MEASINFO`` field keyword to have this
        reference.
    """
    data = table.getcol(name, start, count)
    return _get(table, name, data, casacore_units, astropy_units, measinfo_type, measinfo_ref)


def _getcolslice(table, name, blc, trc, inc=[], start=0, count=-1,
                 casacore_units=None, astropy_units=None,
                 measinfo_type=None, measinfo_ref=None):
    """Wrap the getcol method to fetch a batch of data from a column with
    slicing, applying scaling to the expected units. See :func:`_get` for
    details.

    Parameters
    ----------
    table : `casacore.tables.table`
        Table from which to load
    name : str
        Column name
    blc : list
        Bottom left corner for array slice
    trc : list
        Top right corner for array slice (inclusive)
    ind : list
        Strides
    start : int
        First row to load
    count : int
        Number of rows to load
    casacore_units : str, optional
        Expected unit. A column with compatible units will be converted.
        Use ``None`` for unitless quantities.
    astropy_units : :class:`astropy.units.Unit`, optional
        Astropy unit equivalent to `casacore_units`, used for the return
        value. Use ``None`` for unitless quantities (the return value will
        then be a plain numpy array).
    measinfo_type : str, optional
        If specified, requires the ``MEASINFO`` field keyword to have this
        type.
    measinfo_ref : str, optional
        If specified, requires the ``MEASINFO`` field keyword to have this
        reference.
    """
    data = table.getcolslice(name, blc, trc, inc, start, count)
    return _get(table, name, data, casacore_units, astropy_units, measinfo_type, measinfo_ref)


def _getcolchannels(table, name, start_channel, stop_channel, start=0, count=-1,
                    casacore_units=None, astropy_units=None,
                    measinfo_type=None, measinfo_ref=None):
    """Convenience wrapper around :func:`_getcolslice` for columns where the
    shape is channels by polarizations, and a range of channels is desired.
    """
    return _getcolslice(table, name, [start_channel, -1], [stop_channel - 1, -1], [],
                        start, count, casacore_units, astropy_units,
                        measinfo_type, measinfo_ref)


def _getcell(table, name, row,
             casacore_units=None, astropy_units=None,
             measinfo_type=None, measinfo_ref=None):
    """Like :meth:`_getcol`, but for a single cell"""
    data = table.getcell(name, row)
    # Temporary add a leading dimension, required by _get
    data = data[np.newaxis, :]
    data = _getcol(table, name, row, 1, casacore_units, astropy_units,
                   measinfo_type, measinfo_ref)
    out = _get(table, name, data, casacore_units, astropy_units, measinfo_type, measinfo_ref)
    return out[0]


def _fix_cache_size(table, column):
    """Workaround for https://github.com/casacore/casacore/issues/581.
    We set the column cache size to be large enough for two tiles worth of
    rows.
    """
    cache_size = 0
    try:
        for item in table.getdminfo(column)['SPEC']['HYPERCUBES'].values():
            cube_shape = tuple(item['CubeShape'])
            tile_shape = tuple(item['TileShape'])
            cache_shape = cube_shape[:2] + tile_shape[2:]
            cache_size = max(cache_size, 2 * int(np.product(cache_shape)))
    except (TypeError, KeyError, IndexError, ValueError):
        pass
    if cache_size == 0:
        # No tile size info found - pick some default
        _logger.warning('Could not get tile info for column %s', column)
        cache_size = 1024 * 1024
    _logger.debug('Using cache size %d for column %s', cache_size, column)
    table.setmaxcachesize(column, cache_size)


class LoaderMS(katsdpimager.loader_core.LoaderBase):
    def __init__(self, filename, options):
        super().__init__(filename, options)
        parser = argparse.ArgumentParser(
            prog='Measurement set options',
            usage='Measurement set options: [-i data=COLUMN] [-i field=FIELD] ...')
        parser.add_argument('--data', type=str, metavar='COLUMN', default='DATA',
                            help='Column containing visibilities to image [%(default)s]')
        parser.add_argument('--data-desc', type=int, default=0,
                            help='Data description ID to image [%(default)s]')
        parser.add_argument('--field', type=int, default=0,
                            help='Field to image [%(default)s]')
        parser.add_argument('--pol-frame', choices=['sky', 'feed'], default='sky',
                            help='Reference frame for polarization [%(default)s]')
        parser.add_argument('--uvw', choices=['casa', 'strict'], default='casa',
                            help='UVW sign convention [%(default)s]')
        args = parser.parse_args(options)
        self._strict_uvw = (args.uvw == 'strict')
        self._main = casacore.tables.table(filename, ack=False)
        _fix_cache_size(self._main, 'FLAG')
        self._antenna = casacore.tables.table(self._main.getkeyword('ANTENNA'), ack=False)
        self._data_description = casacore.tables.table(self._main.getkeyword('DATA_DESCRIPTION'),
                                                       ack=False)
        self._field = casacore.tables.table(self._main.getkeyword('FIELD'), ack=False)
        self._spectral_window = casacore.tables.table(self._main.getkeyword('SPECTRAL_WINDOW'),
                                                      ack=False)
        self._polarization = casacore.tables.table(self._main.getkeyword('POLARIZATION'), ack=False)
        self._feed = casacore.tables.table(self._main.getkeyword('FEED'), ack=False)
        self._data_col = args.data
        self._field_id = args.field
        self._data_desc_id = args.data_desc
        if self._data_col not in self._main.colnames():
            raise ValueError('{} has no column named {}'.format(
                filename, self._data_col))
        if args.field < 0 or args.field >= self._field.nrows():
            raise ValueError('Field {} is out of range'.format(args.field))
        if args.data_desc < 0 or args.data_desc >= self._data_description.nrows():
            raise ValueError('Data description {} is out of range'.format(args.data_desc))
        self._polarization_id = self._data_description.getcell(
            'POLARIZATION_ID', args.data_desc)
        self._spectral_window_id = self._data_description.getcell(
            'SPECTRAL_WINDOW_ID', args.data_desc)
        # Detect whether we have a usable WEIGHT_SPECTRUM column. Measurement
        # sets in the wild sometimes have the column but with 0x0 shape.
        try:
            self._main.getcellslice('WEIGHT_SPECTRUM', 0, [0, 0], [0, 0])
        except RuntimeError:
            self._has_weight_spectrum = False
        else:
            self._has_weight_spectrum = True

        self._feed_angle_correction = args.pol_frame == 'feed'
        if self._feed_angle_correction:
            # Load per-antenna feed angles. For now, only a constant value per
            # antenna is supported, rather than per feed or per receptor within a
            # feed. This avoids the need for more per-visibility indexing.
            antenna_id = _getcol(self._feed, 'ANTENNA_ID')
            receptor_angle = _getcol(self._feed, 'RECEPTOR_ANGLE', 0, -1, 'rad', units.rad)
            antenna_angle = [None] * (np.max(antenna_id) + 1)
            for i in range(len(antenna_id)):
                for angle in receptor_angle[i]:
                    if (antenna_angle[antenna_id[i]] is not None
                            and not np.allclose(antenna_angle[antenna_id[i]],
                                                angle, atol=1e-8 * units.rad)):
                        raise ValueError('Multiple feed angles for one antenna is not supported')
                    antenna_angle[antenna_id[i]] = angle
            self._antenna_angle = units.Quantity(antenna_angle)
        else:
            self._antenna_angle = None
        self._average_time = None     # Will be set by data_iter
        self._observation_ids = set()

    @classmethod
    def match(cls, filename):
        return filename.lower().endswith('.ms')

    def antenna_diameters(self):
        return _getcol(self._antenna, 'DISH_DIAMETER', 0, -1, 'm', units.m)

    def antenna_positions(self):
        return _getcol(self._antenna, 'POSITION', 0, -1, 'm', units.m, 'position', 'ITRF')

    def phase_centre(self):
        value = _getcell(self._field, 'PHASE_DIR', self._field_id,
                         'rad', units.rad, 'direction', 'J2000')
        if tuple(value.shape) != (1, 2):
            raise ValueError('Unsupported shape for PHASE_DIR: {}'.format(value.shape))
        return value[0, :]

    def num_channels(self):
        return len(_getcell(self._spectral_window, 'CHAN_FREQ', self._spectral_window_id,
                            'Hz', units.Hz))

    def frequency(self, channel):
        return _getcell(self._spectral_window, 'CHAN_FREQ', self._spectral_window_id,
                        'Hz', units.Hz)[channel]

    def band(self):
        name = self._spectral_window.getcell('NAME', self._spectral_window_id)
        if not name or name.lower() == 'none':
            return None
        return name

    def polarizations(self):
        return _getcell(self._polarization, 'CORR_TYPE', self._polarization_id)

    def has_feed_angles(self):
        return self._feed_angle_correction

    def extra_fits_headers(self):
        headers = astropy.io.fits.Header()
        obsgeo = np.mean(self.antenna_positions().to_value(units.m), axis=0)
        obsgeo_comment = 'Average of antenna positions'
        headers['OBSGEO-X'] = (obsgeo[0], obsgeo_comment)
        headers['OBSGEO-Y'] = (obsgeo[1], obsgeo_comment)
        headers['OBSGEO-Z'] = (obsgeo[2], obsgeo_comment)

        if self._average_time is not None:
            headers['DATE-AVG'] = self._average_time.utc.isot

        meas_freq_ref = self._spectral_window.getcell('MEAS_FREQ_REF', self._spectral_window_id)
        if meas_freq_ref in _MEAS_FREQ_REF_MAP:
            headers['SPECSYS'] = _MEAS_FREQ_REF_MAP[meas_freq_ref]

        if len(self._observation_ids) == 1:
            row = list(self._observation_ids)[0]
            table = casacore.tables.table(self._main.getkeyword('OBSERVATION'), ack=False)
            with closing(table):
                time_range = _getcell(table, 'TIME_RANGE', row,
                                      's', None, 'epoch', 'UTC')
                observer = table.getcell('OBSERVER', row)
                telescope = table.getcell('TELESCOPE_NAME', row)
            start_time = astropy.time.Time(time_range[0] / 86400.0, format='mjd', scale='utc')
            headers['DATE-OBS'] = start_time.utc.isot
            headers['TELESCOP'] = telescope
            headers['OBSERVER'] = observer
        elif len(self._observation_ids) > 1:
            _logger.warning('Multiple OBSERVATION_IDs; will not add FITS headers for observation')

        return headers

    def data_iter(self, start_channel, stop_channel, max_chunk_vis=None):
        if max_chunk_vis is None:
            max_chunk_vis = self._main.nrows()
        num_channels = stop_channel - start_channel
        max_chunk_rows = self._main.nrows()
        if max_chunk_vis is not None:
            max_chunk_rows = max(1, max_chunk_vis // num_channels)
        # PHASE_DIR is used as an approximation to the antenna pointing
        # direction, for the purposes of parallactic angle correction. This
        # probably doesn't generalise well beyond dishes with single-pixel
        # feeds.
        pointing = self.phase_centre()
        pointing = astropy.coordinates.SkyCoord(ra=pointing[0], dec=pointing[1], frame='fk5')
        pos = self.antenna_positions()
        pos = astropy.coordinates.EarthLocation.from_geocentric(pos[:, 0], pos[:, 1], pos[:, 2])
        # Construct a point that is displaced from the pointing by a small
        # quantity northwards. It's necessary to use a small finite difference
        # rather than the pole itself, because the transformation to AzEl is
        # not rigid (does not preserve great circles).
        pole = pointing.directional_offset_by(0 * units.rad, 1e-5 * units.rad)
        time_sum = 0.0
        time_count = 0
        for start in range(0, self._main.nrows(), max_chunk_rows):
            end = min(self._main.nrows(), start + max_chunk_rows)
            nrows = end - start
            flag_row = _getcol(self._main, 'FLAG_ROW', start, nrows)
            field_id = _getcol(self._main, 'FIELD_ID', start, nrows)
            data_desc_id = _getcol(self._main, 'DATA_DESC_ID', start, nrows)
            antenna1 = _getcol(self._main, 'ANTENNA1', start, nrows)
            antenna2 = _getcol(self._main, 'ANTENNA2', start, nrows)
            observation_id = _getcol(self._main, 'OBSERVATION_ID', start, nrows)
            valid = np.logical_not(flag_row)
            valid = np.logical_and(valid, field_id == self._field_id)
            valid = np.logical_and(valid, data_desc_id == self._data_desc_id)
            valid = np.logical_and(valid, antenna1 != antenna2)
            self._observation_ids.update(observation_id)
            antenna1 = antenna1[valid]
            antenna2 = antenna2[valid]
            data = _getcolchannels(self._main, self._data_col, start_channel, stop_channel,
                                   start, nrows, 'Jy', units.Jy)[valid, ...]
            feed_angle1 = None
            feed_angle2 = None
            time_full = _getcol(self._main, 'TIME_CENTROID', start, nrows, 's', None,
                                'epoch', 'UTC')[valid, ...]
            # Each time will be repeated per baseline, but we do not need to repeat all the
            # calculations for each time. Extract just the unique times.
            time, inverse = np.unique(time_full, return_inverse=True)
            time_sum += np.sum(time)
            time_count += len(time)
            # Update average time as we go, in case the caller doesn't exhaust the generator
            average_time = time_sum / time_count
            self._average_time = astropy.time.Time(
                average_time / 86400.0, format='mjd', scale='utc')
            if self._feed_angle_correction:
                # Convert time from MJD seconds to MJD. We do this here rather
                # than by passing 'd' to _getcol, because not all measurement sets
                # specify the units and we want to assume seconds if not specified.
                time = astropy.time.Time(time / 86400.0, format='mjd', scale='utc')
                # Convert to CIRS, which is closer to AltAz in astropy's
                # conversion graph. This avoids duplicating precession-nutation
                # calculations for every antenna.
                cirs_frame = astropy.coordinates.CIRS(obstime=time)
                pole_cirs = pole.transform_to(cirs_frame)
                pointing_cirs = pointing.transform_to(cirs_frame)
                feed_angle = units.Quantity(
                    np.empty((len(time), len(pos))), unit=units.rad, copy=False)
                for i in range(len(pos)):
                    altaz_frame = astropy.coordinates.AltAz(location=pos[i], obstime=time)
                    pole_altaz = pole_cirs.transform_to(altaz_frame)
                    pointing_altaz = pointing_cirs.transform_to(altaz_frame)
                    pa = pointing_altaz.position_angle(pole_altaz)
                    feed_angle[:, i] = pa + self._antenna_angle[i]
                feed_angle1 = feed_angle[inverse, antenna1].astype(np.float32)
                feed_angle2 = feed_angle[inverse, antenna2].astype(np.float32)
            uvw = _getcol(self._main, 'UVW', start, nrows, 'm', units.m, 'uvw')[valid, ...]
            if not self._strict_uvw:
                uvw = -uvw
            if self._has_weight_spectrum:
                weight = _getcolchannels(self._main, 'WEIGHT_SPECTRUM', start_channel, stop_channel,
                                         start, nrows)[valid, ...]
            else:
                weight = _getcol(self._main, 'WEIGHT', start, nrows)[valid, ...]
                # Fake a channel axis
                weight = weight[:, np.newaxis, :]
                weight = np.broadcast_to(weight, data.shape)
            flag = _getcolchannels(self._main, 'FLAG', start_channel, stop_channel,
                                   start, nrows)[valid, ...]
            weight = weight * np.logical_not(flag)
            baseline = (antenna1 * self._antenna.nrows() + antenna2)
            ret = dict(uvw=uvw,
                       weights=np.swapaxes(weight, 0, 1),
                       baselines=baseline,
                       vis=np.swapaxes(data, 0, 1),
                       progress=end,
                       total=self._main.nrows())
            if self._feed_angle_correction:
                ret['feed_angle1'] = feed_angle1
                ret['feed_angle2'] = feed_angle2
            yield ret

    @property
    def raw_data(self):
        return self._main

    def close(self):
        self._main.close()
        self._antenna.close()
        self._data_description.close()
        self._field.close()
        self._spectral_window.close()
        self._polarization.close()
        self._feed.close()
