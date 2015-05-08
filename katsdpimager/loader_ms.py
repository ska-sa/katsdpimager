"""Data loading backend for CASA Measurement Sets."""

import katsdpimager.loader_core
import casacore.tables
import numpy as np
import argparse
import astropy.units as units

class LoaderMS(katsdpimager.loader_core.LoaderBase):
    def __init__(self, filename, options):
        super(LoaderMS, self).__init__(filename, options)
        parser = argparse.ArgumentParser(prog='Measurement set options', usage=
            '''Measurement set options: [-i data=COLUMN] [-i field=FIELD]''')
        parser.add_argument('--data', type=str, metavar='COLUMN', default='DATA', help='Column containing visibilities to image [%(default)s]')
        parser.add_argument('--field', type=int, default=0, help='Field to image [%(default)s]')
        args = parser.parse_args(options)
        self._main = casacore.tables.table(filename, ack=False)
        self._antenna = casacore.tables.table(filename + '::ANTENNA', ack=False)
        self._field = casacore.tables.table(filename + '::FIELD', ack=False)
        self._spw = casacore.tables.table(filename + '::SPECTRAL_WINDOW', ack=False)
        self._data_col = args.data
        self._field_id = args.field
        if self._data_col not in self._main.colnames():
            raise ValueError('{} has no column named {}'.format(
                filename, self._data_col))
        if args.field < 0 or args.field >= self._field.nrows():
            raise ValueError('Field {} is out of range'.format(args.field))

    @classmethod
    def match(cls, filename):
        return filename.lower().endswith('.ms')

    def antenna_diameters(self):
        # TODO: process QuantumUnits
        return self._antenna.getcol('DISH_DIAMETER') * units.m

    def antenna_positions(self):
        # TODO: process QuantumUnits
        return self._antenna.getcol('POSITION') * units.m

    def phase_centre(self):
        keywords = self._field.getcolkeywords('PHASE_DIR')
        measinfo = keywords.get('MEASINFO')
        quantum_units = keywords.get('QuantumUnits')
        if measinfo is not None:
            if measinfo['Ref'] != 'J2000' or measinfo['type'] != 'direction':
                raise ValueError('Unsupported MEASINFO for PHASE_DIR: {}'.format(measinfo))
        if quantum_units is not None:
            if quantum_units != ['rad', 'rad']:
                raise ValueError('Unsupported QuantumUnits for PHASE_DIR: {}'.format(quantum_units))
        value = self._field.getcell('PHASE_DIR', self._field_id)
        if tuple(value.shape) != (1, 2):
            raise ValueError('Unsupported shape for PHASE_DIR: {}'.format(value.shape))
        return value[0, :] * units.rad

    def frequency(self, channel):
        if self._spw.nrows() != 1:
            raise ValueError('Multiple spectral windows are not yet supported')
        keywords = self._spw.getcolkeywords('CHAN_FREQ')
        quantum_units = keywords.get('QuantumUnits')
        if quantum_units is not None and quantum_units != ['Hz']:
            raise ValueError('Unsupported QuantumUnits for CHAN_FREQ: {}'.format(quantum_units))
        return self._spw.getcol('CHAN_FREQ')[0, channel] * units.Hz

    def data_iter(self, channel, max_rows=None):
        if max_rows is None:
            max_rows = self._main.nrows()
        for start in xrange(0, self._main.nrows(), max_rows):
            end = min(self._main.nrows(), start + max_rows)
            flag_row = self._main.getcol('FLAG_ROW', start, end - start)
            field_id = self._main.getcol('FIELD_ID', start, end - start)
            valid = np.logical_and(np.logical_not(flag_row), field_id == self._field_id)
            data = self._main.getcol(self._data_col, start, end - start)[valid, channel, ...]
            uvw = self._main.getcol('UVW', start, end - start)[valid, ...]
            uvw = units.Quantity(uvw, units.m, copy=False)
            if 'WEIGHT_SPECTRUM' in self._main.colnames():
                weight = self._main.getcol('WEIGHT_SPECTRUM', start, end - start)[valid, channel, ...]
            else:
                weight = self._main.getcol('WEIGHT', start, end - start)[valid, ...]
            flag = self._main.getcol('FLAG', start, end - start)[valid, ...]
            weight = weight * np.logical_not(flag[:, channel, :])
            # For now, only image first polarisation
            data = data[..., 0:1]
            weight = weight[..., 0:1]
            yield dict(uvw=uvw, weights=weight, vis=data)

    def close(self):
        self._main.close()
        self._antenna.close()
        self._field.close()
        self._spw.close()
