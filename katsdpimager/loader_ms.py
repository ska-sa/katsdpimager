"""Data loading backend for CASA Measurement Sets."""

import katsdpimager.loader_core
import casacore.tables
import numpy as np

class LoaderMS(katsdpimager.loader_core.LoaderBase):
    def __init__(self, filename):
        super(LoaderMS, self).__init__(filename)
        self._main = casacore.tables.table(filename, ack=False)
        self._antenna = casacore.tables.table(filename + '::ANTENNA', ack=False)

    @classmethod
    def match(cls, filename):
        return filename.lower().endswith('.ms')

    def antenna_diameters(self):
        return self._antenna.getcol('DISH_DIAMETER')

    def antenna_positions(self):
        return self._antenna.getcol('POSITION')

    def data_iter(self, channel, max_rows=None):
        if max_rows is None:
            max_rows = self._main.nrows()
        for start in xrange(0, self._main.nrows(), max_rows):
            end = min(self._main.nrows(), start + max_rows)
            valid = np.logical_not(self._main.getcol('FLAG_ROW', start, end - start))
            data = self._main.getcol('DATA', start, end - start)[valid, ...]
            uvw = self._main.getcol('UVW', start, end - start)[valid, ...]
            weight = self._main.getcol('WEIGHT', start, end - start)[valid, ...]
            flag = self._main.getcol('FLAG', start, end - start)[valid, ...]
            weight = weight * np.logical_not(flag[:, channel, :])
            yield dict(uvw=uvw, weights=weight, vis=data)

    def close(self):
        self._main.close()
        self._antenna.close()
