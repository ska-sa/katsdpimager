"""Base classes used by loader modules"""
import numpy as np
import astropy.units as units
from . import parameters

class LoaderBase(object):
    def __init__(self, filename, options):
        """Open the file"""
        self.filename = filename

    @classmethod
    def match(cls, filename):
        """Return True if this loader can handle the file"""
        raise NotImplementedError('Abstract base class')

    def antenna_diameters(self):
        """Effective diameters of the antennas."""
        raise NotImplementedError('Abstract base class')

    def antenna_diameter(self):
        """Return the common diameter of all dishes.

        Raises
        ------
        ValueError
            if the dishes do not all have the same diameter
        """
        diameters = self.antenna_diameters()
        D = diameters[0]
        if not np.all(diameters == D):
            raise ValueError('Diameters are not all equal')
        return D

    def antenna_positions(self):
        """Return the antenna positions. The coordinate system is arbitrary
        since it is used only to compute baseline lengths.

        Returns
        -------
        Quantity, shape (n, 3)
            XYZ coordinates for each antenna.
        """
        raise NotImplementedError('Abstract base class')

    def longest_baseline(self):
        """Return the length of the longest baseline."""
        positions = self.antenna_positions()
        ans = 0.0 * units.m
        for i in range(len(positions)):
            for j in range(i):
                baseline = positions[i] - positions[j]
                ans = max(ans, np.sqrt(baseline.dot(baseline)))
        return ans

    def array_parameters(self):
        """Return ArrayParameters object for the array used in the observation."""
        return parameters.ArrayParameters(
            self.antenna_diameter(), self.longest_baseline())

    def frequency(self, channel):
        """Return frequency for a given channel.

        Returns
        -------
        Quantity
            Frequency, in appropriate reference frame for transforming UVW
            coordinates from distance to wavelength count.
        """
        raise NotImplementedError('Abstract base class')

    def phase_centre(self):
        """Return direction corresponding to l=0, m=0.

        Returns
        -------
        Quantity of shape (2,)
            RA and DEC for phase centre, in J2000 epoch.

            TODO: use katpoint or astropy.coordinates quantity instead?
        """
        raise NotImplementedError('Abstract base class')

    def polarizations(self):
        """Return polarizations stored in the data.

        Returns
        -------
        list
            List of polarization constants from
            :py:mod:`katsdpsigproc.parameters`.
        """
        raise NotImplementedError('Abstract base class')

    def data_iter(self, channel, max_rows=None):
        """Return an iterator that yields the data in chunks. Each chunk is a
        dictionary containing numpy arrays with the following keys:

         - 'uvw': UVW coordinates (position1 - position2), as a Quantity
         - 'vis': visibilities
         - 'weights': imaging weights
         - 'progress': progress made through the file, in some arbitrary units
         - 'total': size of the file, in same units as 'progress'

        .. note::

           The sign convention for UVW matches the white book and AIPS, but is
           opposite_ to that used in Measurement Sets.

        .. _opposite: http://casa.nrao.edu/Memos/CoordConvention.pdf

        The arrays are indexed first by a 1D time/baseline coordinate. The second
        index is x/y/z for 'uvw' and polarization product for 'vis' and 'weights'.
        Flags are not explicitly returned: they are either omitted entirely
        (if all pols are flagged) or indicated with a zero weight.

        If `max_rows` is given, it limits the number of rows to return in each
        chunk.
        """
        raise NotImplementedError('Abstract base class')

    def close(self):
        """Close any open file handles. The object must not be used after this."""
        pass
