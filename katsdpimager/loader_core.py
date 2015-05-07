"""Base classes used by loader modules"""
import numpy as np

class LoaderBase(object):
    def __init__(self, filename, options):
        """Open the file"""
        self.filename = filename

    @classmethod
    def match(cls, filename):
        """Return True if this loader can handle the file"""
        raise NotImplementedError('Abstract base class')

    def antenna_diameters(self):
        """Effective diameters of the antennas, in metres."""
        raise NotImplementedError('Abstract base class')

    def antenna_diameter(self):
        """Return the common diameter of all dishes.

        Raises
        ------
        ValueError if the dishes do not all have the same diameter
        """
        diameters = self.antenna_diameters()
        D = diameters[0]
        if not np.all(diameters == D):
            raise ValueError('Diameters are not all equal')
        return D

    def antenna_positions(self):
        """Return a list antenna positions. Each is given as a 3-tuple of
        (x, y, z), in metres. The coordinate system is arbitrary since it is
        used only to compute baseline lengths."""
        raise NotImplementedError('Abstract base class')

    def longest_baseline(self):
        """Return the length of the longest baseline, in metres"""
        positions = self.antenna_positions()
        ans = 0.0
        for i in range(len(positions)):
            for j in range(i):
                baseline = np.array(positions[i]) - np.array(positions[j])
                ans = max(ans, np.linalg.norm(baseline))
        return ans

    def phase_centre(self):
        """Return direction (RA and DEC in J2000, radians) corresponding to
        l=0, m=0."""
        raise NotImplementedError('Abstract base class')

    def data_iter(self, channel, max_rows=None):
        """Return an iterator that yields the data in chunks. Each chunk is a
        dictionary containing numpy arrays with the following keys:

         - 'uvw': UVW coordinates (target - source) in metres
         - 'vis': visibilities
         - 'weights': imaging weights

        The arrays are indexed first by a 1D time/baseline coordinate. The second
        index is x/y/z for 'uvw' and polarisation product for 'vis' and 'weights'.
        Flags are not explicitly returned: they are either omitted entirely
        (if all pols are flagged) or indicated with a zero weight.

        If `max_rows` is given, it limits the number of rows to return in each
        chunk.
        """
        raise NotImplementedError('Abstract base class')

    def close(self):
        """Close any open file handles. The object must not be used after this."""
        pass
