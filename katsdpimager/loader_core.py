# -*- coding: utf-8 -*-

"""Base classes used by loader modules"""
import numpy as np
from astropy import units
import astropy.io.fits

from . import parameters, sky_model


class LoaderBase:
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

    def num_channels(self):
        """Return total number of channels, which are assumed to be contiguous."""
        raise NotImplementedError('Abstract base class')

    def frequency(self, channel):
        """Return frequency for a given channel.

        Returns
        -------
        Quantity
            Frequency, in appropriate reference frame for transforming UVW
            coordinates from distance to wavelength count.
        """
        raise NotImplementedError('Abstract base class')

    def band(self):
        """Return name for the frequency band in use.

        This is used for looking up externally-defined beam models. If the
        band name is not known, return None.
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

    def has_feed_angles(self):
        """Return whether the data iterator will return `feed_angle1` and
        `feed_angle2`.
        """
        raise NotImplementedError('Abstract base class')

    def data_iter(self, start_channel, stop_channel, max_chunk_vis=None):
        """Return an iterator that yields the data in chunks. Each chunk is a
        dictionary containing numpy arrays with the following keys:

         - ``uvw``: UVW coordinates (position2 - position1), as a Quantity (N×3)
         - ``vis``: visibilities (C×N×P for C channels and P polarizations)
         - ``weights``: imaging weights (C×N×P for C channels and P polarizations)
         - ``baselines``: arbitrary integer baseline IDs; negative IDs indicate autocorrelations
         - ``feed_angle1``: angle between feed and sky (parallactic angle plus a fixed
           offset for the feed), for the first antenna in the baseline (N).
         - ``feed_angle2``: angle between feed and sky for the second antenna in
           the baseline (N).
         - ``progress``: progress made through the file, in some arbitrary units
         - ``total``: size of the file, in same units as ``progress``

        .. note::

           The visibilities are assumed to use a convention in which the phase
           of the electric field *increases* with time, which is consistent with
           [HB1996]_.

           The sign convention for UVW matches the definition for measurement
           sets, but is opposite to that actually used by CASA_ (and thus by
           other imagers that give consistent results with CASA).

           .. [HB1996]

              Understanding radio polarimetry. III. Interpreting the IAU/IEEE definitions
              of the Stokes parameters. J. P. Hamaker and J. D. Bregman. Astron.
              Astrophys. Suppl. Ser., 117 1 (1996) 161-165.

           .. _CASA: http://casa.nrao.edu/Memos/CoordConvention.pdf

        The arrays are indexed first by channel (where applicable) then by a 1D
        time/baseline coordinate. The second index is x/y/z for 'uvw' and
        polarization product for 'vis' and 'weights'.  Flags are not explicitly
        returned: they are either omitted entirely (if all pols are flagged) or
        indicated with a zero weight.

        If :meth:`has_feed_angles` returns ``False``, then `feed_angle1` and
        `feed_angle2` will be absent.

        Parameters
        ----------
        start_channel,stop_channel : int
            Half-open range of channels for which to return data
        max_chunk_vis : int, optional
            Maximum number of full-pol visibilities to return in each chunk.
            If not specified, there is no bound. This is a soft limit that
            may be exceeded if the natural unit of the storage format (e.g. row
            in a measurement set) exceeds this size.
        """
        raise NotImplementedError('Abstract base class')

    def sky_model(self):
        """Get the stored sky model, if any.

        Returns
        -------
        sky_model : :class:`~katsdpimager.sky_model.SkyModel`
            Sky model from the data set

        Raises
        ------
        sky_model.NoSkyModelError
            If there is no sky model in the data set
        """
        return sky_model.NoSkyModelError('This input format does not support sky models')

    def extra_fits_headers(self):
        """Get loader-specific FITS headers to add to the output.

        This is only called after iterating over the data with
        :meth:`data_iter`, so it is possible for :meth:`data_iter` to compute
        data that will be used here. However, note that when :opt:`--vis-limit`
        is specified the data iterator will be closed early.

        Returns
        -------
        headers : astropy.io.fits.Header
            Extra FITS headers to insert into output files. The headers are passed
            to :py:class:`astropy.io.fits.Header`, so for example the value can be
            a (value, comment) tuple.
        """
        return astropy.io.fits.Header()

    @property
    def raw_data(self):
        """Return a handle to the the underlying class-specific data set."""
        raise NotImplementedError('Abstract base class')

    def close(self):
        """Close any open file handles. The object must not be used after this."""
        pass
