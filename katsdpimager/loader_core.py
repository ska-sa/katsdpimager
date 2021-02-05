# -*- coding: utf-8 -*-

"""Base classes used by loader modules"""

from abc import ABC, abstractmethod

import numpy as np
from astropy import units
import astropy.io.fits

from . import parameters, sky_model


class LoaderBase(ABC):
    def __init__(self, filename, options, start_channel, stop_channel):
        """Open the data set.

        See :func:`katsdpimager.loader.load` for details of the parameters.
        """
        self.filename = filename

    @abstractmethod
    def command_line_options(self):
        """Return command-line options, in string form.

        Turn the `options` passed to the constructor into a canonical string
        form e.g. ``['-i', 'foo=bar', '-i', 'blah']``.
        """

    @classmethod
    @abstractmethod
    def match(cls, filename):
        """Return True if this loader can handle the file"""

    @abstractmethod
    def antenna_diameters(self):
        """Effective diameters of the antennas."""

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

    @abstractmethod
    def antenna_positions(self):
        """Return the antenna positions. The coordinate system is arbitrary
        since it is used only to compute baseline lengths.

        Returns
        -------
        Quantity, shape (n, 3)
            XYZ coordinates for each antenna.
        """

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

    @abstractmethod
    def num_channels(self):
        """Return total number of channels, which are assumed to be contiguous."""

    @abstractmethod
    def frequency(self, channel):
        """Return frequency for a given channel.

        Returns
        -------
        Quantity
            Frequency, in appropriate reference frame for transforming UVW
            coordinates from distance to wavelength count.
        """

    @abstractmethod
    def band(self):
        """Return name for the frequency band in use.

        This is used for looking up externally-defined beam models. If the
        band name is not known, return None.
        """

    @abstractmethod
    def phase_centre(self):
        """Return direction corresponding to l=0, m=0.

        Returns
        -------
        Quantity of shape (2,)
            RA and DEC for phase centre, in J2000 epoch.

            TODO: use katpoint or astropy.coordinates quantity instead?
        """

    @abstractmethod
    def polarizations(self):
        """Return polarizations stored in the data.

        Returns
        -------
        list
            List of polarization constants from
            :py:mod:`katsdpsigproc.parameters`.
        """

    @abstractmethod
    def has_feed_angles(self):
        """Return whether the data iterator will return `feed_angle1` and `feed_angle2`."""

    def weight_scale(self):
        """Get scale factor between weights and inverse square noise.

        The return value is the RMS noise (in Jansky) on a single real
        correlator channel for a visibility of unit weight, or ``None``
        if it is not known. This has no direct effect on imaging, but if
        available is used to report statistics.
        """
        return None

    def channel_enabled(self, channel):
        """Whether the channel should be imaged.

        This can be used to implement a loader-specific channel mask. For
        efficiency, data for these channels should all be masked by
        :meth:`data_iter`.
        """
        return True

    @abstractmethod
    def data_iter(self, start_channel, stop_channel, max_chunk_vis=None):
        """Return an iterator that yields the data in chunks. Each chunk is a
        dictionary containing numpy arrays with the following keys:

         - ``uvw``: UVW coordinates (position2 - position1), as a Quantity (N×3)
         - ``vis``: visibilities (C×N×P for C channels and P polarizations)
         - ``weights``: imaging weights (C×N×P for C channels and P polarizations)
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
        time/baseline coordinate (which for best performance should be
        baseline-major). The second index is x/y/z for 'uvw' and
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
        data that will be used here. However, note that when :option:`--vis-limit`
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
    @abstractmethod
    def raw_data(self):
        """Return a handle to the the underlying class-specific data set."""

    def close(self):
        """Close any open file handles. The object must not be used after this."""
        pass
