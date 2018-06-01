"""Load a local sky model from file and transfer it to a model image

At present the only file format supported is a katpoint catalogue file, but
other formats (e.g. Tigger) may be added later.
"""

from __future__ import print_function, division, absolute_import

import re

import six
import numpy as np
import astropy.wcs
import astropy.units as units
import katpoint

from . import polarization


class SkyModel(object):
    """A sky model which can be used to generate images at specified frequencies.

    Parameters
    ----------
    filename : str or file-like
        File with the catalogue. If it is a string it is assumed to be a
        filename, otherwise it is treated as a file-like object.
    file_format : {'katpoint'}
        File format

    Raises
    ------
    IOError, OSError
        if there was a low-level error reading the file
    ValueError
        if `file_format` was not recognised
    """
    def __init__(self, filename, file_format):
        if file_format != 'katpoint':
            raise ValueError('file_format "{}" is not recognised'.format(file_format))
        if isinstance(filename, six.string_types):
            with open(filename) as f:
                self._init_katpoint(f)
        else:
            self._init_katpoint(filename)

    @classmethod
    def open(cls, filename):
        """Open a sky model from a filename string.

        The string has the format :samp:`{filename}` or
        :samp:`{format}:{filename}`. If the format is not specified, it
        defaults to ``katpoint``.
        """
        match = re.match('^([-a-z_.]+):(.*)$', filename)
        if match:
            file_format = match.group(1)
            filename = match.group(2)
        else:
            file_format = 'katpoint'
        return cls(filename, file_format)

    def _init_katpoint(self, f):
        self._catalogue = katpoint.Catalogue(f)
        positions = units.Quantity(np.empty((len(self._catalogue), 2)),
                                   unit=units.deg)
        for i, source in enumerate(self._catalogue):
            positions[i] = units.Quantity(source.astrometric_radec(), unit=units.rad)
        self._positions = positions

    def add_to_image(self, image, image_p, phase_centre, scale=1.0):
        """Add the source fluxes to an image.

        Parameters
        ----------
        image : array-like
            Image to be incremented, of shape (polarizations, height, width).
        image_p : :class:`.ImageParameters`
            Coordinate system transformations for the image
        phase_centre : Quantity
            RA and DEC of centre of the image
        scale : float
            Scale factor applied to flux before addition
        """
        wcs = astropy.wcs.WCS(naxis=2)
        delta = np.arcsin(image_p.pixel_size).to(units.deg).value
        # The +1 is because FITS counts pixels from 1
        wcs.wcs.crpix = [image_p.pixels / 2 + 1, image_p.pixels / 2 + 1]
        wcs.wcs.cdelt = [delta, delta]
        wcs.wcs.crval = phase_centre.to(units.deg).value
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        wcs.wcs.cunit = ['deg', 'deg']
        freq_MHz = image_p.wavelength.to(units.MHz, equivalencies=units.spectral()).value
        # For each image polarization, find the corresponding index in IQUV
        # for advanced indexing.
        pol_index = [polarization.STOKES_IQUV.index(pol) for pol in image_p.polarizations]
        coords = wcs.wcs_world2pix(self._positions, 0)
        # Snap to the grid. Points more than 90 degrees from the phase centre
        # will be NaN, which when converted to integer becomes INT_MIN and
        # hence are safely rejected.
        for source, coord in zip(self._catalogue, coords):
            # Points more than 90 degrees from the phase centre will be transformed
            # to NaN by the SIN projections. This test is written in a way that these
            # points will be rejected. We also discard sources that fall right on the
            # border rather than deal with the edge cases.
            if not all(c > 0.5 and c < image_p.pixels - 1.5 for c in coord):
                continue    # Falls outside image
            flux = source.flux_density_stokes(freq_MHz)
            if np.all(np.isfinite(flux)):
                # Update a 2x2 region of pixels, as a crude bi-linear
                # anti-aliasing filter.
                flux = flux[pol_index] * scale
                lm = np.trunc(coord)
                frac = coord - lm
                l, m = lm.astype(np.int)
                image[:, m, l] += flux * (1.0 - frac[0]) * (1.0 - frac[1])
                image[:, m, l + 1] += flux * frac[0] * (1.0 - frac[1])
                image[:, m + 1, l] += flux * (1.0 - frac[0]) * frac[1]
                image[:, m + 1, l + 1] += flux * frac[0] * frac[1]
