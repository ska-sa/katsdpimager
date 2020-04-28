"""Miscellaneous file format routines"""

import logging

import numpy as np
from astropy import units
import astropy.io.fits as fits
from astropy.time import Time

import katsdpimager
import katsdpimager.polarization as polarization


_logger = logging.getLogger(__name__)


#: Maps internal polarization constants to those used in FITS
# FITS swaps the meaning of X and Y relative to IEEE definitions (see AIPS
# memo 114).
_FITS_POLARIZATIONS = {
    polarization.STOKES_I: 1,
    polarization.STOKES_Q: 2,
    polarization.STOKES_U: 3,
    polarization.STOKES_V: 4,
    polarization.STOKES_RR: -1,
    polarization.STOKES_LL: -2,
    polarization.STOKES_RL: -3,
    polarization.STOKES_LR: -4,
    polarization.STOKES_YY: -5,
    polarization.STOKES_XX: -6,
    polarization.STOKES_YX: -7,
    polarization.STOKES_XY: -8
}


def _fits_polarizations(header, axis, polarizations):
    """Add keywords to a FITS header describing the polarization axis.

    Parameters
    ----------
    header : :py:class:`astropy.io.fits.Header`
        Header which will be updated
    axis : int
        Axis number containing the polarizations, in FITS enumeration
    polarizations : sequence
        List of polarization enums

    Returns
    -------
    ndarray
        Permutation that must be applied to the axis corresponding to
        `polarizations` to make things match.

    Raises
    ------
    ValueError
        If the set of `polarizations` cannot be represented as a linear
        transform in the FITS header.
    """

    # FITS and CASA enumerate polarizations differently. It might be possible
    # to provide an arbitrary mapping from pixel coordinates to world
    # coordinates, but the common cases can all be handled by a permutation
    # and a linear transformation.
    polarizations = np.array([_FITS_POLARIZATIONS[i] for i in polarizations])
    if polarizations[0] >= 0:
        pol_permute = np.argsort(polarizations)
    else:
        # FITS numbers non-IQUV polarizations in a decreasing manner
        pol_permute = np.argsort(-polarizations)
    polarizations = polarizations[pol_permute]
    pol_ref = polarizations[0]
    if len(polarizations) > 1:
        pol_delta = polarizations[1] - polarizations[0]
    else:
        pol_delta = 1
    if np.any(polarizations != np.arange(len(polarizations)) * pol_delta + pol_ref):
        raise ValueError('Polarizations do not form a linear sequence in FITS enumeration')
    header['CTYPE{}'.format(axis)] = 'STOKES'
    header['CRPIX{}'.format(axis)] = 1.0
    header['CRVAL{}'.format(axis)] = float(pol_ref)
    header['CDELT{}'.format(axis)] = float(pol_delta)
    return pol_permute


def write_fits_image(dataset, image, image_parameters, filename, channel,
                     beam=None, bunit='Jy/beam'):
    """Write an image to a FITS file.

    Parameters
    ----------
    dataset : :class:`katsdpimager.loader_core.LoaderBase`
        Source dataset (used to set metadata such as phase centre)
    image : :class:`numpy.ndarray`
        Image data in Jy/beam, indexed by polarization, m, l. For
        a 2M x 2N image, the phase centre is at coordinates (M, N).
    image_parameters : :class:`katsdpimager.parameters.ImageParameters`
        Metadata associated with the image
    filename : str
        File to write. It is silently overwritten if already present.
    channel : int
        Channel number to substitute into `filename` with printf formatting.
    beam : :class:`katsdpimager.beam.Beam`, optional
        Synthesized beam model to write to the header
    bunit : str, optional
        Value for the ``BUNIT`` header in the file. It can be explicitly set
        to ``None`` to avoid writing this key.

    Raises
    ------
    ValueError
        If the set of `polarizations` cannot be represented as a linear
        transform in the FITS header.
    """
    header = fits.Header()
    if bunit is not None:
        header['BUNIT'] = bunit
    header['ORIGIN'] = 'katsdpimager'
    header['HISTORY'] = f'Created by katsdpimager {katsdpimager.__version__}'
    header['TIMESYS'] = 'UTC'
    header['DATE'] = Time.now().utc.isot

    # Transformation from pixel coordinates to intermediate world coordinates,
    # which are taken to be l, m coordinates. The reference point is currently
    # taken to be the centre of the image (actually half a pixel beyond the
    # centre, because of the way fftshift works).  Note that astropy.io.fits
    # reverses the axis order. The X coordinate is computed differently
    # because the X axis is flipped to allow RA to increase right-to-left.
    header['CRPIX1'] = image.shape[2] * 0.5
    header['CRPIX2'] = image.shape[1] * 0.5 + 1.0
    header['CRPIX4'] = 1.0
    # FITS uses degrees; and RA increases right-to-left
    delt = np.arcsin(image_parameters.pixel_size).to(units.deg).value
    header['CDELT1'] = -delt
    header['CDELT2'] = delt
    header['CDELT4'] = 1.0

    # Transformation from intermediate world coordinates to world
    # coordinates (celestial coordinates in this case).
    # TODO: get equinox from input
    phase_centre = dataset.phase_centre()
    header['EQUINOX'] = 2000.0
    header['RADESYS'] = 'FK5'   # Julian equinox
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CUNIT4'] = 'Hz'
    header['CTYPE1'] = 'RA---SIN'
    header['CTYPE2'] = 'DEC--SIN'
    header['CTYPE4'] = 'FREQ    '
    header['CRVAL1'] = phase_centre[0].to(units.deg).value
    header['CRVAL2'] = phase_centre[1].to(units.deg).value
    header['CRVAL4'] = image_parameters.wavelength.to(
        units.Hz, equivalencies=units.spectral()).value
    if beam is not None:
        major = beam.major * image_parameters.pixel_size * units.rad
        minor = beam.minor * image_parameters.pixel_size * units.rad
        header['BMAJ'] = major.to(units.deg).value
        header['BMIN'] = minor.to(units.deg).value
        header['BPA'] = beam.theta.to(units.deg).value
    _fits_polarizations(header, 3, image_parameters.polarizations)
    datamin = float(np.nanmin(image))
    datamax = float(np.nanmax(image))
    if not np.isnan(datamin):
        header['DATAMIN'] = datamin
        header['DATAMAX'] = datamax

    header.update(dataset.extra_fits_headers())

    # l axis is reversed, because RA increases right-to-left.
    # Explicitly converting to big-endian has two advantages:
    # 1. It returns a contiguous array, which allows for a much faster path
    #    through writeto.
    # 2. If the image is little-endian, then astropy.io.fits will convert to
    #    big endian, write, and convert back again.
    # The disadvantage is an increase in memory usage.
    #
    # The np.newaxis adds an axis for frequency. While it's not required for
    # the FITS file to be valid (it's legal for the WCS transformations to
    # reference additional virtual axes), aplpy 1.1.1 doesn't handle it
    # (https://github.com/aplpy/aplpy/issues/350).
    image = np.require(image[np.newaxis, :, :, ::-1], image.dtype.newbyteorder('>'), 'C')
    hdu = fits.PrimaryHDU(image, header)
    hdu.writeto(filename, overwrite=True)


def _split_array(x, dtype):
    """Return a view of x which has one extra dimension. Each element in x is
    treated as some number of elements of type `dtype`, whose size must divide
    into the element size of `x`."""
    in_dtype = x.dtype
    out_dtype = np.dtype(dtype)
    if in_dtype.hasobject or out_dtype.hasobject:
        raise ValueError('dtypes containing objects are not supported')
    if in_dtype.itemsize % out_dtype.itemsize != 0:
        raise ValueError('item size does not evenly divide')

    interface = dict(x.__array_interface__)
    if interface.get('mask', None) is not None:
        raise ValueError('masked arrays are not supported')
    interface['shape'] = x.shape + (in_dtype.itemsize // out_dtype.itemsize,)
    if interface['strides'] is not None:
        interface['strides'] = x.strides + (out_dtype.itemsize,)
    interface['typestr'] = out_dtype.str
    interface['descr'] = out_dtype.descr
    return np.asarray(np.lib.stride_tricks.DummyArray(interface, base=x))


def write_fits_grid(grid, image_parameters, filename, channel):
    """Writes a UV grid to a FITS file.

    Parameters
    ----------
    grid : ndarray of complex
        Grid data indexed by polarization, m, l
    image_parameters : :class:`katsdpimager.parameters.ImageParameters`
        Metadata used to set headers
    filename : str
        File to write. It is silently overwritten if already present.
    channel : int
        Channel number to substitute into `filename` with printf formatting.

    Raises
    ------
    ValueError
        If the set of `polarizations` cannot be represented as a linear
        transform in the FITS header.
    """
    grid = _split_array(grid, image_parameters.real_dtype)
    grid = grid.transpose(3, 0, 1, 2)

    header = fits.Header()
    header['BUNIT'] = 'Jy'
    header['ORIGIN'] = 'katsdpimager'
    header['CUNIT1'] = 'm'
    header['CRPIX1'] = grid.shape[3] // 2 + 1.0
    header['CRVAL1'] = 0.0
    header['CDELT1'] = float(image_parameters.cell_size / units.m)
    header['CUNIT2'] = 'm'
    header['CRPIX2'] = grid.shape[2] // 2 + 1.0
    header['CRVAL2'] = 0.0
    header['CDELT2'] = float(image_parameters.cell_size / units.m)
    pol_permute = _fits_polarizations(header, 3, image_parameters.polarizations)
    header['CTYPE4'] = 'COMPLEX'
    header['CRPIX4'] = 1.0
    header['CRVAL4'] = 1.0
    header['CDELT4'] = 1.0

    hdu = fits.PrimaryHDU(grid[:, pol_permute, :, :], header)
    hdu.writeto(filename, overwrite=True)
