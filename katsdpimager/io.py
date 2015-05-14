"""Miscellaneous file format routines"""

from __future__ import division, print_function
import numpy as np
import astropy.units as units
import astropy.io.fits as fits
import katsdpimager.polarization as polarization

#: Maps internal polarization constants to those used in FITS
_FITS_POLARIZATIONS = {
    polarization.STOKES_I: 1,
    polarization.STOKES_Q: 2,
    polarization.STOKES_U: 3,
    polarization.STOKES_V: 4,
    polarization.STOKES_RR: -1,
    polarization.STOKES_LL: -2,
    polarization.STOKES_RL: -3,
    polarization.STOKES_LR: -4,
    polarization.STOKES_XX: -5,
    polarization.STOKES_YY: -6,
    polarization.STOKES_XY: -7,
    polarization.STOKES_YX: -8
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
        pol_delta = 0
    if np.any(polarizations != np.arange(len(polarizations)) * pol_delta + pol_ref):
        raise ValueError('Polarizations do not form a linear sequence in FITS enumeration')
    header['CTYPE{}'.format(axis)] = 'STOKES'
    header['CRPIX{}'.format(axis)] = 1.0
    header['CRVAL{}'.format(axis)] = float(pol_ref)
    header['CDELT{}'.format(axis)] = float(pol_delta)
    return pol_permute


def write_fits_image(dataset, image, image_parameters, filename):
    """Write an image to a FITS file.

    Parameters
    ----------
    dataset : :class:`katsdpimager.loader_core.LoaderBase`
        Source dataset (used to set metadata such as phase centre)
    image : :class:`numpy.ndarray`
        Image data in Jy/beam, indexed by m, l and polarization. For
        a 2M x 2N image, the phase centre is at coordinates (M, N).
    image_parameters : :class:`katsdpimager.parameters.ImageParameters`
        Metadata associated with the image
    filename : `str`
        File to write. It is silently overwritten if already present.

    Raises
    ------
    ValueError
        If the set of `polarizations` cannot be represented as a linear
        transform in the FITS header.
    """
    header = fits.Header()
    header['BUNIT'] = 'JY/BEAM'
    header['ORIGIN'] = 'katsdpimager'

    # Transformation from pixel coordinates to intermediate world coordinates,
    # which are taken to be l, m coordinates. The reference point is current
    # taken to be the centre of the image (actually half a pixel beyond the
    # centre, because of the way fftshift works).  Note that astropy.io.fits
    # reverses the axis order. The X coordinate is computed differently
    # because the X axis is flipped to allow RA to increase right-to-left.
    header['CRPIX1'] = image.shape[1] * 0.5
    header['CRPIX2'] = image.shape[0] * 0.5 + 1.0
    # FITS uses degrees; and RA increases right-to-left
    delt = np.arcsin(image_parameters.pixel_size).to(units.deg).value
    header['CDELT1'] = -delt
    header['CDELT2'] = delt

    # Transformation from intermediate world coordinates to world
    # coordinates (celestial coordinates in this case).
    # TODO: get equinox from input
    phase_centre = dataset.phase_centre()
    header['EQUINOX'] = 2000.0
    header['RADESYS'] = 'FK5' # Julian equinox
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CTYPE1'] = 'RA---SIN'
    header['CTYPE2'] = 'DEC--SIN'
    header['CRVAL1'] = phase_centre[0].to(units.deg).value
    header['CRVAL2'] = phase_centre[1].to(units.deg).value
    pol_permute = _fits_polarizations(header, 3, image_parameters.polarizations)

    # Second axis is reversed, because RA increases right-to-left.
    # The permutation axis is rolled to the front, because Tigger doesn't
    # correctly handle the file otherwise.
    hdu = fits.PrimaryHDU(np.rollaxis(image[:, ::-1, pol_permute], 2), header)
    hdu.writeto(filename, clobber=True)

def _split_array(x, dtype):
    """Return a view of x which has one extra dimension. Each element is x is
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

def write_fits_grid(grid, image_parameters, filename):
    """Writes a UV grid to a FITS file.

    Parameters
    ----------
    grid : ndarray of complex
        Grid data indexed by m, l, polarization
    image_parameters : :class:`katsdpimager.parameters.ImageParameters`
        Metadata used to set headers
    filename : str
        File to write. It is silently overwritten if already present.

    Raises
    ------
    ValueError
        If the set of `polarizations` cannot be represented as a linear
        transform in the FITS header.
    """
    grid = _split_array(grid, np.float32)
    grid = grid.transpose(3, 2, 0, 1)

    header = fits.Header()
    header['BUNIT'] = 'JY'
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
    hdu.writeto(filename, clobber=True)
