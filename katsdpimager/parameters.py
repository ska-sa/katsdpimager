"""Assorted equations for computing appropriate imaging parameters. The
functions take physical quantities as Astropy quantities, rather than
specifying any specific units.

Most formulae are taken from SKA-TEL-SDP-0000003.
"""

import math

from astropy import units
import numpy as np

import katsdpimager.types
from . import clean, weight


def is_smooth(x):
    """Whether x is a good candidate for FFT. We heuristically require
    it to be a multiple of 8 and a product of powers of 2, 3, 5 and 7."""
    if x % 8 != 0:
        return False
    for d in [2, 3, 5, 7]:
        while x % d == 0:
            x = x // d
    return x == 1


class ArrayParameters:
    """Physical attributes of an interferometric array."""
    def __init__(self, antenna_diameter, longest_baseline):
        assert antenna_diameter.unit.physical_type == 'length'
        assert longest_baseline.unit.physical_type == 'length'
        self.antenna_diameter = antenna_diameter
        self.longest_baseline = longest_baseline


class ImageParameters:
    """Physical properties associated with an image.

    At present, only single-frequency images are supported.

    Parameters
    ----------
    q_fov : float
        Scale factor for field of view. 1.0 specifies the first null of the
        primary beam (computed just from an Airy disk, not a measured beam
        model).
    image_oversample : float
        Number of pixels per beam
    frequency : Quantity
        Representative frequency, for converting UVW between metres and
        wavelengths. It may also be specified as a wavelength.
    array : :class:`ArrayParameters`
        Properties of the array. It is not needed if both `pixel_size` and
        `pixels` are specified.
    polarizations : list
        List of polarizations that will appear in the image
    dtype : {np.float32, np.complex64}
        Floating-point type for image and grid
    pixel_size : Quantity or float, optional
        Angular size of a single pixel, or dimensionless to specify l or m
        size directly. If specified, `image_oversample` is ignored.
    pixels : int, optional
        Number of pixels in the image. If specified, `q_fov` is ignored.
    """
    def __init__(self, q_fov, image_oversample, frequency, array, polarizations,
                 dtype, pixel_size=None, pixels=None):
        self.wavelength = frequency.to(units.m, equivalencies=units.spectral())
        # Compute pixel size
        if pixel_size is None:
            if image_oversample < 3.0:
                raise ValueError('image_oversample is too small '
                                 'to capture all visibilities in the UV plane')
            # Size of the UV image plane
            uv_size = (2.0 / 3.0 * image_oversample) * array.longest_baseline
            self.pixel_size = (self.wavelength / uv_size).decompose()
        else:
            # Ensure pixel_size is a Quantity and not just a float
            pixel_size = units.Quantity(pixel_size)
            if pixel_size.unit.physical_type == 'angle':
                pixel_size = np.sin(pixel_size)
            self.pixel_size = pixel_size
        # Compute number of pixels
        if pixels is None:
            # These are just a preliminary cell and pixel size, to compute pixels
            cell_size = array.antenna_diameter * (math.pi / (7.6634 * q_fov))
            image_size = self.wavelength / cell_size
            # Allow image to be slightly smaller if it makes the Fourier transform easier
            pixels = int(0.98 * image_size / self.pixel_size)
            while not is_smooth(pixels):
                pixels += 1
        else:
            if not is_smooth(pixels):
                recommended = pixels
                while not is_smooth(recommended):
                    recommended += 1
                raise ValueError("Image size {} not supported - try {}".format(pixels, recommended))
        assert pixels % 2 == 0
        self.real_dtype = np.dtype(dtype)
        self.complex_dtype = katsdpimager.types.real_to_complex(dtype)
        self.pixels = pixels
        self.image_size = self.pixel_size * pixels
        self.cell_size = self.wavelength / self.image_size
        self.polarizations = polarizations

    def __str__(self):
        from . import polarization
        return """\
Pixel size: {:.3f}
Pixels: {}
FOV: {:.3f}
Cell size: {:.3f}
Wavelength: {:.3f}
Polarizations: {}
Precision: {} bit
""".format(np.arcsin(self.pixel_size).to(units.arcsec),
           self.pixels,
           np.arcsin(self.pixel_size * self.pixels).to(units.deg),
           self.cell_size, self.wavelength,
           ','.join([polarization.STOKES_NAMES[i] for i in self.polarizations]),
           32 if self.real_dtype == np.float32 else 64)


def w_kernel_width(image_parameters, w, eps_w, antialias_width=0):
    """Determine the width (in UV cells) for a W kernel. This is Eq 9 of
    SKA-TEL-SDP-0000003.

    Parameters
    ----------
    image_parameters : :class:`ImageParameters`
        Image parameters, from which wavelength and image size are used
    w : Quantity
        W value for the kernel, as a distance
    eps_w : float
        Fraction of peak at which to truncate the kernel
    antialias_width : float, optional
        If provided, the return value is for a combined W and antialias
        kernel, where the sizes of the individual kernels are combined
        in quadrature.
    """
    fov = image_parameters.image_size
    wl = float(w / image_parameters.wavelength)
    # Squared size of the w part
    wk2 = 4 * fov**2 * (
        (wl * image_parameters.image_size / 2)**2
        + wl**1.5 * fov / (2 * math.pi * eps_w))
    return np.sqrt(wk2 + antialias_width**2)


def w_slices(image_parameters, max_w, eps_w, kernel_width, antialias_width=0):
    lo = 0
    hi = 1
    # Each slice is corrected to its center, so maximum W deviation from the
    # slice center is only half the slice thickness.
    max_w = max_w * 0.5
    # Find a number of slices that is definitely big enough. The first slice is
    # only half-width, to allow the (possibly numerous) visibilities with small
    # W to have better accuracy.

    def measure(slices):
        return w_kernel_width(image_parameters, max_w / (slices - 0.5), eps_w, antialias_width)

    while measure(hi) > kernel_width:
        hi *= 2
    # Binary search
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if measure(mid) < kernel_width:
            hi = mid
        else:
            lo = mid
    return hi


class WeightParameters:
    """Parameters affecting imaging weight calculations.

    Parameters
    ----------
    weight_type : :class:`weight.WeightType`
        Image weighting scheme
    robustness : float, optional
        Robustness parameter for robust weighting
    """   # noqa: E501
    def __init__(self, weight_type, robustness=0.0):
        self.weight_type = weight_type
        self.robustness = robustness

    def __str__(self):
        if self.weight_type == weight.WeightType.ROBUST:
            ans = 'robust ({:.3f})'.format(self.robustness)
        else:
            ans = self.weight_type.name.lower()
        return 'Image weights: ' + ans


class GridParameters:
    """Parameters affecting gridding algorithm.

    Parameters
    ----------
    antialias_width : float
        Support of the antialiasing kernel
    oversample : int
        Number of UV sub-cells per cell, for sampling kernels
    image_oversample : int
        Oversampling in image plane during kernel generation
    w_slices : int
        Number of slices for w-stacking
    w_planes : int
        Number of samples to take in w within each slice
    max_w : Quantity
        Maximum absolute w value, as a distance quantity
    kernel_width : int
        Number of UV cells corresponding to the combined W+antialias kernel.
    degrid : bool, optional
        If true, use degridding, otherwise use direct prediction.
    beams : :class:`primary_beam.BeamModelSet`, optional
        Primary beam models for correction.
    """
    def __init__(self, antialias_width, oversample, image_oversample,
                 w_slices, w_planes, max_w, kernel_width, degrid=False, beams=None):
        if max_w.unit.physical_type != 'length':
            raise TypeError('max W must be specified as a length')
        self.antialias_width = antialias_width
        self.oversample = oversample
        self.image_oversample = image_oversample
        self.w_slices = w_slices
        self.w_planes = w_planes
        self.max_w = max_w
        self.kernel_width = kernel_width
        self.degrid = degrid
        self.beams = beams

    def __str__(self):
        prediction = 'degridding' if self.degrid else 'direct'
        beam_correction = 'yes' if self.beams else 'no'
        return """\
Grid oversampling: {self.oversample}
Image oversample: {self.image_oversample}
W slices: {self.w_slices}
W planes per slice: {self.w_planes}
Maximum W: {self.max_w:.3f}
Antialiasing support: {self.antialias_width} cells
Kernel support: {self.kernel_width} cells
Prediction: {prediction}
Primary beam correction: {beam_correction}""".format(**locals())


class CleanParameters:
    def __init__(self, minor, loop_gain, major_gain, threshold, mode,
                 psf_cutoff, psf_limit, border):
        self.minor = minor
        self.loop_gain = loop_gain
        self.major_gain = major_gain
        self.threshold = threshold
        self.mode = mode
        self.psf_cutoff = psf_cutoff
        if self.psf_cutoff >= 1.0:
            raise ValueError('PSF cutoff must be less than 1')
        self.psf_limit = psf_limit
        self.border = border

    def __str__(self):
        return """\
Loop gain: {self.loop_gain}
Major cycle gain: {self.major_gain}
Threshold: {self.threshold} sigma
Max minor cycles: {self.minor}
PSF cutoff: {self.psf_cutoff}
PSF limit: {self.psf_limit} pixels
Peak function: {mode}
Border: {self.border} pixels""".format(
            self=self, mode='I' if self.mode == clean.CLEAN_I else 'I^2+Q^2+U^2+V^2')
