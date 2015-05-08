"""Assorted equations for computing appropriate imaging parameters. The
functions take physical quantities as Astropy quantities, rather than
specifying any specific units.

Most formulae are taken from SKA-TEL-SDP-0000003.
"""

from __future__ import division
import astropy.units as units
import math
import numpy as np

def is_smooth(x):
    """Whether x is a good candidate for FFT. We heuristically require
    it to be a multiple of 64 and a product of powers of 2, 3 and 5."""
    if x % 64 != 0:
        return False
    for d in [2, 3, 5]:
        while x % d == 0:
            x = x // d
    return x == 1


class ArrayParameters(object):
    """Physical attributes of an interferometric array."""
    def __init__(self, diameter, longest_baseline):
        assert diameter.unit.physical_type == 'length'
        assert longest_baseline.unit.physical_type == 'length'
        self.diameter = diameter
        self.longest_baseline = longest_baseline


class ImageParameters(object):
    """Physical properties associated with an image. At present, only
    single-frequency images are supported.

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
    pixel_size : Quantity or float, optional
        Angular size of a single pixel, or dimensionless to specify l or m
        size directly. If specified, `image_oversample` is ignored.
    pixel : int, optional
        Number of pixels in the image. If specified, `q_fov` is ignored.
    """
    def __init__(self, q_fov, image_oversample, frequency, array, pixel_size=None, pixels=None):
        self.wavelength = frequency.to(units.m, equivalencies=units.spectral())
        # Compute pixel size
        if pixel_size is None:
            if image_oversample < 3.0:
                raise ValueError('image_oversample is too small to capture all visibilities in the UV plane')
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
            cell_size = array.diameter * (math.pi / (7.6634 * q_fov))
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
        self.pixels = pixels
        self.image_size = self.pixel_size * pixels
        self.cell_size = self.wavelength / self.image_size

    def __str__(self):
        return "Pixel size: {:.3f}\nPixels: {}\nFOV: {:.3f}\nCell size: {:.3f}\nWavelength: {:.3f}\n".format(
            np.arcsin(self.pixel_size).to(units.arcsec),
            self.pixels,
            np.arcsin(self.pixel_size * self.pixels).to(units.deg),
            self.cell_size, self.wavelength)


class GridParameters(object):
    def __init__(self, antialias_size, oversample):
        self.antialias_size = antialias_size
        self.oversample = oversample
