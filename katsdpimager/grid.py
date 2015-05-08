"""Convolutional gridding"""

from __future__ import division, print_function
from . import parameters
import numpy as np
import math

def kaiser_bessel(x, width, beta):
    return np.i0(beta * np.sqrt(np.maximum(0.0, 1 - (2.0 * x / width)**2))) / np.i0(beta)

def antialias_kernel(width, oversample, beta=None):
    r"""Generate anti-aliasing kernel. The return value is 4-dimensional.
    The first two dimensions select the subpixel position, the second two
    select the pixel position.

    The returned kernels all have the same size, but it is not necessarily
    the same size as `width` (in fact, `width` may be non-integer). Given
    a real-valued :math:`x` coordinate and a grid point :math:`x_g` to be
    updated, the indices are computed as follows:

    .. math::

       x_0 &= \lfloor x\rfloor\\
       s &= \lfloor (x - x_0)\cdot\text{oversample}\rfloor\\
       u &= x_g - x_0 + \tfrac12 \lvert \text{kernel}\rvert - 1

    Then `s` is the subpixel index and `u` is the pixel index.

    TODO: normalisation so that integral is 1?
    """
    if beta is None:
        # Puts the first null of the taper function at the edge of the image
        beta = math.pi * math.sqrt(0.25 * width**2 - 1.0)
    hsize = int(math.ceil(0.5 * width))
    size = 2 * hsize
    # The kernel only contains real values, but we store complex so that we
    # are ready for W projection.
    kernel = np.empty((oversample, oversample, size, size), np.complex64)
    for s in range(oversample):
        for t in range(oversample):
            x_bias = (s + 0.5) / oversample + hsize - 1
            y_bias = (t + 0.5) / oversample + hsize - 1
            x_kernel = kaiser_bessel(np.arange(size) - x_bias, width, beta)
            y_kernel = kaiser_bessel(np.arange(size) - y_bias, width, beta)
            kernel[s, t, ...] = np.outer(x_kernel, y_kernel)
    return kernel

def subpixel_coord(x, oversample):
    x0 = np.floor(x)
    return np.floor((x - x0) * oversample)


class Gridder(object):
    def __init__(self, image_parameters, grid_parameters):
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self.kernel = antialias_kernel(grid_parameters.antialias_size,
                                       grid_parameters.oversample)
        # TODO: compute taper function (FT of kernel)
        # See http://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html

    def grid(self, grid, uvw, weights, vis):
        """Add visibilities to a grid, with convolutional gridding using the
        anti-aliasing filter.

        Parameters
        ----------
        grid : 3D ndarray of complex
            Grid, indexed by m, l and pol. The DC term is at the centre.
        uvw : 2D Quantity array
            UVW coordinates for visibilities, indexed by sample then u/v/w
        weight: 2D ndarray of real
            Weights for visibilities, indexed by sample and pol.
        vis : 2D ndarray of complex
            Visibility data, indexed by sample and pol.
        """
        assert uvw.unit.physical_type == 'length'
        pixels = self.image_parameters.pixels
        ksize = self.kernel.shape[2]
        assert grid.shape[0] == pixels
        assert grid.shape[1] == pixels
        # Offset to bias coordinates such that l,m=0 translates to the first
        # pixel to update in the grid.
        offset = pixels // 2 - (ksize - 1) // 2
        for row in range(uvw.shape[0]):
            # l and m are measured in cells
            l = float(uvw[row, 0] / self.image_parameters.cell_size) + offset
            m = float(uvw[row, 1] / self.image_parameters.cell_size) + offset
            sub_l = subpixel_coord(l, self.grid_parameters.oversample)
            sub_m = subpixel_coord(m, self.grid_parameters.oversample)
            l = int(math.floor(l))
            m = int(math.floor(m))
            sample = weights[row, :] * vis[row, :]
            sub_kernel = self.kernel[sub_m, sub_l, ..., np.newaxis]
            grid[m : m+ksize, l : l+ksize, :] += sample * sub_kernel
