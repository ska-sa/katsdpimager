"""Convolutional gridding"""

from __future__ import division, print_function
from . import parameters
import numpy as np
import math
import pkg_resources
import astropy.units as units
import katsdpsigproc.accel as accel
import katsdpsigproc.tune as tune


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


class GridderTemplate(object):
    def __init__(self, context, grid_parameters, num_polarizations, dtype, tuning=None):
        if tuning is None:
            tuning = self.autotune(
                context, grid_parameters.antialias_size, grid_parameters.oversample,
                num_polarizations)
        self.grid_parameters = grid_parameters
        self.dtype = dtype
        convolve_kernel = antialias_kernel(grid_parameters.antialias_size,
                                           grid_parameters.oversample)
        self.convolve_kernel_size = convolve_kernel.shape[2]
        ksize = self.convolve_kernel_size
        self.wgs_x = 8
        self.wgs_y = 8
        self.multi_x = 1
        self.multi_y = 1
        self.num_polarizations = num_polarizations
        tile_x = self.wgs_x * self.multi_x
        tile_y = self.wgs_y * self.multi_y
        assert ksize <= tile_y
        assert ksize <= tile_x
        self.convolve_kernel = accel.SVMArray(
            context,
            (grid_parameters.oversample, grid_parameters.oversample, tile_y, tile_x),
            np.complex64)
        self.convolve_kernel.fill(0)
        self.convolve_kernel[:, :, :ksize, :ksize] = convolve_kernel
        self.program = accel.build(
            context, "imager_kernels/grid.mako",
            {
                'real_type': ('float' if dtype == np.complex64 else 'double'),
                'convolve_kernel_row_stride': self.convolve_kernel.padded_shape[3],
                'convolve_kernel_slice_stride':
                    self.convolve_kernel.padded_shape[2] * self.convolve_kernel.padded_shape[3],
                'convolve_kernel_oversample': self.convolve_kernel.shape[0],
                'multi_x': self.multi_x,
                'multi_y': self.multi_y,
                'wgs_x': self.wgs_x,
                'wgs_y': self.wgs_y,
                'num_polarizations': self.num_polarizations
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    @classmethod
    @tune.autotuner(test={})
    def autotune(cls, context, antialias_size, oversample, num_polarizations):
        # Nothing to autotune yet
        return {}

    def instantiate(self, *args, **kwargs):
        return Gridder(self, *args, **kwargs)


class Gridder(accel.Operation):
    def __init__(self, template, command_queue, image_parameters, array_parameters,
                 max_vis, allocator=None):
        super(Gridder, self).__init__(command_queue, allocator)
        if len(image_parameters.polarizations) != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        if image_parameters.complex_dtype != template.dtype:
            raise ValueError('Mismatch in data type')
        # Check that longest baseline won't cause an out-of-bounds access
        max_uv_src = float(array_parameters.longest_baseline / image_parameters.cell_size)
        max_uv = max_uv_src + template.convolve_kernel_size / 2
        if max_uv >= image_parameters.pixels // 2 - 1 - 1e-3:
            raise ValueError('image_oversample is too small to capture all visibilities in the UV plane')
        self.template = template
        self.image_parameters = image_parameters
        self.max_vis = max_vis
        self.slots['grid'] = accel.IOSlot(
            (image_parameters.pixels, image_parameters.pixels,
                accel.Dimension(template.num_polarizations, exact=True)),
            template.dtype)
        self.slots['uvw'] = accel.IOSlot(
            (max_vis, accel.Dimension(3, exact=True)), np.float32)
        self.slots['vis'] = accel.IOSlot(
            (max_vis, accel.Dimension(template.num_polarizations, exact=True)), np.complex64)
        self.kernel = template.program.get_kernel('grid')
        self._num_vis = 0
        cell_size_m = image_parameters.cell_size.to(units.m).value
        self.uv_scale = template.grid_parameters.oversample / cell_size_m
        # Offset to bias coordinates such that l,m=0 translates to the first
        # pixel to update in the grid, measured in subpixels
        uv_bias_pixels = image_parameters.pixels // 2 - (template.convolve_kernel_size - 1) // 2
        self.uv_bias = float(uv_bias_pixels) * template.grid_parameters.oversample

    def set_num_vis(self, n):
        if n < 0 or n > self.max_vis:
            raise ValueError('Number of visibilities {} is out of range 0..{}'.format(
                n, self.max_vis))
        self._num_vis = n

    def clear(self):
        """TODO: implement on GPU"""
        self.buffer('grid').fill(0)

    def grid(self, uvw, vis):
        """Add visibilities to the grid, with convolutional gridding using the
        anti-aliasing filter."""
        self.set_num_vis(len(uvw))
        self.buffer('uvw')[:len(uvw)] = uvw
        self.buffer('vis')[:len(vis)] = vis
        self()

    def _run(self):
        if self._num_vis == 0:
            return
        grid = self.buffer('grid')
        workgroups = 256  # TODO: tune this in some way
        vis_per_workgroup = accel.divup(self._num_vis, workgroups)
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                grid.buffer,
                np.int32(grid.padded_shape[1]),
                self.buffer('uvw').buffer,
                self.buffer('vis').buffer,
                self.template.convolve_kernel.buffer,
                np.float32(self.uv_scale),
                np.float32(self.uv_bias),
                np.int32(vis_per_workgroup),
                np.int32(self._num_vis),
            ],
            global_size=(self.template.wgs_x * workgroups, self.template.wgs_y),
            local_size=(self.template.wgs_x, self.template.wgs_y)
        )

    def parameters(self):
        return {
            'grid_parameters': self.template.grid_parameters,
            'image_parameters': self.image_parameters
        }


class GridderHost(object):
    def __init__(self, image_parameters, grid_parameters):
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self.kernel = antialias_kernel(grid_parameters.antialias_size,
                                       grid_parameters.oversample)
        pixels = image_parameters.pixels
        shape = (pixels, pixels, len(image_parameters.polarizations))
        self.values = np.empty(shape, image_parameters.complex_dtype)
        # TODO: compute taper function (FT of kernel)
        # See http://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html

    def clear(self):
        self.values.fill(0)

    def grid(self, uvw, vis):
        """Add visibilities to the grid, with convolutional gridding using the
        anti-aliasing filter.

        Parameters
        ----------
        uvw : 2D Quantity array
            UVW coordinates for visibilities, indexed by sample then u/v/w
        vis : 2D ndarray of complex
            Visibility data, indexed by sample and polarization, and
            pre-multiplied by all weights
        """
        assert uvw.unit.physical_type == 'length'
        pixels = self.image_parameters.pixels
        ksize = self.kernel.shape[2]
        # Offset to bias coordinates such that l,m=0 translates to the first
        # pixel to update in the grid.
        offset = pixels // 2 - (ksize - 1) // 2
        for row in range(uvw.shape[0]):
            # l and m are measured in cells
            l = np.float32(uvw[row, 0] / self.image_parameters.cell_size) + offset
            m = np.float32(uvw[row, 1] / self.image_parameters.cell_size) + offset
            sub_l = subpixel_coord(l, self.grid_parameters.oversample)
            sub_m = subpixel_coord(m, self.grid_parameters.oversample)
            l = int(math.floor(l))
            m = int(math.floor(m))
            sample = vis[row, :]
            sub_kernel = self.kernel[sub_m, sub_l, ..., np.newaxis]
            self.values[m : m+ksize, l : l+ksize, :] += sample * sub_kernel
