# -*- coding: utf-8 -*-

r"""Convolutional gridding with W projection.

Gridding
--------

The GPU approach is based on [Rom12_]. Each workgroup (thread block) handles a
contiguous range of visibilities, with the threads cooperating to grid each
visibility.  The grid is divided into *bins*, whose size is the same as the
size of the convolution function. These bins are further divided into *tiles*,
which are divided into *blocks*. A workitem (thread) handles all blocks that
are at the same offset within their bin. Similarly, a workgroup handles all
tiles that are at the same offset within their bin.

.. tikz:: Mapping of workitems and workgroups to group points. One workgroup
   contributes to all the green cells. One workitem contributes to all the
   cells marked with a red dot. Here blocks are 2×2, tiles are 4×4 and bins
   are 8×8. The dashed lines indicate the footprint of the convolution kernel
   for a single visibility, showing that the thread contributes four values.

   [x=0.5cm, y=0.5cm]
   \foreach \i in {4, 12}
       \foreach \j in {0, 8}
       {
           \fill[shift={(\i,\j)},green!50!white] (0, 0) rectangle (4, 4);
       }
   \foreach \i in {4,5, 12, 13}
       \foreach \j in {2,3, 10,11}
       {
           \node[shift={(\i,\j)},circle,inner sep=0pt, minimum size=0.2cm,fill=red] at (0.5, 0.5) {};
       }
   \draw[help lines] (0, 0) grid[step=0.5cm] (16, 16);
   \draw[very thick] (0, 0) grid[step=4cm] (16, 16);
   \draw[very thick,dashed] (2, 3) rectangle (10, 11);

This means that each visibility is processed by multiple work-groups.
Specifically, the number of work-groups loading a visibility is the bin size
over the tile size.

Each thread maintains a number of accumulators: one per position within a
block, per polarization. When moving to the next visibility, any counters that
now correspond to a grid element outside the footprint are flushed to global
memory (with an atomic add), and reset to a zero value at the new position
that falls inside the footprint. Provided that the footprint only moves
slowly, this reduces memory traffic.

Visibilities are loaded in *batches*, whose size equals the number of
workitems in a workgroup. First, all workitems cooperatively load a batch,
preprocess it to determine pixel and subpixel coordinates, and store the
values in shared memory. Then, each thread iterates over the visibilities in
the batch. This amortises the costs of the global memory loads and address
computations.

Degridding
----------
Degridding is done similarly to gridding, with a few significant changes:

- Each workgroup is split into "subgroups", each of which functions
  independently as if it were a workgroup. This is done to allow for good
  performance even when the subgroup size is below the minimum workgroup size
  to achieve full occupancy on NVIDIA GPUs (this is not an issue on AMD GCN
  GPUs, which can achieve full occupancy a wavefront per workgroup). Smaller
  subgroups reduce the cost of reductions, particular if they are warp-sized
  and can use warp shuffle instructions. On the other hand, the smaller the
  subgroup, the fewer visibilities can be loaded in each batch.
- Each subgroup loops over the tiles in a bin, instead of the tiles being
  split into different subgroups. This allows each visibility to be completely
  calculated in local memory with no atomics, rather than accumulated in
  global memory. On the other hand, it reduces parallelism. Note, this
  variation is also being considered for gridding.

Future planned changes
----------------------
At present, the footprint is allowed to move one grid cell at a time. If it
were forced to align to the blocks, then a number of address
computations could be amortized over a whole block. This would require that
the kernel function be padded slightly (by one less than the block size) so
that the support of the function always falls within the footprint, even when
the footprint has been snapped to the block grid.

There are two possible ways to handle tiles. The current implementation puts
each tile position into a different workgroup. This has the advantage that it
is not necessary to flush all the accumulators at the end of each batch, but
the disadvantage that every visibility is loaded from global memory multiple
times. A possible alternative (for which further analysis would be needed) is
to load a batch into shared memory, and then process multiple tiles in series
inside the workgroup, reusing the values in shared memory each time.

Another reason to use separate workgroups for each tile is that it may be
possible to eliminate atomic operations on small devices (such as a Tegra K1
or X1) by having each workgroup process *all* the visibilities, rather than
splitting them up to create more workgroups. This eliminates the potential for
race conditions, because each workgroup is handling a disjoint part of the
grid. This does limit the amount of available parallelism to the number of
blocks per bin, which is why it is probably only practical on embedded
devices, and even then probably requires a block size of 1.

.. [Rom12] Romein, John W. 2012. An efficient work-distribution strategy for
   gridding radio-telescope data on GPUs. In *Proceedings of the 26th ACM International
   Conference on Supercomputing (ICS '12)*, 321-330.
   http://www.astron.nl/~romein/papers/ICS-12/gridding.pdf
"""

from __future__ import division, print_function
from . import parameters
import numpy as np
import math
import pkg_resources
import astropy.units as units
import katsdpsigproc.accel as accel
import katsdpsigproc.tune as tune
import katsdpimager.types
import numba
import logging


logger = logging.getLogger(__name__)


def kaiser_bessel(x, width, beta):
    r"""Evaluate Kaiser-Bessel window function. Refer to
    http://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html
    for details.

    Parameters
    ----------
    x : array-like, float
        Sample positions
    width : float
        The kernel has support :math:`[-\frac{1}{2}W, \frac{1}{2}W]`
    beta : float
        Shape parameter
    """
    param = 1 - (2 * x / width)**2
    # The np.maximum is to protect against runtime warnings for taking
    # sqrt of negative values. The actual values in this situation are
    # irrelvent due to the np.select call.
    values = np.i0(beta * np.sqrt(np.maximum(0, param))) / np.i0(beta)
    return np.select([param >= 0], [values])


def kaiser_bessel_fourier(f, width, beta, out=None):
    r"""
    Evaluate the continuous Fourier transform of :func:`kaiser_bessel`.
    Note that since the function is even and real, this is also the inverse
    Fourier transform.

    Parameters
    ----------
    f : array-like, float
        Sample positions (frequency)
    width : float
        The kernel has support :math:`[-\frac{1}{2}W, \frac{1}{2}W]`
    beta : float
        Shape parameter
    out : array-like, float
        If specified, result is written into it
    """
    alpha = beta / math.pi
    # np.lib.scimath.sqrt returns complex values for negative inputs, whereas
    # np.sqrt returns NaN. We take the real component because np.sinc returns
    # complex if it has a complex input, even though the imaginary part is 0.
    ans = width / np.i0(beta) * np.sinc(np.lib.scimath.sqrt((width * f)**2 - alpha * alpha)).real
    if out is not None:
        out[:] = ans
        return out
    else:
        return ans


def antialias_kernel(width, oversample, beta=None):
    r"""Generate 1D anti-aliasing kernel. The return value is 2-dimensional.
    The first dimension selects the subpixel position, the second
    selects the pixel position.

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

    Parameters
    ----------
    width : float
        Kernel support
    oversample : int
        Number of samples per unit step
    beta : float, optional
        Shape parameter for Kaiser-Bessel window
    """
    if beta is None:
        # Puts the first null of the taper function at the edge of the image
        beta = math.pi * math.sqrt(0.25 * width**2 - 1.0)
        # Move the null outside the image, to avoid numerical instabilities.
        # This will cause a small amount of aliasing at the edges, which
        # ideally should be handled by clipping the image.
        beta *= 1.2
    hsize = int(math.ceil(0.5 * width))
    size = 2 * hsize
    # The kernel only contains real values, but we store complex so that we
    # are ready for W projection.
    kernel = np.empty((oversample, size), np.complex64)
    for s in range(oversample):
        bias = (s + 0.5) / oversample + hsize - 1
        kernel[s, :] = kaiser_bessel(np.arange(size) - bias, width, beta)
    return kernel


def antialias_w_kernel(
        cell_wavelengths, w, width,
        oversample, antialias_width, image_oversample, beta,
        out=None):
    r"""Computes a combined anti-aliasing and W-projection kernel.

    The format of the returned kernel is similar to :func:`antialias_kernel`.
    In particular, only a 1D kernel is returned.  Note that while the W kernel
    is not truly separable, the small-angle approximation
    :math:`\sqrt{1-l^2-m^2}-1 \approx (\sqrt{1-l^2}-1) + (\sqrt{1-m^2}-1)`
    makes it makes it very close to separable [#]_.

    .. [#] The error in the approximation above is on the order of :math:`10^{-8}`
       for a 1 degree field of view.

    We multiply the inverse Fourier transform of the (idealised) antialiasing
    kernel with the W correction in image space, to obtain a closed-form
    function we can evaluate in image space. This is then Fourier-transformed
    to UV space. To reduce aliasing, the image-space function is oversampled by
    a factor of `uv_oversample`, and the transform result is then truncated.

    All the continuous functions that we are approximating in UV space are
    expressed as functions of the number of wavelengths.

    The output kernel is sampled at the centres of bins, which are at
    half-subpixel offsets. To create this shift in UV space, we multiply by the
    appropriate complex exponential in image space.

    Parameters
    ----------
    cell_wavelengths : float
        Size of a UV cell in wavelengths
    w : float
        w component of baseline, in wavelengths
    width : int
        Support of combined kernel, in cells
    oversample : int
        Number of samples per unit step in UV space
    antialias_width : float
        Support for anti-alias kernel, in cells
    image_oversample : int
        Oversampling factor in image space. Larger values reduce aliasing in
        the output and increase time and memory required for computation, but
        do not affect the size of the resulting kernel.
    beta : float, optional
        Shape parameter for Kaiser-Bessel window
    out : array-like, optional
        If specified, the output is returned in this array
    """
    def image_func(l):
        # The kaiser_bessel function is designed around units of cells rather
        # than wavelengths. We thus want the Fourier transform of
        # kaiser_bessel(u / cell_wavelengths).
        aa_factor = cell_wavelengths * kaiser_bessel_fourier(l * cell_wavelengths, antialias_width, beta)
        shift_arg = shift_by * l
        w_arg = -w * (np.sqrt(1 - l*l) - 1)
        return aa_factor * np.exp(2j * math.pi * (w_arg + shift_arg))

    out_pixels = oversample * width
    assert out_pixels % 2 == 0, "Odd number of pixels is not tested"
    pixels = out_pixels * image_oversample
    # Convert uv-space width to wavelengths
    uv_width = width * cell_wavelengths * image_oversample
    # Compute other support and step sizes
    image_step = 1 / uv_width
    uv_step = uv_width / pixels
    image_width = image_step * pixels
    # Determine sample points in image space
    l = (np.arange(pixels) - (pixels // 2)) * image_step
    # Evaluate function in image space
    shift_by = -0.5 * cell_wavelengths / oversample
    image_values = image_func(l)
    # Convert to UV space. The multiplication is because we're using a DFT to
    # approximate a continuous FFT.
    uv_values = np.fft.fft(np.fft.ifftshift(image_values)) * image_step
    # Crop to area of interest, and swap halves to put DC in the middle
    uv_values = np.concatenate((uv_values[-(out_pixels // 2):], uv_values[:(out_pixels // 2)]))
    # Split up into subkernels. Since the subpixel index indicates the subpixel
    # position of the visibility, rather than the kernel tap, it runs backwards
    # in the kernel indexing.
    kernel = np.reshape(uv_values, (oversample, width), order='F')[::-1, :]
    # Convert to C memory layout
    if out is None:
        out = np.empty_like(kernel)
    out[:] = kernel
    return out


@numba.jit(nopython=True)
def subpixel_coord(x, oversample):
    """Return pixel and subpixel index, as described in :func:`antialias_kernel`"""
    xs = int(np.floor(x * oversample))
    return xs // oversample, xs % oversample


class ConvolutionKernel(object):
    """Separable convolution kernel with metadata. The kernel combines anti-aliasing
    with W correction.

    Parameters
    ----------
    image_parameters : :class:`katsdpimager.parameters.ImageParameters`
        Command-line parameters with image properties
    grid_parameters : :class:`katsdpimager.parameters.GridParameters`
        Gridding parameters
    data : array-like
        If specified, used as a backing data store. Otherwise, a plain numpy
        array is used.
    """
    def __init__(self, image_parameters, grid_parameters, data=None):
        self.grid_parameters = grid_parameters
        if data is None:
            self.data = np.empty(
                (grid_parameters.w_planes, grid_parameters.oversample, grid_parameters.kernel_width),
                np.complex64)
        else:
            self.data = data
        cell_wavelengths = float(image_parameters.cell_size / image_parameters.wavelength)
        # Separation in w between slices
        w_slice_wavelengths = float(grid_parameters.max_w / (grid_parameters.w_slices * image_parameters.wavelength))
        # Separation in w between planes
        w_plane_wavelengths = w_slice_wavelengths / grid_parameters.w_planes
        # Puts the first null of the taper function at the edge of the image
        self.beta = math.pi * math.sqrt(0.25 * grid_parameters.antialias_width**2 - 1.0)
        # Move the null outside the image, to avoid numerical instabilities.
        # This will cause a small amount of aliasing at the edges, which
        # ideally should be handled by clipping the image.
        self.beta *= 1.2
        # TODO: use sqrt(w) scaling as in Cornwell, Golap and Bhatnagar (2008)?
        # TODO: can halve work and memory by exploiting conjugate symmetry
        # Find w for the midpoint of the final plane
        max_w_wavelengths = (w_slice_wavelengths - w_plane_wavelengths) * 0.5
        for i, w in enumerate(np.linspace(-max_w_wavelengths, max_w_wavelengths, grid_parameters.w_planes)):
            antialias_w_kernel(
                cell_wavelengths, w, grid_parameters.kernel_width,
                grid_parameters.oversample,
                grid_parameters.antialias_width,
                grid_parameters.image_oversample,
                self.beta, out=self.data[i, ...])

    @classmethod
    def bin_size(cls, grid_parameters, tile_x, tile_y, pad):
        """Determine appropriate bin size given alignment restrictions."""
        size = max(tile_x, tile_y)
        # Round up to a power of 2
        while size < grid_parameters.kernel_width + pad:
            size *= 2
        if size != grid_parameters.kernel_width + pad:
            logger.info("kernel size rounded up to %d", size)
        assert size % tile_x == 0
        assert size % tile_y == 0
        return size

    def taper(self, N, out=None):
        """Return the Fourier transform of the antialiasing kernel for
        an N-pixel 1D image.

        Parameters
        ----------
        N : int
            Number of pixels in the image
        out : array-like, optional
            If provided, is used to store the result
        """
        x = np.arange(N) / N - 0.5
        return kaiser_bessel_fourier(x, self.grid_parameters.antialias_width, self.beta, out)


class ConvolutionKernelDevice(ConvolutionKernel):
    """A :class:`ConvolutionKernel` that stores data in a device SVM array."""
    def __init__(self, context, image_parameters, grid_parameters, pad=0):
        out = accel.SVMArray(context,
            (grid_parameters.w_planes,
             grid_parameters.oversample,
             grid_parameters.kernel_width + 2 * pad),
            np.complex64)
        out.fill(0)
        super(ConvolutionKernelDevice, self).__init__(
            image_parameters, grid_parameters, out[:, :, pad:grid_parameters.kernel_width + pad])
        self.padded_data = out
        self.pad = pad

    @property
    def bin_size(self):
        return self.data.shape[-1] + self.pad

    def parameters(self):
        """Parameters for templating CUDA/OpenCL kernels"""
        w_scale = float(units.m / self.grid_parameters.max_w) * (self.grid_parameters.w_planes - 1)
        return {
            'convolve_kernel_slice_stride':
                self.padded_data.padded_shape[2],
            'convolve_kernel_oversample': self.data.shape[1],
            'convolve_kernel_w_stride': np.product(self.padded_data.padded_shape[1:]),
            'convolve_kernel_w_scale': w_scale,
            'convolve_kernel_max_w': float(self.grid_parameters.max_w / units.m),
            'bin_x': self.bin_size,
            'bin_y': self.bin_size,
        }


class GridderTemplate(object):
    autotune_version = 1

    def __init__(self, context, image_parameters, grid_parameters, tuning=None):
        if tuning is None:
            tuning = self.autotune(
                context, grid_parameters.antialias_width, grid_parameters.oversample,
                len(image_parameters.polarizations))
        self.grid_parameters = grid_parameters
        self.image_parameters = image_parameters
        # These must be powers of 2. TODO: autotune
        self.wgs_x = 16
        self.wgs_y = 16
        self.multi_x = 2
        self.multi_y = 2
        min_pad = max(self.multi_x, self.multi_y) - 1
        self.tile_x = self.wgs_x * self.multi_x
        self.tile_y = self.wgs_y * self.multi_y
        bin_size = ConvolutionKernel.bin_size(grid_parameters,
            self.tile_x, self.tile_y, min_pad)
        pad = bin_size - grid_parameters.kernel_width
        self.convolve_kernel = ConvolutionKernelDevice(
            context, image_parameters, grid_parameters, pad)
        real_dtype = katsdpimager.types.complex_to_real(image_parameters.complex_dtype)
        parameters = {
            'real_type': katsdpimager.types.dtype_to_ctype(real_dtype),
            'multi_x': self.multi_x,
            'multi_y': self.multi_y,
            'wgs_x': self.wgs_x,
            'wgs_y': self.wgs_y,
            'num_polarizations': len(image_parameters.polarizations)
        }
        parameters.update(self.convolve_kernel.parameters())
        self.program = accel.build(
            context, "imager_kernels/grid.mako", parameters,
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    @classmethod
    @tune.autotuner(test={})
    def autotune(cls, context, antialias_width, oversample, num_polarizations):
        # Nothing to autotune yet
        return {}

    def instantiate(self, *args, **kwargs):
        return Gridder(self, *args, **kwargs)


class GridDegrid(accel.Operation):
    """Base class for :class:`Grid` and :class:`Degrid`.

    .. rubric:: Slots

    **grid** : array of pols × height × width, complex
        Grid (output for :class:`Grid`, input for :class:`Degrid`)
    **uv** : array of int16×4
        The first two elements for each visibility are the
        UV coordinates of the first grid cell to be updated. The other two are
        the subpixel U and V coordinates.
    **w_plane** : array of int16
        W plane index per visibility, clamped to the range of allocated w planes
    **vis** : array of complex64 × pols
        Visibilities. For gridding these are pre-multiplied by weights (input),
        while for degridding they are unweighted (output).

    Parameters
    ----------
    template : :class:`GridTemplate` or :class:`DegridTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    array_parameters : :class:`~katsdpimager.parameters.ArrayParameters`
        Array parameters
    max_vis : int
        Number of visibilities that can be supported per kernel invocation
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Raises
    ------
    ValueError
        If the longest baseline is too large to find within the grid
    """
    def __init__(self, template, command_queue, array_parameters,
                 max_vis, allocator=None):
        super(GridDegrid, self).__init__(command_queue, allocator)
        # Check that longest baseline won't cause an out-of-bounds access
        max_uv_src = float(array_parameters.longest_baseline / template.image_parameters.cell_size)
        convolve_kernel_size = template.convolve_kernel.padded_data.shape[-1]
        max_uv = max_uv_src + convolve_kernel_size / 2
        if max_uv >= template.image_parameters.pixels // 2 - 1 - 1e-3:
            raise ValueError('image_oversample is too small to capture all visibilities in the UV plane')
        self.template = template
        self.max_vis = max_vis
        num_polarizations = len(template.image_parameters.polarizations)
        pixels = template.image_parameters.pixels
        self.slots['grid'] = accel.IOSlot(
            (num_polarizations, pixels, pixels),
            template.image_parameters.complex_dtype)
        self.slots['uv'] = accel.IOSlot(
            (max_vis, accel.Dimension(4, exact=True)), np.int16)
        self.slots['w_plane'] = accel.IOSlot((max_vis,), np.int16)
        self.slots['vis'] = accel.IOSlot(
            (max_vis, accel.Dimension(num_polarizations, exact=True)), np.complex64)
        self._num_vis = 0

    @property
    def num_vis(self):
        return self._num_vis

    @num_vis.setter
    def num_vis(self, n):
        """Change the number of actual visibilities stored in the buffers."""
        if n < 0 or n > self.max_vis:
            raise ValueError('Number of visibilities {} is out of range 0..{}'.format(
                n, self.max_vis))
        self._num_vis = n

    def parameters(self):
        return {
            'grid_parameters': self.template.grid_parameters,
            'image_parameters': self.template.image_parameters
        }


class Gridder(GridDegrid):
    """Instantiation of :class:`GridderTemplate`. See :class:`GridDegrid` for
    details.
    """

    def __init__(self, *args, **kwargs):
        super(Gridder, self).__init__(*args, **kwargs)
        self._kernel = self.template.program.get_kernel('grid')

    def grid(self, uv, sub_uv, w_plane, vis):
        """Add visibilities to the grid, with convolutional gridding using the
        anti-aliasing filter."""
        N = len(uv)
        if len(sub_uv) != N or len(w_plane) != N or len(vis) != N:
            raise ValueError('Lengths do not match')
        self.num_vis = N
        self.buffer('uv')[:N, 0:2] = uv
        self.buffer('uv')[:N, 2:4] = sub_uv
        self.buffer('w_plane')[:N] = w_plane
        self.buffer('vis')[:N] = vis
        return self()

    def _run(self):
        if self.num_vis == 0:
            return
        grid = self.buffer('grid')
        workgroups = 256  # TODO: tune this in some way
        vis_per_workgroup = accel.divup(self.num_vis, workgroups)
        kernel_width = self.template.grid_parameters.kernel_width
        bin_size = self.template.convolve_kernel.bin_size
        tiles_x = bin_size // self.template.tile_x
        tiles_y = bin_size // self.template.tile_y
        uv_bias = (kernel_width - 1) // 2 + self.template.convolve_kernel.pad
        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                grid.buffer,
                np.int32(grid.padded_shape[2]),
                np.int32(grid.padded_shape[1] * grid.padded_shape[2]),
                self.buffer('uv').buffer,
                self.buffer('w_plane').buffer,
                self.buffer('vis').buffer,
                self.template.convolve_kernel.padded_data.buffer,
                np.int32(uv_bias),
                np.int32(vis_per_workgroup),
                np.int32(self.num_vis)
            ],
            global_size=(self.template.wgs_x * workgroups,
                         self.template.wgs_y * tiles_x,
                         tiles_y),
            local_size=(self.template.wgs_x, self.template.wgs_y, 1)
        )


class DegridderTemplate(object):
    def __init__(self, context, image_parameters, grid_parameters, tuning=None):
        # TODO: autotuning
        self.grid_parameters = grid_parameters
        self.image_parameters = image_parameters
        self.wgs_x = 8
        self.wgs_y = 4
        self.wgs_z = 4
        self.multi_x = 2
        self.multi_y = 2
        self.tile_x = self.wgs_x * self.multi_x
        self.tile_y = self.wgs_y * self.multi_y
        min_pad = max(self.multi_x, self.multi_y) - 1
        bin_size = ConvolutionKernel.bin_size(grid_parameters, self.tile_x, self.tile_y, min_pad)
        pad = bin_size - grid_parameters.kernel_width
        # Note: we can't necessarily use the same kernel as for gridding,
        # because different tuning parameters will affect the kernel size.
        # TODO: fix this so that the memory support is decoupled from the
        # mathematical support.
        self.convolve_kernel = ConvolutionKernelDevice(
            context, image_parameters, grid_parameters, pad)
        real_dtype = katsdpimager.types.complex_to_real(image_parameters.complex_dtype)
        parameters = {
            'real_type': katsdpimager.types.dtype_to_ctype(real_dtype),
            'multi_x': self.multi_x,
            'multi_y': self.multi_y,
            'wgs_x': self.wgs_x,
            'wgs_y': self.wgs_y,
            'wgs_z': self.wgs_z,
            'num_polarizations': len(image_parameters.polarizations)
        }
        parameters.update(self.convolve_kernel.parameters())
        self.program = accel.build(
            context, "imager_kernels/degrid.mako", parameters,
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return Degridder(self, *args, **kwargs)


class Degridder(GridDegrid):
    """Instantiation of :class:`DegridderTemplate`. See :class:`GridDegrid` for
    details.
    """

    def __init__(self, *args, **kwargs):
        super(Degridder, self).__init__(*args, **kwargs)
        self._kernel = self.template.program.get_kernel('degrid')

    def degrid(self, uv, sub_uv, w_plane, vis):
        """Degrid visibilities into `vis`."""
        N = len(uv)
        if len(sub_uv) != N or len(w_plane) != N or len(vis) != N:
            raise ValueError('Lengths do not match')
        self.num_vis = N
        self.buffer('uv')[:N, 0:2] = uv
        self.buffer('uv')[:N, 2:4] = sub_uv
        self.buffer('w_plane')[:N] = w_plane
        self()
        self.command_queue.finish()
        vis[:] = self.buffer('vis')[:N]

    def _run(self):
        if self.num_vis == 0:
            return
        grid = self.buffer('grid')
        workgroups = 256   # TODO: tune this in some way
        subgroups = workgroups * self.template.wgs_z
        vis_per_subgroup = accel.divup(self.num_vis, subgroups)
        kernel_width = self.template.grid_parameters.kernel_width
        bin_size = self.template.convolve_kernel.bin_size
        uv_bias = (kernel_width - 1) // 2 + self.template.convolve_kernel.pad
        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                grid.buffer,
                np.int32(grid.padded_shape[2]),
                np.int32(grid.padded_shape[1] * grid.padded_shape[2]),
                self.buffer('uv').buffer,
                self.buffer('w_plane').buffer,
                self.buffer('vis').buffer,
                self.template.convolve_kernel.padded_data.buffer,
                np.int32(uv_bias),
                np.int32(vis_per_subgroup),
                np.int32(self.num_vis)
            ],
            global_size=(self.template.wgs_x * workgroups,
                         self.template.wgs_y,
                         self.template.wgs_z),
            local_size=(self.template.wgs_x, self.template.wgs_y, self.template.wgs_z)
        )


@numba.jit(nopython=True)
def _grid(kernel, values, uv, sub_uv, w_plane, vis, sample):
    """Internal implementation of :meth:`GridderHost.grid`, split out so that
    Numba can JIT it.
    """
    ksize = kernel.shape[2]
    uv_bias = (ksize - 1) // 2
    for row in range(uv.shape[0]):
        u0 = uv[row, 0] - uv_bias
        v0 = uv[row, 1] - uv_bias
        sub_u, sub_v = sub_uv[row]
        for i in range(vis.shape[1]):
            sample[i] = vis[row, i]
        # u and v are converted to cells, w to planes
        for j in range(ksize):
            for k in range(ksize):
                kernel_sample = kernel[w_plane[row], sub_v, j] * kernel[w_plane[row], sub_u, k]
                weight = np.conj(kernel_sample)
                for pol in range(values.shape[0]):
                    values[pol, int(v0 + j), int(u0 + k)] += sample[pol] * weight


class GridderHost(object):
    def __init__(self, image_parameters, grid_parameters):
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        kernel_size = int(math.ceil(grid_parameters.kernel_width))
        self.kernel = ConvolutionKernel(image_parameters, grid_parameters, kernel_size)
        pixels = image_parameters.pixels
        shape = (len(image_parameters.polarizations), pixels, pixels)
        self.values = np.empty(shape, image_parameters.complex_dtype)

    def clear(self):
        self.values.fill(0)

    def grid(self, uv, sub_uv, w_plane, vis):
        """Add visibilities to the grid, with convolutional gridding.

        Parameters
        ----------
        uv : 2D array, integer
            Preprocessed grid UV coordinates
        sub_uv : 2D array, integer
            Preprocessed grid UV sub-pixel coordinates
        w_plane : 1D array, integer
            Preprocessed grid W plane coordinates
        vis : 2D ndarray of complex or real
            Visibility data, indexed by sample and polarization, and
            pre-multiplied by all weights
        """
        _grid(self.kernel.data, self.values,
              uv, sub_uv, w_plane, vis,
              np.empty((vis.shape[1],), self.values.dtype))
