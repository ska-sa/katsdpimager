# -*- coding: utf-8 -*-

r"""Convolutional gridding with W projection.

Gridding
--------
The GPU approach is based on [Rom12]_. Each workgroup (thread block) handles a
contiguous range of visibilities, with the threads cooperating to grid each
visibility.  The grid is divided into *bins*. The size of a bin is the size of
the convolution kernel, plus some padding. These bins are further divided into
*tiles*, which are divided into *blocks*. A workitem (thread) handles all
blocks that are at the same offset within their bin. Similarly, a workgroup
handles all tiles that are at the same offset within their bin. At present,
the blocks, tiles and bins are all power-of-two sizes.

For each visibility, the kernel footprint is found, and a bin-sized,
block-aligned bounding box is placed around it. The requirement to be able to
block-align this rectangle means that bins may need to be slightly larger than
the kernel. Every grid point into this box will be updated for this
visibility, and hence the kernel function needs zero padding (on all sides) of
the same amount as the bin padding.

.. tikz:: Mapping of workitems and workgroups to grid points. One workgroup
   contributes to all the green cells. One workitem contributes to all the
   cells marked with a red dot. Here blocks are 2×2, tiles are 4×4 and bins
   are 8×8. The dashed lines show a kernel footprint, and the blue box shows
   the bounding rectangle.

   [x=0.5cm, y=0.5cm]
   \foreach \i in {4, 12}
       \foreach \j in {0, 8}
       {
           \fill[shift={(\i,\j)},green!50!white] (0, 0) rectangle (4, 4);
       }
   \foreach \i in {4,5, 12, 13}
       \foreach \j in {2,3, 10,11}
       {
           \node[shift={(\i,\j)},circle,inner sep=0pt, minimum size=0.2cm,fill=red]
               at (0.5, 0.5) {};
       }
   \draw[help lines] (0, 0) grid[step=0.5cm] (16, 16);
   \draw[very thick] (0, 0) grid[step=4cm] (16, 16);
   \draw[very thick,blue] (2, 2) rectangle (10, 10);
   \draw[very thick,dashed] (2, 3) rectangle (9, 10);

This means that each visibility is processed by multiple work-groups.
Specifically, the number of work-groups loading a visibility is the bin size
over the tile size.

Each thread maintains a number of accumulators: one per position within a
block, per polarization. When moving to the next visibility, any counters that
now correspond to a grid element outside the bounding box are flushed to global
memory (with an atomic add), and reset to a zero value at the new position
that falls inside the bounding box. Provided that the bounding box only moves
slowly, this reduces memory traffic. Note that because the bounding box is
block-aligned, each block is written all at the same time. This allows a lot
of coordinate calculations to be done per-block instead of per-cell, which
improves performance.

Visibilities are loaded in *batches*, whose size equals the number of
workitems in a workgroup. First, all workitems cooperatively load a batch and
store the values in shared memory. Then, each workitem iterates over the
visibilities in the batch. This amortises the costs of the global memory loads
and address computations.

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

Weights
-------
There are two sources of weighting: statistical weights and imaging weights
(the latter combining density and taper weights) [Bri95]_. The statistical
weights are pre-multiplied into the visibilities before gridding (see
:py:mod:`.preprocess`), and are handled outside this module for degridding. The
imaging weights are stored in a grid and multiplied by the visibilities as
they are loaded, and have no effect on degridding.

Future planned changes
----------------------
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
devices, and even then may limit the block size.

.. [Rom12] Romein, John W. 2012. An efficient work-distribution strategy for
   gridding radio-telescope data on GPUs. In *Proceedings of the 26th ACM International
   Conference on Supercomputing (ICS '12)*, 321-330.
   http://www.astron.nl/~romein/papers/ICS-12/gridding.pdf

.. include:: macros.rst
"""

import math
import logging

import numpy as np
import pkg_resources

import katsdpsigproc.accel as accel
import katsdpsigproc.tune as tune

import katsdpimager.types
from . import numba


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
    # irrelevent due to the np.select call.
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
    :math:`\sqrt{1-l^2-m^2}-1 \approx -\frac{1}{2}(l^2+m^2)-\frac{5}{24}(l^4+m^4)`
    makes it very close to separable [#]_.

    .. [#] The error in the approximation above is on the order of :math:`5\times 10^{-6}`
       for a 10 degree field of view, and less than :math:`10^{-8}` for a 2 degree FOV.

    We multiply the inverse Fourier transform of the (idealised) antialiasing
    kernel with the W correction in image space, to obtain a closed-form
    function we can evaluate in image space. This is then Fourier-transformed
    to UV space. To reduce aliasing, the image-space function is oversampled by
    a factor of `image_oversample`, and the transform result is then truncated.

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
        scale_l = l * cell_wavelengths
        aa_factor = cell_wavelengths * kaiser_bessel_fourier(scale_l, antialias_width, beta)
        shift_arg = shift_by * l
        l2 = l * l
        l4 = l2 * l2
        w_arg = -w * (-0.5 * l2 - 5.0/24.0 * l4)
        return aa_factor * np.exp(2j * math.pi * (w_arg + shift_arg))

    out_pixels = oversample * width
    assert out_pixels % 2 == 0, "Odd number of pixels is not tested"
    pixels = out_pixels * image_oversample
    # Convert uv-space width to wavelengths
    uv_width = width * cell_wavelengths * image_oversample
    # Compute other support and step sizes
    image_step = 1 / uv_width
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


class ConvolutionKernel:
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
        shape = (grid_parameters.w_planes, grid_parameters.oversample, grid_parameters.kernel_width)
        if data is None:
            self.data = np.empty(shape, np.complex64)
        else:
            self.data = data
        cell_wavelengths = float(image_parameters.cell_size / image_parameters.wavelength)
        # Separation in w between slices
        w_slice_wavelengths = float(grid_parameters.max_w
                                    / (grid_parameters.w_slices * image_parameters.wavelength))
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
        ws = np.linspace(-max_w_wavelengths, max_w_wavelengths, grid_parameters.w_planes)
        for i, w in enumerate(ws):
            antialias_w_kernel(
                cell_wavelengths, w, grid_parameters.kernel_width,
                grid_parameters.oversample,
                grid_parameters.antialias_width,
                grid_parameters.image_oversample,
                self.beta, out=self.data[i, ...])

    @classmethod
    def get_bin_size(cls, grid_parameters, tile_x, tile_y, pad):
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

        The combined filter is sampled and then applied as a piecewise constant
        function, which corresponds to convolution with a rect. In image space,
        this corresponds to aliasing, then multiplication by a sinc. We need to
        include the sinc in the Fourier transform.

        Parameters
        ----------
        N : int
            Number of pixels in the image
        out : array-like, optional
            If provided, is used to store the result
        """
        x = np.arange(N) / N - 0.5
        out = kaiser_bessel_fourier(x, self.grid_parameters.antialias_width, self.beta, out)
        out *= np.sinc(x / self.grid_parameters.oversample)
        return out


class ConvolutionKernelDevice(ConvolutionKernel):
    """A :class:`ConvolutionKernel` that stores data in a device array."""
    def __init__(self, context, image_parameters, grid_parameters, pad=0, allocator=None):
        if allocator is None:
            allocator = accel.DeviceAllocator(context)
        out = allocator.allocate(
            (grid_parameters.w_planes,
             grid_parameters.oversample,
             grid_parameters.kernel_width + 2 * pad),
            np.complex64)
        if isinstance(out, accel.SVMArray):
            host = out
        else:
            host = out.empty_like()
        host.fill(0)
        super().__init__(
            image_parameters, grid_parameters, host[:, :, pad:grid_parameters.kernel_width + pad])
        if not isinstance(out, accel.SVMArray):
            queue = context.create_command_queue()
            out.set(queue, host)
        self.padded_data = out
        self.pad = pad

    @property
    def bin_size(self):
        return self.data.shape[-1] + self.pad

    def parameters(self):
        """Parameters for templating CUDA/OpenCL kernels"""
        return {
            'convolve_kernel_slice_stride':
                self.padded_data.padded_shape[2],
            'convolve_kernel_w_stride': np.product(self.padded_data.padded_shape[1:]),
            'bin_x': self.bin_size,
            'bin_y': self.bin_size,
        }


def _autotune_uv(context, pixels, bin_size, oversample):
    """Creates UV test data for autotuning gridding and degridding.

    The visibilities and weights are irrelevant, but the UV coordinates are
    important because the performance is heavily data-dependent. Compression
    ensures that samples in the same subgrid cell are compressed, but each
    sample is typically close to the previous one, moving along an elliptic
    track. Rather than go to the full complexity of modelling such tracks, we
    use a number of linear tracks at a variety of slopes.

    The return value has the format required by the *uv* slot in
    :class:`Gridder`.
    """
    # Set bounds of the UV coordinates in pixels. This is a slightly
    # conservative bound.
    low = bin_size
    high = pixels - bin_size
    track_length = (high - low) * oversample
    tracks = 256 * 1024 // track_length
    out = np.empty((tracks, track_length, 4), np.int16)
    for i in range(tracks):
        angle = 2 * math.pi * i / tracks
        dx = math.cos(angle)
        dy = math.sin(angle)
        scale = max(abs(dx), abs(dy))
        # step is either (1, slope), (-1, slope), (slope, 1) or (slope, -1)
        step = np.array([dx / scale, dy / scale])
        indices = np.arange(track_length) - track_length / 2
        sub_uv = np.outer(indices, step)
        sub_uv = np.round(sub_uv).astype(np.int32)
        out[i, :, 0:2] = sub_uv // oversample
        out[i, :, 2:4] = sub_uv % oversample
    # Flatten the separate tracks into one dimension
    return out.reshape((-1, 4))


def _autotune_arrays(command_queue, oversample, real_dtype, num_polarizations, bin_size, pad):
    """Creates device arrays for autotuning gridding and degridding. The
    UV coordinates are provided by :func:`_autotune_uv`, while the other
    arrays are zeroed out.

    Parameters
    ----------
    command_queue : |CommandQueue|
        Command queue for zeroing the arrays
    oversample : int
        Grid oversampling
    real_dtype : {np.float32, np.complex64}
        Floating-point type for grid value
    num_polarizations : int
        Number of polarizations in grid
    bin_size : int
        Bin size
    pad : int
        Amount of padding around edges of convolution kernel to allow for
        blocking. This must be at greater than or equal to the maximum
        block size (X or Y) minus one.

    Returns
    -------
    grid,weights_grid,uv,w_plane,vis,convolve_kernel : :class:`katsdpsigproc.accel.DeviceArray`
        Device arrays to be passed to the kernels
    """
    context = command_queue.context
    pixels = 256
    complex_dtype = katsdpimager.types.real_to_complex(real_dtype)
    uv_host = _autotune_uv(context, pixels, bin_size, oversample)
    num_vis = len(uv_host)
    uv = accel.DeviceArray(context, uv_host.shape, np.int16)
    grid = accel.DeviceArray(context, (num_polarizations, pixels, pixels), complex_dtype)
    weights_grid = accel.DeviceArray(context, (num_polarizations, pixels, pixels), np.float32)
    w_plane = accel.DeviceArray(context, (num_vis,), np.int16)
    vis = accel.DeviceArray(context, (num_vis, num_polarizations), np.complex64)
    convolve_kernel = accel.DeviceArray(context, (1, oversample, bin_size + pad), np.complex64)
    uv.set(command_queue, uv_host)
    grid.zero(command_queue)
    weights_grid.zero(command_queue)
    w_plane.zero(command_queue)
    vis.zero(command_queue)
    convolve_kernel.zero(command_queue)
    return grid, weights_grid, uv, w_plane, vis, convolve_kernel


class GridderTemplate:
    autotune_version = 6

    def __init__(self, context, image_parameters, grid_parameters, tuning=None):
        if tuning is None:
            tuning = self.autotune(
                context, grid_parameters.oversample, image_parameters.real_dtype,
                len(image_parameters.polarizations))
        self.grid_parameters = grid_parameters
        self.image_parameters = image_parameters
        self.wgs_x = tuning['wgs_x']
        self.wgs_y = tuning['wgs_y']
        self.multi_x = tuning['multi_x']
        self.multi_y = tuning['multi_y']
        min_pad = max(self.multi_x, self.multi_y) - 1
        self.tile_x = self.wgs_x * self.multi_x
        self.tile_y = self.wgs_y * self.multi_y
        bin_size = ConvolutionKernel.get_bin_size(
            grid_parameters, self.tile_x, self.tile_y, min_pad)
        pad = bin_size - grid_parameters.kernel_width
        self.convolve_kernel = ConvolutionKernelDevice(
            context, image_parameters, grid_parameters, pad)
        parameters = {
            'real_type': katsdpimager.types.dtype_to_ctype(image_parameters.real_dtype),
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
    @tune.autotuner(test={'wgs_x': 16, 'wgs_y': 8, 'multi_x': 2, 'multi_y': 2})
    def autotune(cls, context, oversample, real_dtype, num_polarizations):
        # This autotuning code is unusually complex because it tries to avoid
        # actually creating the convolution kernels. Doing so is slow, would
        # require fake values of many parameters to be carefully chosen, and
        # could potentially lead to different bin sizes for different tuning
        # parameters. Instead, we directly build the program and data
        # structures.
        queue = context.create_tuning_command_queue()
        bin_size = 32
        pad = 3
        grid, weights_grid, uv, w_plane, vis, convolve_kernel = _autotune_arrays(
            queue, oversample, real_dtype, num_polarizations, bin_size, pad)
        num_vis = uv.shape[0]

        def generate(multi_x, multi_y, wgs_x, wgs_y):
            assert multi_x <= pad + 1 and multi_y <= pad + 1
            # No point having less than a warp per workgroup
            if wgs_x * wgs_y < context.device.simd_group_size:
                return None
            # Only the total size really matters, so it is not necessary
            # to try all possible shapes. For block shape, it is important
            # to be as square as possible.
            if wgs_x != wgs_y and wgs_y != wgs_x * 2:
                return None
            if multi_x != multi_y and multi_x != multi_y * 2:
                return None
            # This can cause launch timeouts due to excessive register pressure
            # and spilling.
            if multi_x * multi_y * num_polarizations * real_dtype.itemsize > 256:
                return None
            tile_x = wgs_x * multi_x
            tile_y = wgs_y * multi_y
            if tile_x > bin_size or tile_y > bin_size:
                return None
            parameters = {
                'real_type': katsdpimager.types.dtype_to_ctype(real_dtype),
                'multi_x': multi_x,
                'multi_y': multi_y,
                'wgs_x': wgs_x,
                'wgs_y': wgs_y,
                'bin_x': bin_size,
                'bin_y': bin_size,
                'num_polarizations': num_polarizations,
                'convolve_kernel_slice_stride': convolve_kernel.padded_shape[2],
                'convolve_kernel_w_stride': np.product(convolve_kernel.padded_shape[1:])
            }
            program = accel.build(
                context, "imager_kernels/grid.mako", parameters,
                extra_dirs=[pkg_resources.resource_filename(__name__, '')])
            kernel = program.get_kernel('grid')
            uv_bias = (bin_size - 1) // 2 - grid.shape[-1] // 2

            def fn():
                Gridder.static_run(
                    queue, kernel, wgs_x, wgs_y, tile_x, tile_y, bin_size,
                    num_vis, uv_bias,
                    grid, weights_grid, uv, w_plane, vis, convolve_kernel)
            return tune.make_measure(queue, fn)

        return tune.autotune(
            generate,
            multi_x=[1, 2, 4],
            multi_y=[1, 2, 4],
            wgs_x=[4, 8, 16],
            wgs_y=[4, 8, 16])

    def instantiate(self, *args, **kwargs):
        return Gridder(self, *args, **kwargs)


class VisOperation(accel.Operation):
    """Base for operation classes that store visibility data in GPU buffers.

    .. rubric:: Slots

    **uv** : array of int16×4
        The first two elements for each visibility are the
        UV coordinates of the first grid cell to be updated. The other two are
        the subpixel U and V coordinates.
    **w_plane** : array of int16
        W plane index per visibility, clamped to the range of allocated w planes
    **vis** : array of complex64 × pols
        Visibilities, which are pre-multiplied by statistical weights. For
        imaging these are inputs.  For prediction these are visibilities on
        input and residual visibilities on output.

    Parameters
    ----------
    command_queue : |CommandQueue|
        Command queue for the operation
    num_polarizations : int
        Size of vis buffer on polarization dimension
    max_vis : int
        Number of visibilities that can be supported per kernel invocation
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, command_queue, num_polarizations, max_vis, allocator=None):
        super().__init__(command_queue, allocator)
        self.max_vis = max_vis
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

    def set_coordinates(self, uv, sub_uv, w_plane):
        """Set the UVW coordinates.

        Before calling, first set :attr:`num_vis`.
        """
        N = self.num_vis
        if len(uv) != N or len(sub_uv) != N or len(w_plane) != N:
            raise ValueError('Lengths do not match')
        self.buffer('uv')[:N, 0:2] = uv
        self.buffer('uv')[:N, 2:4] = sub_uv
        self.buffer('w_plane')[:N] = w_plane

    def set_vis(self, vis):
        """Set input visibilities.

        Before calling, first set :attr:`num_vis`.
        """
        N = self.num_vis
        if len(vis) != N:
            raise ValueError('Lengths do not match')
        self.buffer('vis')[:N] = vis


class GridDegrid(VisOperation):
    """Base class for :class:`Gridder` and :class:`Degridder`.

    .. rubric:: Slots

    In addition to those in :class:`VisOperation`:

    **grid** : array of pols × height × width, complex
        Grid (output for :class:`Gridder`, input for :class:`Degridder`)
    **weights_grid** : array of pols × height × width, float32
        Grid of density weights (input for :class:`Gridder`, absent for :class:`Degridder`)
    **weights** : array of float32 × pols
        Statistical weights for visibilities (:class:`Degridder` only).

    Parameters
    ----------
    template : :class:`GridderTemplate` or :class:`DegridderTemplate`
        Operation template
    command_queue : |CommandQueue|
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
        num_polarizations = len(template.image_parameters.polarizations)
        super().__init__(command_queue, num_polarizations, max_vis, allocator)
        # Check that longest baseline won't cause an out-of-bounds access
        max_uv_src = float(array_parameters.longest_baseline / template.image_parameters.cell_size)
        convolve_kernel_size = template.convolve_kernel.padded_data.shape[-1]
        # I don't think the + 1 is actually needed, but it's a safety factor in
        # case I've made an off-by-one error in the maths.
        grid_pixels = 2 * (int(max_uv_src) + convolve_kernel_size // 2 + 1)
        if grid_pixels > template.image_parameters.pixels:
            raise ValueError('image_oversample is too small '
                             'to capture all visibilities in the UV plane')
        self.template = template
        self.slots['grid'] = accel.IOSlot(
            (num_polarizations, grid_pixels, grid_pixels),
            template.image_parameters.complex_dtype)

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
        super().__init__(*args, **kwargs)
        self.slots['weights_grid'] = accel.IOSlot(self.slots['grid'].shape, np.float32)
        self._kernel = self.template.program.get_kernel('grid')

    @classmethod
    def static_run(cls, command_queue, kernel,
                   wgs_x, wgs_y, tile_x, tile_y, bin_size, num_vis, uv_bias,
                   grid, weights_grid, uv, w_plane, vis, convolve_kernel):
        """Implementation of :meth:`_run` with all the parameters explicit.
        This is not intended for direct use, but is used for autotuning to
        execute the kernel while bypassing a lot of unwanted setup.

        Parameters
        ----------
        command_queue : |CommandQueue|
            Command queue for the operation
        kernel : :class:`katsdpsigproc.cuda.Kernel` or :class:`katsdpsigproc.opencl.Kernel`
            Compiled kernel to run
        wgs_x,wgs_y,tile_x,tile_y,bin_size : int
            Tuning parameters (see :class:`GridTemplate`)
        num_vis : int
            Number of visibilities to grid
        uv_bias : int
            Bias for UV coordinates to center the kernel
        grid,weights_grid,uv,w_plane,vis,convolve_kernel : \
                :class:`~katsdpsigproc.accel.DeviceArray`-like
            Data passed to the kernel
        """
        if num_vis == 0:
            return
        tiles_x = bin_size // tile_x
        tiles_y = bin_size // tile_y
        vis_per_workgroup = 1024
        workgroups_z = accel.divup(num_vis, vis_per_workgroup)
        # Recompute vis_per_workgroup: a smaller value may now be possible
        # for the given value of workgroups_z.
        vis_per_workgroup = accel.divup(num_vis, workgroups_z)
        # Make sure that vis_per_workgroup is a multiple of the batch size.
        # If it is not, then every workgroup will finish with a partial batch,
        # rather than only the last workgroup.
        vis_per_workgroup = accel.roundup(vis_per_workgroup, wgs_x * wgs_y)
        half_u = weights_grid.shape[1] // 2
        half_v = weights_grid.shape[2] // 2
        weights_address_bias = half_v * weights_grid.padded_shape[2] + half_u
        command_queue.enqueue_kernel(
            kernel,
            [
                grid.buffer,
                np.int32(grid.padded_shape[2]),
                np.int32(grid.padded_shape[1] * grid.padded_shape[2]),
                weights_grid.buffer,
                np.int32(weights_grid.padded_shape[2]),
                np.int32(weights_grid.padded_shape[1] * weights_grid.padded_shape[2]),
                uv.buffer,
                w_plane.buffer,
                vis.buffer,
                convolve_kernel.buffer,
                np.int32(uv_bias),
                np.int32(weights_address_bias),
                np.int32(vis_per_workgroup),
                np.int32(num_vis)
            ],
            global_size=(wgs_x * tiles_x,
                         wgs_y * tiles_y,
                         workgroups_z),
            local_size=(wgs_x, wgs_y, 1)
        )

    def _run(self):
        kernel_width = self.template.grid_parameters.kernel_width
        uv_bias = ((kernel_width - 1) // 2 + self.template.convolve_kernel.pad
                   - self.slots['grid'].shape[-1] // 2)
        self.static_run(
            self.command_queue, self._kernel,
            self.template.wgs_x, self.template.wgs_y,
            self.template.tile_x, self.template.tile_y,
            self.template.convolve_kernel.bin_size,
            self.num_vis,
            uv_bias,
            self.buffer('grid'),
            self.buffer('weights_grid'),
            self.buffer('uv'),
            self.buffer('w_plane'),
            self.buffer('vis'),
            self.template.convolve_kernel.padded_data)


class DegridderTemplate:
    autotune_version = 3

    def __init__(self, context, image_parameters, grid_parameters, tuning=None):
        if tuning is None:
            tuning = self.autotune(
                context, grid_parameters.oversample, image_parameters.real_dtype,
                len(image_parameters.polarizations))
        self.grid_parameters = grid_parameters
        self.image_parameters = image_parameters
        self.wgs_x = tuning['wgs_x']
        self.wgs_y = tuning['wgs_y']
        self.wgs_z = tuning['wgs_z']
        self.multi_x = tuning['multi_x']
        self.multi_y = tuning['multi_y']
        self.tile_x = self.wgs_x * self.multi_x
        self.tile_y = self.wgs_y * self.multi_y
        min_pad = max(self.multi_x, self.multi_y) - 1
        bin_size = ConvolutionKernel.get_bin_size(grid_parameters, self.tile_x, self.tile_y,
                                                  min_pad)
        pad = bin_size - grid_parameters.kernel_width
        # Note: we can't necessarily use the same kernel as for gridding,
        # because different tuning parameters will affect the kernel padding.
        # TODO: should still reuse the work done to compute the function, and
        # simply adjust the padding.
        self.convolve_kernel = ConvolutionKernelDevice(
            context, image_parameters, grid_parameters, pad)
        parameters = {
            'real_type': katsdpimager.types.dtype_to_ctype(image_parameters.real_dtype),
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

    @classmethod
    @tune.autotuner(test={'wgs_x': 8, 'wgs_y': 4, 'wgs_z': 4, 'multi_x': 2, 'multi_y': 4})
    def autotune(cls, context, oversample, real_dtype, num_polarizations):
        queue = context.create_tuning_command_queue()
        bin_size = 32
        grid, weights_grid, uv, w_plane, vis, convolve_kernel = _autotune_arrays(
            queue, oversample, real_dtype, num_polarizations, bin_size, 3)
        weights = accel.DeviceArray(context, vis.shape, np.complex64)
        num_vis = uv.shape[0]

        def generate(multi_x, multi_y, wgs_x, wgs_y, wgs_z):
            # No point having less than a warp per workgroup
            if wgs_x * wgs_y * wgs_z < context.device.simd_group_size:
                return None
            # Avoid subgroups bigger than a warp, since they can't use
            # shuffle reduction
            if wgs_x * wgs_y > context.device.simd_group_size:
                return None
            # Cut down number of configurations to test
            if multi_x != multi_y and multi_x != multi_y * 2:
                return None
            if wgs_y != wgs_x and wgs_y != wgs_x * 2:
                return None
            if multi_x * multi_y * num_polarizations * real_dtype.itemsize > 128:
                return None
            # Eliminate configurations that don't fit in a bin
            if wgs_x * multi_x > bin_size or wgs_y * multi_y > bin_size:
                return None
            parameters = {
                'real_type': katsdpimager.types.dtype_to_ctype(real_dtype),
                'multi_x': multi_x,
                'multi_y': multi_y,
                'wgs_x': wgs_x,
                'wgs_y': wgs_y,
                'wgs_z': wgs_z,
                'num_polarizations': num_polarizations,
                'convolve_kernel_slice_stride': convolve_kernel.padded_shape[2],
                'convolve_kernel_w_stride': np.product(convolve_kernel.padded_shape[1:]),
                'bin_x': bin_size,
                'bin_y': bin_size
            }
            program = accel.build(
                context, "imager_kernels/degrid.mako", parameters,
                extra_dirs=[pkg_resources.resource_filename(__name__, '')])
            kernel = program.get_kernel('degrid')
            uv_bias = (bin_size - 1) // 2 - grid.shape[-1] // 2

            def fn():
                Degridder.static_run(
                    queue, kernel, wgs_x, wgs_y, wgs_z,
                    num_vis, uv_bias,
                    grid, uv, w_plane, weights, vis, convolve_kernel)
            return tune.make_measure(queue, fn)

        return tune.autotune(
            generate,
            multi_x=[1, 2, 4],
            multi_y=[1, 2, 4],
            wgs_x=[4, 8],
            wgs_y=[4, 8],
            wgs_z=[1, 2, 4, 8])

    def instantiate(self, *args, **kwargs):
        return Degridder(self, *args, **kwargs)


class Degridder(GridDegrid):
    """Instantiation of :class:`DegridderTemplate`. See :class:`GridDegrid` for
    details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_polarizations = len(self.template.image_parameters.polarizations)
        self.slots['weights'] = accel.IOSlot(
            (self.max_vis, accel.Dimension(num_polarizations, exact=True)), np.float32)
        self._kernel = self.template.program.get_kernel('degrid')

    def set_weights(self, weights):
        """Set statistical weights on visibilities.

        Before calling, set :attr:`num_vis`.
        """
        N = self.num_vis
        if len(weights) != N:
            raise ValueError('Lengths do not match')
        self.buffer('weights')[:N] = weights

    @classmethod
    def static_run(cls, command_queue, kernel,
                   wgs_x, wgs_y, wgs_z, num_vis, uv_bias,
                   grid, uv, w_plane, weights, vis, convolve_kernel):
        if num_vis == 0:
            return
        batch_size = wgs_x * wgs_y
        subgroups = accel.divup(num_vis, batch_size)
        workgroups = accel.divup(subgroups, wgs_z)
        command_queue.enqueue_kernel(
            kernel,
            [
                grid.buffer,
                np.int32(grid.padded_shape[2]),
                np.int32(grid.padded_shape[1] * grid.padded_shape[2]),
                uv.buffer,
                w_plane.buffer,
                weights.buffer,
                vis.buffer,
                convolve_kernel.buffer,
                np.int32(uv_bias),
                np.int32(num_vis)
            ],
            global_size=(wgs_x * workgroups,
                         wgs_y,
                         wgs_z),
            local_size=(wgs_x, wgs_y, wgs_z)
        )

    def _run(self):
        kernel_width = self.template.grid_parameters.kernel_width
        uv_bias = ((kernel_width - 1) // 2 + self.template.convolve_kernel.pad
                   - self.slots['grid'].shape[-1] // 2)
        self.static_run(
            self.command_queue, self._kernel,
            self.template.wgs_x, self.template.wgs_y, self.template.wgs_z,
            self.num_vis,
            uv_bias,
            self.buffer('grid'),
            self.buffer('uv'),
            self.buffer('w_plane'),
            self.buffer('weights'),
            self.buffer('vis'),
            self.template.convolve_kernel.padded_data)


@numba.jit(nopython=True)
def _grid(kernel, grid, weights_grid, uv, sub_uv, w_plane, vis, sample):
    """Internal implementation of :meth:`GridderHost.grid`, split out so that
    Numba can JIT it.
    """
    ksize = kernel.shape[2]
    uv_bias = (ksize - 1) // 2 - grid.shape[2] // 2
    for row in range(uv.shape[0]):
        u0 = uv[row, 0] - uv_bias
        v0 = uv[row, 1] - uv_bias
        sub_u, sub_v = sub_uv[row]
        weights_u = uv[row, 0] + weights_grid.shape[2] // 2
        weights_v = uv[row, 1] + weights_grid.shape[1] // 2
        for i in range(vis.shape[1]):
            sample[i] = vis[row, i] * weights_grid[i, weights_v, weights_u]
        for j in range(ksize):
            for k in range(ksize):
                kernel_sample = kernel[w_plane[row], sub_v, j] * kernel[w_plane[row], sub_u, k]
                weight = np.conj(kernel_sample)
                for pol in range(grid.shape[0]):
                    grid[pol, int(v0 + j), int(u0 + k)] += sample[pol] * weight


class VisOperationHost:
    """Equivalent to :class:`VisOperation` on the host."""
    def __init__(self):
        self._num_vis = 0
        self.uv = None
        self.sub_uv = None
        self.w_plane = None
        self.vis = None

    @property
    def num_vis(self):
        return self._num_vis

    @num_vis.setter
    def num_vis(self, value):
        self._num_vis = value
        self.uv = None
        self.sub_uv = None
        self.w_plane = None
        self.vis = None

    def set_coordinates(self, uv, sub_uv, w_plane):
        """Set UVW coordinates for the visibilities.

        Parameters
        ----------
        uv : 2D array, integer
            Preprocessed grid UV coordinates
        sub_uv : 2D array, integer
            Preprocessed grid UV sub-pixel coordinates
        w_plane : 1D array, integer
            Preprocessed grid W plane coordinates
        """
        N = self.num_vis
        if len(uv) != N or len(sub_uv) != N or len(w_plane) != N:
            raise ValueError('Lengths do not match')
        self.uv = uv
        self.sub_uv = sub_uv
        self.w_plane = w_plane

    def set_vis(self, vis):
        """Set input visibility data.

        vis : 2D ndarray of complex or real
            Visibility data, indexed by sample and polarization, and
            pre-multiplied by all weights
        """
        if len(vis) != self.num_vis:
            raise ValueError('Lengths do not match')
        self.vis = vis


class GridDegridHost(VisOperationHost):
    """Common code shared by :class:`GridderHost` and :class:`DegridderHost`."""
    def __init__(self, image_parameters, grid_parameters):
        super().__init__()
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self.kernel = ConvolutionKernel(image_parameters, grid_parameters)
        pixels = image_parameters.pixels
        shape = (len(image_parameters.polarizations), pixels, pixels)
        self.values = np.empty(shape, image_parameters.complex_dtype)


class GridderHost(GridDegridHost):
    def __init__(self, image_parameters, grid_parameters):
        super().__init__(image_parameters, grid_parameters)
        self.weights_grid = np.empty(self.values.shape, np.float32)

    def clear(self):
        self.values.fill(0)

    def __call__(self):
        """Add visibilities to the grid, with convolutional gridding.

        Parameters
        ----------
        """
        _grid(self.kernel.data, self.values, self.weights_grid,
              self.uv, self.sub_uv, self.w_plane, self.vis,
              np.empty((self.vis.shape[1],), self.values.dtype))


@numba.jit(nopython=True)
def _degrid(kernel, values, uv, sub_uv, w_plane, weights, vis, sample):
    ksize = kernel.shape[2]
    uv_bias = (ksize - 1) // 2 - values.shape[2] // 2
    for row in range(uv.shape[0]):
        u0 = uv[row, 0] - uv_bias
        v0 = uv[row, 1] - uv_bias
        sub_u, sub_v = sub_uv[row]
        for i in range(vis.shape[1]):
            sample[i] = 0
        for j in range(ksize):
            for k in range(ksize):
                weight = kernel[w_plane[row], sub_v, j] * kernel[w_plane[row], sub_u, k]
                for pol in range(values.shape[0]):
                    sample[pol] += weight * values[pol, v0 + j, u0 + k]
        for i in range(vis.shape[1]):
            vis[row, i] -= weights[row, i] * sample[i]


class DegridderHost(GridDegridHost):
    def __init__(self, image_parameters, grid_parameters):
        super().__init__(image_parameters, grid_parameters)
        self.weights = None

    @VisOperationHost.num_vis.setter
    def num_vis(self, value):
        VisOperationHost.num_vis.fset(self, value)
        self.weights = None

    def set_weights(self, weights):
        """Set statistical weights"""
        if len(weights) != self.num_vis:
            raise ValueError('Lengths do not match')
        self.weights = weights

    def __call__(self):
        """Compute visibilities from the grid. See :meth:`GridderHost.grid`
        for details of the parameters.
        """
        _degrid(self.kernel.data, self.values,
                self.uv, self.sub_uv, self.w_plane, self.weights, self.vis,
                np.empty((self.vis.shape[1],), self.values.dtype))
