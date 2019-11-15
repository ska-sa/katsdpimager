"""Deconvolution routines based on CLEAN.

Both the CPU and GPU versions use an acceleration structure to speed up
peak-finding when the point spread function (PSF) patch in use is
significantly smaller than the image. The image is divided into tiles. For
each tile, the peak value and the position of that value are maintained.
Finding the global peak requires only finding the best tile peak. Subtracting
the PSF requires only updating those tiles intersected by the shifted PSF.

The GPU implementation currently round-trips to the CPU on each minor cycle.
It could be done entirely on the GPU, but round-tripping will make it easier
to put in a threshold later. It should still be possible to do batches of
minor cycles if the launch overheads become an issue.

.. include:: macros.rst
"""

import math

import numpy as np
from katsdpsigproc import accel
import pkg_resources

import katsdpimager.types
from katsdpimager import numba

#: Use only Stokes I to find peaks
CLEAN_I = 0
#: Use the sum of squares of available Stokes components to find peaks
CLEAN_SUMSQ = 1


class PsfPatchTemplate:
    """Examines a PSF to determine how big a PSF patch is required to
    enclosure all values above some threshold.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    dtype : {`np.float32`, `np.float64`}
        Precision of image
    num_polarizations : int
        Number of polarizations stored in the dirty image
    tuning : dict, optional
        Tuning parameters (unused for now)
    """
    def __init__(self, context, dtype, num_polarizations, tuning=None):
        # TODO: autotuning
        self.num_polarizations = num_polarizations
        self.dtype = dtype
        self.wgsx = 16
        self.wgsy = 16
        self.program = accel.build(
            context, "imager_kernels/clean/psf_patch.mako",
            {
                'real_type': katsdpimager.types.dtype_to_ctype(dtype),
                'wgsx': self.wgsx,
                'wgsy': self.wgsy,
                'num_polarizations': num_polarizations
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return PsfPatch(self, *args, **kwargs)


class PsfPatch(accel.Operation):
    """Instantiation of :class:`PsfPatchTemplate`.

    The implementation works by dividing the PSF into tiles, with a workgroup
    per tile, and doing a workgroup-level reduction in each tile. The
    inter-tile reduction is done on the host.

    The returned patch size is generally odd, but will never exceed the PSF size
    (so if the side lobes extend all the way to the edge, it will be equal to the
    PSF size, which is even).

    .. rubric:: Slots

    **psf** : array of shape(num_polarizations, width, height), real
        PSF, already scaled so that the central value is 1.
    **bound** : array, int
        Array of implementation-dependent size containing lower bounds on the
        l and m distance from the centre, in pixels.

    Parameters
    ----------
    template : :class:`PsfPatchTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    shape : tuple of ints
        Shape for the PSF
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """

    def __init__(self, template, command_queue, shape, allocator=None):
        if shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        super().__init__(command_queue, allocator)
        max_wg_x = accel.divup(shape[2], template.wgsx)
        max_wg_y = accel.divup(shape[1], template.wgsy)
        polarizations = accel.Dimension(template.num_polarizations, exact=True)
        self.slots['psf'] = accel.IOSlot([polarizations, shape[1], shape[2]], template.dtype)
        self.slots['bound'] = accel.IOSlot([
            accel.Dimension(max_wg_y * max_wg_x),
            accel.Dimension(2, exact=True)], np.int32)
        self._bound_host = accel.HostArray((max_wg_y * max_wg_x, 2), np.int32,
                                           context=command_queue.context)
        self.template = template
        self.kernel = template.program.get_kernel('psf_patch')

    def _run(self):
        # Never used because we override __call__
        pass        # pragma: nocover

    def __call__(self, threshold, limit=None, **kwargs):
        self.bind(**kwargs)
        self.ensure_all_bound()
        psf = self.buffer('psf')
        bound = self.buffer('bound')
        min_x = 0
        min_y = 0
        max_x = psf.shape[2] - 1
        max_y = psf.shape[1] - 1
        mid_x = psf.shape[2] // 2
        mid_y = psf.shape[1] // 2
        if limit is not None:
            hlimit = (limit - 1) // 2
            min_x = max(min_x, mid_x - hlimit)
            min_y = max(min_y, mid_y - hlimit)
            max_x = min(max_x, mid_x + hlimit)
            max_y = min(max_y, mid_y + hlimit)
        wg_x = accel.divup(max_x - min_x + 1, self.template.wgsx)
        wg_y = accel.divup(max_y - min_y + 1, self.template.wgsy)
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                psf.buffer,
                np.int32(psf.padded_shape[2]),
                np.int32(psf.padded_shape[1] * psf.padded_shape[2]),
                bound.buffer,
                np.int32(min_x), np.int32(min_y),
                np.int32(max_x), np.int32(max_y),
                np.int32(mid_x), np.int32(mid_y),
                np.float32(threshold)
            ],
            global_size=(wg_x * self.template.wgsx, wg_y * self.template.wgsy),
            local_size=(self.template.wgsx, self.template.wgsy)
        )
        if isinstance(bound, accel.SVMArray):
            self.command_queue.finish()
        bound.get(self.command_queue, self._bound_host)
        # Turn distances from the centre into a symmetric bounding box size.
        box = 2 * np.max(self._bound_host[: wg_x * wg_y], axis=0) + 1
        return (self.template.num_polarizations,
                int(min(box[1], psf.shape[1])),
                int(min(box[0], psf.shape[2])))


def metric_to_power(mode, metric):
    """Convert a peak-finding metric to a value that scales linearly with
    power (e.g. Jy/beam).
    """
    if mode == CLEAN_I:
        return metric
    elif mode == CLEAN_SUMSQ:
        return math.sqrt(metric)
    else:
        raise ValueError('Invalid mode {}'.format(mode))


def power_to_metric(mode, power):
    """Inverse of :func:`metric_to_power`."""
    if mode == CLEAN_I:
        return power
    elif mode == CLEAN_SUMSQ:
        return power * power
    else:
        raise ValueError('Invalid mode {}'.format(mode))


class NoiseEstTemplate:
    """Robust estimation of the noise (as a standard deviation) in an image.

    The noise is estimated by computing the median absolute value via binary
    search.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    dtype : {`np.float32`, `np.float64`}
        Image precision
    num_polarizations : int
        number of polarizations stored in the dirty/residual image
    mode : {:data:`CLEAN_I`, :data:`CLEAN_SUMSQ`}
        Metric for scoring pixels
    tuning : dict, optional
        Tuning parameters (unused for now)
    """
    def __init__(self, context, dtype, num_polarizations, mode, tuning=None):
        self.context = context
        self.dtype = np.dtype(dtype)
        self.num_polarizations = num_polarizations
        self.mode = mode
        self.wgsx = 32
        self.wgsy = 8
        self.tilex = 32
        self.tiley = 32
        self.program = accel.build(
            context, "imager_kernels/clean/rank.mako",
            {
                'real_type': katsdpimager.types.dtype_to_ctype(dtype),
                'wgsx': self.wgsx,
                'wgsy': self.wgsy,
                'tilex': self.tilex,
                'tiley': self.tiley,
                'num_polarizations': num_polarizations,
                'clean_mode': mode
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return NoiseEst(self, *args, **kwargs)


class NoiseEst(accel.Operation):
    """Instantiation of :class:`NoiseEstTemplate`

    .. rubric:: Slots

    **dirty** : array of shape (num_polarizations, height, width), real
        Dirty image
    **rank** : implementation-defined
        Temporary buffer used during operation. It holds a rank of a test
        value for each workgroup.

    Parameters
    ----------
    template : :class:`NoiseEstTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    image_shape : tuple of ints
        Shape for the dirty image
    border : int
        Distance from each edge of dirty image to ignore in ranking
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """

    def __init__(self, template, command_queue, image_shape, border, allocator=None):
        if image_shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        if border * 2 >= min(image_shape[1], image_shape[2]):
            raise ValueError('Border must be less than half the image size')
        super().__init__(command_queue, allocator)
        self._num_tiles_x = accel.divup(image_shape[2] - 2 * border, template.tilex)
        self._num_tiles_y = accel.divup(image_shape[1] - 2 * border, template.tiley)
        self.template = template
        self.border = border
        self.slots['dirty'] = accel.IOSlot([
            accel.Dimension(template.num_polarizations, exact=True),
            image_shape[1], image_shape[2]], template.dtype)
        self.slots['rank'] = accel.IOSlot([
            accel.Dimension(self._num_tiles_y, exact=True),
            accel.Dimension(self._num_tiles_x, exact=True)], np.uint32)
        self.kernel = template.program.get_kernel('compute_rank')

    def _run(self):
        # Never used because we override __call__
        pass        # pragma: nocover

    def __call__(self, **kwargs):
        self.bind(**kwargs)
        self.ensure_all_bound()
        dtype = self.template.dtype
        if dtype == np.float32:
            itype = np.uint32
        elif dtype == np.float64:
            itype = np.uint64
        else:
            raise TypeError('dtype {} is not supported'.format(dtype))
        # We do binary search on the bit representation of the floating point
        # value.
        low = dtype.type(0)
        high = dtype.type(np.inf)
        dirty = self.buffer('dirty')
        rank = self.buffer('rank')
        rank_host = rank.empty_like()
        median_rank = (dirty.shape[1] - 2 * self.border) * (dirty.shape[2] - 2 * self.border) // 2
        # We don't need a super-accurate estimate down to the last bit.
        while high > np.finfo(dtype).tiny and high > low * 1.0001:
            # Average the integer bit representations, which will effectively
            # binary search the exponent.
            ilow = low.view(itype)
            ihigh = high.view(itype)
            if ihigh - ilow == itype(1):
                # This can happen in some corner cases despite the ratio chunk
                # e.g. if low is zero or high is inf
                break
            imid = ilow + (ihigh - ilow) // itype(2)
            mid = imid.view(dtype)
            self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    dirty.buffer,
                    np.int32(dirty.padded_shape[2]),
                    np.int32(dirty.padded_shape[1] * dirty.padded_shape[2]),
                    np.int32(dirty.shape[2] - self.border),
                    np.int32(dirty.shape[1] - self.border),
                    np.int32(self.border),
                    rank.buffer,
                    mid
                ],
                global_size=(self._num_tiles_x * self.template.wgsx,
                             self._num_tiles_y * self.template.wgsy),
                local_size=(self.template.wgsx, self.template.wgsy))
            if isinstance(rank, accel.SVMArray):
                self.command_queue.finish()
            rank.get(self.command_queue, rank_host)
            cur_rank = rank_host.sum()
            if cur_rank < median_rank:
                low = mid
            else:
                high = mid
        # low and high are very close, but we use low so that if the input is
        # more than 50% zeros then the output is zero.
        # The magic number is the ratio between median absolute deviation and
        # standard deviation of a Gaussian.
        return metric_to_power(self.template.mode, low) * 1.48260222


class _UpdateTilesTemplate:
    """Operation template to compute the peak (including location) for each
    tile intersecting a window.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    dtype : {`np.float32`, `np.float64`}
        Precision of image
    num_polarizations : int
        Number of polarizations stored in the dirty image
    mode : {:data:`CLEAN_I`, :data:`CLEAN_SUMSQ`}
        Function determining peak
    tuning : dict, optional
        Tuning parameters (unused for now)
    """
    def __init__(self, context, dtype, num_polarizations, mode, tuning=None):
        # TODO: autotuning
        self.num_polarizations = num_polarizations
        self.dtype = dtype
        self.wgsx = 32
        self.wgsy = 8
        self.tilex = 32
        self.tiley = 32
        self.program = accel.build(
            context, "imager_kernels/clean/update_tiles.mako",
            {
                'real_type': katsdpimager.types.dtype_to_ctype(dtype),
                'wgsx': self.wgsx,
                'wgsy': self.wgsy,
                'tilex': self.tilex,
                'tiley': self.tiley,
                'num_polarizations': num_polarizations,
                'clean_mode': mode
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return _UpdateTiles(self, *args, **kwargs)


class _UpdateTiles(accel.Operation):
    """Instantiation of :class:`_UpdateTileTemplate`

    .. rubric:: Slots

    **dirty** : array of shape (num_polarizations, height, width), real
        Dirty image
    **tile_max** : array of shape (height, width)
        Internal storage of maximum score for each tile
    **tile_pos** : array of shape (height, width, 2)
        Internal storage of best position for each tile, with each position
        stored as (row, col).

    Parameters
    ----------
    template : :class:`_UpdateTilesTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    image_shape : tuple of ints
        Shape for the dirty image
    border : int
        Distance from each edge of dirty image where tiles start
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """

    def __init__(self, template, command_queue, image_shape, border, allocator=None):
        if image_shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        if border * 2 >= min(image_shape[1], image_shape[2]):
            raise ValueError('Border must be less than half the image size')
        super().__init__(command_queue, allocator)
        num_tiles_x = accel.divup(image_shape[2] - 2 * border, template.tilex)
        num_tiles_y = accel.divup(image_shape[1] - 2 * border, template.tiley)
        image_width = accel.Dimension(image_shape[2])
        image_height = accel.Dimension(image_shape[1])
        image_pols = accel.Dimension(template.num_polarizations, exact=True)
        tiles_width = accel.Dimension(num_tiles_x)
        tiles_height = accel.Dimension(num_tiles_y)
        self.template = template
        self.border = border
        self.slots['dirty'] = accel.IOSlot([image_pols, image_height, image_width], template.dtype)
        self.slots['tile_max'] = accel.IOSlot([tiles_height, tiles_width], template.dtype)
        self.slots['tile_pos'] = accel.IOSlot(
            [tiles_height, tiles_width, accel.Dimension(2, exact=True)], np.int32)
        self.kernel = template.program.get_kernel('update_tiles')

    def _run(self):
        # Never used because we override __call__
        pass        # pragma: nocover

    def __call__(self, x0, y0, x1, y1, **kwargs):
        """Update all tiles intersected by the pixel range [x0, x1) by [y0, y1)"""
        self.bind(**kwargs)
        self.ensure_all_bound()
        tile_max = self.buffer('tile_max')
        tile_pos = self.buffer('tile_pos')
        # Map pixel range to a tile range
        x0 = max((x0 - self.border) // self.template.tilex, 0)
        y0 = max((y0 - self.border) // self.template.tiley, 0)
        x1 = min(accel.divup(x1 - self.border, self.template.tilex), tile_max.shape[1])
        y1 = min(accel.divup(y1 - self.border, self.template.tiley), tile_max.shape[0])
        if x0 < x1 and y0 < y1:
            dirty = self.buffer('dirty')
            self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    dirty.buffer,
                    np.int32(dirty.padded_shape[2]),
                    np.int32(dirty.padded_shape[1] * dirty.padded_shape[2]),
                    np.int32(dirty.shape[2] - self.border),
                    np.int32(dirty.shape[1] - self.border),
                    np.int32(self.border),
                    tile_max.buffer,
                    tile_pos.buffer,
                    np.int32(tile_max.padded_shape[1]),
                    np.int32(x0), np.int32(y0)
                ],
                global_size=((x1 - x0) * self.template.wgsx, (y1 - y0) * self.template.wgsy),
                local_size=(self.template.wgsx, self.template.wgsy))


class _FindPeakTemplate:
    """Find the global peak from per-tile peaks.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    dtype : {`np.float32`, `np.float64`}
        Image precision
    tuning : dict, optional
        Tuning parameters (currently unused)
    """

    def __init__(self, context, dtype, num_polarizations, tuning=None):
        # TODO: autotuning
        self.wgsx = 16
        self.wgsy = 16
        self.dtype = dtype
        self.num_polarizations = num_polarizations
        self.program = accel.build(
            context, "imager_kernels/clean/find_peak.mako",
            {
                'real_type': katsdpimager.types.dtype_to_ctype(dtype),
                'num_polarizations': self.num_polarizations,
                'wgsx': self.wgsx,
                'wgsy': self.wgsy,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return _FindPeak(self, *args, **kwargs)


class _FindPeak(accel.Operation):
    """Instantiation of :class:`_FindPeak`.

    .. rubric:: Slots

    **dirty** : array of shape (num_polarizations, height, width), real
        Dirty image, used to return the original values at the peak
    **tile_max** : array of shape (height, width)
        Internal storage of maximum score for each tile
    **tile_pos** : array of shape (height, width, 2)
        Internal storage of best position for each tile, with each position
        stored as (row, col).
    **peak_value** : array of shape (1,), real
        Score for the peak position (output)
    **peak_pos** : array of shape (2,), int32
        Row and column of the peak position (output)
    **peak_pixel** : array of shape (num_polarizations,), real
        Dirty image sampled at the peak (output)

    Parameters
    ----------
    template : :class:`_FindPeakTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    image_shape : tuple of int
        Shape of the dirty image, as (num_polarizations, height, width)
    tile_shape : tuple of int
        Shape of the tile array, as (height, width)
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, image_shape, tile_shape, allocator=None):
        if image_shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        super().__init__(command_queue, allocator)
        self.template = template
        image_dims = [accel.Dimension(image_shape[0], exact=True),
                      accel.Dimension(image_shape[1]),
                      accel.Dimension(image_shape[2])]
        tile_dims = [accel.Dimension(tile_shape[0]), accel.Dimension(tile_shape[1])]
        pair = accel.Dimension(2, exact=True)
        self.slots['dirty'] = accel.IOSlot(image_dims, template.dtype)
        self.slots['tile_max'] = accel.IOSlot(tile_dims, template.dtype)
        self.slots['tile_pos'] = accel.IOSlot(tile_dims + [pair], np.int32)
        self.slots['peak_value'] = accel.IOSlot([1], template.dtype)
        self.slots['peak_pos'] = accel.IOSlot([2], np.int32)
        self.slots['peak_pixel'] = accel.IOSlot([template.num_polarizations], template.dtype)
        self.kernel = template.program.get_kernel('find_peak')

    def _run(self):
        dirty = self.buffer('dirty')
        tile_max = self.buffer('tile_max')
        tile_pos = self.buffer('tile_pos')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                dirty.buffer,
                np.int32(dirty.padded_shape[2]),
                np.int32(dirty.padded_shape[1] * dirty.padded_shape[2]),
                tile_max.buffer,
                tile_pos.buffer,
                np.int32(tile_max.padded_shape[1]),
                np.int32(tile_max.shape[1]), np.int32(tile_max.shape[0]),
                self.buffer('peak_value').buffer,
                self.buffer('peak_pos').buffer,
                self.buffer('peak_pixel').buffer
            ],
            global_size=(accel.roundup(tile_max.shape[0], self.template.wgsx),
                         accel.roundup(tile_max.shape[1], self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy))


class _SubtractPsfTemplate:
    """Subtract a multiple of the point spread function from the dirty image,
    and add a corresponding single pixel to the model image.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    dtype : {`np.float32`, `np.float64`}
        Image precision
    num_polarizations : int
        number of polarizations stored in the dirty image and PSF
    tuning : dict, optional
        Tuning parameters (unused for now)
    """
    def __init__(self, context, dtype, num_polarizations, tuning=None):
        # TODO: autotuning
        self.wgsx = 16
        self.wgsy = 16
        self.dtype = dtype
        self.num_polarizations = num_polarizations
        self.program = accel.build(
            context, "imager_kernels/clean/subtract_psf.mako",
            {
                'real_type': katsdpimager.types.dtype_to_ctype(dtype),
                'num_polarizations': num_polarizations,
                'wgsx': self.wgsx,
                'wgsy': self.wgsy
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return _SubtractPsf(self, *args, **kwargs)


class _SubtractPsf(accel.Operation):
    """Instantiation of :class:`_SubtractPsfTemplate`. The dirty and model
    image have the same shape and padding, while the PSF may be smaller.

    .. rubric:: Slots

    **dirty** : array of shape (num_polarizations, height, width), real
        Dirty image
    **model** : array of shape (num_polarizations, height, width), real
        Model image
    **psf** : array of shape (num_polarizations, height, width), real
        Point spread function, with center at (height // 2, width // 2)
    **peak_pixel** : array of shape (num_polarizations,), real
        Current dirty image value at the position where subtract is happening

    Parameters
    ----------
    template : :class:`_SubtractPsfTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    loop_gain : float
        Scale factor for subtraction. The PSF is scaled by both `loop_gain` and
        the per-polarization value in the **peak_pixel** slot before subtraction.
    image_shape : tuple of int
        Shape for the dirty and model images, as (num_polarizations, height, width)
    psf_shape : tuple of int
        Shape for the point spread function, as (num_polarizations, height, width)
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Raises
    ------
    ValueError
        if the number of polarizations in `image_shape` or `psf_shape` does not
        match `template`
    """
    def __init__(self, template, command_queue, loop_gain, image_shape, psf_shape, allocator=None):
        super().__init__(command_queue, allocator)
        pol_dim = accel.Dimension(template.num_polarizations, exact=True)
        if image_shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        if psf_shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        image_dims = [pol_dim, accel.Dimension(image_shape[1]), accel.Dimension(image_shape[2])]
        psf_dims = [pol_dim, accel.Dimension(psf_shape[1]), accel.Dimension(psf_shape[2])]
        self.slots['dirty'] = accel.IOSlot(image_dims, template.dtype)
        self.slots['model'] = accel.IOSlot(image_dims, template.dtype)
        self.slots['psf'] = accel.IOSlot(psf_dims, template.dtype)
        self.slots['peak_pixel'] = accel.IOSlot([pol_dim], template.dtype)
        self.loop_gain = loop_gain
        self.template = template
        self.kernel = template.program.get_kernel('subtract_psf')

    def _run(self):
        # Never used because we override __call__
        pass        # pragma: nocover

    def __call__(self, pos, psf_patch, **kwargs):
        """Execute the operation.

        Parameters
        ----------
        pos : tuple
            row, col in image at which to center the PSF
        psf_patch : tuple
            num_polarizations, height, width of central area of PSF to subtract
            (num_polarizations is ignored)
        """
        self.bind(**kwargs)
        self.ensure_all_bound()
        dirty = self.buffer('dirty')
        model = self.buffer('model')
        psf = self.buffer('psf')
        psf_y = psf.shape[1] // 2 - psf_patch[1] // 2
        psf_x = psf.shape[2] // 2 - psf_patch[2] // 2
        psf_addr_offset = psf_y * psf.padded_shape[2] + psf_x
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                dirty.buffer,
                model.buffer,
                np.int32(dirty.padded_shape[2]),
                np.int32(dirty.padded_shape[1] * dirty.padded_shape[2]),
                np.int32(dirty.shape[2]),
                np.int32(dirty.shape[1]),
                psf.buffer,
                np.int32(psf.padded_shape[2]),
                np.int32(psf.padded_shape[1] * psf.padded_shape[2]),
                np.int32(psf_patch[2]),
                np.int32(psf_patch[1]),
                np.int32(psf_addr_offset),
                self.buffer('peak_pixel').buffer,
                np.int32(pos[1]), np.int32(pos[0]),
                np.int32(pos[1] - psf_patch[2] // 2),
                np.int32(pos[0] - psf_patch[1] // 2),
                np.float32(self.loop_gain)
            ],
            global_size=(accel.roundup(psf_patch[2], self.template.wgsx),
                         accel.roundup(psf_patch[1], self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy))


class CleanTemplate:
    """Composite template for the CLEAN minor cycles.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    clean_parameters : :class:`katsdpimager.parameters.CleanParameters`
        Command-line parameters for CLEAN
    dtype : {`np.float32`, `np.float64`}
        Image precision
    num_polarizations : int
        Number of polarizations stored in the image
    """
    def __init__(self, context, clean_parameters, dtype, num_polarizations):
        self.clean_parameters = clean_parameters
        self.dtype = dtype
        self.num_polarizations = num_polarizations
        self._update_tiles = _UpdateTilesTemplate(context, dtype, num_polarizations,
                                                  clean_parameters.mode)
        self._find_peak = _FindPeakTemplate(context, dtype, num_polarizations)
        self._subtract_psf = _SubtractPsfTemplate(context, dtype, num_polarizations)

    def instantiate(self, *args, **kwargs):
        return Clean(self, *args, **kwargs)


class Clean(accel.OperationSequence):
    """Instantiation of :class:`CleanTemplate`.

    .. rubric:: Slots

    **dirty** : array of shape (num_polarizations, height, width), real
        Dirty image
    **model** : array of shape (num_polarizations, height, width), real
        Model image
    **psf** : array of shape (num_polarizations, height, width), real
        Point spread function, with center at (height // 2, width // 2)
    **tile_max** : array of shape (height, width)
        Internal storage of maximum score for each tile
    **tile_pos** : array of shape (height, width, 2)
        Internal storage of best position for each tile, with each position
        stored as (row, col).
    **peak_value** : array of shape (1,), real
        Score for the peak position (output)
    **peak_pos** : array of shape (2,), int32
        Row and column of the peak position (output)
    **peak_pixel** : array of shape (num_polarizations,), real
        Dirty image sampled at the peak (output)

    The external interface consists of **dirty**, **model** and **psf**. The
    others are exposed only so that the memory can be reused for other
    purposes.

    Parameters
    ----------
    template : :class:`_UpdateTilesTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    image_parameters : :class:`katsdpimager.parameters.ImageParameters`
        Command-line parameters with image properties
    border : int
        Size of border in which no components may be placed
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Raises
    ------
    ValueError
        if `image_parameters` is inconsistent with the template
    """
    def __init__(self, template, command_queue, image_parameters, allocator=None):
        if image_parameters.real_dtype != template.dtype:
            raise ValueError('dtype mismatch')
        image_shape = (len(image_parameters.polarizations),
                       image_parameters.pixels, image_parameters.pixels)
        self._update_tiles = template._update_tiles.instantiate(
            command_queue, image_shape, template.clean_parameters.border, allocator)
        tile_shape = self._update_tiles.slots['tile_max'].shape
        self._find_peak = template._find_peak.instantiate(
            command_queue, image_shape, tile_shape, allocator)
        self._subtract_psf = template._subtract_psf.instantiate(
            command_queue, template.clean_parameters.loop_gain, image_shape, image_shape,
            allocator)
        ops = [
            ('update_tiles', self._update_tiles),
            ('find_peak', self._find_peak),
            ('subtract_psf', self._subtract_psf)
        ]
        compounds = {
            'dirty': ['update_tiles:dirty', 'find_peak:dirty', 'subtract_psf:dirty'],
            'model': ['subtract_psf:model'],
            'psf': ['subtract_psf:psf'],
            'tile_max': ['update_tiles:tile_max', 'find_peak:tile_max'],
            'tile_pos': ['update_tiles:tile_pos', 'find_peak:tile_pos'],
            'peak_value': ['find_peak:peak_value'],
            'peak_pos': ['find_peak:peak_pos'],
            'peak_pixel': ['find_peak:peak_pixel', 'subtract_psf:peak_pixel']
        }
        super().__init__(command_queue, ops, compounds, allocator=allocator)
        peak_value = self.slots['peak_value']
        peak_pos = self.slots['peak_pos']
        self._peak_value_host = accel.HostArray(
            peak_value.shape, peak_value.dtype, peak_value.required_padded_shape(),
            context=command_queue.context)
        self._peak_pos_host = accel.HostArray(
            peak_pos.shape, peak_pos.dtype, peak_pos.required_padded_shape(),
            context=command_queue.context)

    def reset(self):
        """Call after populating the buffers but before the first minor cycle."""
        self.ensure_all_bound()
        dirty = self.buffer('dirty')
        self._update_tiles(0, 0, dirty.shape[2], dirty.shape[1])

    def __call__(self, psf_patch, threshold=0.0):
        """Run a single minor CLEAN cycle.

        Parameters
        ----------
        psf_patch : tuple
            Shape of the PSF patch to use for subtraction
        threshold : float, optional
            If specified, skip the cycle if the peak value metric is less
            than this threshold. Note that this value must be chosen relative
            to the clean mode, since the units are different.

        Returns
        -------
        peak_value : float
            The clean mode metric for the peak, or ``None`` if it was
            below the threshold and so no subtraction was done.
        """
        self.ensure_all_bound()
        self._find_peak()
        # Copy the peak position and value back to the host
        peak_value_device = self.buffer('peak_value')
        peak_pos_device = self.buffer('peak_pos')
        if (isinstance(peak_value_device, accel.SVMArray)
                or isinstance(peak_pos_device, accel.SVMArray)):
            self.command_queue.finish()
        peak_value = peak_value_device.get_async(self.command_queue, self._peak_value_host)
        peak_pos = peak_pos_device.get_async(self.command_queue, self._peak_pos_host)
        peak_pos = [int(x) for x in peak_pos]
        self.command_queue.finish()
        if peak_value[0] < threshold:
            return None

        self._subtract_psf(peak_pos, psf_patch)
        # Update the tiles
        x0 = peak_pos[1] - psf_patch[2] // 2
        x1 = x0 + psf_patch[2]
        y0 = peak_pos[0] - psf_patch[1] // 2
        y1 = y0 + psf_patch[1]
        self._update_tiles(x0, y0, x1, y1)
        return peak_value[0]


def psf_patch_host(psf, threshold, limit=None):
    """Compute the size of the bounding box of the PSF required to contain all
    values above `threshold`.

    This is a CPU-only equivalent to :class:`PsfPatch`.

    Parameters
    ----------
    psf : ndarray
        Point spread function, with shape (polarizations, height, width)
    threshold : float
        Minimum value that must be included inside the box
    limit : float, optional
        Upper bound on the return value. Pixels outside this range are not
        examined.

    Returns
    -------
    box : tuple
        Shape of the central part of `psf` to retain, with the same
        dimensions. The size is always even.
    """
    if limit is not None:
        hlimit = (limit - 1) // 2
        mid_x = psf.shape[2] // 2
        mid_y = psf.shape[1] // 2
        min_x = max(0, mid_x - hlimit)
        min_y = max(0, mid_y - hlimit)
        max_x = min(psf.shape[2] - 1, mid_x + hlimit)
        max_y = min(psf.shape[1] - 1, mid_y + hlimit)
        psf = psf[:, min_y : max_y + 1, min_x : max_x + 1]

    nz = np.nonzero(np.abs(psf) >= threshold)
    if len(nz[0]) == 0:
        # No values above threshold at all. This should never happen, because
        # the peak should be 1.0, but return a 1x1 box.
        return (psf.shape[0], 1, 1)
    y_dist = np.max(np.abs(nz[1] - psf.shape[1] // 2))
    x_dist = np.max(np.abs(nz[2] - psf.shape[2] // 2))
    y_size = min(psf.shape[1], 2 * y_dist + 1)
    x_size = min(psf.shape[2], 2 * x_dist + 1)
    return (psf.shape[0], y_size, x_size)


def noise_est_host(image, border, mode):
    """Host implementation of :class:`NoiseEstTemplate`."""
    image = image[:, border:-border, border:-border]
    if mode == CLEAN_I:
        metric = np.abs(image)
    elif mode == CLEAN_SUMSQ:
        metric = np.sum(image**2, axis=0)
    median = np.median(metric)
    return metric_to_power(mode, median) * 1.48260222


@numba.jit(nopython=True)
def _tile_peak(y0, x0, y1, x1, image, mode, zero):
    """Implementation of :meth:`CleanHost._update_tile`, split out as a
    function to allow numba to fully JIT it."""
    best_pos = (x0, y0)
    best_value = zero
    if mode == CLEAN_I:
        for y in range(y0, y1):
            for x in range(x0, x1):
                value = np.abs(image[0, y, x])
                if value > best_value:
                    best_value = value
                    best_pos = (y, x)
    else:
        for y in range(y0, y1):
            for x in range(x0, x1):
                value = zero
                for pol in range(image.shape[0]):
                    value += image[pol, y, x]**2
                if value > best_value:
                    best_value = value
                    best_pos = (y, x)
    return best_pos, best_value


class CleanHost:
    """CPU-only equivalent to :class:`Clean`. The class keeps references to
    the provided arrays, and only examines modifies them when :meth:`reset`
    or :meth:`__call__` is invoked.

    Parameters
    ----------
    image_parameters : :class:`katsdpimager.parameters.ImageParameters`
        Command-line parameters with image properties
    clean_parameters : :class:`katsdpimager.parameters.CleanParameters`
        Command-line parameters for CLEAN
    image : ndarray, (height, width, num_polarizations)
        Dirty image, modified as cleaning proceeds
    psf : ndarray, (height, width, num_polarizations), real
        Point spread function (may be smaller than `image`)
    model : ndarray, (height, width, num_polarizations), real
        Model image, which is updated with found CLEAN components
    """
    def __init__(self, image_parameters, clean_parameters, image, psf, model):
        self.clean_parameters = clean_parameters
        self.image_parameters = image_parameters
        self.image = image
        self.model = model
        self.psf = psf
        self.tile_size = 32
        border = self.clean_parameters.border
        tiles_x = accel.divup(image.shape[2] - 2 * border, self.tile_size)
        tiles_y = accel.divup(image.shape[1] - 2 * border, self.tile_size)
        self._tile_max = np.zeros((tiles_y, tiles_x), self.image_parameters.real_dtype)
        self._tile_pos = np.empty((tiles_y, tiles_x, 2), np.int32)

    def _update_tile(self, y, x):
        border = self.clean_parameters.border
        x0 = x * self.tile_size + border
        y0 = y * self.tile_size + border
        x1 = min(x0 + self.tile_size, self.image.shape[2] - border)
        y1 = min(y0 + self.tile_size, self.image.shape[1] - border)
        best_pos, best_value = _tile_peak(
            y0, x0, y1, x1, self.image, self.clean_parameters.mode,
            self.image.dtype.type(0))
        self._tile_max[y, x] = best_value
        self._tile_pos[y, x] = best_pos

    def _subtract_psf(self, y, x, psf_patch):
        psf_size_x = self.psf.shape[2]
        psf_size_y = self.psf.shape[1]
        psf_patch_x = psf_patch[2]
        psf_patch_y = psf_patch[1]
        image_size_x = self.image.shape[2]
        image_size_y = self.image.shape[1]
        # Centre point to subtract at (x, y) in image
        psf_x = psf_size_x // 2
        psf_y = psf_size_y // 2
        x0 = x - psf_patch_x // 2
        x1 = x0 + psf_patch_x
        y0 = y - psf_patch_y // 2
        y1 = y0 + psf_patch_y
        psf_x0 = psf_x - psf_patch_x // 2
        psf_y0 = psf_y - psf_patch_y // 2
        psf_x1 = psf_x0 + psf_patch_x
        psf_y1 = psf_y0 + psf_patch_y
        if x0 < 0:
            psf_x0 -= x0
            x0 = 0
        if y0 < 0:
            psf_y0 -= y0
            y0 = 0
        if x1 > image_size_x:
            psf_x1 -= (x1 - image_size_x)
            x1 = image_size_x
        if y1 > image_size_y:
            psf_y1 -= (y1 - image_size_y)
            y1 = image_size_y
        scale = self.clean_parameters.loop_gain * self.image[:, y, x]
        self.image[..., y0:y1, x0:x1] -= (
            scale[:, np.newaxis, np.newaxis] * self.psf[..., psf_y0:psf_y1, psf_x0:psf_x1])
        self.model[..., y, x] += scale
        return (y0, x0, y1, x1)

    def reset(self):
        """Call when the dirty image has changed (including before the first
        use) to recompute the internal caches. Note that this does *not* clear
        the model image: the caller is responsible for setting the initial
        values.
        """
        for y in range(self._tile_max.shape[0]):
            for x in range(self._tile_max.shape[1]):
                self._update_tile(y, x)

    def __call__(self, psf_patch, threshold=0.0):
        """Execute a single CLEAN minor cycle."""
        peak_tile = np.unravel_index(np.argmax(self._tile_max), self._tile_max.shape)
        peak_pos = self._tile_pos[peak_tile]
        peak_value = self._tile_max[peak_tile]
        if peak_value < threshold:
            return None
        (y0, x0, y1, x1) = self._subtract_psf(peak_pos[0], peak_pos[1], psf_patch)
        border = self.clean_parameters.border
        tile_y0 = max((y0 - border) // self.tile_size, 0)
        tile_x0 = max((x0 - border) // self.tile_size, 0)
        tile_y1 = min(accel.divup(y1 - border, self.tile_size), self._tile_max.shape[0])
        tile_x1 = min(accel.divup(x1 - border, self.tile_size), self._tile_max.shape[1])
        for y in range(tile_y0, tile_y1):
            for x in range(tile_x0, tile_x1):
                self._update_tile(y, x)
        return peak_value
