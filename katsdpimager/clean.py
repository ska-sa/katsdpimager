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
"""

from __future__ import division, print_function
import numpy as np
import numba
import katsdpsigproc.accel as accel
import katsdpimager.types
import pkg_resources

#: Use only Stokes I to find peaks
CLEAN_I = 0
#: Use the sum of squares of available Stokes components to find peaks
CLEAN_SUMSQ = 1


class _UpdateTilesTemplate(object):
    """Operation template to compute the peak (including location) for each
    tile intersecting a window.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
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
                'clean_sumsq': (mode == CLEAN_SUMSQ)
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
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    image_shape : tuple of ints
        Shape for the dirty image
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """

    def __init__(self, template, command_queue, image_shape, allocator=None):
        if image_shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        super(_UpdateTiles, self).__init__(command_queue, allocator)
        num_tiles_x = accel.divup(image_shape[2], template.tilex)
        num_tiles_y = accel.divup(image_shape[1], template.tiley)
        image_width = accel.Dimension(image_shape[2])
        image_height = accel.Dimension(image_shape[1])
        image_pols = accel.Dimension(template.num_polarizations, exact=True)
        tiles_width = accel.Dimension(num_tiles_x)
        tiles_height = accel.Dimension(num_tiles_y)
        self.template = template
        self.slots['dirty'] = accel.IOSlot([image_pols, image_height, image_width], template.dtype)
        self.slots['tile_max'] = accel.IOSlot([tiles_height, tiles_width], template.dtype)
        self.slots['tile_pos'] = accel.IOSlot(
                [tiles_height, tiles_width, accel.Dimension(2, exact=True)], np.int32)
        self.kernel = template.program.get_kernel('update_tiles')

    def __call__(self, x0, y0, x1, y1, **kwargs):
        """Update all tiles intersected by the pixel range [x0, x1) by [y0, y1)"""
        self.bind(**kwargs)
        self.ensure_all_bound()
        tile_max = self.buffer('tile_max')
        tile_pos = self.buffer('tile_pos')
        # Map pixel range to a tile range
        x0 = max(x0 // self.template.tilex, 0)
        y0 = max(y0 // self.template.tiley, 0)
        x1 = min(accel.divup(x1, self.template.tilex), tile_max.shape[1])
        y1 = min(accel.divup(y1, self.template.tiley), tile_max.shape[0])
        if x0 < x1 and y0 < y1:
            dirty = self.buffer('dirty')
            self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    dirty.buffer,
                    np.int32(dirty.padded_shape[2]),
                    np.int32(dirty.padded_shape[1] * dirty.padded_shape[2]),
                    np.int32(dirty.shape[1]),
                    np.int32(dirty.shape[2]),
                    tile_max.buffer,
                    tile_pos.buffer,
                    np.int32(tile_max.padded_shape[1]),
                    np.int32(x0), np.int32(y0)
                ],
                global_size=((x1 - x0) * self.template.wgsx, (y1 - y0) * self.template.wgsy),
                local_size=(self.template.wgsx, self.template.wgsy))


class _FindPeakTemplate(object):
    """Find the global peak from per-tile peaks.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
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
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
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
        super(_FindPeak, self).__init__(command_queue, allocator)
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


class _SubtractPsfTemplate(object):
    """Subtract a multiple of the point spread function from the dirty image,
    and add a corresponding single pixel to the model image.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
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
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
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
        super(_SubtractPsf, self).__init__(command_queue, allocator)
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

    def __call__(self, pos, **kwargs):
        """Execute the operation, centering the PSF at `pos` in the image. Note
        that `pos` is specified as (row, col).
        """
        self.bind(**kwargs)
        self.ensure_all_bound()
        dirty = self.buffer('dirty')
        model = self.buffer('model')
        psf = self.buffer('psf')
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
                np.int32(psf.shape[2]),
                np.int32(psf.shape[1]),
                self.buffer('peak_pixel').buffer,
                np.int32(pos[1]), np.int32(pos[0]),
                np.int32(pos[1] - psf.shape[2] // 2),
                np.int32(pos[0] - psf.shape[1] // 2),
                np.float32(self.loop_gain)
            ],
            global_size=(accel.roundup(psf.shape[2], self.template.wgsx),
                         accel.roundup(psf.shape[1], self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy))


class CleanTemplate(object):
    """Composite template for the CLEAN minor cycles.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
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
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    image_parameters : :class:`katsdpimager.parameters.ImageParameters`
        Command-line parameters with image properties
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
        psf_patch = min(image_parameters.pixels, template.clean_parameters.psf_patch)
        psf_shape = (template.num_polarizations, psf_patch, psf_patch)
        self._update_tiles = template._update_tiles.instantiate(
            command_queue, image_shape, allocator)
        tile_shape = self._update_tiles.slots['tile_max'].shape
        self._find_peak = template._find_peak.instantiate(
            command_queue, image_shape, tile_shape, allocator)
        self._subtract_psf = template._subtract_psf.instantiate(
            command_queue, template.clean_parameters.loop_gain, image_shape, psf_shape,
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
        super(Clean, self).__init__(command_queue, ops, compounds, allocator=allocator)
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

    def __call__(self):
        """Run a single minor CLEAN cycle"""
        self.ensure_all_bound()
        self._find_peak()
        # Copy the peak position and value back to the host
        peak_value_device = self.buffer('peak_value')
        peak_pos_device = self.buffer('peak_pos')
        if (isinstance(peak_value_device, accel.SVMArray) or
                isinstance(peak_pos_device, accel.SVMArray)):
            self.command_queue.finish()
        peak_value = peak_value_device.get_async(self.command_queue, self._peak_value_host)
        peak_pos = peak_pos_device.get_async(self.command_queue, self._peak_pos_host)
        self.command_queue.finish()

        self._subtract_psf(peak_pos)
        # Update the tiles
        psf_shape = self.buffer('psf').shape
        x0 = peak_pos[1] - psf_shape[2] // 2
        x1 = x0 + psf_shape[2]
        y0 = peak_pos[0] - psf_shape[1] // 2
        y1 = y0 + psf_shape[1]
        self._update_tiles(x0, y0, x1, y1)
        return peak_value[0]


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


class CleanHost(object):
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
        tiles_x = accel.divup(image.shape[2], self.tile_size)
        tiles_y = accel.divup(image.shape[1], self.tile_size)
        self._tile_max = np.zeros((tiles_y, tiles_x), self.image_parameters.real_dtype)
        self._tile_pos = np.empty((tiles_y, tiles_x, 2), np.int32)

    def _update_tile(self, y, x):
        x0 = x * self.tile_size
        y0 = y * self.tile_size
        x1 = min(x0 + self.tile_size, self.image.shape[2])
        y1 = min(y0 + self.tile_size, self.image.shape[1])
        best_pos, best_value = _tile_peak(
            y0, x0, y1, x1, self.image, self.clean_parameters.mode,
            self.image.dtype.type(0))
        self._tile_max[y, x] = best_value
        self._tile_pos[y, x] = best_pos

    def _subtract_psf(self, y, x):
        psf_size_x = self.psf.shape[2]
        psf_size_y = self.psf.shape[1]
        image_size_x = self.image.shape[2]
        image_size_y = self.image.shape[1]
        # Centre point to subtract at (x, y) in image
        psf_x = psf_size_x // 2
        psf_y = psf_size_y // 2
        x0 = x - psf_x
        x1 = x0 + psf_size_x
        y0 = y - psf_y
        y1 = y0 + psf_size_y
        psf_x0 = 0
        psf_y0 = 0
        psf_x1 = psf_size_x
        psf_y1 = psf_size_y
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

    def __call__(self):
        """Execute a single CLEAN minor cycle."""
        peak_tile = np.unravel_index(np.argmax(self._tile_max), self._tile_max.shape)
        peak = self._tile_max[peak_tile]
        peak_pos = self._tile_pos[peak_tile]
        (y0, x0, y1, x1) = self._subtract_psf(peak_pos[0], peak_pos[1])
        tile_y0 = y0 // self.tile_size
        tile_x0 = x0 // self.tile_size
        tile_y1 = accel.divup(y1, self.tile_size)
        tile_x1 = accel.divup(x1, self.tile_size)
        for y in range(tile_y0, tile_y1):
            for x in range(tile_x0, tile_x1):
                self._update_tile(y, x)
