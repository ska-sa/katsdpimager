"""Deconvolution routines based on CLEAN"""

from __future__ import division, print_function
import numpy as np
import katsdpsigproc.accel as accel
import katsdpimager.types
import pkg_resources

#: Use only Stokes I to find peaks
CLEAN_I = 0
#: Use the sum of squares of available Stokes components to find peaks
CLEAN_SUMSQ = 1


class _UpdateTilesTemplate(object):
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
    def __init__(self, template, command_queue, image_shape, allocator=None):
        if image_shape[2] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        super(_UpdateTiles, self).__init__(command_queue, allocator)
        num_tiles_x = accel.divup(image_shape[1], template.tilex)
        num_tiles_y = accel.divup(image_shape[0], template.tiley)
        image_width = accel.Dimension(image_shape[1])
        image_height = accel.Dimension(image_shape[0])
        image_pols = accel.Dimension(template.num_polarizations, exact=True)
        tiles_width = accel.Dimension(num_tiles_x)
        tiles_height = accel.Dimension(num_tiles_y)
        self.template = template
        self.slots['dirty'] = accel.IOSlot([image_height, image_width, image_pols], template.dtype)
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
                    np.int32(dirty.padded_shape[1]),
                    np.int32(dirty.shape[0]),
                    np.int32(dirty.shape[1]),
                    tile_max.buffer,
                    tile_pos.buffer,
                    np.int32(tile_max.padded_shape[1]),
                    np.int32(x0), np.int32(y0)
                ],
                global_size=((x1 - x0) * self.template.wgsx, (y1 - y0) * self.template.wgsy),
                local_size=(self.template.wgsx, self.template.wgsy))


class _FindPeakTemplate(object):
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
    def __init__(self, template, command_queue, image_shape, tile_shape, allocator=None):
        if image_shape[2] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        super(_FindPeak, self).__init__(command_queue, allocator)
        self.template = template
        image_dims = [accel.Dimension(image_shape[0]), accel.Dimension(image_shape[1]),
                      accel.Dimension(image_shape[2], exact=True)]
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
                np.int32(dirty.padded_shape[1]),
                tile_max.buffer,
                tile_pos.buffer,
                np.int32(tile_max.shape[1]), np.int32(tile_max.shape[0]),
                np.int32(tile_max.padded_shape[1]),
                self.buffer('peak_value').buffer,
                self.buffer('peak_pos').buffer,
                self.buffer('peak_pixel').buffer
            ],
            global_size=(accel.roundup(tile_max.shape[0], self.template.wgsx),
                         accel.roundup(tile_max.shape[1], self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy))


class _SubtractPsfTemplate(object):
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
    def __init__(self, template, command_queue, loop_gain, image_shape, psf_shape, allocator=None):
        super(_SubtractPsf, self).__init__(command_queue, allocator)
        pol_dim = accel.Dimension(template.num_polarizations, exact=True)
        image_dims = [accel.Dimension(image_shape[0]), accel.Dimension(image_shape[1]), pol_dim]
        psf_dims = [accel.Dimension(psf_shape[0]), accel.Dimension(psf_shape[1]), pol_dim]
        self.slots['dirty'] = accel.IOSlot(image_dims, template.dtype)
        self.slots['model'] = accel.IOSlot(image_dims, template.dtype)
        self.slots['psf'] = accel.IOSlot(psf_dims, template.dtype)
        self.slots['peak_pixel'] = accel.IOSlot([pol_dim], template.dtype)
        self.loop_gain = loop_gain
        self.template = template
        self.kernel = template.program.get_kernel('subtract_psf')

    def __call__(self, pos, **kwargs):
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
                np.int32(dirty.shape[1]),
                np.int32(dirty.shape[0]),
                np.int32(dirty.padded_shape[1]),
                psf.buffer,
                np.int32(psf.shape[1]),
                np.int32(psf.shape[0]),
                np.int32(psf.padded_shape[1]),
                self.buffer('peak_pixel').buffer,
                np.int32(pos[1]), np.int32(pos[0]),
                np.int32(pos[1] - psf.shape[1] // 2),
                np.int32(pos[0] - psf.shape[0] // 2),
                np.float32(self.loop_gain)
            ],
            global_size=(accel.roundup(psf.shape[1], self.template.wgsx),
                         accel.roundup(psf.shape[0], self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy))


class CleanTemplate(object):
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
    def __init__(self, template, command_queue, image_parameters, allocator=None):
        if image_parameters.dtype != template.dtype:
            raise ValueError('dtype mismatch')
        image_shape = (image_parameters.pixels, image_parameters.pixels,
                       len(image_parameters.polarizations))
        psf_patch = min(image_parameters.pixels, template.clean_parameters.psf_patch)
        psf_shape = (psf_patch, psf_patch, template.num_polarizations)
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
        super(Clean, self).__init__(command_queue, ops, compounds)
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
        self._update_tiles(0, 0, self.buffer('dirty').shape[1], self.buffer('dirty').shape[0])

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
        x0 = peak_pos[1] - psf_shape[1] // 2
        x1 = x0 + psf_shape[1]
        y0 = peak_pos[0] - psf_shape[0] // 2
        y1 = y0 + psf_shape[0]
        self._update_tiles(x0, y0, x1, y1)
        return peak_value[0]


class CleanHost(object):
    def __init__(self, image_parameters, clean_parameters, image, psf, model):
        self.clean_parameters = clean_parameters
        self.image_parameters = image_parameters
        self.image = image
        self.model = model
        self.psf = psf
        tile_size = self.clean_parameters.tile_size
        tiles_x = accel.divup(image.shape[1], tile_size)
        tiles_y = accel.divup(image.shape[0], tile_size)
        self._tile_max = np.zeros((tiles_y, tiles_x), self.image_parameters.dtype)
        self._tile_pos = np.empty((tiles_y, tiles_x, 2), np.int32)

    def _update_tile(self, y, x):
        tile_size = self.clean_parameters.tile_size
        x0 = x * tile_size
        y0 = y * tile_size
        x1 = min(x0 + tile_size, self.image.shape[1])
        y1 = min(y0 + tile_size, self.image.shape[0])
        tile = self.image[y0:y1, x0:x1, ...]
        if self.clean_parameters.mode == CLEAN_I:
            tile_fn = np.abs(tile[..., 0])
        else:
            tile_fn = np.sum(tile * tile, axis=2)
        pos = np.unravel_index(np.argmax(tile_fn), tile_fn.shape)
        self._tile_max[y, x] = tile_fn[pos]
        self._tile_pos[y, x] = (pos[0] + y0, pos[1] + x0)

    def _subtract_psf(self, y, x):
        psf_x = self.psf.shape[1] // 2
        psf_y = self.psf.shape[0] // 2
        x0 = x - psf_x
        x1 = x0 + self.psf.shape[1]
        y0 = y - psf_y
        y1 = y0 + self.psf.shape[0]
        psf_x0 = 0
        psf_y0 = 0
        psf_x1 = self.psf.shape[1]
        psf_y1 = self.psf.shape[0]
        if x0 < 0:
            psf_x0 -= x0
            x0 = 0
        if y0 < 0:
            psf_y0 -= y0
            y0 = 0
        if x1 > self.image.shape[1]:
            psf_x1 -= (x1 - self.image.shape[1])
            x1 = self.image.shape[1]
        if y1 > self.image.shape[0]:
            psf_y1 -= (y1 - self.image.shape[0])
            y1 = self.image.shape[0]
        scale = self.clean_parameters.loop_gain * self.image[y, x]
        self.image[y0:y1, x0:x1, ...] -= scale * self.psf[psf_y0:psf_y1, psf_x0:psf_x1, ...]
        self.model[y, x] += scale
        return (y0, x0, y1, x1)

    def reset(self):
        for y in range(self._tile_max.shape[0]):
            for x in range(self._tile_max.shape[1]):
                self._update_tile(y, x)

    def __call__(self):
        tile_size = self.clean_parameters.tile_size
        peak_tile = np.unravel_index(np.argmax(self._tile_max), self._tile_max.shape)
        peak = self._tile_max[peak_tile]
        peak_pos = self._tile_pos[peak_tile]
        (y0, x0, y1, x1) = self._subtract_psf(peak_pos[0], peak_pos[1])
        tile_y0 = y0 // tile_size
        tile_x0 = x0 // tile_size
        tile_y1 = accel.divup(y1, tile_size)
        tile_x1 = accel.divup(x1, tile_size)
        for y in range(tile_y0, tile_y1):
            for x in range(tile_x0, tile_x1):
                self._update_tile(y, x)
