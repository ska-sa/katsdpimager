"""Top-level objects for accelerated imaging, that glue the lower-level
objects together."""

from __future__ import print_function, division
import numpy as np
from katsdpsigproc import accel, fill
from . import grid, fft, clean, types

class ImagingTemplate(object):
    """Template holding all the other templates for imaging."""

    def __init__(self, command_queue, array_parameters, image_parameters,
                 grid_parameters, clean_parameters):
        self.command_queue = command_queue
        self.array_parameters = array_parameters
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self.clean_parameters = clean_parameters
        context = command_queue.context
        image_shape = (len(image_parameters.polarizations), image_parameters.pixels, image_parameters.pixels)
        grid_shape = image_shape
        # Currently none of the kernels accessing the grid need any padding.
        # It would be nice if there was a cleaner way to handle this; possibly
        # by deferring creating of the FFT plan until instantiation.
        padded_grid_shape = grid_shape
        self.gridder = grid.GridderTemplate(context, image_parameters, grid_parameters)
        self.grid_to_image = fft.GridToImageTemplate(
            command_queue, grid_shape, padded_grid_shape,
            image_shape, image_parameters.real_dtype)
        self.clean = clean.CleanTemplate(
            context, clean_parameters, image_parameters.real_dtype, image_shape[0])
        self.scale = fft.ScaleTemplate(
            context, image_parameters.real_dtype, image_shape[0])
        self.clear_grid = fill.FillTemplate(
            context, image_parameters.complex_dtype,
            types.dtype_to_ctype(image_parameters.complex_dtype))
        self.clear_image = fill.FillTemplate(
            context, image_parameters.real_dtype,
            types.dtype_to_ctype(image_parameters.real_dtype))
        self.taper1d = accel.SVMArray(context, (image_parameters.pixels,), image_parameters.real_dtype)
        self.gridder.taper(image_parameters.pixels, self.taper1d)


    def instantiate(self, *args, **kwargs):
        return Imaging(self, *args, **kwargs)


class Imaging(accel.OperationSequence):
    def __init__(self, template, max_vis, allocator=None):
        lm_scale = float(template.image_parameters.pixel_size)
        lm_bias = -0.5 * template.image_parameters.pixels * lm_scale
        image_shape = (len(template.image_parameters.polarizations),
                       template.image_parameters.pixels,
                       template.image_parameters.pixels)
        self._gridder = template.gridder.instantiate(
            template.command_queue, template.array_parameters, max_vis, allocator)
        self._grid_to_image = template.grid_to_image.instantiate(
            lm_scale, lm_bias, allocator)
        self._clean = template.clean.instantiate(
            template.command_queue, template.image_parameters, allocator)
        self._scale = template.scale.instantiate(
            template.command_queue, image_shape, allocator)
        self._clear_grid = template.clear_grid.instantiate(
            template.command_queue, image_shape, allocator)
        self._clear_dirty = template.clear_image.instantiate(
            template.command_queue, image_shape, allocator)
        self._clear_model = template.clear_image.instantiate(
            template.command_queue, image_shape, allocator)
        self._grid_to_image.bind(kernel1d=template.taper1d)
        operations = [
            ('gridder', self._gridder),
            ('grid_to_image', self._grid_to_image),
            ('clean', self._clean),
            ('scale', self._scale),
            ('clear_grid', self._clear_grid),
            ('clear_dirty', self._clear_dirty),
            ('clear_model', self._clear_model)
        ]
        compounds = {
            'uv': ['gridder:uv'],
            'w_plane': ['gridder:w_plane'],
            'vis': ['gridder:vis'],
            'grid': ['gridder:grid', 'grid_to_image:grid', 'clear_grid:data'],
            'layer': ['grid_to_image:layer'],
            'dirty': ['grid_to_image:image', 'clean:dirty', 'clear_dirty:data', 'scale:data'],
            'model': ['clean:model', 'clear_model:data'],
            'psf': ['clean:psf'],
            'tile_max': ['clean:tile_max'],
            'tile_pos': ['clean:tile_pos'],
            'peak_value': ['clean:peak_value'],
            'peak_pos': ['clean:peak_pos'],
            'peak_pixel': ['clean:peak_pixel']
        }
        super(Imaging, self).__init__(template.command_queue, operations, compounds, allocator=allocator)

    def __call__(self):
        raise NotImplementedError()

    def clear_grid(self):
        self.ensure_all_bound()
        self._clear_grid()

    def clear_dirty(self):
        self.ensure_all_bound()
        self._clear_dirty()

    def clear_model(self):
        self.ensure_all_bound()
        self._clear_model()

    def grid(self, *args, **kwargs):
        self.ensure_all_bound()
        self._gridder.grid(*args, **kwargs)

    def grid_to_image(self, w):
        self.ensure_all_bound()
        self._grid_to_image.set_w(w)
        self._grid_to_image()

    def scale_dirty(self, scale_factor):
        self.ensure_all_bound()
        self._scale.set_scale_factor(scale_factor)
        self._scale()

    def clean_reset(self):
        self.ensure_all_bound()
        self._clean.reset()

    def clean_cycle(self):
        self.ensure_all_bound()
        self._clean()


class ImagingHost(object):
    """Host-only equivalent to :class:`Imaging`."""

    def __init__(self, image_parameters, grid_parameters, clean_parameters):
        lm_scale = float(image_parameters.pixel_size)
        lm_bias = -0.5 * image_parameters.pixels * lm_scale
        psf_shape = (len(image_parameters.polarizations),
                     clean_parameters.psf_patch, clean_parameters.psf_patch)
        self._gridder = grid.GridderHost(image_parameters, grid_parameters)
        self._grid = self._gridder.values
        self._layer = np.empty(self._grid.shape, image_parameters.complex_dtype)
        self._dirty = np.empty(self._grid.shape, image_parameters.real_dtype)
        self._model = np.empty(self._grid.shape, image_parameters.real_dtype)
        self._psf = np.empty(psf_shape, image_parameters.real_dtype)
        self._grid_to_image = fft.GridToImageHost(
            self._grid, self._layer, self._dirty,
            self._gridder.taper(image_parameters.pixels), lm_scale, lm_bias)
        self._clean = clean.CleanHost(image_parameters, clean_parameters,
                                      self._dirty, self._psf, self._model)
        self._buffer = {
            'psf': self._psf,
            'dirty': self._dirty,
            'model': self._model,
            'grid': self._grid
        }

    def buffer(self, name):
        return self._buffer[name]

    def clear_grid(self):
        self._grid.fill(0)

    def clear_dirty(self):
        self._dirty.fill(0)

    def clear_model(self):
        self._model.fill(0)

    def grid(self, *args, **kwargs):
        self._gridder.grid(*args, **kwargs)

    def grid_to_image(self, w):
        self._grid_to_image.set_w(w)
        self._grid_to_image()

    def scale_dirty(self, scale_factor):
        self._dirty *= scale_factor[:, np.newaxis, np.newaxis]

    def clean_reset(self):
        self._clean.reset()

    def clean_cycle(self):
        self._clean()
