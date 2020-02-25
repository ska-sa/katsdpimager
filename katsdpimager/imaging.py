"""Top-level objects for accelerated imaging, that glue the lower-level
objects together."""

import numpy as np
from katsdpsigproc import accel

from . import grid, predict, weight, image, clean


class ImagingTemplate:
    """Template holding all the other templates for imaging."""

    def __init__(self, command_queue, array_parameters, image_parameters,
                 weight_parameters, grid_parameters, clean_parameters):
        self.command_queue = command_queue
        self.array_parameters = array_parameters
        self.image_parameters = image_parameters
        self.weight_parameters = weight_parameters
        self.grid_parameters = grid_parameters
        self.clean_parameters = clean_parameters
        context = command_queue.context
        num_polarizations = len(image_parameters.polarizations)
        image_shape = (num_polarizations,
                       image_parameters.pixels,
                       image_parameters.pixels)
        # Currently none of the kernels accessing the layer need any padding.
        # It would be nice if there was a cleaner way to handle this; possibly
        # by deferring creation of the FFT plan until instantiation.
        padded_layer_shape = layer_shape = image_shape[1:]
        self.weights = weight.WeightsTemplate(
            context, weight_parameters.weight_type, num_polarizations)
        self.gridder = grid.GridderTemplate(context, image_parameters, grid_parameters)
        self.predict = predict.PredictTemplate(
            context, image_parameters.real_dtype, num_polarizations)
        self.grid_image = image.GridImageTemplate(
            command_queue, layer_shape, padded_layer_shape, image_parameters.real_dtype)
        self.psf_patch = clean.PsfPatchTemplate(
            context, image_parameters.real_dtype, num_polarizations)
        self.noise_est = clean.NoiseEstTemplate(
            context, image_parameters.real_dtype, num_polarizations, clean_parameters.mode)
        self.clean = clean.CleanTemplate(
            context, clean_parameters, image_parameters.real_dtype, num_polarizations)
        self.scale = image.ScaleTemplate(
            context, image_parameters.real_dtype, num_polarizations)
        self.taper1d = accel.SVMArray(
            context, (image_parameters.pixels,), image_parameters.real_dtype)
        self.gridder.convolve_kernel.taper(image_parameters.pixels, self.taper1d)
        if grid_parameters.degrid:
            self.untaper1d = accel.SVMArray(
                context, (image_parameters.pixels,), image_parameters.real_dtype)
            self.degridder = grid.DegridderTemplate(context, image_parameters, grid_parameters)
            self.degridder.convolve_kernel.taper(image_parameters.pixels, self.untaper1d)
        else:
            self.untaper1d = None
            self.degridder = None

    def instantiate(self, *args, **kwargs):
        return Imaging(self, *args, **kwargs)


class Imaging(accel.OperationSequence):
    def __init__(self, template, max_vis, max_sources, major, allocator=None):
        self.template = template
        lm_scale = float(template.image_parameters.pixel_size)
        lm_bias = -0.5 * template.image_parameters.pixels * lm_scale
        image_shape = (len(template.image_parameters.polarizations),
                       template.image_parameters.pixels,
                       template.image_parameters.pixels)
        self._gridder = template.gridder.instantiate(
            template.command_queue, template.array_parameters, max_vis, allocator)
        self._continuum_predict = template.predict.instantiate(
            template.command_queue, template.image_parameters, template.grid_parameters,
            max_vis, max_sources, allocator)
        grid_shape = self._gridder.slots['grid'].shape
        self._weights = template.weights.instantiate(
            template.command_queue, grid_shape, max_vis, allocator)
        self._weights.robustness = template.weight_parameters.robustness
        self._grid_to_image = template.grid_image.instantiate_grid_to_image(
            grid_shape, lm_scale, lm_bias, allocator)
        self._psf_patch = template.psf_patch.instantiate(
            template.command_queue, image_shape, allocator)
        self._noise_est = template.noise_est.instantiate(
            template.command_queue, image_shape, template.clean_parameters.border, allocator)
        self._clean = template.clean.instantiate(
            template.command_queue, template.image_parameters, allocator)
        self._scale = template.scale.instantiate(
            template.command_queue, image_shape, allocator)
        self._grid_to_image.bind(kernel1d=template.taper1d)
        if template.grid_parameters.degrid:
            self._predict = template.degridder.instantiate(
                template.command_queue, template.array_parameters, max_vis, allocator)
            degrid_shape = self._predict.slots['grid'].shape
            self._image_to_grid = template.grid_image.instantiate_image_to_grid(
                degrid_shape, lm_scale, lm_bias, allocator)
            self._image_to_grid.bind(kernel1d=template.untaper1d)
        else:
            max_components = min(template.image_parameters.pixels**2,
                                 (major - 1) * template.clean_parameters.minor)
            self._predict = template.predict.instantiate(
                template.command_queue, template.image_parameters, template.grid_parameters,
                max_vis, max_components, allocator)
        operations = [
            ('weights', self._weights),
            ('gridder', self._gridder),
            ('predict', self._predict),
            ('continuum_predict', self._continuum_predict),
            ('grid_to_image', self._grid_to_image),
            ('psf_patch', self._psf_patch),
            ('noise_est', self._noise_est),
            ('clean', self._clean),
            ('scale', self._scale)
        ]
        compounds = {
            'weights': ['weights:weights'],
            'weights_grid': ['weights:grid', 'gridder:weights_grid'],
            'uv': ['weights:uv', 'gridder:uv', 'predict:uv', 'continuum_predict:uv'],
            'w_plane': ['gridder:w_plane', 'predict:w_plane', 'continuum_predict:w_plane'],
            'vis': ['gridder:vis', 'predict:vis', 'continuum_predict:vis'],
            'predict_weights': ['predict:weights', 'continuum_predict:weights'],
            'grid': ['gridder:grid', 'grid_to_image:grid'],
            'layer': ['grid_to_image:layer'],
            'dirty': ['grid_to_image:image', 'noise_est:dirty', 'clean:dirty', 'scale:data'],
            'model': ['clean:model'],
            'psf': ['clean:psf', 'psf_patch:psf'],
            'tile_max': ['clean:tile_max'],
            'tile_pos': ['clean:tile_pos'],
            'peak_value': ['clean:peak_value'],
            'peak_pos': ['clean:peak_pos'],
            'peak_pixel': ['clean:peak_pixel']
        }
        if template.grid_parameters.degrid:
            operations.append(('image_to_grid', self._image_to_grid))
            compounds['degrid'] = ['predict:grid', 'image_to_grid:grid']
            compounds['layer'].append('image_to_grid:layer')
            compounds['model'].append('image_to_grid:image')
        # TODO: could alias weights with something, since it's only needed
        # early on while setting up weights_grid.
        # TODO: could alias noise_est:rank with something, since it's only
        # needed at the start of the minor cycles (it is small though).
        super().__init__(
            template.command_queue, operations, compounds, allocator=allocator)
        # dirty_to_psf swaps the dirty and PSF images, so they need to have
        # compatible padding.
        for x, y in zip(self.slots['dirty'].dimensions, self.slots['psf'].dimensions):
            x.link(y)

    def __call__(self, **kwargs):
        raise NotImplementedError()

    @property
    def num_vis(self):
        return self._gridder.num_vis

    @num_vis.setter
    def num_vis(self, value):
        self._gridder.num_vis = value
        self._predict.num_vis = value
        self._continuum_predict.num_vis = value

    def clear_weights(self):
        self.ensure_all_bound()
        self._weights.clear()

    def grid_weights(self, uv, weights):
        self.ensure_all_bound()
        self._weights.grid(uv, weights)

    def finalize_weights(self):
        self.ensure_all_bound()
        return self._weights.finalize()

    def clear_grid(self):
        self.ensure_all_bound()
        self.buffer('grid').zero(self.command_queue)

    def clear_dirty(self):
        self.ensure_all_bound()
        self.buffer('dirty').zero(self.command_queue)

    def clear_model(self):
        self.ensure_all_bound()
        self.buffer('model').zero(self.command_queue)

    def set_coordinates(self, *args, **kwargs):
        self.ensure_all_bound()
        # The gridder and predictors share their coordinates, so we can
        # use any here.
        self._gridder.set_coordinates(*args, **kwargs)

    def set_vis(self, *args, **kwargs):
        self.ensure_all_bound()
        # The gridder and predicters share their visibilities, so we
        # can use any here.
        self._gridder.set_vis(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        """Set statistical weights for prediction"""
        self.ensure_all_bound()
        # The predicters share their weights, so we can use any here.
        self._predict.set_weights(*args, **kwargs)

    def grid(self):
        self.ensure_all_bound()
        self._gridder()

    def predict(self, w):
        self.ensure_all_bound()
        if not self.template.grid_parameters.degrid:
            self._predict.set_w(w)
        self._predict()

    def continuum_predict(self, w):
        self.ensure_all_bound()
        self._continuum_predict.set_w(w)
        self._continuum_predict()

    def set_sky_model(self, sky_model, phase_centre):
        self.ensure_all_bound()
        self._continuum_predict.set_sky_model(sky_model, phase_centre)

    def grid_to_image(self, w):
        self.ensure_all_bound()
        self._grid_to_image.set_w(w)
        self._grid_to_image()

    def model_to_grid(self, w):
        if not self._image_to_grid:
            raise RuntimeError('Can only use model_to_grid with degridding')
        self.ensure_all_bound()
        self._image_to_grid.set_w(w)
        self._image_to_grid()

    def model_to_predict(self):
        if self.template.grid_parameters.degrid:
            raise RuntimeError('Can only use model_to_predict with direct prediction')
        # It's computed on the CPU, so we need to be sure that device work is complete
        self._predict.command_queue.finish()
        self._predict.set_sky_image(self.buffer('model'))

    def scale_dirty(self, scale_factor):
        self.ensure_all_bound()
        self._scale.set_scale_factor(scale_factor)
        self._scale()

    def dirty_to_psf(self):
        dirty = self.buffer('dirty')
        psf = self.buffer('psf')
        self.bind(dirty=psf, psf=dirty)

    def psf_patch(self):
        self.ensure_all_bound()
        return self._psf_patch(self.template.clean_parameters.psf_cutoff,
                               self.template.clean_parameters.psf_limit)

    def noise_est(self):
        self.ensure_all_bound()
        return self._noise_est()

    def clean_reset(self):
        self.ensure_all_bound()
        self._clean.reset()

    def clean_cycle(self, psf_patch, threshold=0.0):
        self.ensure_all_bound()
        return self._clean(psf_patch, threshold)


class ImagingHost:
    """Host-only equivalent to :class:`Imaging`."""

    def __init__(self, image_parameters, weight_parameters, grid_parameters, clean_parameters):
        lm_scale = float(image_parameters.pixel_size)
        lm_bias = -0.5 * image_parameters.pixels * lm_scale
        self._clean_parameters = clean_parameters
        self._image_parameters = image_parameters
        self._gridder = grid.GridderHost(image_parameters, grid_parameters)
        self._grid = self._gridder.values
        self._continuum_predict = predict.PredictHost(image_parameters, grid_parameters)
        self._weights_grid = self._gridder.weights_grid
        self._weights = weight.WeightsHost(weight_parameters.weight_type, self._weights_grid)
        self._weights.robustness = weight_parameters.robustness
        self._layer = np.empty(self._grid.shape, image_parameters.complex_dtype)
        self._dirty = np.empty(self._grid.shape, image_parameters.real_dtype)
        self._model = np.empty(self._grid.shape, image_parameters.real_dtype)
        self._psf = np.empty(self._grid.shape, image_parameters.real_dtype)
        self._grid_to_image = image.GridToImageHost(
            self._grid, self._layer, self._dirty,
            self._gridder.kernel.taper(image_parameters.pixels), lm_scale, lm_bias)
        self._clean = clean.CleanHost(image_parameters, clean_parameters,
                                      self._dirty, self._psf, self._model)
        if grid_parameters.degrid:
            self._predict = grid.DegridderHost(image_parameters, grid_parameters)
            self._degrid = self._predict.values
            self._image_to_grid = image.ImageToGridHost(
                self._degrid, self._layer, self._model,
                self._predict.kernel.taper(image_parameters.pixels), lm_scale, lm_bias)
        else:
            self._predict = predict.PredictHost(image_parameters, grid_parameters)
            self._degrid = None
            self._image_to_grid = None
        self._buffer = {
            'psf': self._psf,
            'dirty': self._dirty,
            'model': self._model,
            'grid': self._grid,
            'weights_grid': self._weights_grid
        }
        if self._degrid is not None:
            self._buffer['degrid'] = self._degrid

    def buffer(self, name):
        return self._buffer[name]

    @property
    def num_vis(self):
        return self._gridder.num_vis

    @num_vis.setter
    def num_vis(self, value):
        self._gridder.num_vis = value
        self._predict.num_vis = value
        self._continuum_predict.num_vis = value

    def clear_weights(self):
        self._weights_grid.fill(0)

    def grid_weights(self, uv, weights):
        self._weights.grid(uv, weights)

    def finalize_weights(self):
        return self._weights.finalize()

    def clear_grid(self):
        self._grid.fill(0)

    def clear_dirty(self):
        self._dirty.fill(0)

    def clear_model(self):
        self._model.fill(0)

    def set_sky_model(self, sky_model, phase_centre):
        self._continuum_predict.set_sky_model(sky_model, phase_centre)

    def set_coordinates(self, *args, **kwargs):
        self._gridder.set_coordinates(*args, **kwargs)
        self._predict.set_coordinates(*args, **kwargs)
        self._continuum_predict.set_coordinates(*args, **kwargs)

    def set_vis(self, *args, **kwargs):
        self._gridder.set_vis(*args, **kwargs)
        self._predict.set_vis(*args, **kwargs)
        self._continuum_predict.set_vis(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        self._predict.set_weights(*args, **kwargs)
        self._continuum_predict.set_weights(*args, **kwargs)

    def grid(self):
        self._gridder()

    def predict(self, w):
        if self._degrid is None:
            self._predict.set_w(w)
        self._predict()

    def continuum_predict(self, w):
        self._continuum_predict.set_w(w)
        self._continuum_predict()

    def grid_to_image(self, w):
        self._grid_to_image.set_w(w)
        self._grid_to_image()

    def model_to_grid(self, w):
        if self._degrid is None:
            raise RuntimeError('Can only use model_to_grid with degridding')
        self._image_to_grid.set_w(w)
        self._image_to_grid()

    def model_to_predict(self):
        if self._degrid is not None:
            raise RuntimeError('Can only use model_to_predict with direct prediction')
        self._predict.set_sky_image(self._model)

    def scale_dirty(self, scale_factor):
        self._dirty *= scale_factor[:, np.newaxis, np.newaxis]

    def dirty_to_psf(self):
        self._psf[:] = self._dirty

    def psf_patch(self):
        return clean.psf_patch_host(self._psf,
                                    self._clean_parameters.psf_cutoff,
                                    self._clean_parameters.psf_limit)

    def noise_est(self):
        return clean.noise_est_host(self._dirty, self._clean_parameters.border,
                                    self._clean_parameters.mode)

    def clean_reset(self):
        self._clean.reset()

    def clean_cycle(self, psf_patch, threshold=0.0):
        return self._clean(psf_patch, threshold)
