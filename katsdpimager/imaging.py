"""Top-level objects for accelerated imaging, that glue the lower-level
objects together."""

import numpy as np
from katsdpsigproc import accel

from . import grid, predict, weight, image, clean
from .profiling import profile_function, profile_device


class ImagingTemplate:
    """Template holding all the other templates for imaging."""

    @profile_function()
    def __init__(self, context, array_parameters, fixed_image_parameters,
                 weight_parameters, fixed_grid_parameters, clean_parameters):
        self.context = context
        self.array_parameters = array_parameters
        self.fixed_image_parameters = fixed_image_parameters
        self.weight_parameters = weight_parameters
        self.fixed_grid_parameters = fixed_grid_parameters
        self.clean_parameters = clean_parameters
        num_polarizations = len(fixed_image_parameters.polarizations)
        self.weights = weight.WeightsTemplate(
            context, weight_parameters.weight_type, num_polarizations)
        self.gridder = grid.GridderTemplate(
            context, fixed_image_parameters, fixed_grid_parameters)
        self.predict = predict.PredictTemplate(
            context, fixed_image_parameters.real_dtype, num_polarizations)
        self.grid_image = image.GridImageTemplate(
            context, fixed_image_parameters.real_dtype)
        self.psf_patch = clean.PsfPatchTemplate(
            context, fixed_image_parameters.real_dtype, num_polarizations)
        self.noise_est = clean.NoiseEstTemplate(
            context, fixed_image_parameters.real_dtype, num_polarizations)
        self.clean = clean.CleanTemplate(
            context, clean_parameters, fixed_image_parameters.real_dtype, num_polarizations)
        self.scale = image.ScaleTemplate(
            context, fixed_image_parameters.real_dtype, num_polarizations)
        self.add_image = image.AddImageTemplate(
            context, fixed_image_parameters.real_dtype, num_polarizations)
        self.apply_primary_beam = image.ApplyPrimaryBeamTemplate(
            context, fixed_image_parameters.real_dtype, num_polarizations)
        if fixed_grid_parameters.degrid:
            self.degridder = grid.DegridderTemplate(
                context, fixed_image_parameters, fixed_grid_parameters)
        else:
            self.degridder = None

    def instantiate(self, *args, **kwargs):
        return Imaging(self, *args, **kwargs)


class _HostBuffer:
    """A host array with supporting utilities for synchronisation."""

    def __init__(self, context, slot):
        self.array = accel.HostArray(slot.shape, slot.dtype, slot.required_padded_shape(),
                                     context=context)
        self.transfer_event = None


def _get_uv(coords):
    """Extract view of uv and sub_uv fields as an Nx4 array."""
    # Check that types are as expected and contiguous
    assert coords.dtype['uv'] == np.dtype(('i2', (2,)))
    assert coords.dtype['sub_uv'] == np.dtype(('i2', (2,)))
    assert (coords.dtype.fields['sub_uv'][1]
            == coords.dtype.fields['uv'][1] + coords.dtype['uv'].itemsize)
    # Create a dtype alias where these are just a single field
    new_dtype = np.dtype(dict(
        names=['uv_sub_uv'],
        formats=[('i2', (4,))],
        offsets=[coords.dtype.fields['uv'][1]],
        itemsize=coords.dtype.itemsize
    ))
    alias = coords.view(new_dtype)
    return alias['uv_sub_uv']


class Imaging(accel.OperationSequence):
    @profile_function()
    def __init__(
            self, template, command_queue,
            image_parameters, grid_parameters,
            max_vis, max_sources, major, allocator=None):
        assert image_parameters.fixed == template.fixed_image_parameters
        assert grid_parameters.fixed == template.fixed_grid_parameters
        self.template = template
        lm_scale = float(image_parameters.pixel_size)
        lm_bias = -0.5 * image_parameters.pixels * lm_scale
        image_shape = (len(image_parameters.fixed.polarizations),
                       image_parameters.pixels,
                       image_parameters.pixels)
        # Currently none of the kernels accessing the layer need any padding.
        # It would be nice if there was a cleaner way to handle this; possibly
        # by deferring creation of the FFT plan until instantiation.
        padded_layer_shape = layer_shape = image_shape[1:]
        fft_plan = template.grid_image.make_fft_plan(layer_shape, padded_layer_shape)

        self._gridder = template.gridder.instantiate(
            command_queue,
            template.array_parameters,
            image_parameters,
            grid_parameters,
            max_vis,
            allocator)
        self._continuum_predict = template.predict.instantiate(
            command_queue, image_parameters, grid_parameters,
            max_vis, max_sources, allocator)
        grid_shape = self._gridder.slots['grid'].shape
        self._weights = template.weights.instantiate(
            command_queue, grid_shape, max_vis, allocator)
        self._weights.robustness = template.weight_parameters.robustness
        self._grid_to_image = template.grid_image.instantiate_grid_to_image(
            command_queue, grid_shape, lm_scale, lm_bias, fft_plan, allocator)
        self._psf_patch = template.psf_patch.instantiate(
            command_queue, image_shape, allocator)
        self._noise_est = template.noise_est.instantiate(
            command_queue, image_shape, template.clean_parameters.border, allocator)
        self._clean = template.clean.instantiate(
            command_queue, image_parameters, allocator)
        self._scale = template.scale.instantiate(
            command_queue, image_shape, allocator)
        self._add_image = template.add_image.instantiate(
            command_queue, image_shape, allocator)
        # Thresholds are set later by self.apply_primary_beam
        self._apply_primary_beam_model = template.apply_primary_beam.instantiate(
            command_queue, image_shape, 0.0, 0.0, allocator)
        self._apply_primary_beam_dirty = template.apply_primary_beam.instantiate(
            command_queue, image_shape, 0.0, np.nan, allocator)

        # TODO: handle taper1d/untaper1d as standard slots
        taper1d = accel.DeviceArray(
            template.context,
            (image_parameters.pixels,),
            image_parameters.fixed.real_dtype)
        taper1d_host = taper1d.empty_like()
        self._gridder.convolve_kernel.taper(image_parameters.pixels, taper1d_host)
        taper1d.set(command_queue, taper1d_host)
        self._grid_to_image.bind(kernel1d=taper1d)
        del taper1d_host
        if grid_parameters.fixed.degrid:
            untaper1d = accel.DeviceArray(
                template.context,
                (image_parameters.pixels,),
                image_parameters.fixed.real_dtype)
            untaper1d_host = untaper1d.empty_like()
            self._predict = template.degridder.instantiate(
                command_queue,
                template.array_parameters,
                image_parameters,
                grid_parameters,
                max_vis,
                allocator)
            self._predict.convolve_kernel.taper(image_parameters.pixels, untaper1d_host)
            untaper1d.set(command_queue, untaper1d_host)
            del untaper1d_host

            degrid_shape = self._predict.slots['grid'].shape
            self._image_to_grid = template.grid_image.instantiate_image_to_grid(
                command_queue, degrid_shape, lm_scale, lm_bias, fft_plan, allocator)
            self._image_to_grid.bind(kernel1d=untaper1d)
        else:
            max_components = min(image_parameters.pixels**2,
                                 (major - 1) * template.clean_parameters.minor)
            self._predict = template.predict.instantiate(
                command_queue, image_parameters, grid_parameters,
                max_vis, max_components, allocator)
        self._model_components = {}
        operations = [
            ('weights', self._weights),
            ('gridder', self._gridder),
            ('predict', self._predict),
            ('continuum_predict', self._continuum_predict),
            ('grid_to_image', self._grid_to_image),
            ('psf_patch', self._psf_patch),
            ('noise_est', self._noise_est),
            ('clean', self._clean),
            ('scale', self._scale),
            ('add_image', self._add_image),
            ('apply_primary_beam_model', self._apply_primary_beam_model),
            ('apply_primary_beam_dirty', self._apply_primary_beam_dirty)
        ]
        compounds = {
            'weights': ['weights:weights', 'predict:weights', 'continuum_predict:weights'],
            'weights_grid': ['weights:grid', 'gridder:weights_grid'],
            'uv': ['weights:uv', 'gridder:uv', 'predict:uv', 'continuum_predict:uv'],
            'w_plane': ['gridder:w_plane', 'predict:w_plane', 'continuum_predict:w_plane'],
            'vis': ['gridder:vis', 'predict:vis', 'continuum_predict:vis'],
            'grid': ['gridder:grid', 'grid_to_image:grid'],
            'layer': ['grid_to_image:layer'],
            'dirty': ['grid_to_image:image', 'noise_est:dirty', 'clean:dirty', 'scale:data',
                      'add_image:dest', 'apply_primary_beam_dirty:data'],
            'model': ['clean:model', 'apply_primary_beam_model:data', 'add_image:src'],
            'psf': ['clean:psf', 'psf_patch:psf'],
            'tile_max': ['clean:tile_max'],
            'tile_pos': ['clean:tile_pos'],
            'peak_value': ['clean:peak_value'],
            'peak_pos': ['clean:peak_pos'],
            'peak_pixel': ['clean:peak_pixel'],
            'beam_power': ['apply_primary_beam_model:beam_power',
                           'apply_primary_beam_dirty:beam_power']
        }
        if grid_parameters.fixed.degrid:
            operations.append(('image_to_grid', self._image_to_grid))
            compounds['degrid'] = ['predict:grid', 'image_to_grid:grid']
            compounds['layer'].append('image_to_grid:layer')
            compounds['model'].append('image_to_grid:image')
        # TODO: could alias weights with something, since it's only needed
        # early on while setting up weights_grid.
        # TODO: could alias noise_est:rank with something, since it's only
        # needed at the start of the minor cycles (it is small though).
        super().__init__(
            command_queue, operations, compounds, allocator=allocator)
        # dirty_to_psf swaps the dirty and PSF images, so they need to have
        # compatible padding.
        for x, y in zip(self.slots['dirty'].dimensions, self.slots['psf'].dimensions):
            x.link(y)

        self.host_buffer = {}
        for name in ['weights', 'uv', 'w_plane', 'vis']:
            if name in self.slots:
                self.host_buffer[name] = _HostBuffer(command_queue.context, self.slots[name])

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

    @profile_function()
    def clear_weights(self):
        self._weights.clear()

    @profile_function()
    def grid_weights(self, uv, weights):
        self._set_buffer('uv', len(uv), uv, (np.s_[:2],))
        self._set_buffer('weights', len(uv), weights)
        self._weights.grid(len(uv))

    @profile_function()
    def finalize_weights(self):
        return self._weights.finalize()

    @profile_function()
    def clear_grid(self):
        with profile_device(self.command_queue, 'clear_grid'):
            self.buffer('grid').zero(self.command_queue)

    @profile_function()
    def clear_dirty(self):
        with profile_device(self.command_queue, 'clear_dirty'):
            self.buffer('dirty').zero(self.command_queue)

    @profile_function()
    def clear_model(self):
        with profile_device(self.command_queue, 'clear_model'):
            self.buffer('model').zero(self.command_queue)
        self._model_components.clear()

    def _set_buffer(self, name, N, data, extra_index=()):
        if len(data) != N:
            raise ValueError('Lengths do not match')
        host = self.host_buffer[name]
        device = self.buffer(name)
        if host.transfer_event is not None:
            host.transfer_event.wait()
        idx = (np.s_[:N],) + extra_index
        host.array[idx] = data
        device.set_region(self.command_queue, host.array, idx, idx, blocking=False)
        host.transfer_event = self.command_queue.enqueue_marker()

    def _set_uv(self, coords):
        N = self.num_vis
        if len(coords) != N:
            raise ValueError('Lengths do not match')
        host = self.host_buffer['uv']
        device = self.buffer('uv')
        if host.transfer_event is not None:
            host.transfer_event.wait()
        host.array[:N] = _get_uv(coords)
        device.set_region(self.command_queue, host.array, np.s_[:N], np.s_[:N], blocking=False)
        host.transfer_event = self.command_queue.enqueue_marker()

    @profile_function()
    def set_coordinates(self, coords):
        """Set UVW coordinates.

        Parameters
        ----------
        coords
            Structured array with fields ``uv``, ``sub_uv`` and ``w_plane``. The
            former two must each be a 2-element int16 array, and they must be
            contiguous in the structure.
        """
        self._set_uv(coords)
        self._set_buffer('w_plane', self.num_vis, coords['w_plane'])

    @profile_function()
    def set_vis(self, vis):
        self._set_buffer('vis', self.num_vis, vis)

    @profile_function()
    def set_weights(self, weights):
        """Set statistical weights for prediction"""
        self._set_buffer('weights', self.num_vis, weights)

    @profile_function()
    def grid(self):
        self._gridder()

    @profile_function()
    def predict(self, w):
        if not self.template.fixed_grid_parameters.degrid:
            self._predict.set_w(w)
        self._predict()

    @profile_function()
    def continuum_predict(self, w):
        self._continuum_predict.set_w(w)
        self._continuum_predict()

    def set_sky_model(self, sky_model, phase_centre):
        self._continuum_predict.set_sky_model(sky_model, phase_centre)

    @profile_function()
    def grid_to_image(self, w):
        self._grid_to_image.set_w(w)
        self._grid_to_image()

    @profile_function()
    def model_to_grid(self, w):
        if not self._image_to_grid:
            raise RuntimeError('Can only use model_to_grid with degridding')
        self._image_to_grid.set_w(w)
        self._image_to_grid()

    @profile_function()
    def model_to_predict(self):
        if self.template.fixed_grid_parameters.degrid:
            raise RuntimeError('Can only use model_to_predict with direct prediction')
        self._predict.set_sky_image(self._model_components)

    @profile_function()
    def scale_dirty(self, scale_factor):
        self._scale.set_scale_factor(scale_factor)
        self._scale()

    @profile_function()
    def add_model_to_dirty(self):
        self._add_image()

    @profile_function()
    def apply_primary_beam(self, threshold):
        """Applies primary beam power to both model and dirty images."""
        self._apply_primary_beam_model.threshold = threshold
        self._apply_primary_beam_model()
        self._apply_primary_beam_dirty.threshold = threshold
        self._apply_primary_beam_dirty()

    @profile_function()
    def dirty_to_psf(self):
        dirty = self.buffer('dirty')
        psf = self.buffer('psf')
        self.bind(dirty=psf, psf=dirty)

    @profile_function()
    def psf_patch(self):
        return self._psf_patch(self.template.clean_parameters.psf_cutoff,
                               self.template.clean_parameters.psf_limit)

    @profile_function()
    def noise_est(self):
        return self._noise_est()

    @profile_function()
    def clean_reset(self):
        self._clean.reset()

    @profile_function()
    def clean_cycle(self, psf_patch, threshold=0.0):
        peak_value, peak_pos, model_pixel = self._clean(psf_patch, threshold)
        if peak_pos is not None:
            try:
                self._model_components[peak_pos] += model_pixel
            except KeyError:
                self._model_components[peak_pos] = model_pixel
        return peak_value

    @profile_function(labels=['name'])
    def get_buffer(self, name):
        """Get the contents of a buffer as a numpy array."""
        return self.buffer(name).get(self.command_queue)

    @profile_function(labels=['name'])
    def set_buffer(self, name, data):
        """Copy a numpy array to a buffer (blocking).

        Because this is blocking, it is not considered a high performance
        path. It exists to simplify compatibility with :class:`ImagingHost`.
        """
        self.buffer(name).set(self.command_queue, data)

    def free_buffer(self, name):
        """Free the memory associated with a given buffer.

        It is the caller's responsibility to ensure that the buffer is not
        needed in future operations.
        """
        if name in self.slots:
            self.slots[name].bind(None)


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
        self._layer = np.empty(self._grid.shape, image_parameters.fixed.complex_dtype)
        self._dirty = np.empty(self._grid.shape, image_parameters.fixed.real_dtype)
        self._model = np.empty(self._grid.shape, image_parameters.fixed.real_dtype)
        self._psf = np.empty(self._grid.shape, image_parameters.fixed.real_dtype)
        self._beam_power = np.empty(self._grid.shape[1:], image_parameters.fixed.real_dtype)
        self._grid_to_image = image.GridToImageHost(
            self._grid, self._layer, self._dirty,
            self._gridder.kernel.taper(image_parameters.pixels), lm_scale, lm_bias)
        self._clean = clean.CleanHost(image_parameters, clean_parameters,
                                      self._dirty, self._psf, self._model)
        self._model_components = {}
        if grid_parameters.fixed.degrid:
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
            'weights_grid': self._weights_grid,
            'beam_power': self._beam_power
        }
        if self._degrid is not None:
            self._buffer['degrid'] = self._degrid

    def buffer(self, name):
        return self._buffer[name]

    def get_buffer(self, name):
        return self._buffer[name]

    def set_buffer(self, name, data):
        self._buffer[name][()] = data

    def free_buffer(self, name):
        if name in self._buffer:
            self._buffer[name] = None

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
        self._model_components.clear()

    def set_sky_model(self, sky_model, phase_centre):
        self._continuum_predict.set_sky_model(sky_model, phase_centre)

    def set_coordinates(self, coords):
        self._gridder.set_coordinates(coords.uv, coords.sub_uv, coords.w_plane)
        self._predict.set_coordinates(coords.uv, coords.sub_uv, coords.w_plane)
        self._continuum_predict.set_coordinates(coords.uv, coords.sub_uv, coords.w_plane)

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
        self._predict.set_sky_image(self._model_components)

    def scale_dirty(self, scale_factor):
        self._dirty *= scale_factor[:, np.newaxis, np.newaxis]

    def add_model_to_dirty(self):
        self._dirty += self._model

    def apply_primary_beam(self, threshold):
        mask = (self._beam_power < threshold)[np.newaxis, ...]
        self._model /= self._beam_power
        self._model[mask] = 0.0
        self._dirty /= self._beam_power
        self._dirty[mask] = np.nan

    def dirty_to_psf(self):
        self._psf[:] = self._dirty

    def psf_patch(self):
        return clean.psf_patch_host(self._psf,
                                    self._clean_parameters.psf_cutoff,
                                    self._clean_parameters.psf_limit)

    def noise_est(self):
        return clean.noise_est_host(self._dirty, self._clean_parameters.border)

    def clean_reset(self):
        self._clean.reset()

    def clean_cycle(self, psf_patch, threshold=0.0):
        peak_value, peak_pos, model_pixel = self._clean(psf_patch, threshold)
        if peak_pos is not None:
            try:
                self._model_components[peak_pos] += model_pixel
            except KeyError:
                self._model_components[peak_pos] = model_pixel
        return peak_value
