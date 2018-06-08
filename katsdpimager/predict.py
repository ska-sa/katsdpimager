"""Predict visibilities from a sky model by directly evaluating the RIME"""

from __future__ import division, print_function, absolute_import
import logging

import numpy as np
import pkg_resources

import katsdpsigproc.accel as accel

from . import polarization, types, grid


logger = logging.getLogger(__name__)


class PredictTemplate(object):
    autotune_version = 0

    """Predict visibilities from a sky model by directly evaluating the RIME."""
    def __init__(self, context, real_dtype, num_polarizations, tuning=None):
        if tuning is None:
            tuning = {'wgs': 128}   # TODO autotune
        self.wgs = tuning['wgs']
        self.real_dtype = real_dtype
        self.num_polarizations = num_polarizations
        parameters = {
            'real_type': types.dtype_to_ctype(real_dtype),
            'wgs': self.wgs,
            'num_polarizations': self.num_polarizations
        }
        self.program = accel.build(
            context, 'imager_kernels/predict.mako', parameters,
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return Predict(self, *args, **kwargs)


class Predict(grid.VisOperation):
    def __init__(self, template, command_queue, image_parameters, grid_parameters,
                 max_vis, max_sources, allocator=None):
        if len(image_parameters.polarizations) != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        super(Predict, self).__init__(command_queue, template.num_polarizations, max_vis,
                                      allocator)
        self.template = template
        pol_dim = accel.Dimension(template.num_polarizations, exact=True)
        sources_dim = max(1, max_sources)   # Cannot allocate 0-byte buffer
        self.slots['lmn'] = accel.IOSlot((sources_dim, accel.Dimension(3, exact=True)), np.float32)
        self.slots['flux'] = accel.IOSlot((sources_dim, pol_dim), template.real_dtype)
        self.slots['weights'] = accel.IOSlot(
            (max_vis, accel.Dimension(template.num_polarizations, exact=True)), np.float32)
        self._kernel = self.template.program.get_kernel('predict')
        self._num_sources = 0
        self.max_sources = max_sources
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self._w = 0.0

    def set_sky_model(self, model, phase_centre):
        """Set the sky model.

        This copies data to the device, so if `model` is altered, a subsequent
        call is needed to update the device buffers.

        TODO: could use some optimisation
        """
        lmn = model.lmn(phase_centre)
        # Actually want n-1, not n
        flux = model.flux_density(self.image_parameters.wavelength)
        N = lmn.shape[0]
        if N > self.max_sources:
            raise ValueError('too many sources ({} > {})'.format(N, self.max_sources))
        # For each image polarization, find the corresponding index in IQUV
        # for advanced indexing.
        pol_index = [polarization.STOKES_IQUV.index(pol)
                     for pol in self.image_parameters.polarizations]
        self._num_sources = N
        lmn[:, 2] -= 1
        self.buffer('lmn')[:N] = lmn
        self.buffer('flux')[:N] = flux[:, pol_index]

    @property
    def num_sources(self):
        return self._num_sources

    def set_weights(self, weights):
        """Set statistical weights on visibilities.

        Before calling, set :attr:`num_vis`.
        """
        N = self.num_vis
        if len(weights) != N:
            raise ValueError('Lengths do not match')
        self.buffer('weights')[:N] = weights

    def set_w(self, w):
        self._w = w

    def _run(self):
        if self.num_vis == 0 or self.num_sources == 0:
            return
        uv = self.buffer('uv')
        w_plane = self.buffer('w_plane')
        lmn = self.buffer('lmn')
        flux = self.buffer('flux')
        vis = self.buffer('vis')
        weights = self.buffer('weights')

        # Compute scale factors to reverse the effects of preprocess.cpp
        gp = self.grid_parameters
        # uv is first computed in subpixels, hence need to / by oversample
        uv_scale = self.image_parameters.cell_size / gp.oversample
        w_scale = gp.max_w / ((gp.w_slices - 0.5) * gp.w_planes)
        w_bias = (0.5 - 0.5 * gp.w_planes) * w_scale

        # The above give uvw in length, but we want it in wavelengths. We then
        # convert the (unitless, but possibly not scale-free) Quantity to a
        # plain float.
        uv_scale = float(uv_scale / self.image_parameters.wavelength)
        w_scale = float(w_scale / self.image_parameters.wavelength)
        w_bias = self._w + float(w_bias / self.image_parameters.wavelength)

        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                vis.buffer,
                uv.buffer,
                w_plane.buffer,
                weights.buffer,
                lmn.buffer,
                flux.buffer,
                np.int32(self.num_vis),
                np.int32(self.num_sources),
                np.int32(gp.oversample), np.float32(uv_scale),
                np.float32(w_scale), np.float32(w_bias)
            ],
            global_size=(accel.roundup(self.num_vis, self.template.wgs),),
            local_size=(self.template.wgs,)
        )
