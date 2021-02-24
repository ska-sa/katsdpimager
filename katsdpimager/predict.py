# -*- coding: utf-8 -*-
"""Predict visibilities from a sky model by directly evaluating the RIME.

It uses a sky model to predict visibilities and subtract them from previous
values. No effects are modelled other than geometric delay (K Jones).
Direction-independent antenna effects (G Jones) should be already incorporated
into the visibilities which are being subtracted from, and time- and
antenna-independent DDEs should be included in the sky model. General DDEs are
not supported at all.

The visibilities are assumed to have already been passed through the
:mod:`preprocess` module, and hence UVW coordinates will be quantised.
"""

import logging

import numpy as np
import numba
import pkg_resources
from astropy import units
from katsdpsigproc import accel, tune

from . import polarization, types, grid, parameters
from .profiling import profile_function, profile_device


logger = logging.getLogger(__name__)


def _extract_sky_model(image_parameters, grid_parameters, model, phase_centre):
    """Extract lmn coordinates and fluxes from a sky model.

    Parameters
    ----------
    image_parameters : :class:`~.ImageParameters`
        Image parameters
    grid_parameters : :class:`~.GridParameters`
        Grid parameters for the corresponding UVW coordinates, for aliasing correction.
    model : :class:`~.SkyModel`
        The input sky model

    Returns
    -------
    lmn : array of float32
        Array of shape N×3, containing l, m, n-1
    flux : array of float32
        Array of shape N×P, containing the Stokes parameters specified by
        `image_parameters` for each source.
    """
    ip = image_parameters
    lmn = model.lmn(phase_centre)
    # n -> n-1. Note that it can't be in-place because lmn returns a read-only
    # array from a cache.
    lmn = lmn - np.array([[0, 0, 1]], dtype=lmn.dtype)
    flux = model.flux_density(ip.wavelength)
    # When we convert the subtracted visibilities back to a dirty image,
    # we compensate for the UV coordinate quantisation. However, in this case
    # we're already predicting with the quantised coordinates, so we need to
    # reverse that taper.
    #
    # I'm not sure that this is mathematically defensible, but the results
    # look good.
    taper = np.sinc(lmn[:, 0:2] / float(ip.image_size * grid_parameters.fixed.oversample))
    flux *= np.product(taper, axis=1, keepdims=True)

    # For each image polarization, find the corresponding index in IQUV
    # for advanced indexing.
    pol_index = [polarization.STOKES_IQUV.index(pol) for pol in ip.fixed.polarizations]
    flux = flux[:, pol_index]
    return lmn.astype(np.float32), flux.astype(np.float32)


def _extract_sky_image(image_parameters, grid_parameters, components):
    """Turn a sky model image into a model for direct prediction.

    The return values have the same meanings as for :func:`_extract_sky_model`.

    Parameters
    ----------
    image_parameters : :class:`~.ImageParameters`
        Image parameters
    grid_parameters : :class:`~.GridParameters`
        Grid parameters for the corresponding UVW coordinates, for aliasing correction.
    components : Mapping[tuple[int, int], np.ndarray]
        CLEAN components, indexed by (y, x) coordinates, with each value being a 1D
        array of per-polarization values.

    Returns
    -------
    lmn : array of float32
        Array of shape N×3, containing l, m, n-1
    flux : array of float
        Array of shape N×P, containing the Stokes parameters specified by
        `image_parameters` for each source.
    """
    dtype = image_parameters.fixed.real_dtype
    pols = len(image_parameters.fixed.polarizations)
    N = len(components)
    lmn = np.empty((N, 3), np.float32)
    flux = np.empty((N, pols), dtype)

    # Note: These are currently done in double precision to avoid cancellation
    # issues in computing n-1.
    pixel_size = float(image_parameters.pixel_size)   # unitless Quantity -> plain float
    x = np.array([pos[1] for pos in components])
    y = np.array([pos[0] for pos in components])
    l = (x - 0.5 * image_parameters.pixels) * pixel_size
    m = (y - 0.5 * image_parameters.pixels) * pixel_size
    n1 = np.sqrt(1.0 - (np.square(l) + np.square(m))) - 1.0
    lmn[:, 0] = l
    lmn[:, 1] = m
    lmn[:, 2] = n1
    flux[:] = list(components.values())
    # See :func:`_extract_sky_model` for explanation of this tapering
    taper_scale = float(image_parameters.image_size * grid_parameters.fixed.oversample)
    # TODO: single precision probably suffices for l and m here
    taper = np.sinc(l / taper_scale) * np.sinc(m / taper_scale)
    flux *= taper[:, np.newaxis]
    return lmn, flux


def _uvw_scale_bias(image_parameters, grid_parameters):
    r"""Compute scale and bias values to turn UVW indices back into coordinates.

    The :mod:`preprocess` module quantises UVW coordinates; the values computed
    by this function can be used to convert them back to coordinates in
    wavelengths. Given quantised UV with grid cell :math:`g` and subpixel `s`,
    and W plane of :math:`w_p` in a W slice with centre :math:`w_0` wavelengths,
    the coordinates in wavelengths are

    .. math::

       uv &= uv_{scale}(oversample\cdot g + s + 0.5)\\
       w &= w_0 + w_{scale}w_p + w_bias
    """
    ip = image_parameters
    gp = grid_parameters

    # uv is first computed in subpixels, hence need to / by oversample
    uv_scale = ip.cell_size / gp.fixed.oversample
    w_scale = gp.fixed.max_w / ((gp.w_slices - 0.5) * gp.w_planes)

    # The above give uvw in length, but we want it in wavelengths. We then
    # convert the (unitless, but possibly not scale-free) Quantity to a
    # plain float.
    uv_scale = float(uv_scale / ip.wavelength)
    w_scale = float(w_scale / ip.wavelength)
    w_bias = (0.5 - 0.5 * gp.w_planes) * w_scale
    return uv_scale, w_scale, w_bias


class PredictTemplate:
    """Predict visibilities from a sky model by directly evaluating the RIME.

    Parameters
    ----------
    context : :class:`katsdpsigproc.abc.AbstractContext`
        Context for which kernels will be compiled
    real_dtype : {np.float32, np.float64}
        Data type for internal accumulation of values. This does not affect
        the buffer sizes.
    num_polarizations : int
        Number of polarizations
    tuning : dict, optional
        Kernel tuning parameters. The keys are:

        wgs
            Work group size
    """

    autotune_version = 0

    def __init__(self, context, real_dtype, num_polarizations, tuning=None):
        if tuning is None:
            tuning = self.autotune(context, real_dtype, num_polarizations)
        self.wgs = tuning['wgs']
        self.real_dtype = real_dtype
        self.num_polarizations = num_polarizations
        params = {
            'real_type': types.dtype_to_ctype(real_dtype),
            'wgs': self.wgs,
            'num_polarizations': self.num_polarizations
        }
        self.program = accel.build(
            context, 'imager_kernels/predict.mako', params,
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    @classmethod
    @tune.autotuner(test={'wgs': 128})
    def autotune(cls, context, real_dtype, num_polarizations):
        queue = context.create_tuning_command_queue()
        num_vis = 1000000
        # Need at least as many sources as the workgroup size to achieve full
        # throughput.
        num_sources = 1024
        vis = accel.DeviceArray(context, (num_vis, num_polarizations), dtype=np.complex64)
        uv = accel.DeviceArray(context, (num_vis, 4), dtype=np.int16)
        w_plane = accel.DeviceArray(context, (num_vis,), dtype=np.int16)
        weights = accel.DeviceArray(context, (num_vis, num_polarizations), dtype=np.float32)
        lmn = accel.DeviceArray(context, (num_sources, 3), dtype=np.float32)
        flux = accel.DeviceArray(context, (num_sources, num_polarizations), dtype=np.float32)
        # The values don't really matter, but we want to avoid non-finites
        # which would skew performance.
        vis.zero(queue)
        uv.zero(queue)
        w_plane.zero(queue)
        weights.zero(queue)
        lmn.zero(queue)
        flux.zero(queue)

        # The values don't make any difference to auto-tuning; they just affect
        # scale factors that have no impact on control flow.
        image_parameters = parameters.ImageParameters(
            parameters.FixedImageParameters(
                polarizations=polarization.STOKES_IQUV[:num_polarizations],
                dtype=real_dtype
            ),
            q_fov=1.0,
            image_oversample=3.0,
            frequency=0.21 * units.m,
            array=None,
            pixel_size=1 * units.arcsec,
            pixels=4096
        )
        grid_parameters = parameters.GridParameters(
            parameters.FixedGridParameters(
                antialias_width=7.0,
                oversample=8,
                image_oversample=5,
                max_w=1 * units.m,
                kernel_width=64
            ),
            w_slices=5,
            w_planes=5
        )

        def generate(wgs):
            template = cls(context, real_dtype, num_polarizations, {'wgs': wgs})
            fn = template.instantiate(queue, image_parameters, grid_parameters,
                                      num_vis, num_sources)
            fn.bind(vis=vis, uv=uv, w_plane=w_plane, weights=weights, lmn=lmn, flux=flux)
            fn.num_vis = num_vis
            fn._num_sources = num_sources   # Skip actually creating a sky model
            return tune.make_measure(queue, fn)

        return tune.autotune(generate, wgs=[64, 128, 256, 512, 1024])

    def instantiate(self, *args, **kwargs):
        return Predict(self, *args, **kwargs)


class Predict(grid.VisOperation):
    """Instantiation of :class:`PredictTemplate`.

    Before using a constructed instance, it's necessary to first

    1. Configure the visibilities, by setting :attr:`num_vis` and
       :meth:`set_w` and populating the visibility-related buffers.
    2. Configure the sources, by calling :meth:`set_sky_model` or
       :meth:`set_sky_image`.

    .. rubric:: Slots

    In addition to those specified in :class:`~.grid.VisOperation`:

    **lmn** : S×3, float32
        For each source, coordinates l, m, n-1
    **flux** : S×P, float32
        For each source, the perceived flux density (in Jy) per polarisation
    **weights** : N×P, float32
        The statistical weights associated with the visibilities. The input
        visibilities are assumed to be pre-weighted, and the predicted
        visibility is scaled by the weight before subtraction.

    Parameters
    ----------
    template : :class:`PredictTemplate`
        The template for the operation
    command_queue : :class:`katsdpsigproc.abc.AbstractCommandQueue`
        The command queue for the operation
    image_parameters : :class:`~.ImageParameters`
        Parameters that determine the UVW quantisation
    grid_parameters : :class:`~.GridParameters`
        Parameters that determine the UVW quantisation
    max_vis : int
        Maximum number of visibilities this instance can support
    max_sources : int
        Maximum number of sources this instance can support
    allocator : :class:`~katsdpsigproc.accel.AbstractAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, image_parameters, grid_parameters,
                 max_vis, max_sources, allocator=None):
        if len(image_parameters.fixed.polarizations) != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        super().__init__(command_queue, template.num_polarizations, max_vis,
                         allocator)
        self.template = template
        pol_dim = accel.Dimension(template.num_polarizations, exact=True)
        sources_dim = max(1, max_sources)   # Cannot allocate 0-byte buffer
        self.slots['lmn'] = accel.IOSlot((sources_dim, accel.Dimension(3, exact=True)), np.float32)
        self.slots['flux'] = accel.IOSlot((sources_dim, pol_dim), np.float32)
        self.slots['weights'] = accel.IOSlot(
            (max_vis, accel.Dimension(template.num_polarizations, exact=True)), np.float32)
        self._kernel = self.template.program.get_kernel('predict')
        self._num_sources = 0
        self.max_sources = max_sources
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self._w = 0.0
        self._host_lmn = accel.HostArray(
            (max_sources, 3), np.float32, context=command_queue.context)
        self._host_flux = accel.HostArray(
            (max_sources, template.num_polarizations), np.float32, context=command_queue.context)
        self._transfer_event = None

    def _copy_lmn_flux(self):
        if self._transfer_event is not None:
            # Wait for the previous iteration's transfer to complete
            self._transfer_event.wait()
        self.buffer('lmn').set_region(
            self.command_queue, self._host_lmn,
            np.s_[:self._num_sources], np.s_[:self._num_sources],
            blocking=False)
        self.buffer('flux').set_region(
            self.command_queue, self._host_flux,
            np.s_[:self._num_sources], np.s_[:self._num_sources],
            blocking=False)
        self._transfer_event = self.command_queue.enqueue_marker()

    @profile_function()
    def set_sky_model(self, model, phase_centre):
        """Set the sky model.

        This copies data to the device, so if `model` is altered, a subsequent
        call is needed to update the device buffers.

        TODO: could use some optimisation
        """
        N = len(model)
        if N > self.max_sources:
            raise ValueError('too many sources ({} > {})'.format(N, self.max_sources))
        lmn, flux = _extract_sky_model(self.image_parameters, self.grid_parameters,
                                       model, phase_centre)
        self._host_lmn[:N] = lmn
        self._host_flux[:N] = flux
        self._num_sources = N
        self._copy_lmn_flux()

    @profile_function()
    def set_sky_image(self, components):
        """Set the sky model from a component list.

        This extracts components from the image, so if the image is altered a
        subsequent call is needed.

        Parameters
        ----------
        components : Mapping[tuple[int, int], np.ndarray]
            CLEAN components, indexed by (y, x) coordinates, with each value being a 1D
            array of per-polarization values.
        """
        lmn, flux = _extract_sky_image(self.image_parameters, self.grid_parameters, components)
        N = len(lmn)
        if N > self.max_sources:
            raise ValueError('too many components ({} > {})'.format(N, self.max_sources))
        # TODO: have _extract_sky_image write directly to the buffer
        self._host_lmn[:N] = lmn
        self._host_flux[:N] = flux
        self._num_sources = N
        self._copy_lmn_flux()

    @property
    def num_sources(self):
        return self._num_sources

    def set_w(self, w):
        """Set the W slice.

        The W plane coordinates set by :meth:`set_coordinates` are taken
        to lie within a slice centred at `w`, which is specified in
        wavelengths.
        """
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

        uv_scale, w_scale, w_bias = _uvw_scale_bias(self.image_parameters, self.grid_parameters)
        w_bias += self._w

        with profile_device(self.command_queue, 'predict'):
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
                    np.int32(self.grid_parameters.fixed.oversample), np.float32(uv_scale),
                    np.float32(w_scale), np.float32(w_bias)
                ],
                global_size=(accel.roundup(self.num_vis, self.template.wgs),),
                local_size=(self.template.wgs,)
            )


@numba.jit(nopython=True)
def _predict_host(vis, uv, sub_uv, w_plane, weights, lmn, flux,
                  oversample, uv_scale, w_scale, w_bias, accum):
    N = vis.shape[0]
    S = lmn.shape[0]
    P = vis.shape[1]
    m2pi = np.complex64(-2j * np.pi)
    uvw = np.empty(3, np.float32)
    for i in range(N):
        uvw[0] = (uv[i, 0] * oversample + sub_uv[i, 0] + np.float32(0.5)) * uv_scale
        uvw[1] = (uv[i, 1] * oversample + sub_uv[i, 1] + np.float32(0.5)) * uv_scale
        uvw[2] = w_plane[i] * w_scale + w_bias
        accum[:] = 0
        for j in range(S):
            phase = np.dot(lmn[j], uvw)
            rot = np.exp(m2pi * phase)
            for p in range(P):
                accum[p] += rot * flux[j, p]
        accum *= weights[i]
        vis[i] -= accum


class PredictHost(grid.VisOperationHost):
    def __init__(self, image_parameters, grid_parameters):
        super().__init__()
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self.lmn = None
        self.flux = None
        self._w = 0

    @grid.VisOperationHost.num_vis.setter
    def num_vis(self, value):
        grid.VisOperationHost.num_vis.fset(self, value)
        self.weights = None

    def set_weights(self, weights):
        """Set statistical weights"""
        if len(weights) != self.num_vis:
            raise ValueError('Lengths do not match')
        self.weights = weights

    def set_w(self, w):
        self._w = w

    def set_sky_model(self, model, phase_centre):
        """Set the sky model.

        This copies data, so if `model` is altered, a subsequent
        call is needed to update the internal structures.
        """
        self.lmn, self.flux = _extract_sky_model(self.image_parameters,
                                                 self.grid_parameters,
                                                 model, phase_centre)

    def set_sky_image(self, components):
        """Set the sky model from an image.

        This copies data, so if `components` is altered, a subsequent
        call is needed to update the internal structures.
        """
        self.lmn, self.flux = _extract_sky_image(self.image_parameters,
                                                 self.grid_parameters,
                                                 components)

    def __call__(self):
        """Subtract predicted visibilities from existing values"""
        uv_scale, w_scale, w_bias = _uvw_scale_bias(self.image_parameters, self.grid_parameters)
        w_bias += self._w
        _predict_host(self.vis, self.uv, self.sub_uv, self.w_plane, self.weights,
                      self.lmn, self.flux,
                      np.float32(self.grid_parameters.fixed.oversample),
                      np.float32(uv_scale), np.float32(w_scale), np.float32(w_bias),
                      np.zeros(len(self.image_parameters.fixed.polarizations),
                               self.image_parameters.fixed.complex_dtype))
