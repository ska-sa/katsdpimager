from abc import abstractmethod
import atexit
import concurrent.futures
import contextlib
import logging
import math
import os
import tempfile
from typing import List, Iterable
import warnings

import numpy as np
import numba
from astropy import units
import astropy.wcs
import katsdpsigproc.accel as accel

from . import (
    loader, loader_core, parameters, polarization, preprocess, clean, weight, sky_model,
    imaging, progress, beam, primary_beam, arguments
)
from .profiling import profile, profile_function
from .fast_math import nansum


logger = logging.getLogger(__name__)


@profile_function(labels=('start_channel', 'stop_channel'))
def preprocess_visibilities(dataset, args, start_channel, stop_channel,
                            image_parameters, grid_parameters, polarization_matrices):
    bar = None
    if args.tmp_file:
        handle, filename = tempfile.mkstemp('.h5')
        os.close(handle)
        atexit.register(os.remove, filename)
        collector = preprocess.VisibilityCollectorHDF5(
            filename, image_parameters, grid_parameters, args.vis_block,
            max_cache_size=args.max_cache_size)
    else:
        collector = preprocess.VisibilityCollectorMem(
            image_parameters, grid_parameters, args.vis_block)

    def reap():
        nonlocal add_future
        if add_future is not None:
            with profile('concurrent.futures.result'):
                bar_progress = add_future.result()
            bar.goto(bar_progress)
            add_future = None

    with contextlib.ExitStack() as exit_stack:
        exit_stack.callback(collector.close)
        # We overlap data loading with preprocessing by using a separate
        # thread for preprocessing. To limit memory usage, we only allow
        # one outstanding chunk at a time.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        exit_stack.enter_context(executor)
        add_future = None
        for chunk in loader.data_iter(dataset, args.vis_limit, args.vis_load,
                                      start_channel, stop_channel):
            if bar is None:
                bar = progress.make_progressbar("Preprocessing vis", max=chunk['total'])
                exit_stack.callback(bar.finish)
            reap()

            @profile_function()
            def add_chunk(chunk):
                collector.add(
                    chunk['uvw'], chunk['weights'], chunk['vis'],
                    chunk.get('feed_angle1'), chunk.get('feed_angle2'),
                    *polarization_matrices)
                return chunk['progress']

            add_future = executor.submit(add_chunk, chunk)
        reap()

    logger.info("Compressed %d visibilities to %d (%.2f%%)",
                collector.num_input, collector.num_output,
                100.0 * collector.num_output / collector.num_input)
    return collector


@profile_function()
def make_weights(queue, reader, rel_channel, imager, weight_type, vis_block, weight_scale):
    imager.clear_weights()
    total = 0
    for w_slice in range(reader.num_w_slices(rel_channel)):
        total += reader.len(rel_channel, w_slice)
    bar = progress.make_progressbar('Computing weights', max=total)
    with bar:
        if weight_type != weight.WeightType.NATURAL:
            for w_slice in range(reader.num_w_slices(rel_channel)):
                for chunk in reader.iter_slice(rel_channel, w_slice, vis_block):
                    imager.grid_weights(chunk.uv, chunk.weights)
                    bar.next(len(chunk.uv))
        else:
            bar.next(total)
        noise, normalized_noise = imager.finalize_weights()
        if noise is not None and weight_scale is not None:
            noise *= weight_scale
    if noise is not None:
        logger.info('Thermal RMS noise (from weights): %g', noise)
    logger.info('Normalized thermal RMS noise: %g', normalized_noise)
    return noise, normalized_noise


@profile_function()
def make_dirty(queue, reader, rel_channel, name, field, imager, mid_w, vis_block, degrid,
               full_cycle=False, subtract_model=None):
    imager.clear_dirty()
    if full_cycle and not degrid:
        with progress.step('Extract components'):
            imager.model_to_predict()
    for w_slice in range(reader.num_w_slices(rel_channel)):
        N = reader.len(rel_channel, w_slice)
        if N == 0:
            logger.info("Skipping slice %d which has no visibilities", w_slice + 1)
            continue
        label = '{} {}/{}'.format(name, w_slice + 1, reader.num_w_slices(rel_channel))
        if full_cycle and degrid:
            with progress.step('FFT {}'.format(label)):
                imager.model_to_grid(mid_w[w_slice])
        bar = progress.make_progressbar('Grid {}'.format(label), max=N)
        imager.clear_grid()
        with bar:
            for chunk in reader.iter_slice(rel_channel, w_slice, vis_block):
                imager.num_vis = len(chunk.uv)
                imager.set_coordinates(chunk)
                imager.set_vis(chunk[field])
                if full_cycle or subtract_model:
                    imager.set_weights(chunk.weights)
                if subtract_model:
                    imager.continuum_predict(mid_w[w_slice])
                if full_cycle:
                    imager.predict(mid_w[w_slice])
                imager.grid()
                bar.next(len(chunk))

        with progress.step('IFFT {}'.format(label)):
            imager.grid_to_image(mid_w[w_slice])


@profile_function()
def extract_psf(queue, psf, psf_patch):
    """Extract the central region of a PSF.

    Parameters
    ----------
    queue : :class:`katsdpsigproc.abc.AbstractCommandQueue`
        Command queue for the copy (ignored if `psf` is already on the host)
    psf : :class:`katsdpsigproc.accel.DeviceArray` or :class:`np.ndarray`
        Point spread function (shape pols × height × width). Only the first
        polarization is returned.
    psf_patch : tuple[int, int]
        Size to extract, in the y and x directions respectively
    """
    y0 = (psf.shape[1] - psf_patch[0]) // 2
    y1 = y0 + psf_patch[0]
    x0 = (psf.shape[2] - psf_patch[1]) // 2
    x1 = x0 + psf_patch[1]
    if isinstance(psf, accel.DeviceArray):
        out = accel.HostArray((y1 - y0, x1 - x0), psf.dtype, context=queue.context)
        psf.get_region(queue, out, np.s_[0, y0:y1, x0:x1], np.s_[:, :])
        return out
    else:
        return psf[0, y0:y1, x0:x1]


@profile_function()
@numba.jit(nopython=True)
def find_peak(image, pbeam, noise):
    """Heuristically find the peak of the data in the image.

    When primary beam correction is used, the beam-corrected noise near the
    null of the primary beam can be quite large and possibly larger than any
    true emission. To avoid picking up noise, limit the search to pixels
    whose absolute value is more than 7.5σ.
    """
    # TODO: it may be worth doing this on the GPU. On the other hand,
    # the primary beam isn't currently on the GPU.
    peak = image.dtype.type(0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                v = np.abs(image[i, j, k])
                if v > peak:
                    pb = pbeam[j, k]
                    if v * pb > 7.5 * noise:
                        peak = v
    if peak == 0:
        peak = np.nan
    return peak


@profile_function()
def get_totals(image_parameters, image, restoring_beam):
    """Compute total flux density in each polarization."""
    sums = nansum(image, axis=(1, 2), dtype=np.float64)   # Sum separately per polarization
    # Area under the restoring beam. It is a Gaussian with peak of 1, and
    # hence the area under it is 2πσ_xσ_y. The Beam class holds FWHM (in
    # pixels) not standard deviations, hence the extra factor of 8*log 2.
    beam_area = 2 * math.pi * restoring_beam.major * restoring_beam.minor / (8 * math.log(2))
    sums /= beam_area
    return {
        polarization.STOKES_NAMES[pol]: float(s)
        for pol, s in zip(image_parameters.fixed.polarizations, sums)
    }


def log_parameters(name, params):
    if logger.isEnabledFor(logging.INFO):
        s = str(params)
        lines = s.split('\n')
        logger.info('%s:', name)
        for line in lines:
            if line:
                logger.info('    %s', line)


class ChannelParameters:
    """Collects imaging parameters for a single channel.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments
    dataset : :class:`loader_core.LoaderBase`
        Input dataset
    channel : int
        Index of this channel in the dataset
    array_p : :class:`katsdpimager.parameters.ArrayParameters`
        Array parameters from the input file

    Attributes
    ----------
    channel : int
        Index of this channel in the dataset
    image_p : :class:`katsdpimager.parameters.ImageParameters`
        Image parameters
    grid_p : :class:`katsdpimager.parameters.GridParameters`
        Gridding parameters
    """

    def __init__(self, args, dataset, channel, array_p, fixed_image_p, fixed_grid_p):
        self.channel = channel
        self.image_p = parameters.ImageParameters(
            fixed_image_p, args.q_fov, args.image_oversample,
            dataset.frequency(channel), array_p,
            args.pixel_size, args.pixels)
        if args.w_slices is None:
            w_slices = parameters.w_slices(self.image_p, fixed_grid_p.max_w, args.eps_w,
                                           args.kernel_width, args.aa_width)
        else:
            w_slices = args.w_slices
        if args.w_step.unit.physical_type == 'length':
            w_planes = float(fixed_grid_p.max_w / args.w_step)
        elif args.w_step.unit.physical_type == 'dimensionless':
            w_step = args.w_step * self.image_p.cell_size / args.grid_oversample
            w_planes = float(fixed_grid_p.max_w / w_step)
        else:
            raise ValueError('--w-step must be dimensionless or a length')
        w_planes = int(np.ceil(w_planes / w_slices))
        self.grid_p = parameters.GridParameters(fixed_grid_p, w_slices, w_planes)

    def log_parameters(self, suffix=''):
        log_parameters("Image parameters" + suffix, self.image_p)
        log_parameters("Grid parameters" + suffix, self.grid_p)


def prepend_dashes(value):
    return '--' + value


def add_options(parser):
    group = parser.add_argument_group('Input selection')
    group.add_argument('--input-option', '-i',
                       type=prepend_dashes, action='append', default=[], metavar='KEY=VALUE',
                       help='Backend-specific input parsing option')
    group.add_argument('--start-channel', '-c', type=int, default=0,
                       help='Index of first channel to process [%(default)s]')
    group.add_argument('--stop-channel', '-C', type=int,
                       help='Index past last channel to process [#channels]')
    group.add_argument('--subtract', metavar='URL',
                       help='Sky model with sources to subtract at the start')

    group = parser.add_argument_group('Image options')
    group.add_argument('--q-fov', type=float, default=1.0,
                       help='Field of view to image, relative to main lobe of beam [%(default)s]')
    group.add_argument('--image-oversample', type=float, default=5,
                       help='Pixels per beam [%(default)s]')
    group.add_argument('--pixel-size', type=units.Quantity,
                       help='Size of each image pixel [computed from array]')
    group.add_argument('--pixels', type=int,
                       help='Number of pixels in image [computed from array]')
    group.add_argument('--stokes', type=polarization.parse_stokes,
                       default='I',
                       help='Stokes parameters to image e.g. IQUV for full-Stokes [%(default)s]')
    group.add_argument('--precision', choices=['single', 'double'], default='single',
                       help='Internal floating-point precision [%(default)s]')

    group = parser.add_argument_group('Weighting options')
    group.add_argument('--weight-type',
                       choices=[weight_type.name.lower() for weight_type in weight.WeightType],
                       default='natural',
                       help='Imaging density weights [%(default)s]')
    group.add_argument('--robustness', type=float, default=0.0,
                       help='Robustness parameter for --weight-type=robust [%(default)s]')

    group = parser.add_argument_group('Gridding options')
    group.add_argument('--grid-oversample', type=int, default=8,
                       help='Oversampling factor for convolution kernels [%(default)s]')
    group.add_argument('--kernel-image-oversample', type=int, default=4,
                       help='Oversampling factor for kernel generation [%(default)s]')
    group.add_argument('--w-slices', type=int,
                       help='Number of W slices [computed from --kernel-width]')
    group.add_argument('--w-step', type=units.Quantity, default=units.Quantity(1.0),
                       help='Separation between W planes, in subgrid cells or a distance '
                            '[%(default)s]')
    group.add_argument('--max-w', type=units.Quantity,
                       help='Largest w, in units of distance [longest baseline]')
    group.add_argument('--aa-width', type=float, default=7,
                       help='Support of anti-aliasing kernel [%(default)s]')
    group.add_argument('--kernel-width', type=int, default=60,
                       help='Support of combined anti-aliasing + w kernel [computed]')
    group.add_argument('--eps-w', type=float, default=0.001,
                       help='Level at which to truncate W kernel [%(default)s]')
    group.add_argument('--degrid', action='store_true',
                       help='Use degridding rather than direct prediction (less accurate)')
    group.add_argument('--primary-beam', choices=['meerkat', 'meerkat:1', 'none'], default='none',
                       help='Primary beam model for the telescope')
    group.add_argument('--primary-beam-cutoff', type=float, default=0.1,
                       help='Primary beam power level below which output pixels are discarded')

    group = parser.add_argument_group('Cleaning options')
    group.add_argument('--psf-cutoff', type=float, default=0.01,
                       help='fraction of PSF peak at which to truncate PSF [%(default)s]')
    group.add_argument('--psf-limit', type=float, default=0.5,
                       help='maximum fraction of image to use for PSF [%(default)s]')
    group.add_argument('--loop-gain', type=float, default=0.1,
                       help='Loop gain for cleaning [%(default)s]')
    group.add_argument('--major-gain', type=float, default=0.85,
                       help='Fraction of peak to clean in each major cycle [%(default)s]')
    group.add_argument('--threshold', type=float, default=5.0,
                       help='CLEAN threshold in sigma [%(default)s]')
    group.add_argument('--major', type=int, default=1,
                       help='Major cycles [%(default)s]')
    group.add_argument('--minor', type=int, default=10000,
                       help='Max minor cycles per major cycle [%(default)s]')
    group.add_argument('--border', type=float, default=0.02,
                       help='CLEAN border as a fraction of image size [%(default)s]')
    group.add_argument('--clean-mode', choices=['I', 'IQUV'], default='IQUV',
                       help='Stokes parameters to consider for peak-finding [%(default)s]')

    group = parser.add_argument_group('Performance tuning options')
    group.add_argument('--vis-block', type=int, default=1048576,
                       help='Number of visibilities to load and grid at a time [%(default)s]')
    group.add_argument('--vis-load', type=int, default=32 * 1048576,
                       help='Number of visibilities to load from file at a time [%(default)s]')
    group.add_argument('--channel-batch', type=int, default=16,
                       help='Number of channels to fully process before starting next batch '
                            '[%(default)s]')
    group.add_argument('--no-tmp-file', dest='tmp_file', action='store_false', default=True,
                       help='Keep preprocessed visibilities in memory')
    group.add_argument('--max-cache-size', type=int, default=None,
                       help='Limit HDF5 cache size for preprocessing')


def command_line_options(args: arguments.SmartNamespace,
                         dataset: loader_core.LoaderBase,
                         exclude: Iterable[str]) -> List[str]:
    """Reconstruct an equivalent command line from parsed arguments."""
    dataset_args = dataset.command_line_options()
    global_args = arguments.unparse_args(
        args, exclude,
        arg_handlers={
            'stokes': lambda name, value: ['--stokes=' + polarization.unparse_stokes(value)]
        })
    return dataset_args + global_args


class Writer:
    """Abstract class that handles writing grids/images to files"""

    def channel_already_done(self, dataset, channel):
        """Determine whether a channel has already been imaged in a previous run.

        If this returns true, it will be skipped for imaging this time.
        """
        return False

    @abstractmethod
    def needs_fits_image(self, name):
        """Tell the caller whether it wants a call to :meth:`write_fits_image` for `name`.

        The caller may still make the call even if this method returns False.
        This is just an optimisation hint to avoid the need to retrieve the
        data if it is not needed.
        """

    @abstractmethod
    def needs_fits_grid(self, name):
        """Like :meth:`needs_fits_image`, but for grids."""

    @abstractmethod
    def write_fits_image(self, name, description, dataset, image, image_parameters, channel,
                         beam=None, bunit='Jy/beam'):
        """Optionally output a FITS image.

        `name` is a machine-readable name (consistent with the argument naming
        in image.py, while `description` is human-readable. The other
        arguments have the same meaning as for :meth:`.io.write_fits_image`.
        """

    @abstractmethod
    def write_fits_grid(self, name, description, fftshift, grid_data, image_parameters, channel):
        """Optional output a FITS image showing the UV plane.

        `name` is a machine-readable name (consistent with the argument naming
        in image.py, while `description` is human-readable. If `fftshift` is
        true, the data needs to be fft-shifted on the u and v axes. The other
        arguments have the same meaning as for :meth:`.io.write_fits_grid`.
        """

    def skip_channel(self, dataset, image_parameters, channel):
        """Called to indicate that a channel was skipped due to lack of data."""

    def statistics(self, dataset, channel, **kwargs):
        """Report statistics of the image or imaging process.

        The statistics reported will evolve over time. Currently, they are

        noise
          Estimated noise in the residual image, in Jy/beam
        weights_noise
          Estimated noise computed from the visibility and image weights,
          in Jy/beam. It can be ``None`` if not available.
        normalized_noise
          Increase in noise due to use of non-natural weights (unitless)
        peak
          Largest absolute value of pixel in final image that is not simply
          noise (according to a heuristic), or NaN if there is no such pixel.
        totals
          Total non-NaN flux density in each Stokes image, as a dictionary
          keyed by string name of the Stokes parameter.
        major
          Number of major cycles performed.
        minor
          Number of minor cycles (CLEAN components) performed.
        psf_patch_size
          Number of a pixels in the PSF patch, as a tuple (x, y).
        compressed_vis
          Number of compressed visibilities.
        image_parameters
          The :class:`ImageParameters` used for the channel.
        grid_parameters
          The :class:`GridParameters` used for the channel.
        clean_parameters
          The :class:`CleanParameters` used for the channel.
        """


@profile_function(labels={'channel': lambda bound_args: bound_args.arguments['channel_p'].channel})
def process_channel(dataset, args, start_channel,
                    context, queue, imager_template,
                    reader, writer,
                    channel_p, array_p, weight_p, clean_p,
                    subtract_model):
    channel = channel_p.channel
    rel_channel = channel - start_channel
    image_p = channel_p.image_p
    grid_p = channel_p.grid_p

    # Check if there is anything to do
    if writer.channel_already_done(dataset, channel):
        logger.info('Skipping channel %d because it has already been done', channel)
        return
    if not dataset.channel_enabled(channel_p.channel):
        logger.info('Skipping channel %d which is masked', channel)
        return
    if not any(reader.len(rel_channel, w_slice)
               for w_slice in range(reader.num_w_slices(rel_channel))):
        logger.info('Skipping channel %d which has no data', channel)
        writer.skip_channel(dataset, image_p, channel)
        return

    logger.info('Processing channel %d', channel)
    # Create data and operation instances
    if args.host:
        imager = imaging.ImagingHost(image_p, weight_p, grid_p, clean_p)
    else:
        n_sources = len(subtract_model) if subtract_model else 0
        imager = imager_template.instantiate(
            queue, image_p, grid_p, args.vis_block, n_sources, args.major)
        imager.ensure_all_bound()
    imager.clear_model()

    # Compute imaging weights
    weights_noise, normalized_noise = make_weights(queue, reader, rel_channel,
                                                   imager, weight_p.weight_type, args.vis_block,
                                                   dataset.weight_scale())
    if writer.needs_fits_image('weights'):
        writer.write_fits_image(
            'weights', 'image weights',
            dataset, imager.get_buffer('weights_grid'), image_p, channel, bunit=None)

    # Create PSF
    slice_w_step = float(grid_p.fixed.max_w / image_p.wavelength / (grid_p.w_slices - 0.5))
    mid_w = np.arange(grid_p.w_slices) * slice_w_step
    make_dirty(queue, reader, rel_channel,
               'PSF', 'weights', imager, mid_w, args.vis_block, args.degrid)
    # Normalization
    dirty = imager.buffer('dirty')
    if args.host:
        psf_peak = dirty[..., dirty.shape[1] // 2, dirty.shape[2] // 2]
    else:
        psf_peak = accel.HostArray((dirty.shape[0],), dirty.dtype, context=context)
        dirty.get_region(
            queue, psf_peak,
            np.s_[:, dirty.shape[1] // 2, dirty.shape[2] // 2],
            np.s_[:])
    if np.any(psf_peak == 0):
        logger.info('Skipping channel %d which has no usable data', channel)
        writer.skip_channel(dataset, image_p, channel)
        return
    scale = np.reciprocal(psf_peak)
    imager.scale_dirty(scale)
    imager.dirty_to_psf()
    # dirty_to_psf works by swapping, so re-fetch the buffer pointers
    psf_patch = imager.psf_patch()
    logger.info('Using %dx%d patch for PSF', psf_patch[2], psf_patch[1])
    # Extract the patch for beam fitting
    psf_core = extract_psf(queue, imager.buffer('psf'), psf_patch[1:])
    restoring_beam = beam.fit_beam(psf_core)
    if writer.needs_fits_image('psf'):
        writer.write_fits_image(
            'psf', 'PSF', dataset, imager.get_buffer('psf'), image_p, channel, restoring_beam)

    # Imaging
    if subtract_model:
        imager.set_sky_model(subtract_model, dataset.phase_centre())
    major = 0
    minor = 0
    for i in range(args.major):
        logger.info("Starting major cycle %d/%d", i + 1, args.major)
        make_dirty(queue, reader, rel_channel,
                   'image', 'vis', imager, mid_w, args.vis_block, args.degrid,
                   i != 0, subtract_model)
        imager.scale_dirty(scale)
        if i == 0:
            if writer.needs_fits_grid('grid'):
                writer.write_fits_grid(
                    'grid', 'grid', not args.host, imager.get_buffer('grid'), image_p, channel)
            if writer.needs_fits_image('dirty'):
                writer.write_fits_image(
                    'dirty', 'dirty image', dataset, imager.get_buffer('dirty'), image_p,
                    channel, restoring_beam)
        major += 1

        # Deconvolution
        noise = imager.noise_est()
        imager.clean_reset()
        peak_value = imager.clean_cycle(psf_patch)
        peak_power = clean.metric_to_power(clean_p.mode, peak_value)
        noise_threshold = noise * clean.noise_threshold_scale(
            clean_p.mode, clean_p.threshold, len(image_p.fixed.polarizations))
        mgain_threshold = (1.0 - clean_p.major_gain) * peak_power
        logger.info('Threshold from noise estimate: %g', noise_threshold)
        logger.info('Threshold from mgain:          %g', mgain_threshold)
        threshold = max(noise_threshold, mgain_threshold)
        if peak_power <= threshold:
            logger.info('Threshold reached, terminating')
            break
        logger.info('CLEANing to threshold:         %g', threshold)
        threshold_metric = clean.power_to_metric(clean_p.mode, threshold)
        with progress.make_progressbar('CLEAN', max=clean_p.minor - 1) as bar:
            for j in bar.iter(range(clean_p.minor - 1)):
                value = imager.clean_cycle(psf_patch, threshold_metric)
                minor += 1
                if value is None:
                    break
        if i == args.major - 1:
            # Update the noise estimate for output stats
            noise = imager.noise_est()

    # Scale by primary beam
    model = imager.buffer('model')
    if grid_p.fixed.beams:
        pbeam_model = grid_p.fixed.beams.sample()
        # Sample beam model at the pixel grid. It's circularly symmetric, so
        # we don't need to worry about parallactic angle rotations or the
        # different sign conventions for azimuth versus RA.
        start = -image_p.pixels / 2 * image_p.pixel_size
        pbeam = pbeam_model.sample(start, image_p.pixel_size, image_p.pixels,
                                   start, image_p.pixel_size, image_p.pixels,
                                   [image_p.wavelength])
        # Ignore polarization and length-1 frequency axis.
        pbeam = pbeam[0, 0, 0]
        # Square to convert voltage to power.
        if pbeam.dtype.kind == 'c':
            pbeam = np.square(pbeam.real) + np.square(pbeam.imag)
        else:
            pbeam = np.square(pbeam)
        # Ensure the units match the destination buffer.
        pbeam = pbeam.astype(imager.buffer('beam_power').dtype, copy=False)
        imager.set_buffer('beam_power', pbeam)
        imager.apply_primary_beam(args.primary_beam_cutoff)
        writer.write_fits_image(
            'primary_beam', 'primary beam', dataset,
            np.broadcast_to(pbeam, model.shape), image_p, channel)
    else:
        pbeam = np.broadcast_to(np.ones(1, model.dtype), model.shape[-2:])

    if writer.needs_fits_image('model'):
        writer.write_fits_image(
            'model', 'model', dataset, imager.get_buffer('model'), image_p, channel)
    if writer.needs_fits_image('residuals'):
        writer.write_fits_image(
            'residuals', 'residuals', dataset, imager.get_buffer('dirty'), image_p,
            channel, restoring_beam)

    model = imager.buffer('model')
    # Try to free up memory for the beam convolution
    for name in ['weights_grid', 'grid', 'layer', 'psf', 'beam_power']:
        imager.free_buffer(name)

    # Convolve with restoring beam, and add residuals back in
    if args.host:
        beam.convolve_beam(model, restoring_beam, model)
    else:
        restore = beam.ConvolveBeamTemplate(
            context, model.shape[1:], model.dtype).instantiate(queue)
        restore.beam = restoring_beam
        restore.ensure_all_bound()
        restore_image = restore.buffer('image')
        for pol in range(model.shape[0]):
            # TODO: eliminate these copies, and work directly on model
            model.copy_region(queue, restore_image, np.s_[pol], ())
            restore()
            restore_image.copy_region(queue, model, (), np.s_[pol])

    # Combine the restored model image with the residuals
    imager.add_model_to_dirty()
    del model
    final_image = imager.get_buffer('dirty')

    writer.write_fits_image('clean', 'clean image', dataset, final_image, image_p,
                            channel, restoring_beam)
    peak = find_peak(final_image, pbeam, noise)
    totals = get_totals(image_p, final_image, restoring_beam)
    compressed_vis = sum(reader.len(rel_channel, w_slice)
                         for w_slice in range(reader.num_w_slices(rel_channel)))
    writer.statistics(dataset, channel,
                      major=major, minor=minor,
                      peak=peak, totals=totals, noise=noise,
                      weights_noise=weights_noise,
                      normalized_noise=normalized_noise,
                      psf_patch_size=(psf_patch[2], psf_patch[1]),
                      compressed_vis=compressed_vis,
                      image_parameters=image_p,
                      grid_parameters=grid_p,
                      clean_parameters=clean_p)


@profile_function()
def run(args, context, queue, dataset, writer):
    # Workaround for https://github.com/astropy/astropy/issues/10365
    warnings.filterwarnings(
        'ignore',
        message=r'.*Set OBSGEO-. to .* from OBSGEO-\[XYZ\]',
        category=astropy.wcs.FITSFixedWarning)

    # PyCUDA leaks resources that are freed when the corresponding context is
    # not active. We make it active for the rest of the execution to avoid
    # this.
    with context:
        # Determine parameters
        input_polarizations = dataset.polarizations()
        if dataset.has_feed_angles():
            polarization_matrices = \
                polarization.polarization_matrices(args.stokes, input_polarizations)
        else:
            polarization_matrices = (
                polarization.polarization_matrix(args.stokes, input_polarizations), None)
        array_p = dataset.array_parameters()
        if args.stop_channel is None:
            args.stop_channel = dataset.num_channels()
        if not (0 <= args.start_channel < args.stop_channel <= dataset.num_channels()):
            raise ValueError('Channels are out of range')
        weight_p = parameters.WeightParameters(
            weight.WeightType[args.weight_type.upper()], args.robustness)

        if args.clean_mode == 'I':
            clean_mode = clean.CLEAN_I
        elif args.clean_mode == 'IQUV':
            clean_mode = clean.CLEAN_SUMSQ
        else:
            raise ValueError('Unhandled --clean-mode {}'.format(args.clean_mode))
        clean_p = parameters.CleanParameters(
            args.minor, args.loop_gain, args.major_gain, args.threshold,
            clean_mode, args.psf_cutoff, args.psf_limit, args.border)

        fixed_image_p = parameters.FixedImageParameters(
            args.stokes,
            np.float32 if args.precision == 'single' else np.float64
        )

        if args.max_w is None:
            max_w = array_p.longest_baseline
        else:
            max_w = args.max_w
        if args.primary_beam in {'meerkat', 'meerkat:1'}:
            band = dataset.band()
            if band is None:
                raise ValueError(
                    'Data set does not specify a band, so --primary-beam cannot be used')
            beams = primary_beam.MeerkatBeamModelSet1(band)
        elif args.primary_beam == 'none':
            beams = None
        else:
            raise ValueError(f'Unexpected value {args.primary_beam} for --primary-beam')
        fixed_grid_p = parameters.FixedGridParameters(
            args.aa_width, args.grid_oversample, args.kernel_image_oversample,
            max_w, args.kernel_width, args.degrid, beams
        )

        if args.stop_channel - args.start_channel > 1:
            ChannelParameters(
                args, dataset, args.start_channel,
                array_p, fixed_image_p, fixed_grid_p).log_parameters(' [first channel]')
            ChannelParameters(
                args, dataset, args.stop_channel - 1,
                array_p, fixed_image_p, fixed_grid_p).log_parameters(' [last channel]')
        else:
            ChannelParameters(args, dataset, args.start_channel,
                              array_p, fixed_image_p, fixed_grid_p).log_parameters()
        log_parameters("Weight parameters", weight_p)
        log_parameters("CLEAN parameters", clean_p)

        if args.subtract == 'auto':
            subtract_model = dataset.sky_model()
        elif args.subtract is not None:
            subtract_model = sky_model.open_sky_model(args.subtract)
        else:
            subtract_model = None

        if not args.host:
            imager_template = imaging.ImagingTemplate(
                context, array_p, fixed_image_p, weight_p, fixed_grid_p, clean_p)
        else:
            imager_template = None

        for start_channel in range(args.start_channel, args.stop_channel, args.channel_batch):
            stop_channel = min(args.stop_channel, start_channel + args.channel_batch)
            channels = range(start_channel, stop_channel)
            params = [ChannelParameters(args, dataset, channel,
                                        array_p, fixed_image_p, fixed_grid_p)
                      for channel in channels]
            # Preprocess visibilities
            image_ps = [channel_p.image_p for channel_p in params]
            grid_ps = [channel_p.grid_p for channel_p in params]
            collector = preprocess_visibilities(dataset, args, start_channel, stop_channel,
                                                image_ps, grid_ps, polarization_matrices)
            reader = collector.reader()

            # Do the work
            for channel_p in params:
                process_channel(dataset, args, start_channel,
                                context, queue, imager_template,
                                reader, writer, channel_p, array_p, weight_p, clean_p,
                                subtract_model)
