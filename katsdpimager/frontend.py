import tempfile
import os
import atexit
import logging
import math
from abc import abstractmethod

import numpy as np
from astropy import units
import katsdpsigproc.accel as accel

from . import \
    loader, parameters, polarization, preprocess, clean, weight, sky_model, \
    imaging, progress, beam, primary_beam


logger = logging.getLogger(__name__)


def parse_quantity(str_value):
    """Parse a string into an astropy Quantity. Rather than trying to guess
    where the split occurs, we try every position from the back until we
    succeed."""
    for i in range(len(str_value), 0, -1):
        try:
            value = float(str_value[:i])
            unit = units.Unit(str_value[i:])
            return units.Quantity(value, unit)
        except ValueError:
            pass
    raise ValueError('Could not parse {} as a quantity'.format(str_value))


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
    try:
        for chunk in loader.data_iter(dataset, args.vis_limit, args.vis_load,
                                      start_channel, stop_channel):
            if bar is None:
                bar = progress.make_progressbar("Preprocessing vis", max=chunk['total'])
            collector.add(
                chunk['uvw'], chunk['weights'], chunk['baselines'], chunk['vis'],
                chunk.get('feed_angle1'), chunk.get('feed_angle2'),
                *polarization_matrices)
            bar.goto(chunk['progress'])
    finally:
        if bar is not None:
            bar.finish()
        collector.close()
    logger.info("Compressed %d visibilities to %d (%.2f%%)",
                collector.num_input, collector.num_output,
                100.0 * collector.num_output / collector.num_input)
    return collector


def make_weights(queue, reader, rel_channel, imager, weight_type, vis_block):
    imager.clear_weights()
    total = 0
    for w_slice in range(reader.num_w_slices(rel_channel)):
        total += reader.len(rel_channel, w_slice)
    bar = progress.make_progressbar('Computing weights', max=total)
    queue.finish()
    with progress.finishing(bar):
        if weight_type != weight.WeightType.NATURAL:
            for w_slice in range(reader.num_w_slices(rel_channel)):
                for chunk in reader.iter_slice(rel_channel, w_slice, vis_block):
                    imager.grid_weights(chunk.uv, chunk.weights)
                    # Need to serialise calls to grid, since otherwise the next
                    # call will overwrite the incoming data before the previous
                    # iteration is done with it.
                    queue.finish()
                    bar.next(len(chunk.uv))
        else:
            bar.next(total)
        normalized_rms = imager.finalize_weights()
        queue.finish()
    logger.info('Normalized thermal RMS: %g', normalized_rms)


def make_dirty(queue, reader, rel_channel, name, field, imager, mid_w, vis_block, degrid,
               full_cycle=False, subtract_model=None):
    imager.clear_dirty()
    queue.finish()
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
        queue.finish()
        with progress.finishing(bar):
            for chunk in reader.iter_slice(rel_channel, w_slice, vis_block):
                imager.num_vis = len(chunk.uv)
                imager.set_coordinates(chunk.uv, chunk.sub_uv, chunk.w_plane)
                imager.set_vis(chunk[field])
                if full_cycle or subtract_model:
                    imager.set_weights(chunk.weights)
                if subtract_model:
                    imager.continuum_predict(mid_w[w_slice])
                if full_cycle:
                    imager.predict(mid_w[w_slice])
                imager.grid()
                # Need to serialise calls to grid, since otherwise the next
                # call will overwrite the incoming data before the previous
                # iteration is done with it.
                queue.finish()
                bar.next(len(chunk))

        with progress.step('IFFT {}'.format(label)):
            imager.grid_to_image(mid_w[w_slice])
            queue.finish()


def extract_psf(psf, psf_patch):
    """Extract the central region of a PSF.

    This function is 2D: the polarization axis must have already been removed.
    """
    y0 = (psf.shape[0] - psf_patch[0]) // 2
    y1 = y0 + psf_patch[0]
    x0 = (psf.shape[1] - psf_patch[1]) // 2
    x1 = x0 + psf_patch[1]
    return psf[..., y0:y1, x0:x1]


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
    clean_p : :class:`katsdpimage.parameters.CleanParameters`
        CLEAN parameters
    """

    def __init__(self, args, dataset, channel, array_p):
        self.channel = channel
        self.image_p = parameters.ImageParameters(
            args.q_fov, args.image_oversample,
            dataset.frequency(channel), array_p, args.stokes,
            (np.float32 if args.precision == 'single' else np.float64),
            args.pixel_size, args.pixels)
        if args.max_w is None:
            max_w = array_p.longest_baseline
        elif args.max_w.unit.physical_type == 'dimensionless':
            max_w = args.max_w * self.image_p.wavelength
        else:
            max_w = args.max_w
        if args.w_slices is None:
            w_slices = parameters.w_slices(self.image_p, max_w, args.eps_w,
                                           args.kernel_width, args.aa_width)
        else:
            w_slices = args.w_slices
        if args.w_step.unit.physical_type == 'length':
            w_planes = float(max_w / args.w_step)
        elif args.w_step.unit.physical_type == 'dimensionless':
            w_step = args.w_step * self.image_p.cell_size / args.grid_oversample
            w_planes = float(max_w / w_step)
        else:
            raise ValueError('--w-step must be dimensionless or a length')
        w_planes = int(np.ceil(w_planes / w_slices))
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
        self.grid_p = parameters.GridParameters(
            args.aa_width, args.grid_oversample, args.kernel_image_oversample,
            w_slices, w_planes, max_w, args.kernel_width, args.degrid, beams)
        if args.clean_mode == 'I':
            clean_mode = clean.CLEAN_I
        elif args.clean_mode == 'IQUV':
            clean_mode = clean.CLEAN_SUMSQ
        else:
            raise ValueError('Unhandled --clean-mode {}'.format(args.clean_mode))
        border = int(round(self.image_p.pixels * args.border))
        limit = int(round(self.image_p.pixels * args.psf_limit))
        self.clean_p = parameters.CleanParameters(
            args.minor, args.loop_gain, args.major_gain, args.threshold,
            clean_mode, args.psf_cutoff, limit, border)

    def log_parameters(self, suffix=''):
        log_parameters("Image parameters" + suffix, self.image_p)
        log_parameters("Grid parameters" + suffix, self.grid_p)
        log_parameters("CLEAN parameters" + suffix, self.clean_p)


def add_options(parser):
    group = parser.add_argument_group('Input selection')
    group.add_argument('--input-option', '-i', action='append', default=[], metavar='KEY=VALUE',
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
    group.add_argument('--pixel-size', type=parse_quantity,
                       help='Size of each image pixel [computed from array]')
    group.add_argument('--pixels', type=int,
                       help='Number of pixels in image [computed from array]')
    group.add_argument('--stokes', type=polarization.parse_stokes, default='I',
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
    group.add_argument('--w-step', type=parse_quantity, default='1.0',
                       help='Separation between W planes, in subgrid cells or a distance '
                            '[%(default)s]')
    group.add_argument('--max-w', type=parse_quantity,
                       help='Largest w, as either distance or wavelengths [longest baseline]')
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


class Writer:
    """Abstract class that handles writing grids/images to files"""
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


def process_channel(dataset, args, start_channel,
                    context, queue, reader, writer, channel_p, array_p, weight_p,
                    subtract_model):
    channel = channel_p.channel
    rel_channel = channel - start_channel
    image_p = channel_p.image_p
    grid_p = channel_p.grid_p
    clean_p = channel_p.clean_p

    # Check if there is anything to do
    if not any(reader.len(rel_channel, w_slice)
               for w_slice in range(reader.num_w_slices(rel_channel))):
        logger.info('Skipping channel %d which has no data', channel)
        return

    logger.info('Processing channel %d', channel)
    # Create data and operation instances
    if args.host:
        imager = imaging.ImagingHost(image_p, weight_p, grid_p, clean_p)
    else:
        allocator = accel.SVMAllocator(context)
        imager_template = imaging.ImagingTemplate(
            queue, array_p, image_p, weight_p, grid_p, clean_p)
        n_sources = len(subtract_model) if subtract_model else 0
        imager = imager_template.instantiate(args.vis_block, n_sources, args.major, allocator)
        imager.ensure_all_bound()
    psf = imager.buffer('psf')
    dirty = imager.buffer('dirty')
    model = imager.buffer('model')
    grid_data = imager.buffer('grid')
    imager.clear_model()

    # Compute imaging weights
    make_weights(queue, reader, rel_channel,
                 imager, weight_p.weight_type, args.vis_block)
    writer.write_fits_image('weights', 'image weights',
                            dataset, imager.buffer('weights_grid'), image_p, channel, bunit=None)

    # Create PSF
    slice_w_step = float(grid_p.max_w / image_p.wavelength / (grid_p.w_slices - 0.5))
    mid_w = np.arange(grid_p.w_slices) * slice_w_step
    make_dirty(queue, reader, rel_channel,
               'PSF', 'weights', imager, mid_w, args.vis_block, args.degrid)
    # Normalization
    psf_peak = dirty[..., dirty.shape[1] // 2, dirty.shape[2] // 2]
    if np.any(psf_peak == 0):
        logger.info('Skipping channel %d which has no usable data', channel)
        return
    scale = np.reciprocal(psf_peak)
    imager.scale_dirty(scale)
    queue.finish()
    imager.dirty_to_psf()
    # dirty_to_psf works by swapping, so re-fetch the buffer pointers
    psf = imager.buffer('psf')
    dirty = imager.buffer('dirty')
    psf_patch = imager.psf_patch()
    logger.info('Using %dx%d patch for PSF', psf_patch[2], psf_patch[1])
    # Extract the patch for beam fitting
    psf_core = extract_psf(psf[0], psf_patch[1:])
    restoring_beam = beam.fit_beam(psf_core)
    writer.write_fits_image('psf', 'PSF', dataset, psf, image_p, channel, restoring_beam)

    # Imaging
    if subtract_model:
        imager.set_sky_model(subtract_model, dataset.phase_centre())
    for i in range(args.major):
        logger.info("Starting major cycle %d/%d", i + 1, args.major)
        make_dirty(queue, reader, rel_channel,
                   'image', 'vis', imager, mid_w, args.vis_block, args.degrid,
                   i != 0, subtract_model)
        imager.scale_dirty(scale)
        queue.finish()
        if i == 0:
            writer.write_fits_grid('grid', 'grid', not args.host, grid_data, image_p, channel)
            writer.write_fits_image('dirty', 'dirty image', dataset, dirty, image_p,
                                    channel, restoring_beam)

        # Deconvolution
        noise = imager.noise_est()
        imager.clean_reset()
        peak_value = imager.clean_cycle(psf_patch)
        peak_power = clean.metric_to_power(clean_p.mode, peak_value)
        noise_threshold = noise * clean_p.threshold
        mgain_threshold = (1.0 - clean_p.major_gain) * peak_power
        logger.info('Threshold from noise estimate: %g', noise_threshold)
        logger.info('Threshold from mgain:          %g', mgain_threshold)
        threshold = max(noise_threshold, mgain_threshold)
        if peak_power <= threshold:
            logger.info('Threshold reached, terminating')
            break
        logger.info('CLEANing to threshold:         %g', threshold)
        threshold_metric = clean.power_to_metric(clean_p.mode, threshold)
        bar = progress.make_progressbar('CLEAN', max=clean_p.minor - 1)
        with progress.finishing(bar):
            for j in bar.iter(range(clean_p.minor - 1)):
                value = imager.clean_cycle(psf_patch, threshold_metric)
                if value is None:
                    break
        queue.finish()

    # Scale by primary beam
    if grid_p.beams:
        pbeam_model = grid_p.beams.sample()
        # Sample beam model at the pixel grid. It's circularly symmetric, so
        # we don't need to worry about parallactic angle rotations or the
        # different sign conventions for azimuth versus RA.
        start = -image_p.pixels / 2 * image_p.pixel_size
        pbeam = pbeam_model.sample(start, image_p.pixel_size, image_p.pixels,
                                   start, image_p.pixel_size, image_p.pixels,
                                   [image_p.wavelength])
        # Ignore polarization and length-1 frequency axis; square to
        # convert voltage to power.
        pbeam = np.square(np.abs(pbeam[0, 0, 0]))
        # Where the primary beam power is low, just mask the result rather
        # than emitting massively scaled noise.
        pbeam[pbeam < args.primary_beam_cutoff] = math.nan
        model /= pbeam
        dirty /= pbeam
        writer.write_fits_image(
            'primary_beam', 'primary beam', dataset,
            np.broadcast_to(pbeam, model.shape), image_p, channel)
    else:
        pbeam = None

    writer.write_fits_image('model', 'model', dataset, model, image_p, channel)
    writer.write_fits_image('residuals', 'residuals', dataset, dirty, image_p,
                            channel, restoring_beam)

    # Try to free up memory for the beam convolution
    del pbeam
    del grid_data
    del psf
    del imager

    # NaNs in the model will mess up the convolution, so zero them. They will
    # become NaNs again when the residuals are added (assuming the NaNs are
    # due to the primary beam mask).
    if grid_p.beams:
        model[np.isnan(model)] = 0.0
    # Convolve with restoring beam, and add residuals back in
    if args.host:
        beam.convolve_beam(model, restoring_beam, model)
    else:
        restore = beam.ConvolveBeamTemplate(queue, model.shape[1:], model.dtype).instantiate()
        restore.beam = restoring_beam
        restore.ensure_all_bound()
        restore_image = restore.buffer('image')
        for pol in range(model.shape[0]):
            # TODO: eliminate these copies, and work directly on model
            model.copy_region(queue, restore_image, np.s_[pol], ())
            restore()
            restore_image.copy_region(queue, model, (), np.s_[pol])
        queue.finish()

    model += dirty
    writer.write_fits_image('clean', 'clean image', dataset, model, image_p,
                            channel, restoring_beam)


def run(args, context, queue, dataset, writer):
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

        if args.stop_channel - args.start_channel > 1:
            ChannelParameters(
                args, dataset, args.start_channel, array_p).log_parameters(' [first channel]')
            ChannelParameters(
                args, dataset, args.stop_channel - 1, array_p).log_parameters(' [last channel]')
        else:
            ChannelParameters(args, dataset, args.start_channel, array_p).log_parameters()
        log_parameters("Weight parameters", weight_p)

        if args.subtract == 'auto':
            subtract_model = dataset.sky_model()
        elif args.subtract is not None:
            subtract_model = sky_model.open_sky_model(args.subtract)
        else:
            subtract_model = None

        for start_channel in range(args.start_channel, args.stop_channel, args.channel_batch):
            stop_channel = min(args.stop_channel, start_channel + args.channel_batch)
            channels = range(start_channel, stop_channel)
            params = [ChannelParameters(args, dataset, channel, array_p)
                      for channel in channels]
            # Preprocess visibilities
            image_ps = [channel_p.image_p for channel_p in params]
            grid_ps = [channel_p.grid_p for channel_p in params]
            collector = preprocess_visibilities(dataset, args, start_channel, stop_channel,
                                                image_ps, grid_ps, polarization_matrices)
            reader = collector.reader()

            # Do the work
            for channel_p in params:
                process_channel(dataset, args, start_channel, context, queue,
                                reader, writer, channel_p, array_p, weight_p, subtract_model)
