import tempfile
import os
import atexit
import logging
from contextlib import closing

import numpy as np
import astropy.units as units
import katsdpsigproc.accel as accel

from . import \
    loader, parameters, polarization, preprocess, io, clean, weight, sky_model, \
    imaging, progress, beam


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


def make_dirty(queue, reader, rel_channel, name, field, imager, mid_w, vis_block,
               full_cycle=False, subtract_model=None):
    imager.clear_dirty()
    queue.finish()
    for w_slice in range(reader.num_w_slices(rel_channel)):
        N = reader.len(rel_channel, w_slice)
        if N == 0:
            logger.info("Skipping slice %d which has no visibilities", w_slice + 1)
            continue
        label = '{} {}/{}'.format(name, w_slice + 1, reader.num_w_slices(rel_channel))
        if full_cycle:
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
                    imager.predict(mid_w[w_slice])
                if full_cycle:
                    imager.degrid()
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
        logger.info(name + ":")
        for line in lines:
            if line:
                logger.info('    ' + line)


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
        self.grid_p = parameters.GridParameters(
            args.aa_width, args.grid_oversample, args.kernel_image_oversample,
            w_slices, w_planes, max_w, args.kernel_width)
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


def process_channel(dataset, args, start_channel,
                    context, queue, reader, channel_p, array_p, weight_p,
                    subtract_model):
    channel = channel_p.channel
    rel_channel = channel - start_channel
    image_p = channel_p.image_p
    grid_p = channel_p.grid_p
    clean_p = channel_p.clean_p
    logger.info('Processing channel {}'.format(channel))
    # Create data and operation instances
    if args.host:
        imager = imaging.ImagingHost(image_p, weight_p, grid_p, clean_p)
    else:
        allocator = accel.SVMAllocator(context)
        imager_template = imaging.ImagingTemplate(
            queue, array_p, image_p, weight_p, grid_p, clean_p)
        n_sources = len(subtract_model) if subtract_model else 0
        imager = imager_template.instantiate(args.vis_block, n_sources, allocator)
        imager.ensure_all_bound()
    psf = imager.buffer('psf')
    dirty = imager.buffer('dirty')
    model = imager.buffer('model')
    grid_data = imager.buffer('grid')
    imager.clear_model()

    # Compute imaging weights
    make_weights(queue, reader, rel_channel,
                 imager, weight_p.weight_type, args.vis_block)
    if args.write_weights is not None:
        with progress.step('Write image weights'):
            io.write_fits_image(dataset, imager.buffer('weights_grid'), image_p,
                                args.write_weights, channel, image_p.wavelength, bunit=None)

    # Create PSF
    slice_w_step = float(grid_p.max_w / image_p.wavelength / (grid_p.w_slices - 0.5))
    mid_w = np.arange(grid_p.w_slices) * slice_w_step
    make_dirty(queue, reader, rel_channel,
               'PSF', 'weights', imager, mid_w, args.vis_block)
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
    if args.write_psf is not None:
        with progress.step('Write PSF'):
            io.write_fits_image(dataset, psf, image_p, args.write_psf,
                                channel, image_p.wavelength, restoring_beam)

    # Imaging
    if subtract_model:
        imager.set_sky_model(subtract_model, dataset.phase_centre())
    for i in range(args.major):
        logger.info("Starting major cycle %d/%d", i + 1, args.major)
        make_dirty(queue, reader, rel_channel,
                   'image', 'vis', imager, mid_w, args.vis_block,
                   i != 0, subtract_model)
        imager.scale_dirty(scale)
        queue.finish()
        if i == 0 and args.write_grid is not None:
            with progress.step('Write grid'):
                if args.host:
                    io.write_fits_grid(grid_data, image_p, args.write_grid, channel)
                else:
                    io.write_fits_grid(np.fft.fftshift(grid_data, axes=(1, 2)),
                                       image_p, args.write_grid, channel)
        if i == 0 and args.write_dirty is not None:
            with progress.step('Write dirty image'):
                io.write_fits_image(dataset, dirty, image_p, args.write_dirty,
                                    channel, image_p.wavelength, restoring_beam)

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

    if args.write_model is not None:
        with progress.step('Write model'):
            io.write_fits_image(dataset, model, image_p, args.write_model,
                                channel, image_p.wavelength)
    if args.write_residuals is not None:
        with progress.step('Write residuals'):
            io.write_fits_image(dataset, dirty, image_p, args.write_residuals,
                                channel, image_p.wavelength, restoring_beam)

    # Try to free up memory for the beam convolution
    del grid_data
    del psf
    del imager

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
    with progress.step('Write clean image'):
        io.write_fits_image(dataset, model, image_p, args.output_file,
                            channel, image_p.wavelength, restoring_beam)


def run(context, queue, args):
    # PyCUDA leaks resources that are freed when the corresponding context is
    # not active. We make it active for the rest of the execution to avoid
    # this.
    with context, closing(loader.load(args.input_file, args.input_option)) as dataset:
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

        if args.subtract is not None:
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
                                reader, channel_p, array_p, weight_p, subtract_model)
