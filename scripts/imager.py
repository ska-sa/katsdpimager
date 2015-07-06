#!/usr/bin/env python
from __future__ import print_function, division
import math
import sys
import argparse
import astropy.units as units
import numpy as np
import logging
import colors
import functools
import katsdpsigproc.accel as accel
import katsdpimager.loader as loader
import katsdpimager.parameters as parameters
import katsdpimager.polarization as polarization
import katsdpimager.preprocess as preprocess
import katsdpimager.grid as grid
import katsdpimager.io as io
import katsdpimager.fft as fft
import katsdpimager.clean as clean
import katsdpimager.progress as progress
from contextlib import closing, contextmanager


logger = logging.getLogger()


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

def parse_stokes(str_value):
    ans = []
    for p in str_value:
        if p not in 'IQUV':
            raise ValueError('Invalid Stokes parameter {}'.format(p))
    if not str_value:
        raise ValueError('Empty Stokes parameter list')
    for p in 'IQUV':
        cnt = str_value.count(p)
        if cnt > 1:
            raise ValueError('Stokes parameter {} listed multiple times'.format(p))
        elif cnt > 0:
            ans.append(polarization.STOKES_NAMES.index(p))
    return sorted(ans)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT', help='Input measurement set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT', help='Output FITS file')
    parser.add_argument('--log-level', type=str, default='INFO', metavar='LEVEL', help='Logging level [%(default)s]')
    group = parser.add_argument_group('Input selection')
    group.add_argument('--input-option', '-i', action='append', default=[], metavar='KEY=VALUE', help='Backend-specific input parsing option')
    group.add_argument('--channel', '-c', type=int, default=0, help='Channel number [%(default)s]')
    group = parser.add_argument_group('Image options')
    group.add_argument('--q-fov', type=float, default=1.0, help='Field of view to image, relative to main lobe of beam [%(default)s]')
    group.add_argument('--image-oversample', type=float, default=5, help='Pixels per beam [%(default)s]')
    group.add_argument('--pixel-size', type=parse_quantity, help='Size of each image pixel [computed from array]')
    group.add_argument('--pixels', type=int, help='Number of pixels in image [computed from array]')
    group.add_argument('--stokes', type=parse_stokes, default='I', help='Stokes parameters to image e.g. IQUV for full-Stokes [%(default)s]')
    group.add_argument('--precision', choices=['single', 'double'], default='single', help='Internal floating-point precision [%(default)s]')
    group = parser.add_argument_group('Gridding options')
    group.add_argument('--grid-oversample', type=int, default=8, help='Oversampling factor for convolution kernels [%(default)s]')
    group.add_argument('--kernel-image-oversample', type=int, default=4, help='Oversampling factor for kernel generation [%(default)s]')
    group.add_argument('--w-slices', type=int, help='Number of W slices [computed from --kernel-width]')
    group.add_argument('--w-planes', type=int, default=128, help='Number of W planes per slice [%(default)s]'),
    group.add_argument('--max-w', type=parse_quantity, help='Largest w, as either distance or wavelengths [longest baseline]')
    group.add_argument('--aa-width', type=float, default=7, help='Support of anti-aliasing kernel [%(default)s]')
    group.add_argument('--kernel-width', type=int, default=64, help='Support of combined anti-aliasing + w kernel [computed]')
    group.add_argument('--eps-w', type=float, default=0.01, help='Level at which to truncate W kernel [%(default)s]')
    group = parser.add_argument_group('Cleaning options')
    # TODO: compute from some heuristic if not specified, instead of a hard-coded default
    group.add_argument('--psf-patch', type=int, default=100, help='Pixels in beam patch for cleaning [%(default)s]')
    group.add_argument('--loop-gain', type=float, default=0.1, help='Loop gain for cleaning [%(default)s]')
    group.add_argument('--minor', type=int, default=1000, help='Minor cycles per major cycle [%(default)s]')
    group.add_argument('--clean-mode', choices=['I', 'IQUV'], default='IQUV', help='Stokes parameters to consider for peak-finding [%(default)s]')
    group = parser.add_argument_group('Performance tuning options')
    group.add_argument('--vis-block', type=int, default=1048576, help='Number of visibilities to load at a time [%(default)s]')
    group = parser.add_argument_group('Debugging options')
    group.add_argument('--host', action='store_true', help='Perform operations on the CPU')
    group.add_argument('--write-psf', metavar='FILE', help='Write image of PSF to FITS file')
    group.add_argument('--write-grid', metavar='FILE', help='Write UV grid to FITS file')
    group.add_argument('--write-dirty', metavar='FILE', help='Write dirty image to FITS file')
    group.add_argument('--write-model', metavar='FILE', help='Write model image to FITS file')
    group.add_argument('--write-residuals', metavar='FILE', help='Write image residuals to FITS file')
    group.add_argument('--profile', action='store_true', help='Do profiling on GPU code')
    group.add_argument('--vis-limit', type=int, metavar='N', help='Use only the first N visibilities')
    return parser

def data_iter(dataset, args):
    """Wrapper around :py:meth:`katsdpimager.loader_core.LoaderBase.data_iter`
    that handles truncation to a number of visibilities specified on the
    command line.
    """
    N = args.vis_limit
    for chunk in dataset.data_iter(args.channel, args.vis_block):
        if N is not None:
            if N < len(chunk['uvw']):
                for key in ['uvw', 'weights', 'baselines', 'vis']:
                    if key in chunk:
                        chunk[key] = chunk[key][:N]
                chunk['progress'] = chunk['total']
        yield chunk
        if N is not None:
            N -= len(chunk['uvw'])
            if N == 0:
                return

def preprocess_visibilities(dataset, args, image_parameters, grid_parameters, polarization_matrix):
    bar = None
    collector = preprocess.VisibilityCollectorMem([image_parameters], grid_parameters, args.vis_block)
    try:
        for chunk in data_iter(dataset, args):
            if bar is None:
                bar = progress.make_progressbar("Preprocessing vis", max=chunk['total'])
            collector.add(
                0, chunk['uvw'], chunk['weights'], chunk['baselines'], chunk['vis'],
                polarization_matrix)
            bar.goto(chunk['progress'])
    finally:
        if bar is not None:
            bar.finish()
        collector.close()
    logger.info("Compressed %d visibilities to %d (%.2f%%)",
        collector.num_input, collector.num_output, 100.0 * collector.num_output / collector.num_input)
    return collector


def timer(queue):
    """Decorator that enqueues markers before and after the wrapped function, and
    returns a function that, when called, returns the elapsed time."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            def get_elapsed():
                return end.time_since(start)
            start = queue.enqueue_marker()
            fn(*args, **kwargs)
            end = queue.enqueue_marker()
            return get_elapsed
        return wrapper
    return decorator


def make_dirty(queue, reader, name, field, gridder, grid_to_image, mid_w, vis_block):
    grid_to_image.clear()
    for w_slice in range(reader.num_w_slices):
        N = reader.len(0, w_slice)
        if N == 0:
            logger.info("Skipping slice %d which has no visibilities", w_slice + 1)
            continue
        label = '{} {}/{}'.format(name, w_slice + 1, reader.num_w_slices)
        bar = progress.make_progressbar('Grid {}'.format(label), max=N)
        gridder.clear()
        grid_time = 0.0
        with progress.finishing(bar):
            for chunk in reader.iter_slice(0, w_slice, vis_block):
                t = gridder.grid(chunk.uv, chunk.sub_uv, chunk.w_plane, chunk[field])
                if queue:
                    # Need to serialise calls to grid, since otherwise the next
                    # call will overwrite the incoming data before the previous
                    # iteration is done with it.
                    queue.finish()
                if t is not None:
                    grid_time += t()
                bar.next(len(chunk))
        if grid_time > 0.0:
            logger.info("Gridded %d points in %.3fs (%.3g/s)", N, grid_time, N / grid_time)

        with progress.step('FFT {}'.format(label)):
            grid_to_image.set_w(mid_w[w_slice])
            grid_to_image()
            if queue:
                queue.finish()

def psf_shape(image_parameters, clean_parameters):
    psf_patch = min(image_parameters.pixels, clean_parameters.psf_patch)
    return (len(image_parameters.polarizations), psf_patch, psf_patch)

def extract_psf(image, psf):
    """Copy the central region of `image` to `psf`.

    .. todo::

        Move to clean module, and ideally on the GPU.
    """
    y0 = (image.shape[1] - psf.shape[1]) // 2
    y1 = y0 + psf.shape[1]
    x0 = (image.shape[2] - psf.shape[2]) // 2
    x1 = y0 + psf.shape[2]
    psf[...] = image[..., y0:y1, x0:x1]


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.CRITICAL: colors.red,
        logging.ERROR: colors.red,
        logging.WARNING: colors.magenta,
        logging.INFO: colors.green,
        logging.DEBUG: colors.blue
    }

    def __init__(self, *args, **kwargs):
        super(ColorFormatter, self).__init__(*args, **kwargs)

    def format(self, record):
        msg = super(ColorFormatter, self).format(record)
        if record.levelno in self.COLORS:
            msg = self.COLORS[record.levelno](msg)
        return msg


def configure_logging(args):
    log_handler = logging.StreamHandler()
    fmt = "[%(levelname)s] %(message)s"
    if sys.stderr.isatty():
        log_handler.setFormatter(ColorFormatter(fmt))
    else:
        log_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(log_handler)
    logger.setLevel(args.log_level.upper())

def log_parameters(name, params):
    if logger.isEnabledFor(logging.INFO):
        s = str(params)
        lines = s.split('\n')
        logger.info(name + ":")
        for line in lines:
            if line:
                logger.info('    ' + line)


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.input_option = ['--' + opt for opt in args.input_option]
    configure_logging(args)

    queue = None
    context = None
    if not args.host:
        context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
        queue = context.create_command_queue(profile=args.profile)

    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        #### Determine parameters ####
        input_polarizations = dataset.polarizations()
        output_polarizations = args.stokes
        polarization_matrix = polarization.polarization_matrix(output_polarizations, input_polarizations)
        array_p = dataset.array_parameters()
        image_p = parameters.ImageParameters(
            args.q_fov, args.image_oversample,
            dataset.frequency(args.channel), array_p, output_polarizations,
            (np.float32 if args.precision == 'single' else np.float64),
            args.pixel_size, args.pixels)
        if args.max_w is None:
            args.max_w = array_p.longest_baseline
        elif args.max_w.unit.physical_type == 'dimensionless':
            args.max_w = args.max_w * image_p.wavelength
        if args.w_slices is None:
            args.w_slices = parameters.w_slices(image_p, args.max_w, args.eps_w, args.kernel_width, args.aa_width)
        grid_p = parameters.GridParameters(
            args.aa_width, args.grid_oversample, args.kernel_image_oversample,
            args.w_slices, args.w_planes, args.max_w, args.kernel_width)
        if args.clean_mode == 'I':
            clean_mode = clean.CLEAN_I
        elif args.clean_mode == 'IQUV':
            clean_mode = clean.CLEAN_SUMSQ
        else:
            raise ValueError('Unhandled --clean-mode {}'.format(args.clean_mode))
        clean_p = parameters.CleanParameters(
            args.minor, args.loop_gain, clean_mode,
            args.psf_patch)

        log_parameters("Image parameters", image_p)
        log_parameters("Grid parameters", grid_p)
        log_parameters("CLEAN parameters", clean_p)

        #### Create data and operation instances ####
        lm_scale = float(image_p.pixel_size)
        lm_bias = -0.5 * image_p.pixels * lm_scale
        if args.host:
            gridder = grid.GridderHost(image_p, grid_p)
            grid_data = gridder.values
            layer = np.empty(grid_data.shape, image_p.complex_dtype)
            image = np.empty(grid_data.shape, image_p.real_dtype)
            model = np.empty(grid_data.shape, image_p.real_dtype)
            psf = np.empty(psf_shape(image_p, clean_p), image_p.real_dtype)
            grid_to_image = fft.GridToImageHost(
                grid_data, layer, image,
                gridder.taper(image_p.pixels), lm_scale, lm_bias)
            cleaner = clean.CleanHost(image_p, clean_p, image, psf, model)
        else:
            allocator = accel.SVMAllocator(context)
            # Gridder
            gridder_template = grid.GridderTemplate(context, image_p, grid_p)
            gridder = gridder_template.instantiate(queue, array_p, args.vis_block, allocator)
            gridder.ensure_all_bound()
            if args.profile:
                grid.Gridder.__call__ = timer(queue)(grid.Gridder.__call__)
            grid_data = gridder.buffer('grid')
            # Grid to image
            kernel1d = accel.SVMArray(context, (image_p.pixels,), image_p.real_dtype)
            gridder_template.taper(image_p.pixels, kernel1d)
            # TODO: allocate these from the operations, to ensure alignment
            layer = accel.SVMArray(context, grid_data.shape, image_p.complex_dtype)
            image = accel.SVMArray(context, grid_data.shape, image_p.real_dtype)
            grid_to_image_template = fft.GridToImageTemplate(
                queue, grid_data.shape, grid_data.padded_shape, image.shape, image.dtype)
            grid_to_image = grid_to_image_template.instantiate(lm_scale, lm_bias, allocator)
            grid_to_image.bind(grid=grid_data, layer=layer, image=image, kernel1d=kernel1d)
            # CLEAN
            cleaner_template = clean.CleanTemplate(
                context, clean_p, image_p.real_dtype, len(output_polarizations))
            cleaner = cleaner_template.instantiate(queue, image_p, allocator)
            cleaner.bind(dirty=image)
            cleaner.ensure_all_bound()
            psf = cleaner.buffer('psf')
            model = cleaner.buffer('model')
            model[:] = 0

        #### Preprocess visibilities ####
        collector = preprocess_visibilities(dataset, args, image_p, grid_p, polarization_matrix)
        reader = collector.reader()

        #### Create dirty image ####
        slice_w_step = float(grid_p.max_w / image_p.wavelength / grid_p.w_slices)
        mid_w = np.arange(0.5, grid_p.w_slices) * slice_w_step
        make_dirty(queue, reader, 'PSF', 'weights', gridder, grid_to_image, mid_w, args.vis_block)
        # TODO: all this scaling is hacky. Move it into subroutines somewhere
        scale = np.reciprocal(image[..., image.shape[1] // 2, image.shape[2] // 2])
        scale = scale[:, np.newaxis, np.newaxis]
        image *= scale
        if args.write_psf is not None:
            with progress.step('Write PSF'):
                io.write_fits_image(dataset, image, image_p, args.write_psf)
        extract_psf(image, psf)
        make_dirty(queue, reader, 'image', 'vis', gridder, grid_to_image, mid_w, args.vis_block)
        image *= scale
        if args.write_grid is not None:
            with progress.step('Write grid'):
                if args.host:
                    io.write_fits_grid(grid_data, image_p, args.write_grid)
                else:
                    io.write_fits_grid(np.fft.fftshift(grid_data, axes=(1, 2)),
                                       image_p, args.write_grid)
        if args.write_dirty is not None:
            with progress.step('Write dirty image'):
                io.write_fits_image(dataset, image, image_p, args.write_dirty)

        #### Deconvolution ####
        bar = progress.make_progressbar('CLEAN', max=clean_p.minor)
        cleaner.reset()
        with progress.finishing(bar):
            for i in bar.iter(range(clean_p.minor)):
                cleaner()
        if queue:
            queue.finish()
        # TODO: restoring beam
        if args.write_model is not None:
            with progress.step('Write model'):
                io.write_fits_image(dataset, model, image_p, args.write_model)
        if args.write_residuals is not None:
            with progress.step('Write residuals'):
                io.write_fits_image(dataset, image, image_p, args.write_residuals)
        # Add residuals back in
        model += image
        with progress.step('Write clean image'):
            io.write_fits_image(dataset, model, image_p, args.output_file)

if __name__ == '__main__':
    main()
