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
import tempfile
import atexit
import os
import katsdpsigproc.accel as accel
from katsdpimager import \
    loader, parameters, polarization, preprocess, io, clean, imaging, progress, beam
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
    group.add_argument('--w-step', type=parse_quantity, default='1.0', help='Separation between W planes, in subgrid cells or a distance [%(default)s]'),
    group.add_argument('--max-w', type=parse_quantity, help='Largest w, as either distance or wavelengths [longest baseline]')
    group.add_argument('--aa-width', type=float, default=7, help='Support of anti-aliasing kernel [%(default)s]')
    group.add_argument('--kernel-width', type=int, default=60, help='Support of combined anti-aliasing + w kernel [computed]')
    group.add_argument('--eps-w', type=float, default=0.01, help='Level at which to truncate W kernel [%(default)s]')
    group = parser.add_argument_group('Cleaning options')
    # TODO: compute from some heuristic if not specified, instead of a hard-coded default
    group.add_argument('--psf-patch', type=int, default=100, help='Pixels in beam patch for cleaning [%(default)s]')
    group.add_argument('--loop-gain', type=float, default=0.1, help='Loop gain for cleaning [%(default)s]')
    group.add_argument('--major', type=int, default=1, help='Major cycles [%(default)s]')
    group.add_argument('--minor', type=int, default=1000, help='Minor cycles per major cycle [%(default)s]')
    group.add_argument('--clean-mode', choices=['I', 'IQUV'], default='IQUV', help='Stokes parameters to consider for peak-finding [%(default)s]')
    group = parser.add_argument_group('Performance tuning options')
    group.add_argument('--vis-block', type=int, default=1048576, help='Number of visibilities to load and grid at a time [%(default)s]')
    group.add_argument('--no-tmp-file', dest='tmp_file', action='store_false', default=True, help='Keep preprocessed visibilities in memory')
    group.add_argument('--max-cache-size', type=int, default=None, help='Limit HDF5 cache size for preprocessing')
    group = parser.add_argument_group('Debugging options')
    group.add_argument('--host', action='store_true', help='Perform operations on the CPU')
    group.add_argument('--write-psf', metavar='FILE', help='Write image of PSF to FITS file')
    group.add_argument('--write-grid', metavar='FILE', help='Write UV grid to FITS file')
    group.add_argument('--write-dirty', metavar='FILE', help='Write dirty image to FITS file')
    group.add_argument('--write-model', metavar='FILE', help='Write model image to FITS file')
    group.add_argument('--write-residuals', metavar='FILE', help='Write image residuals to FITS file')
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


class DummyCommandQueue(object):
    """Stub equivalent to CUDA/OpenCL CommandQueue objects, that just provides
    an empty :meth:`finish` method."""
    def finish(self):
        pass


def preprocess_visibilities(dataset, args, image_parameters, grid_parameters, polarization_matrix):
    bar = None
    if args.tmp_file:
        handle, filename = tempfile.mkstemp('.h5')
        os.close(handle)
        atexit.register(os.remove, filename)
        collector = preprocess.VisibilityCollectorHDF5(
            filename, [image_parameters], grid_parameters, args.vis_block,
            max_cache_size=args.max_cache_size)
    else:
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


def make_dirty(queue, reader, name, field, imager, mid_w, vis_block, full_cycle=False):
    imager.clear_dirty()
    queue.finish()
    for w_slice in range(reader.num_w_slices):
        N = reader.len(0, w_slice)
        if N == 0:
            logger.info("Skipping slice %d which has no visibilities", w_slice + 1)
            continue
        label = '{} {}/{}'.format(name, w_slice + 1, reader.num_w_slices)
        if full_cycle:
            with progress.step('FFT {}'.format(label)):
                imager.model_to_grid(mid_w[w_slice])
        bar = progress.make_progressbar('Grid {}'.format(label), max=N)
        imager.clear_grid()
        queue.finish()
        with progress.finishing(bar):
            for chunk in reader.iter_slice(0, w_slice, vis_block):
                if full_cycle:
                    # TODO: this will transfer the coordinate data twice. Also,
                    # it makes unnecessary copies of the visibilities, which
                    # should ideally stay on the GPU.
                    predicted = np.empty_like(chunk[field])
                    imager.degrid(chunk.uv, chunk.sub_uv, chunk.w_plane, predicted)
                    chunk[field] -= predicted * chunk.weights
                imager.grid(chunk.uv, chunk.sub_uv, chunk.w_plane, chunk[field])
                # Need to serialise calls to grid, since otherwise the next
                # call will overwrite the incoming data before the previous
                # iteration is done with it.
                queue.finish()
                bar.next(len(chunk))

        with progress.step('IFFT {}'.format(label)):
            imager.grid_to_image(mid_w[w_slice])
            queue.finish()

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
        logging.DEBUG: colors.cyan
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
        queue = context.create_command_queue()
    else:
        queue = DummyCommandQueue()

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
        if args.w_step.unit.physical_type == 'length':
            w_planes = float(args.max_w / args.w_step)
        elif args.w_step.unit.physical_type == 'dimensionless':
            w_step = args.w_step * image_p.cell_size / args.grid_oversample
            w_planes = float(args.max_w / w_step)
        else:
            raise ValueError('--w-step must be dimensionless or a length')
        w_planes = int(np.ceil(w_planes / args.w_slices))
        grid_p = parameters.GridParameters(
            args.aa_width, args.grid_oversample, args.kernel_image_oversample,
            args.w_slices, w_planes, args.max_w, args.kernel_width)
        if args.clean_mode == 'I':
            clean_mode = clean.CLEAN_I
        elif args.clean_mode == 'IQUV':
            clean_mode = clean.CLEAN_SUMSQ
        else:
            raise ValueError('Unhandled --clean-mode {}'.format(args.clean_mode))
        args.psf_patch = min(args.psf_patch, image_p.pixels)
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
            imager = imaging.ImagingHost(image_p, grid_p, clean_p)
        else:
            allocator = accel.SVMAllocator(context)
            imager_template = imaging.ImagingTemplate(
                queue, array_p, image_p, grid_p, clean_p)
            imager = imager_template.instantiate(args.vis_block, allocator)
            imager.ensure_all_bound()
        psf = imager.buffer('psf')
        dirty = imager.buffer('dirty')
        model = imager.buffer('model')
        grid_data = imager.buffer('grid')

        #### Preprocess visibilities ####
        collector = preprocess_visibilities(dataset, args, image_p, grid_p, polarization_matrix)
        reader = collector.reader()

        #### Create dirty image ####
        slice_w_step = float(grid_p.max_w / image_p.wavelength / (grid_p.w_slices - 0.5))
        mid_w = np.arange(grid_p.w_slices) * slice_w_step
        make_dirty(queue, reader, 'PSF', 'weights', imager, mid_w, args.vis_block)
        # Normalization
        scale = np.reciprocal(dirty[..., dirty.shape[1] // 2, dirty.shape[2] // 2])
        imager.scale_dirty(scale)
        queue.finish()
        extract_psf(dirty, psf)
        restoring_beam = beam.fit_beam(psf[0])
        if args.write_psf is not None:
            with progress.step('Write PSF'):
                io.write_fits_image(dataset, dirty, image_p, args.write_psf, restoring_beam)

        imager.clear_model()
        for i in range(args.major):
            logger.info("Starting major cycle %d/%d", i + 1, args.major)
            make_dirty(queue, reader, 'image', 'vis', imager, mid_w, args.vis_block, i != 0)
            imager.scale_dirty(scale)
            queue.finish()
            if i == 0 and args.write_grid is not None:
                with progress.step('Write grid'):
                    if args.host:
                        io.write_fits_grid(grid_data, image_p, args.write_grid)
                    else:
                        io.write_fits_grid(np.fft.fftshift(grid_data, axes=(1, 2)),
                                           image_p, args.write_grid)
            if i == 0 and args.write_dirty is not None:
                with progress.step('Write dirty image'):
                    io.write_fits_image(dataset, dirty, image_p, args.write_dirty, restoring_beam)

            #### Deconvolution ####
            bar = progress.make_progressbar('CLEAN', max=clean_p.minor)
            imager.clean_reset()
            with progress.finishing(bar):
                for j in bar.iter(range(clean_p.minor)):
                    imager.clean_cycle()
            queue.finish()

        if args.write_model is not None:
            with progress.step('Write model'):
                io.write_fits_image(dataset, model, image_p, args.write_model)
        if args.write_residuals is not None:
            with progress.step('Write residuals'):
                io.write_fits_image(dataset, dirty, image_p, args.write_residuals, restoring_beam)
        # Convolve with restoring beam, and add residuals back in
        beam.convolve_beam(model, restoring_beam, model)
        model += dirty
        with progress.step('Write clean image'):
            io.write_fits_image(dataset, model, image_p, args.output_file, restoring_beam)

if __name__ == '__main__':
    main()
