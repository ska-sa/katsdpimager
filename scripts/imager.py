#!/usr/bin/env python3
import sys
import argparse
import logging
import shlex
from contextlib import closing, contextmanager

import numpy as np
import colors
import katsdpsigproc.accel as accel
import astropy.io.fits as fits

from katsdpimager import frontend, loader, io, progress, arguments, profiling


logger = logging.getLogger()


class Writer(frontend.Writer):
    def __init__(self, args, dataset):
        self.args = args
        options = frontend.command_line_options(
            args, dataset, {'input_file', 'output_file', 'input_option'})
        self.extra_fits_headers = fits.Header()
        self.extra_fits_headers['HISTORY'] = \
            'Command line options: ' + ' '.join(shlex.quote(arg) for arg in options)

    def needs_fits_image(self, name):
        if name == 'clean':
            return True
        else:
            return getattr(self.args, 'write_' + name) is not None

    def needs_fits_grid(self, name):
        return getattr(self.args, 'write_' + name) is not None

    def write_fits_image(self, name, description, dataset, image, image_parameters, channel,
                         beam=None, bunit='Jy/beam'):
        if name == 'clean':
            filename = self.args.output_file
        else:
            filename = getattr(self.args, 'write_' + name)
        if filename is not None:
            if '%' in filename:
                filename = filename % channel
            with progress.step('Write {}'.format(description)):
                io.write_fits_image(dataset, image, image_parameters, filename,
                                    channel, beam, bunit, self.extra_fits_headers)

    def write_fits_grid(self, name, description, fftshift, grid_data, image_parameters, channel):
        filename = getattr(self.args, 'write_' + name)
        if filename is not None:
            if '%' in filename:
                filename = filename % channel
            with progress.step('Write {}'.format(description)):
                if fftshift:
                    grid_data = np.fft.fftshift(grid_data, axes=(1, 2))
                io.write_fits_grid(grid_data, image_parameters, filename, channel)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT',
                        help='Input data set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT',
                        help='Output FITS file')
    parser.add_argument('--log-level', type=str, default='INFO', metavar='LEVEL',
                        help='Logging level [%(default)s]')

    frontend.add_options(parser)

    group = parser.add_argument_group('Debugging options')
    group.add_argument('--host', action='store_true',
                       help='Perform operations on the CPU')
    group.add_argument('--write-weights', metavar='FILE',
                       help='Write imaging weights to FITS file')
    group.add_argument('--write-psf', metavar='FILE',
                       help='Write image of PSF to FITS file')
    group.add_argument('--write-grid', metavar='FILE',
                       help='Write UV grid to FITS file')
    group.add_argument('--write-dirty', metavar='FILE',
                       help='Write dirty image to FITS file')
    group.add_argument('--write-model', metavar='FILE',
                       help='Write model image to FITS file')
    group.add_argument('--write-residuals', metavar='FILE',
                       help='Write image residuals to FITS file')
    group.add_argument('--write-primary-beam', metavar='FILE',
                       help='Write primary beam model to FITS file')
    group.add_argument('--write-profile', metavar='FILE',
                       help='Write profiling information to file')
    group.add_argument('--write-device-profile', metavar='FILE',
                       help='Write device profiling information to file')
    group.add_argument('--vis-limit', type=int, metavar='N',
                       help='Use only the first N visibilities')
    return parser


class DummyCommandQueue:
    """Stub equivalent to CUDA/OpenCL CommandQueue objects, that just provides
    an empty :meth:`finish` method."""
    def finish(self):
        pass


@contextmanager
def dummy_context():
    """Do-nothing context manager used in place of a device context."""
    yield


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.CRITICAL: colors.red,
        logging.ERROR: colors.red,
        logging.WARNING: colors.magenta,
        logging.INFO: colors.green,
        logging.DEBUG: colors.cyan
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        msg = super().format(record)
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
    # In case some other module has decided to configure logging, remove any
    # existing handlers.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(log_handler)
    logger.setLevel(args.log_level.upper())


def main():
    parser = get_parser()
    args = parser.parse_args(namespace=arguments.SmartNamespace())
    if (args.stop_channel is None or args.stop_channel - args.start_channel > 1):
        if '%' not in args.output_file:
            parser.error('More than one channel selected but no %d in output filename')
    configure_logging(args)
    if args.write_profile or args.write_device_profile:
        profiling.Profiler.set_profiler(profiling.FlamegraphProfiler())

    queue = None
    context = None
    if not args.host:
        context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
        queue = context.create_command_queue()
    else:
        context = dummy_context()
        queue = DummyCommandQueue()

    with closing(loader.load(args.input_file, args.input_option,
                             args.start_channel, args.stop_channel)) as dataset:
        frontend.run(args, context, queue, dataset, Writer(args, dataset))

    profiler = profiling.Profiler.get_profiler()
    if args.write_profile:
        with open(args.write_profile, 'w') as f:
            assert isinstance(profiler, profiling.FlamegraphProfiler)
            profiler.write_flamegraph(f)
    if args.write_device_profile:
        with open(args.write_device_profile, 'w') as f:
            assert isinstance(profiler, profiling.FlamegraphProfiler)
            profiler.write_device_flamegraph(f)


if __name__ == '__main__':
    main()
