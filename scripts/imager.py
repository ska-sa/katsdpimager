#!/usr/bin/env python3
import sys
import argparse
import logging
from contextlib import closing, contextmanager

import numpy as np
import colors
import katsdpsigproc.accel as accel

from katsdpimager import frontend, loader, io, progress, numba


logger = logging.getLogger()


class Writer(frontend.Writer):
    def __init__(self, args):
        self.args = args

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
                                    channel, beam, bunit)

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
    args = parser.parse_args()
    args.input_option = ['--' + opt for opt in args.input_option]
    if (args.stop_channel is None or args.stop_channel - args.start_channel > 1):
        if '%' not in args.output_file:
            parser.error('More than one channel selected but no %d in output filename')
    configure_logging(args)

    queue = None
    context = None
    if not args.host:
        context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
        queue = context.create_command_queue()
    else:
        context = dummy_context()
        queue = DummyCommandQueue()
        if not numba.have_numba:
            logger.warning('could not import numba: --host mode will be VERY slow')

    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        frontend.run(args, context, queue, dataset, Writer(args))


if __name__ == '__main__':
    main()
