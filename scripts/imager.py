#!/usr/bin/env python
import sys
import argparse
import logging
from contextlib import contextmanager

import numpy as np
import colors
import katsdpsigproc.accel as accel

from katsdpimager import frontend, polarization, weight, io, progress, numba


logger = logging.getLogger()


class Writer(frontend.Writer):
    def __init__(self, args):
        self.args = args

    def write_fits_image(self, name, description, dataset, image, image_parameters, channel,
                         beam=None, bunit='JY/BEAM'):
        if name == 'clean':
            filename = self.args.output_file
        else:
            filename = getattr(self.args, 'write_' + name)
        if filename is not None:
            with progress.step('Write {}'.format(description)):
                io.write_fits_image(dataset, image, image_parameters, filename,
                                    channel, beam, bunit)

    def write_fits_grid(self, name, description, fftshift, grid_data, image_parameters, channel):
        filename = getattr(self.args, 'write_' + name)
        if filename is not None:
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
    group.add_argument('--pixel-size', type=frontend.parse_quantity,
                       help='Size of each image pixel [computed from array]')
    group.add_argument('--pixels', type=int,
                       help='Number of pixels in image [computed from array]')
    group.add_argument('--stokes', type=polarization.parse_stokes, default='I',
                       help='Stokes parameters to image e.g. IQUV for full-Stokes [%(default)s]')
    group.add_argument('--precision', choices=['single', 'double'], default='single',
                       help='Internal floating-point precision [%(default)s]')

    group = parser.add_argument_group('Weighting options')
    group.add_argument('--weight-type',
                       choices=[name.lower() for name in weight.WeightType.__members__],
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
    group.add_argument('--w-step', type=frontend.parse_quantity, default='1.0',
                       help='Separation between W planes, in subgrid cells or a distance '
                            '[%(default)s]'),
    group.add_argument('--max-w', type=frontend.parse_quantity,
                       help='Largest w, as either distance or wavelengths [longest baseline]')
    group.add_argument('--aa-width', type=float, default=7,
                       help='Support of anti-aliasing kernel [%(default)s]')
    group.add_argument('--kernel-width', type=int, default=60,
                       help='Support of combined anti-aliasing + w kernel [computed]')
    group.add_argument('--eps-w', type=float, default=0.001,
                       help='Level at which to truncate W kernel [%(default)s]')

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
            logger.warn('could not import numba: --host mode will be VERY slow')

    frontend.run(args, context, queue, Writer(args))


if __name__ == '__main__':
    main()
