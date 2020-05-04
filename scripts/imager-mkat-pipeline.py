#!/usr/bin/env python3
import argparse
import logging
import os
import uuid
import shutil
import json
from contextlib import closing
import gc

import numpy as np
from astropy import units
import katsdpservices
import katdal
from katsdpsigproc import accel

from katsdpimager import frontend, loader, io, progress, render, metadata


logger = logging.getLogger()


class Writer(frontend.Writer):
    def __init__(self, args, dataset):
        self.args = args
        self.uuid = uuid.uuid4()

        raw_data = dataset.raw_data
        if not isinstance(raw_data, katdal.DataSet):
            raise RuntimeError('Only katdal data sets are supported')
        self.common_metadata = {
            'ProductType': {
                'ProductTypeName': 'FITSImageProduct',
                'ReductionName': 'Spectral Image'
            },
            **metadata.make_metadata(raw_data, [dataset.raw_target], 1,
                                     'Spectral-line image')
        }
        # katdal's TelstateToStr doesn't support indexed keys yet.
        # The .wrapped can be removed once it does.
        telstate = raw_data.source.telstate.wrapped.root()
        namespace = telstate.join(raw_data.source.capture_block_id, args.stream)
        self.telstate = telstate.view(namespace)

    def write_fits_image(self, name, description, dataset, image, image_parameters, channel,
                         beam=None, bunit='Jy/beam'):
        if name != 'clean':
            return
        # Add a unique component to the directory name so that it will be
        # unique even if the task dies and is re-run.
        output_dir = '{}_{:05}_{}'.format(self.args.prefix, channel, self.uuid)
        output_dir = os.path.join(self.args.output_dir, output_dir)
        tmp_dir = output_dir + '.writing'
        base_filename = '{}_{:05}.fits'.format(self.args.prefix, channel)
        filename = os.path.join(tmp_dir, base_filename)
        freq = image_parameters.wavelength.to(units.Hz, equivalencies=units.spectral()).value
        channel_width = dataset.raw_data.channel_width
        metadata = {
            **self.common_metadata,
            'FITSImageFilename': [base_filename],
            'PNGImageFileName': [base_filename + '.png'],
            'PNGThumbNailFileName': [base_filename + '.tnail.png'],
            'CenterFrequency': freq,
            'MinFreq': freq - 0.5 * channel_width,
            'MaxFreq': freq + 0.5 * channel_width,
            'Run': channel
        }
        os.mkdir(tmp_dir)
        try:
            with progress.step('Write {}'.format(description)):
                io.write_fits_image(dataset, image, image_parameters, filename,
                                    channel, beam, bunit)
            with progress.step('Write PNG'):
                render.write_image(filename, filename + '.png',
                                   width=6500, height=5000,
                                   dpi=10 * render.DEFAULT_DPI)
            with progress.step('Write thumbnail'):
                render.write_image(filename, filename + '.tnail.png', width=650, height=500)
            with open(os.path.join(tmp_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, allow_nan=False, indent=2)
            os.rename(tmp_dir, output_dir)
            sub_key = (dataset.raw_target.description, channel)
            self.telstate.set_indexed('peak', sub_key, float(np.nanmax(image)))
            self.telstate.set_indexed('status', sub_key, 'complete')
        except Exception:
            # Make a best effort to clean up
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        finally:
            # Something in the PNG writing causes memory to not get freed,
            # and the garbage collector is not kicking in properly on its
            # own.
            gc.collect()

    def write_fits_grid(self, name, description, fftshift, grid_data, image_parameters, channel):
        pass

    def skip_channel(self, dataset, image_parameters, channel):
        sub_key = (dataset.raw_target.description, channel)
        self.telstate.set_indexed('status', sub_key, 'no-data')

    def statistics(self, dataset, image_parameters, channel, **kwargs):
        sub_key = (dataset.raw_target.description, channel)
        self.telstate.set_indexed('noise', sub_key, kwargs['noise'])
        self.telstate.set_indexed('normalized_noise', sub_key, kwargs['normalized_noise'])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT',
                        help='Input data set')
    parser.add_argument('output_dir', type=str, metavar='DIR',
                        help='Parent directory for output')
    parser.add_argument('prefix', type=str,
                        help='Prefix for output directories and filenames')
    parser.add_argument('stream', type=str,
                        help='Stream name for telescope state outputs')
    parser.add_argument('--log-level', type=str, metavar='LEVEL',
                        help='Logging level [INFO]')

    frontend.add_options(parser)

    # Arguments only supported by offline imager.py
    parser.set_defaults(host=False, vis_limit=None)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.input_option = ['--' + opt for opt in args.input_option]
    katsdpservices.setup_logging()
    if args.log_level is not None:
        logger.setLevel(args.log_level.upper())

    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        writer = Writer(args, dataset)
        context = accel.create_some_context(interactive=False, device_filter=lambda x: x.is_cuda)
        queue = context.create_command_queue()
        frontend.run(args, context, queue, dataset, writer)


if __name__ == '__main__':
    main()
