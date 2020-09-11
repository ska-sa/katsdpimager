#!/usr/bin/env python3
import argparse
import logging
import os
import io
import uuid
import shutil
import json
from contextlib import closing
import gc
import shlex

import numpy as np
from astropy import units
import katsdpservices
import katsdptelstate
import katdal
from katsdpsigproc import accel
import katsdpimageutils.render
import astropy.io.fits as fits

from katsdpimager import frontend, loader, progress, metadata, arguments, profiling
from katsdpimager.profiling import profile, profile_function
import katsdpimager.io    # Use full name to avoid conflict with stdlib io


logger = logging.getLogger()


class Writer(frontend.Writer):
    def __init__(self, args, dataset):
        self.args = args
        self.uuid = uuid.uuid4()

        options = frontend.command_line_options(
            args, dataset, {'input_file', 'output_dir', 'prefix', 'stream', 'input_option'})
        self.extra_fits_headers = fits.Header()
        self.extra_fits_headers['HISTORY'] = \
            'Command line options: ' + ' '.join(shlex.quote(arg) for arg in options)

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
        telstate = raw_data.source.telstate.root()
        namespace = telstate.join(raw_data.source.capture_block_id, args.stream)
        self.telstate = telstate.view(namespace)

    def channel_already_done(self, dataset, channel):
        sub_key = (dataset.raw_target.description, channel)
        return self.telstate.get_indexed('status', sub_key) is not None

    @profile_function(labels=['name', 'channel'])
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
            'PNGThumbNailFileName': [base_filename + '.tnail.png'],
            'CenterFrequency': freq,
            'MinFreq': freq - 0.5 * channel_width,
            'MaxFreq': freq + 0.5 * channel_width,
            'Run': channel
        }
        os.mkdir(tmp_dir)
        try:
            with progress.step('Write {}'.format(description)):
                katsdpimager.io.write_fits_image(dataset, image, image_parameters, filename,
                                                 channel, beam, bunit, self.extra_fits_headers)
            with progress.step('Write thumbnail'):
                with profile(
                    'katsdpimageutils.render.write_image',
                    labels={'filename': filename + '.tnail.png'}
                ):
                    katsdpimageutils.render.write_image(
                        filename, filename + '.tnail.png',
                        width=650, height=500
                    )
            with open(os.path.join(tmp_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, allow_nan=False, indent=2)
            os.rename(tmp_dir, output_dir)
        except Exception:
            # Make a best effort to clean up
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        finally:
            # Something in the PNG writing causes memory to not get freed,
            # and the garbage collector is not kicking in properly on its
            # own.
            with profile('gc.collect'):
                gc.collect()

    def write_fits_grid(self, name, description, fftshift, grid_data, image_parameters, channel):
        pass

    def skip_channel(self, dataset, image_parameters, channel):
        sub_key = (dataset.raw_target.description, channel)
        self.telstate.set_indexed('status', sub_key, 'no-data')

    def _set_statistic(self, key, sub_key, value):
        # The imaging adds floating-point values in a non-deterministic order
        # and hence produces slightly different results every time. In the
        # unlikely event that we filled in statistics in a previous run but
        # didn't make it to setting the completion flag, this can fail.
        try:
            self.telstate.set_indexed(key, sub_key, value)
        except katsdptelstate.ImmutableKeyError as exc:
            logger.warning("%s", exc)

    @profile_function()
    def statistics(self, dataset, image_parameters, channel, **kwargs):
        sub_key = (dataset.raw_target.description, channel)
        peak = kwargs['peak']
        if np.isfinite(peak):
            self._set_statistic('peak', sub_key, peak)
        for pol, total in kwargs['totals'].items():
            self._set_statistic('total', sub_key + (pol,), total)
        if kwargs.get('weights_noise') is not None:
            self._set_statistic('weights_noise', sub_key, kwargs['weights_noise'])
        for key in ['noise', 'normalized_noise', 'major', 'minor', 'psf_patch_size',
                    'compressed_vis']:
            self._set_statistic(key, sub_key, kwargs[key])
        # statistics() is the last step in process_channel, so if we get this
        # far, the channel is fully processed.
        self.telstate.set_indexed('status', sub_key, 'complete')

    def finalize(self, dataset, start_channel, stop_channel):
        sub_key = (dataset.raw_target.description, start_channel, stop_channel)
        profiler = profiling.Profiler.get_profiler()

        flamegraph = io.StringIO()
        profiler.write_flamegraph(flamegraph)
        self._set_statistic('flamegraph', sub_key, flamegraph.getvalue())

        device_flamegraph = io.StringIO()
        profiler.write_device_flamegraph(device_flamegraph)
        self._set_statistic('device_flamegraph', sub_key, device_flamegraph.getvalue())


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
    args = parser.parse_args(namespace=arguments.SmartNamespace())
    katsdpservices.setup_logging()
    if args.log_level is not None:
        logger.setLevel(args.log_level.upper())

    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        writer = Writer(args, dataset)
        context = accel.create_some_context(interactive=False, device_filter=lambda x: x.is_cuda)
        queue = context.create_command_queue()
        frontend.run(args, context, queue, dataset, writer)
        # frontend.run modifies args.stop_channel in place, so even if it
        # wasn't specified by the user it will now be valid.
        writer.finalize(dataset, args.start_channel, args.stop_channel)


if __name__ == '__main__':
    main()
