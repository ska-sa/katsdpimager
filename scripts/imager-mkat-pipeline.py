#!/usr/bin/env python3
import argparse
import logging
import os
import uuid
import shutil
import json
from datetime import datetime
from contextlib import closing
import gc

from astropy import units
from astropy.coordinates import Angle
import katsdpservices
import katdal
from katsdpsigproc import accel

from katsdpimager import frontend, loader, io, progress, render


logger = logging.getLogger()


class Writer(frontend.Writer):
    def __init__(self, args, dataset):
        self.args = args
        self.uuid = uuid.uuid4()

        raw_data = dataset.raw_data
        if not isinstance(raw_data, katdal.DataSet):
            raise RuntimeError('Only katdal data sets are supported')
        radec_deg = dataset.phase_centre().to(units.deg)
        obs_params = raw_data.obs_params
        self.common_metadata = {
            'ProductType': {
                'ProductTypeName': 'FITSImageProduct',
                'ReductionName': 'Spectral Image'
            },
            'CaptureBlockId': raw_data.source.capture_block_id,
            'ScheduleBlockIdCode': obs_params.get('sb_id_code', 'UNKNOWN'),
            'Description': obs_params.get('description', 'UNKNOWN') + ': Spectral-line image',
            'ProposalId': obs_params.get('proposal_id', 'UNKNOWN'),
            'Observer': obs_params.get('observer', 'UNKNOWN'),
            # Solr doesn't accept +00:00, only Z, so we can't just format a timezone-aware value
            'StartTime': datetime.utcnow().isoformat() + 'Z',
            'Bandwidth': raw_data.channel_width,
            'ChannelWidth': raw_data.channel_width,
            'NumFreqChannels': 1,
            'RightAscension': [Angle(radec_deg[0]).to_string(units.hour, sep=':', pad=True)],
            'Declination': [Angle(radec_deg[1]).to_string(units.deg, sep=':', pad=True)],
            # JSON schema limits format to fixed-point with at most 10 decimal places
            'DecRa': [','.join('{:.10f}'.format(angle) for angle in radec_deg.value[::-1])],
            'Targets': [dataset.raw_target.name],
            'KatpointTargets': [dataset.raw_target.description],
            # katdal gives dump period in seconds, metadata value needs to be in hours
            'IntegrationTime': [raw_data.dump_period * len(raw_data.dumps) / 3600.0]
        }

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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT',
                        help='Input data set')
    parser.add_argument('output_dir', type=str, metavar='DIR',
                        help='Parent directory for output')
    parser.add_argument('prefix', type=str,
                        help='Prefix for output directories and filenames')
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
