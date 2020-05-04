#!/usr/bin/env python3
import argparse
import os
import logging
import shutil
import uuid

import katdal
import katsdpservices

from katsdpimager import report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Input dataset')
    parser.add_argument('output_dir', type=str, help='Parent directory for output')
    parser.add_argument('prefix', type=str, help='Prefix for output directories and filenames')
    parser.add_argument('stream', type=str, help='Stream name for telescope state inputs')
    parser.add_argument('--log-level', type=str, metavar='LEVEL',
                        help='Logging level [INFO]')
    args = parser.parse_args()
    katsdpservices.setup_logging()
    if args.log_level is not None:
        logging.getLogger().setLevel(args.log_level.upper())

    dataset = katdal.open(args.dataset, chunk_store=None, upgrade_flags=False)
    telstate = dataset.source.telstate.wrapped.root()
    telstate = telstate.view(telstate.join(dataset.source.capture_block_id, args.stream))

    output_dir = '{}_{}'.format(args.prefix, uuid.uuid4())
    output_dir = os.path.join(args.output_dir, output_dir)
    filename = args.prefix + '_report.html'
    tmp_dir = output_dir + '.writing'
    os.mkdir(tmp_dir)
    try:
        common_stats, target_stats = report.get_stats(dataset, telstate)
        report.write_report(common_stats, target_stats, os.path.join(tmp_dir, filename))
        report.write_metadata(dataset, common_stats, target_stats,
                              os.path.join(tmp_dir, 'metadata.json'))
        os.rename(tmp_dir, output_dir)
    except Exception:
        # Make a best effort to clean up
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()
