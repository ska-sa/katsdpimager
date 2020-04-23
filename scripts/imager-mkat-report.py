#!/usr/bin/env python3
import argparse
import os
import json
import shutil
from datetime import datetime

import jinja2
import katpoint
import katdal


class TargetStats:
    """Collect all the statistics for a single target."""

    def __init__(self, target):
        self.target = target
        self.status = {}        # Status string for each channel

    @property
    def name(self):
        return self.target.name

    @property
    def description(self):
        return self.target.description


def get_target_stats(telstate):
    stats = [TargetStats(katpoint.Target(target)) for target in telstate.get('targets', {})]
    stats_lookup = {target.description: target for target in stats}
    for ((target_desc, channel), status) in telstate.get('status', {}).items():
        stats_lookup[target_desc].status[channel] = status
    return stats


def write_report(dataset, telstate, target_stats, filename):
    env = jinja2.Environment(
        loader=jinja2.PackageLoader('katsdpimager'),
        autoescape=True,
        undefined=jinja2.StrictUndefined
    )
    context = {
        'dataset': dataset,
        'telstate': telstate,
        'targets': target_stats
    }
    template = env.get_template('report.html.j2')
    template.stream(context).dump(filename)


def write_metadata(dataset, telstate, target_stats, filename):
    # TODO refactor to share code with imager-mkat-pipeline.py
    obs_params = dataset.obs_params

    metadata = {
        'ProductType': {
            'ProductTypeName': 'MeerKATReductionProduct',
            'ReductionName': 'Spectral Imager Report'
        },
        'CaptureBlockId': dataset.source.capture_block_id,
        'ScheduleBlockIdCode': obs_params.get('sb_id_code', 'UNKNOWN'),
        'Description': obs_params.get('description', 'UNKNOWN') + ': Spectral-line imaging report',
        'ProposalId': obs_params.get('proposal_id', 'UNKNOWN'),
        'Observer': obs_params.get('observer', 'UNKNOWN'),
        # Solr doesn't accept +00:00, only Z, so we can't just format a timezone-aware value
        'StartTime': datetime.utcnow().isoformat() + 'Z',
        'ChannelWidth': dataset.channel_width,
        'Targets': [target.name for target in target_stats],
        'KatpointTargets': [target.description for target in target_stats]
    }
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('stream', type=str)
    args = parser.parse_args()

    dataset = katdal.open(args.dataset, chunk_store=None, upgrade_flags=False)
    telstate = dataset.source.telstate.wrapped.root()
    telstate = telstate.view(telstate.join(dataset.source.capture_block_id, args.stream))

    try:
        tmp_dir = args.output_dir + '.writing'
        os.mkdir(tmp_dir)
        target_stats = get_target_stats(telstate)
        write_report(dataset, telstate, target_stats, os.path.join(tmp_dir, 'report.html'))
        write_metadata(dataset, telstate, target_stats, os.path.join(tmp_dir, 'metadata.json'))
        os.rename(tmp_dir, args.output_dir)
    except Exception:
        # Make a best effort to clean up
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()
