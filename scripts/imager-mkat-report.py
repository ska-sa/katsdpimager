#!/usr/bin/env python3
import argparse
import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Tuple

import jinja2
import katpoint
import katdal
import katsdptelstate
import astropy.units as u

import bokeh.embed
import bokeh.plotting
import bokeh.model
import bokeh.resources


class CommonStats:
    """Information extracted from the dataset, independent of target."""

    def __init__(self, dataset: katdal.DataSet, telstate: katsdptelstate.TelescopeState) -> None:
        # TODO: have master controller write the output channel range
        spw = dataset.spectral_windows[dataset.spw]
        self.channels = range(dataset.spectral_windows[dataset.spw].num_chans)
        self.frequencies = [spw.channel_freqs[channel] for channel in self.channels] * u.Hz


class TargetStats:
    """Collect all the statistics for a single target."""

    def __init__(self, common: CommonStats, target: katpoint.Target) -> None:
        self.common = common
        self.target = target
        self.status: Dict[int, str] = {}    # Status string for each channel
        self.plots: Dict[str, str] = {}     # Divs to insert for plots returned by make_plots

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def description(self) -> str:
        return self.target.description

    def make_plot_status(self) -> bokeh.model.Model:
        fig = bokeh.plotting.figure(
            title='Successful channels',
            x_axis_label='Channel',
            y_axis_label='Present')
        fig.line(self.common.channels,
                 [int(self.status.get(channel) == 'complete') for channel in self.common.channels])
        return fig

    def make_plots(self) -> Dict[str, bokeh.model.Model]:
        """Generate Bokeh figures for the plots."""
        return {
            'status': self.make_plot_status()
        }


def get_stats(dataset: katdal.DataSet,
              telstate: katsdptelstate.TelescopeState) -> Tuple[CommonStats, List[TargetStats]]:
    common = CommonStats(dataset, telstate)
    stats = [TargetStats(common, katpoint.Target(target))
             for target in telstate.get('targets', {})]
    stats_lookup = {target.description: target for target in stats}
    for ((target_desc, channel), status) in telstate.get('status', {}).items():
        stats_lookup[target_desc].status[channel] = status
    return common, stats


def write_report(common_stats: CommonStats, target_stats: List[TargetStats],
                 filename: str) -> None:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader('katsdpimager'),
        autoescape=True,
        undefined=jinja2.StrictUndefined
    )

    plots = [target.make_plots() for target in target_stats]
    # Flatten all plots into one list to pass to bokeh
    flat_plots: List[Dict[str, bokeh.model.Model]] = []
    for plot_dict in plots:
        flat_plots.extend(plot_dict.values())
    script, divs = bokeh.embed.components(flat_plots)
    # Distribute divs back to individual objects
    i = 0
    for (target, plot_dict) in zip(target_stats, plots):
        for name in plot_dict:
            target.plots[name] = divs[i]
            i += 1

    resources = bokeh.resources.INLINE.render()
    context = {
        'common': common_stats,
        'targets': target_stats,
        'resources': resources,
        'script': script
    }
    template = env.get_template('report.html.j2')
    template.stream(context).dump(filename)


def write_metadata(dataset: katdal.DataSet,
                   telstate: katsdptelstate.TelescopeState,
                   common_stats: CommonStats,
                   target_stats: List[TargetStats],
                   filename: str) -> None:
    # TODO: check if all parameters are needed
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


def main() -> None:
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
        common_stats, target_stats = get_stats(dataset, telstate)
        write_report(common_stats, target_stats, os.path.join(tmp_dir, 'report.html'))
        write_metadata(dataset, telstate, common_stats, target_stats,
                       os.path.join(tmp_dir, 'metadata.json'))
        os.rename(tmp_dir, args.output_dir)
    except Exception:
        # Make a best effort to clean up
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()
