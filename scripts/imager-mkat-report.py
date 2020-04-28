#!/usr/bin/env python3
import argparse
import os
import json
import shutil
import math
import logging
import uuid
from typing import List, Dict, Tuple

import pkg_resources
import jinja2
import katpoint
import katdal
import katsdptelstate
import katsdpservices
import astropy.units as u

import bokeh.embed
import bokeh.palettes
import bokeh.plotting
import bokeh.model
import bokeh.models
import bokeh.resources

import katsdpimager.metadata


FREQUENCY_PLOT_UNIT = u.MHz
FLUX_PLOT_UNIT = u.mJy / u.beam
PALETTE = bokeh.palettes.colorblind['Colorblind'][8]


logger = logging.getLogger()


class CommonStats:
    """Information extracted from the dataset, independent of target."""

    def __init__(self, dataset: katdal.DataSet, telstate: katsdptelstate.TelescopeState) -> None:
        self.output_channels: List[int] = telstate['output_channels']
        self.channels: int = len(dataset.channels)
        self.channel_width = dataset.channel_width * u.Hz
        self.frequencies: u.Quantity = dataset.freqs * u.Hz


class TargetStats:
    """Collect all the statistics for a single target."""

    def __init__(self, common: CommonStats, target: katpoint.Target) -> None:
        self.common = common
        self.target = target
        # Status string for each channel
        self.status: List[str] = ['masked'] * common.channels
        for channel in common.output_channels:
            self.status[channel] = 'failed'
        # Peak flux per channel (NaN where missing)
        self.peak: u.Quantity = [math.nan] * common.channels * (u.Jy / u.beam)
        # Noise per channel (NaN where missing)
        self.noise: u.Quantity = [math.nan] * common.channels * (u.Jy / u.beam)
        self.plots: Dict[str, str] = {}     # Divs to insert for plots returned by make_plots
        self.frequency_range = bokeh.models.Range1d(
            self.common.frequencies[0].to_value(FREQUENCY_PLOT_UNIT),
            self.common.frequencies[-1].to_value(FREQUENCY_PLOT_UNIT)
        )
        self.channel_range = bokeh.models.Range1d(0, self.common.channels - 1)

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def description(self) -> str:
        return self.target.description

    def _add_channel_range(self, fig: bokeh.plotting.Figure) -> None:
        fig.extra_x_ranges = {'channel': self.channel_range}
        fig.add_layout(bokeh.models.LinearAxis(x_range_name='channel', axis_label='Channel'),
                       'above')

    def make_data_source(self) -> bokeh.models.ColumnDataSource:
        data = {
            'frequency': self.common.frequencies.to_value(FREQUENCY_PLOT_UNIT),
            'status': self.status,
            'noise': self.noise.to_value(FLUX_PLOT_UNIT),
            'peak': self.peak.to_value(FLUX_PLOT_UNIT)
        }
        return bokeh.models.ColumnDataSource(data)

    def make_plot_status(self, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        fig = bokeh.plotting.figure(
            height=200,
            x_axis_label=f'Frequency ({FREQUENCY_PLOT_UNIT})',
            y_axis_label='Status',
            x_range=self.frequency_range,
            y_range=['masked', 'failed', 'no-data', 'complete'],
            sizing_mode='stretch_width', toolbar_location='below'
        )
        fig.cross(x='frequency', y='status', source=source, color=PALETTE[0])
        self._add_channel_range(fig)
        return fig

    def make_plot_flux(self, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        fig = bokeh.plotting.figure(
            x_axis_label=f'Frequency ({FREQUENCY_PLOT_UNIT})',
            y_axis_label=f'Flux ({FLUX_PLOT_UNIT})',
            x_range=self.frequency_range,
            y_range=bokeh.models.DataRange1d(start=0.0),
            sizing_mode='stretch_width',
            tooltips=[
                ('Frequency', f'$x {FREQUENCY_PLOT_UNIT}'),
                ('Channel', '$index'),
                ('Flux', f'$y {FLUX_PLOT_UNIT}')
            ]
        )
        fig.line(x='frequency', y='peak', source=source,
                 line_color=PALETTE[0], legend_label='Peak')
        fig.line(x='frequency', y='noise', source=source,
                 line_color=PALETTE[1], legend_label='Noise')
        self._add_channel_range(fig)
        return fig

    def make_plots(self) -> Dict[str, bokeh.model.Model]:
        """Generate Bokeh figures for the plots."""
        source = self.make_data_source()
        return {
            'status': self.make_plot_status(source),
            'flux': self.make_plot_flux(source)
        }


def get_stats(dataset: katdal.DataSet,
              telstate: katsdptelstate.TelescopeState) -> Tuple[CommonStats, List[TargetStats]]:
    common = CommonStats(dataset, telstate)
    stats = [TargetStats(common, katpoint.Target(target))
             for target in telstate.get('targets', {})]
    stats_lookup = {target.description: target for target in stats}
    for ((target_desc, channel), status) in telstate.get('status', {}).items():
        stats_lookup[target_desc].status[channel] = status
    for ((target_desc, channel), peak) in telstate.get('peak', {}).items():
        stats_lookup[target_desc].peak[channel] = peak * (u.Jy / u.beam)
    for ((target_desc, channel), noise) in telstate.get('noise', {}).items():
        stats_lookup[target_desc].noise[channel] = noise * (u.Jy / u.beam)
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
    static = {}
    for f in pkg_resources.resource_listdir('katsdpimager', 'static'):
        content = pkg_resources.resource_string('katsdpimager', 'static/' + f)
        static[f] = content.decode('utf-8')
    context = {
        'common': common_stats,
        'targets': target_stats,
        'resources': resources,
        'script': script,
        'static': static
    }
    template = env.get_template('report.html.j2')
    template.stream(context).dump(filename)


def write_metadata(dataset: katdal.DataSet,
                   common_stats: CommonStats,
                   target_stats: List[TargetStats],
                   filename: str) -> None:
    half_channel = 0.5 * common_stats.channel_width
    metadata = {
        'ProductType': {
            'ProductTypeName': 'MeerKATReductionProduct',
            'ReductionName': 'SpectralImagerReport'
        },
        **katsdpimager.metadata.make_metadata(
            dataset, [target.target for target in target_stats],
            len(common_stats.output_channels),
            'Spectral-line imaging report'
        ),
        'MinFreq': (common_stats.frequencies[0] - half_channel).to_value(u.Hz),
        'MaxFreq': (common_stats.frequencies[-1] + half_channel).to_value(u.Hz),
        'Run': 0
    }
    with open(filename, 'w') as f:
        json.dump(metadata, f, allow_nan=False, indent=2)


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
        logger.setLevel(args.log_level.upper())

    dataset = katdal.open(args.dataset, chunk_store=None, upgrade_flags=False)
    telstate = dataset.source.telstate.wrapped.root()
    telstate = telstate.view(telstate.join(dataset.source.capture_block_id, args.stream))

    output_dir = '{}_{}'.format(args.prefix, uuid.uuid4())
    output_dir = os.path.join(args.output_dir, output_dir)
    filename = args.prefix + '_report.html'
    tmp_dir = output_dir + '.writing'
    os.mkdir(tmp_dir)
    try:
        common_stats, target_stats = get_stats(dataset, telstate)
        write_report(common_stats, target_stats, os.path.join(tmp_dir, filename))
        write_metadata(dataset, common_stats, target_stats,
                       os.path.join(tmp_dir, 'metadata.json'))
        os.rename(tmp_dir, output_dir)
    except Exception:
        # Make a best effort to clean up
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()
