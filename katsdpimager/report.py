"""Quality report for MeerKAT pipeline."""

import json
import math
import logging
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import pkg_resources
import jinja2
import numpy as np
import katpoint
import katdal
import katsdptelstate
import astropy.units as u
import h5py

import bokeh.embed
import bokeh.palettes
import bokeh.plotting
import bokeh.model
import bokeh.models
import bokeh.resources

import katsdpimager.metadata


FREQUENCY_PLOT_UNIT = u.MHz
FLUX_PLOT_UNIT = u.mJy / u.beam
ANGLE_PLOT_UNIT = u.deg
PALETTE = bokeh.palettes.colorblind['Colorblind'][8]


logger = logging.getLogger()


class SEFDModel:
    """Model of system-equivalent flux density."""

    @staticmethod
    def _load_dataset(dataset: h5py.Dataset) -> u.Quantity:
        return dataset[:] << u.Unit(dataset.attrs['unit'])

    def __init__(self, dish_type: str, band: str) -> None:
        path = f'models/sefd/{dish_type}/v1/sefd_{band}.h5'
        with h5py.File(pkg_resources.resource_stream('katsdpimager', path), 'r') as h5file:
            self.frequencies = self._load_dataset(h5file['frequency'])
            h = self._load_dataset(h5file['H'])
            v = self._load_dataset(h5file['V'])
            # Take quadratic mean to get SEFD for Stokes I
            self.sefd = np.sqrt((np.square(h) + np.square(v)) * 0.5)
            self.correlator_efficiency = h5file.attrs['correlator_efficiency']

    def __call__(self, frequencies: u.Quantity, effective: bool = False) -> u.Quantity:
        """Interpolate to given frequencies.

        If `effective` is true, the correlator efficiency is combined
        with the result.
        """
        frequencies = frequencies.to(self.frequencies.unit, equivalencies=u.spectral())
        ans = np.interp(frequencies, self.frequencies, self.sefd, left=np.nan, right=np.nan)
        if effective:
            ans /= self.correlator_efficiency
        return ans


class CommonStats:
    """Information extracted from the dataset, independent of target."""

    def __init__(self, dataset: katdal.DataSet, telstate: katsdptelstate.TelescopeState) -> None:
        self.output_channels: List[int] = telstate['output_channels']
        self.channels: int = len(dataset.channels)
        self.channel_width = dataset.channel_width * u.Hz
        self.frequencies: u.Quantity = dataset.freqs * u.Hz
        self.antennas = dataset.ants
        # XXX for now this only supports MeerKAT. MeerKAT+ will be heterogeneous.
        self.sefd: Optional[u.Quantity] = None
        band = dataset.spectral_windows[dataset.spw].band
        try:
            sefd_model = SEFDModel('meerkat', band)
        except FileNotFoundError:
            logger.warning('No SEFD model for band %s', band)
        else:
            self.sefd = sefd_model(self.frequencies, effective=True)


class TargetStats:
    """Collect all the statistics for a single target."""

    def __init__(self, dataset: katdal.DataSet,
                 common: CommonStats, target: katpoint.Target) -> None:
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
            self.common.frequencies[-1].to_value(FREQUENCY_PLOT_UNIT),
            bounds='auto'
        )
        self.channel_range = bokeh.models.Range1d(0, self.common.channels - 1, bounds='auto')
        self.time_on_target = katsdpimager.metadata.time_on_target(dataset, target)
        self.predicted_noise: Optional[u.Quantity] = None
        if self.common.sefd is not None and len(self.common.antennas) > 1:
            n = len(self.common.antennas)
            # Correlator efficiency is already folded in to self.common.sefd
            denom = math.sqrt(2 * n * (n - 1) * self.time_on_target * self.common.channel_width)
            self.predicted_noise = self.common.sefd / denom / u.beam

        mask = katsdpimager.metadata.target_mask(dataset, target)
        self.timestamps = dataset.timestamps[mask]
        self.time_range = bokeh.models.Range1d(
            datetime.fromtimestamp(self.timestamps[0], timezone.utc),
            datetime.fromtimestamp(self.timestamps[-1], timezone.utc)
        )
        array_ant = dataset.sensor['Antennas/array/antenna'][0]
        self.elevation = target.azel(timestamp=self.timestamps, antenna=array_ant)[1] << u.rad
        self.parallactic_angle = target.parallactic_angle(
            timestamp=self.timestamps, antenna=array_ant) << u.rad

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

    def make_channel_data_source(self) -> bokeh.models.ColumnDataSource:
        data = {
            'frequency': self.common.frequencies.to_value(FREQUENCY_PLOT_UNIT),
            'status': self.status,
            'noise': self.noise.to_value(FLUX_PLOT_UNIT),
            'peak': self.peak.to_value(FLUX_PLOT_UNIT)
        }
        if self.predicted_noise is not None:
            data['predicted_noise'] = self.predicted_noise.to_value(FLUX_PLOT_UNIT)
        return bokeh.models.ColumnDataSource(data)

    def make_time_data_source(self) -> bokeh.models.ColumnDataSource:
        data = {
            'time': [datetime.fromtimestamp(ts, timezone.utc) for ts in self.timestamps],
            'elevation': self.elevation.to_value(ANGLE_PLOT_UNIT),
            'parallactic_angle': self.parallactic_angle.to_value(ANGLE_PLOT_UNIT)
        }
        return bokeh.models.ColumnDataSource(data)

    def make_plot_status(self, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        fig = bokeh.plotting.figure(
            height=200,
            x_axis_label=f'Frequency ({FREQUENCY_PLOT_UNIT})',
            y_axis_label='Status',
            x_range=self.frequency_range,
            y_range=bokeh.models.FactorRange(factors=['masked', 'failed', 'no-data', 'complete'],
                                             bounds='auto'),
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
                ('Frequency', f'@frequency{{0.0000}} {FREQUENCY_PLOT_UNIT}'),
                ('Channel', '$index'),
                ('Flux', f'@$name {FLUX_PLOT_UNIT}')
            ]
        )
        fig.line(x='frequency', y='peak', source=source, name='peak',
                 line_color=PALETTE[0], legend_label='Peak')
        fig.line(x='frequency', y='noise', source=source, name='noise',
                 line_color=PALETTE[1], legend_label='Noise')
        if self.predicted_noise is not None:
            fig.line(x='frequency', y='predicted_noise', source=source, name='predicted_noise',
                    line_color=PALETTE[2], legend_label='Predicted noise')
        self._add_channel_range(fig)
        return fig

    def make_plot_elevation(self, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        unit_str = ANGLE_PLOT_UNIT.to_string('unicode')  # deg -> °
        fig = bokeh.plotting.figure(
            x_axis_label='Time (UTC)', x_axis_type='datetime',
            y_axis_label=f'Elevation ({unit_str})',
            x_range=self.time_range,
            y_range=[0.0, (90 * u.deg).to_value(ANGLE_PLOT_UNIT)],
            sizing_mode='stretch_width'
        )
        fig.add_tools(bokeh.models.HoverTool(
            tooltips=[
                ('Time', '@time{%Y-%m-%d %H:%M:%S}'),
                ('Elevation', f'@elevation {unit_str}')
            ],
            formatters={
                '@time': 'datetime'
            }
        ))
        fig.circle(x='time', y='elevation', source=source, line_color=PALETTE[0])
        # TODO: plot LST on the opposite axis?
        return fig

    def make_plot_parallactic_angle(
            self, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        unit_str = ANGLE_PLOT_UNIT.to_string('unicode')  # deg -> °
        fig = bokeh.plotting.figure(
            x_axis_label='Time (UTC)', x_axis_type='datetime',
            y_axis_label=f'Parallactic angle ({unit_str})',
            x_range=self.time_range,
            y_range=[(-180 * u.deg).to_value(ANGLE_PLOT_UNIT),
                     (180 * u.deg).to_value(ANGLE_PLOT_UNIT)],
            sizing_mode='stretch_width'
        )
        fig.add_tools(bokeh.models.HoverTool(
            tooltips=[
                ('Time', '@time{%Y-%m-%d %H:%M:%S}'),
                ('Angle', f'@parallactic_angle {unit_str}')
            ],
            formatters={
                '@time': 'datetime'
            }
        ))
        fig.circle(x='time', y='parallactic_angle', source=source, line_color=PALETTE[0])
        # TODO: plot LST on the opposite axis?
        return fig

    def make_plots(self) -> Dict[str, bokeh.model.Model]:
        """Generate Bokeh figures for the plots."""
        channel_source = self.make_channel_data_source()
        time_source = self.make_time_data_source()
        return {
            'status': self.make_plot_status(channel_source),
            'flux': self.make_plot_flux(channel_source),
            'elevation': self.make_plot_elevation(time_source),
            'parallactic_angle': self.make_plot_parallactic_angle(time_source)
        }


def get_stats(dataset: katdal.DataSet,
              telstate: katsdptelstate.TelescopeState) -> Tuple[CommonStats, List[TargetStats]]:
    common = CommonStats(dataset, telstate)
    stats = [TargetStats(dataset, common, katpoint.Target(target))
             for target in telstate.get('targets', {})]
    stats_lookup = {target.description: target for target in stats}
    for ((target_desc, channel), status) in telstate.get('status', {}).items():
        stats_lookup[target_desc].status[channel] = status
    for ((target_desc, channel), peak) in telstate.get('peak', {}).items():
        stats_lookup[target_desc].peak[channel] = peak * (u.Jy / u.beam)
    for ((target_desc, channel), noise) in telstate.get('noise', {}).items():
        stats_lookup[target_desc].noise[channel] = noise * (u.Jy / u.beam)
    return common, stats


def format_duration(duration: u.Quantity) -> str:
    seconds = int(round(duration.to_value(u.s)))
    return '{}:{:02}:{:02}s'.format(seconds // 3600, seconds // 60 % 60, seconds % 60)


def write_report(common_stats: CommonStats, target_stats: List[TargetStats],
                 filename: str) -> None:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader('katsdpimager'),
        autoescape=True,
        undefined=jinja2.StrictUndefined
    )
    env.filters['duration'] = format_duration

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
            'ReductionName': 'Spectral Imager Report'
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
