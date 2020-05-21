"""Quality report for MeerKAT pipeline."""

from abc import abstractmethod, ABC
import json
import math
import logging
from datetime import datetime, timezone
from typing import List, Sequence, Dict, Tuple, Optional

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
FLUX_PLOT_UNIT = u.Jy / u.beam
ANGLE_PLOT_UNIT = u.deg
PALETTE = bokeh.palettes.colorblind['Colorblind'][8]


logger = logging.getLogger()


class SEFDModel(ABC):
    """Abstract class for system-equivalent flux density models."""

    @abstractmethod
    def __call__(self, frequencies: u.Quantity, effective: bool = False) -> u.Quantity:
        """Interpolate to given frequencies.

        If `effective` is true, the correlator efficiency is combined
        with the result.
        """


class PolynomialSEFDModel(SEFDModel):
    """SEFD model that uses a polynomial for each polarization.

    Parameters
    ----------
    min_frequency, max_frequency
        Frequency range over which the model is valid. For frequencies
        outside this range, NaN will be returned.
    coeffs
        2D array of polynomial coefficients. Each column corresponds to a
        single polarization. See :meth:`numpy.polynomial.polynomial.polyval`.
        The units must be flux density units e.g. Jy.
    frequency_unit
        Unit to which frequencies are converted for evaluation of the
        polynomial.
    correlator_efficiency
        Loss in SNR due to quantization in the correlator.
    """

    def __init__(self,
                 min_frequency: u.Quantity, max_frequency: u.Quantity,
                 coeffs: u.Quantity, frequency_unit: u.Unit,
                 correlator_efficiency: float) -> None:
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self._coeffs = coeffs.copy()
        self._frequency_unit = frequency_unit
        self.correlator_efficiency = correlator_efficiency

    def __call__(self, frequencies: u.Quantity, effective: bool = False) -> u.Quantity:
        # Note: don't use to_value, because that can return a value, and we're
        # going to modify x.
        x = frequencies.to(self._frequency_unit).value
        x[(frequencies < self.min_frequency) | (frequencies > self.max_frequency)] = np.nan
        pol_sefd = np.polynomial.polynomial.polyval(
            x, self._coeffs.value, tensor=True) << self._coeffs.unit
        # Take quadratic mean of individual polarizations
        sefd = np.sqrt(np.mean(np.square(pol_sefd), axis=0))
        if effective:
            sefd /= self.correlator_efficiency
        return sefd


def meerkat_sefd_model(band: str) -> SEFDModel:
    if band == 'L':
        coeffs = [
            [2.08778760e+02, 1.08462392e+00, -1.24639611e-03, 4.00344294e-07],  # H
            [7.57838984e+02, -2.24205001e-01, -1.72161897e-04, 1.11118471e-07]  # V
        ]
        return PolynomialSEFDModel(
            900.0 * u.MHz, 1670 * u.MHz,
            (coeffs * u.Jy).T, u.MHz, 0.96)
    else:
        raise ValueError(f'No SEFD model for band {band}')


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
            sefd_model = meerkat_sefd_model(band)
        except ValueError:
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
        # Peak per channel (NaN where missing)
        self.peak: u.Quantity = [math.nan] * common.channels * (u.Jy / u.beam)
        # Noise per channel (NaN where missing)
        self.noise: u.Quantity = [math.nan] * common.channels * (u.Jy / u.beam)
        # Estimate noise from weights (NaN where missing)
        self.weights_noise: u.Quantity = [math.nan] * common.channels * (u.Jy / u.beam)
        # Increase in noise due to imaging weights
        self.normalized_noise = [math.nan] * common.channels
        self.plots: Dict[str, str] = {}     # Divs to insert for plots returned by make_plots
        self.frequency_range = bokeh.models.Range1d(
            self.common.frequencies[0].to_value(FREQUENCY_PLOT_UNIT),
            self.common.frequencies[-1].to_value(FREQUENCY_PLOT_UNIT),
            bounds='auto'
        )
        self.channel_range = bokeh.models.Range1d(0, self.common.channels - 1, bounds='auto')
        self.time_on_target = katsdpimager.metadata.time_on_target(dataset, target)
        self.model_natural_noise: Optional[u.Quantity] = None
        if self.common.sefd is not None and len(self.common.antennas) > 1:
            n = len(self.common.antennas)
            # Correlator efficiency is already folded in to self.common.sefd
            denom = math.sqrt(2 * n * (n - 1) * self.time_on_target * self.common.channel_width)
            self.model_natural_noise = self.common.sefd / denom / u.beam

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

    @staticmethod
    def _line_with_markers(fig: bokeh.plotting.Figure, *args, **kwargs) -> None:
        fig.line(*args, **kwargs)
        if 'line_color' in kwargs:
            kwargs['fill_color'] = kwargs['line_color']
        fig.circle(*args, **kwargs)

    def make_channel_data_source(self) -> bokeh.models.ColumnDataSource:
        data = {
            'frequency': self.common.frequencies.to_value(FREQUENCY_PLOT_UNIT),
            'status': self.status,
            'weights_noise': self.weights_noise.to_value(FLUX_PLOT_UNIT),
            'noise': self.noise.to_value(FLUX_PLOT_UNIT),
            'peak': self.peak.to_value(FLUX_PLOT_UNIT)
        }
        if self.model_natural_noise is not None:
            model_noise = self.model_natural_noise * self.normalized_noise
            data['model_noise'] = model_noise.to_value(FLUX_PLOT_UNIT)
        return bokeh.models.ColumnDataSource(data)

    def make_time_data_source(self) -> bokeh.models.ColumnDataSource:
        data = {
            'time': [datetime.fromtimestamp(ts, timezone.utc) for ts in self.timestamps],
            'elevation': self.elevation.to_value(ANGLE_PLOT_UNIT),
            'parallactic_angle': self.parallactic_angle.to_value(ANGLE_PLOT_UNIT)
        }
        return bokeh.models.ColumnDataSource(data)

    def make_plot_status(self, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        factors = ['masked', 'failed', 'no-data', 'complete']
        fig = bokeh.plotting.figure(
            height=200,
            x_axis_label=f'Frequency ({FREQUENCY_PLOT_UNIT})',
            y_axis_label='Status',
            x_range=self.frequency_range,
            y_range=bokeh.models.FactorRange(factors=factors,
                                             bounds='auto',
                                             min_interval=len(factors)),
            sizing_mode='stretch_width', toolbar_location='below'
        )
        fig.cross(x='frequency', y='status', source=source, color=PALETTE[0])
        self._add_channel_range(fig)
        return fig

    def make_plot_snr(self, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        fig = bokeh.plotting.figure(
            x_axis_label=f'Frequency ({FREQUENCY_PLOT_UNIT})',
            y_axis_label=f'Flux density per beam ({FLUX_PLOT_UNIT})',
            x_range=self.frequency_range,
            y_axis_type='log',
            sizing_mode='stretch_width'
        )
        si_format = bokeh.models.CustomJSHover(code="""
            const threshold = [1e-9, 1e-6, 1e-3, 1e0, 1e3,  1e6,  1e9, 1e12];
            const scale =     [1e12,  1e9,  1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12];
            const suffix =    [ "p",  "n",  "µ", "m",  "",  "k",  "M",  "G", "T"];
            var i = 0;
            if (value != value)
                return "N/A ";
            while (i < threshold.length && value >= threshold[i])
                i++;
            return sprintf(format, value * scale[i]) + " " + suffix[i];
            """)
        fig.add_tools(bokeh.models.HoverTool(
            tooltips=[
                ('Frequency', f'@frequency{{0.0000}} {FREQUENCY_PLOT_UNIT}'),
                ('Channel', '$index'),
                ('Peak', f'@peak{{%.3f}}{FLUX_PLOT_UNIT}'),
                ('Noise', f'@noise{{%.3f}}{FLUX_PLOT_UNIT}'),
                ('Predicted noise (weights)', f'@weights_noise{{%.3f}}{FLUX_PLOT_UNIT}'),
                ('Predicted noise (model)', f'@model_noise{{%.3f}}{FLUX_PLOT_UNIT}')
            ],
            formatters={
                '@peak': si_format,
                '@noise': si_format,
                '@weights_noise': si_format,
                '@model_noise': si_format
            }
        ))
        self._line_with_markers(
            fig, x='frequency', y='peak', source=source, name='peak',
            line_color=PALETTE[0], legend_label='Peak')
        self._line_with_markers(
            fig, x='frequency', y='noise', source=source, name='noise',
            line_color=PALETTE[1], legend_label='Noise')
        self._line_with_markers(
            fig, x='frequency', y='weights_noise', source=source, name='weights_noise',
            line_color=PALETTE[2], legend_label='Predicted noise (weights)')
        if self.model_natural_noise is not None:
            self._line_with_markers(
                fig, x='frequency', y='model_noise', source=source, name='model_noise',
                line_color=PALETTE[3], legend_label='Predicted noise (model)')
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
            'snr': self.make_plot_snr(channel_source),
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
    for ((target_desc, channel), noise) in telstate.get('weights_noise', {}).items():
        stats_lookup[target_desc].weights_noise[channel] = noise * (u.Jy / u.beam)
    for ((target_desc, channel), normalized_noise) in telstate.get('normalized_noise', {}).items():
        stats_lookup[target_desc].normalized_noise[channel] = normalized_noise
    return common, stats


def format_duration(duration: u.Quantity) -> str:
    seconds = int(round(duration.to_value(u.s)))
    return '{}:{:02}:{:02}'.format(seconds // 3600, seconds // 60 % 60, seconds % 60)


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
