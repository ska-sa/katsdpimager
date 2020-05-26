"""Quality report for MeerKAT pipeline."""

from abc import abstractmethod, ABC
import itertools
import io
import json
import math
import logging
import xml.sax.saxutils
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import pkg_resources
import jinja2
import numpy as np
import katpoint
import katdal
import katsdptelstate
import astropy.units as u

import matplotlib
import matplotlib.figure
import matplotlib.patches

import bokeh.embed
import bokeh.palettes
import bokeh.plotting
import bokeh.model
import bokeh.models
import bokeh.resources

import katsdpimager.metadata


FREQUENCY_PLOT_UNIT = u.MHz
FLUX_DENSITY_PLOT_UNIT = u.Jy
PIXEL_PLOT_UNIT = u.Jy / u.beam
ANGLE_PLOT_UNIT = u.deg
PALETTE = bokeh.palettes.colorblind['Colorblind'][8]
SI_FORMAT = bokeh.models.CustomJSHover(code="""
    const threshold = [1e-9, 1e-6, 1e-3, 1e0, 1e3,  1e6,  1e9, 1e12];
    const scale =     [1e12,  1e9,  1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12];
    const suffix =    [ "p",  "n",  "µ", "m",  "",  "k",  "M",  "G", "T"];
    var i = 0;
    if (value != value)
        return "N/A ";
    while (i < threshold.length && Math.abs(value) >= threshold[i])
        i++;
    return sprintf(format, value * scale[i]) + " " + suffix[i];
    """)

logger = logging.getLogger()


class SEFDModel(ABC):
    """Abstract class for system-equivalent flux density models."""

    @abstractmethod
    def __call__(self, frequencies: u.Quantity, effective: bool = False) -> u.Quantity:
        """Interpolate to given frequencies.

        If `effective` is true, the correlator efficiency is combined
        with the result.

        Given frequencies outside the support of the model return NaN.
        """


class PolynomialSEFDModel(SEFDModel):
    """SEFD model that uses a polynomial for each polarization.

    Parameters
    ----------
    min_frequency, max_frequency
        Frequency range over which the model is valid. For frequencies
        outside this range, NaN will be returned.
    coeffs
        2D array of polynomial coefficients. Each row corresponds to a
        single polarization, with coefficients increasing in exponent.
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
            x, self._coeffs.value.T, tensor=True) << self._coeffs.unit
        # Take quadratic mean of individual polarizations
        sefd = np.sqrt(np.mean(np.square(pol_sefd), axis=0))
        if effective:
            sefd /= self.correlator_efficiency
        return sefd


def meerkat_sefd_model(band: str) -> SEFDModel:
    """Look up an SEFD model for MeerKAT.

    Parameters
    ----------
    band
        Frequency band for which the model will be applicable. Note that the model
        will likely not cover the edges of the band.

    Raises
    ------
    ValueError
        If no model is available for the given band
    """
    if band == 'L':
        coeffs = [
            [2.08778760e+02, 1.08462392e+00, -1.24639611e-03, 4.00344294e-07],   # H
            [7.57838984e+02, -2.24205001e-01, -1.72161897e-04, 1.11118471e-07]   # V
        ] * u.Jy
        return PolynomialSEFDModel(900.0 * u.MHz, 1670 * u.MHz, coeffs, u.MHz, 0.96)
    elif band == 'UHF':
        coeffs = [
            [1.20011355e+03, -7.08771871e-01, -1.46789604e-03, 1.29596990e-06],  # H
            [2.06467503e+03, -4.16858417e+00, 3.16131140e-03, -7.31852890e-07]   # V
        ] * u.Jy
        return PolynomialSEFDModel(580 * u.MHz, 1015 * u.MHz, coeffs, u.MHz, 0.96)
    else:
        raise ValueError(f'No SEFD model for band {band}')


class SVGMangler(xml.sax.saxutils.XMLGenerator):
    """Rip out the XML header and width and height attributes from an SVG.

    This is used to massage the SVG written by matplotlib to make it suitable for embedding.
    """

    def startDocument(self):
        pass      # Suppress the <?xml> header

    def startElement(self, name, attrs):
        if name == 'svg':
            # attrs is a non-mutable mapping, so we can't just delete from it
            attrs2 = dict(item for item in attrs.items() if item[0] not in {'width', 'height'})
            super().startElement(name, attrs.__class__(attrs2))
        else:
            super().startElement(name, attrs)


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
        # Total flux density per (polarization, channel) (NaN where missing)
        self.totals: u.Quantity = {
            pol: [math.nan] * common.channels * u.Jy
            for pol in 'IQUV'
        }
        # Noise per channel (NaN where missing)
        self.noise: u.Quantity = [math.nan] * common.channels * (u.Jy / u.beam)
        # Estimate noise from weights (NaN where missing)
        self.weights_noise: u.Quantity = [math.nan] * common.channels * (u.Jy / u.beam)
        # Increase in noise due to imaging weights
        self.normalized_noise = [math.nan] * common.channels
        self.plots: Dict[str, str] = {}     # Divs to insert for plots returned by make_plots
        self.uv_coverage = ''
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
            datetime.fromtimestamp(self.timestamps[0] - 0.5 * dataset.dump_period, timezone.utc),
            datetime.fromtimestamp(self.timestamps[-1] + 0.5 * dataset.dump_period, timezone.utc)
        )
        # Find contiguous time intervals on target
        delta = np.diff(mask, prepend=0, append=0)
        starts = np.nonzero(delta == 1)[0]
        ends = np.nonzero(delta == -1)[0] - 1
        self.time_intervals = list(zip(dataset.timestamps[starts] - 0.5 * dataset.dump_period,
                                       dataset.timestamps[ends] + 0.5 * dataset.dump_period))

        self.array_ant = dataset.sensor['Antennas/array/antenna'][0]
        self.ants = dataset.ants
        self.elevation = target.azel(timestamp=self.timestamps, antenna=self.array_ant)[1] << u.rad
        self.parallactic_angle = target.parallactic_angle(
            timestamp=self.timestamps, antenna=self.array_ant) << u.rad

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
            'weights_noise': self.weights_noise.to_value(PIXEL_PLOT_UNIT),
            'noise': self.noise.to_value(PIXEL_PLOT_UNIT),
            'peak': self.peak.to_value(PIXEL_PLOT_UNIT)
        }
        for pol, total in self.totals.items():
            data[f'total_{pol}'] = total.to_value(FLUX_DENSITY_PLOT_UNIT)
        if self.model_natural_noise is not None:
            model_noise = self.model_natural_noise * self.normalized_noise
            data['model_noise'] = model_noise.to_value(PIXEL_PLOT_UNIT)
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
            y_axis_label=f'Flux density per beam ({PIXEL_PLOT_UNIT})',
            x_range=self.frequency_range,
            y_axis_type='log',
            sizing_mode='stretch_width'
        )
        fig.add_tools(bokeh.models.HoverTool(
            tooltips=[
                ('Frequency', f'@frequency{{0.0000}} {FREQUENCY_PLOT_UNIT}'),
                ('Channel', '$index'),
                ('Peak', f'@peak{{%.3f}}{PIXEL_PLOT_UNIT}'),
                ('Noise', f'@noise{{%.3f}}{PIXEL_PLOT_UNIT}'),
                ('Predicted noise (weights)', f'@weights_noise{{%.3f}}{PIXEL_PLOT_UNIT}'),
                ('Predicted noise (model)', f'@model_noise{{%.3f}}{PIXEL_PLOT_UNIT}')
            ],
            formatters={
                '@peak': SI_FORMAT,
                '@noise': SI_FORMAT,
                '@weights_noise': SI_FORMAT,
                '@model_noise': SI_FORMAT
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

    def make_plot_flux_density(self, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        fig = bokeh.plotting.figure(
            x_axis_label=f'Frequency ({FREQUENCY_PLOT_UNIT})',
            y_axis_label=f'Flux density ({FLUX_DENSITY_PLOT_UNIT})',
            x_range=self.frequency_range,
            sizing_mode='stretch_width'
        )
        fig.add_tools(bokeh.models.HoverTool(
            tooltips=[
                ('Frequency', f'@frequency{{0.0000}} {FREQUENCY_PLOT_UNIT}'),
                ('Channel', '$index')
            ] + [
                (pol, f'@total_{pol}{{%.3f}}{FLUX_DENSITY_PLOT_UNIT}')
                for pol in self.totals.keys()
            ],
            formatters={f'@total_{pol}': SI_FORMAT for pol in self.totals.keys()}
        ))
        for i, pol in enumerate(self.totals.keys()):
            self._line_with_markers(
                fig, x='frequency', y=f'total_{pol}', source=source, name=f'total_{pol}',
                line_color=PALETTE[i], legend_label=pol)
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
            'flux_density': self.make_plot_flux_density(channel_source),
            'elevation': self.make_plot_elevation(time_source),
            'parallactic_angle': self.make_plot_parallactic_angle(time_source)
        }

    def make_uv_coverage(self) -> None:
        """Generate SVG showing UV coverage and store it internally.

        This is implemented in matplotlib (generating an SVG) rather than
        Bokeh because Bokeh doesn't support elliptic arcs.
        """
        matplotlib.use('SVG')
        fig = matplotlib.figure.Figure(figsize=(14, 14))
        ax = fig.subplots()
        ax.set_aspect('equal')

        def split_interval(start, end):
            """Split up long intervals into shorter ones.

            This is to avoid any ambiguities about which direction arcs should
            go if there is a track longer than 12 hours.
            """
            if end - start < 11 * 3600:
                yield start, end
            else:
                mid = (start + end) / 2
                yield from split_interval(start, mid)
                yield from split_interval(mid, end)

        for start, end in itertools.chain.from_iterable(
                split_interval(s, e) for s, e in self.time_intervals):
            # Find the Earth's rotation axis in the UVW basis (it's not a
            # constant due to precession etc, but close enough over a single
            # track). We use an offset of an hour rather than end time to
            # avoid numeric stability issues for very short tracks.
            bases = self.target.uvw_basis([start, start + 3600.0], self.array_ant)
            B = bases[..., 1] - bases[..., 0]
            # The axis of rotation will have the same UVW coordinates at both times,
            # and hence is in the null-space of B - so should correspond to a
            # singular value of zero.
            U, S, VH = np.linalg.svd(B)
            assert S[2] <= 1e-4 <= S[1]
            axis = U[:, 2]    # UVW coordinates of rotation axis (could be North or South)
            # Projection of the rotation axis gives direction of semi-minor
            # axis of the ellipse, and the w coordinate determines the ratio
            # between the axis lengths.
            flatten = abs(axis[2])
            angle = np.arctan2(axis[1], axis[0])

            all_uvw = np.stack(self.target.uvw(self.ants, [start, end], self.array_ant), axis=-1)
            for i in range(len(self.ants)):
                for j in range(i + 1, len(self.ants)):
                    uvw0, uvw1 = all_uvw[:, j] - all_uvw[:, i]  # Start and end uvw
                    # Project onto rotation axis to get centre of the ellipse.
                    # A lot of these calculations could be vectorised, but
                    # in practice I think the matplotlib arc rendering will
                    # dominate any potential savings.
                    mid = np.dot(uvw0, axis) * axis
                    a = np.linalg.norm(uvw0 - mid)
                    b = flatten * a
                    angle0 = np.arctan2(uvw0[1] - mid[1], uvw0[0] - mid[0])
                    angle1 = np.arctan2(uvw1[1] - mid[1], uvw1[0] - mid[0])
                    adiff = (angle1 - angle0) % (2 * np.pi)
                    # Ensure we draw the arc the shorter way around. We've
                    # limited the track length so that this should always be
                    # the correct direction.
                    if adiff >= np.pi:
                        angle0, angle1 = angle1, angle0
                    # matplotlib interprets arc endpoints relative to rotated ellipse.
                    angle0 -= angle
                    angle1 -= angle
                    arc = matplotlib.patches.Arc(
                        (mid[0], mid[1]), 2 * b, 2 * a, np.rad2deg(angle),
                        np.rad2deg(angle0),
                        np.rad2deg(angle1))
                    ax.add_patch(arc)
                    # Mirror image
                    arc = matplotlib.patches.Arc(
                        (-mid[0], -mid[1]), 2 * b, 2 * a, np.rad2deg(angle),
                        np.rad2deg(angle0) + 180,
                        np.rad2deg(angle1) + 180)
                    ax.add_patch(arc)
                    # Uncomment to plot dots on top of the tracks to verify.
                    # u, v, w = self.target.uvw([self.ants[i], self.ants[j]],
                    #                           np.arange(start, end, 256.0),
                    #                           self.array_ant)
                    # u = u[:, 1] - u[:, 0]
                    # v = v[:, 1] - v[:, 0]
                    # ax.plot(u, v, 'g.', markersize=1)
                    # ax.plot(-u, -v, 'g.', markersize=1)

        # Adjust data limits to be square and centred
        data_lim = ax.dataLim
        lim = max(-data_lim.x0, data_lim.x1, -data_lim.y0, data_lim.y1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('m')
        ax.set_ylabel('m')
        raw_svg = io.BytesIO()
        fig.savefig(raw_svg, format='svg', bbox_inches='tight')

        # Clean up the SVG for embedding into the HTML.
        raw_svg.seek(0)
        out_svg = io.StringIO()
        mangler = SVGMangler(out_svg, encoding='utf-8', short_empty_elements=True)
        xml.sax.parse(raw_svg, mangler)
        self.uv_coverage = out_svg.getvalue()


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
    for ((target_desc, channel, pol), total) in telstate.get('total', {}).items():
        stats_lookup[target_desc].totals[pol][channel] = total * u.Jy
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

    uv_coverage = [target.make_uv_coverage() for target in target_stats]
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
        'static': static,
        'uv_coverage': uv_coverage
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
