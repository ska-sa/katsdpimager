"""Data loading backend for katdal.

To avoid confusion, the term "baseline" is used for a pair of antennas, while
"correlation product" is used for a pair of single-pol inputs. katdal works
on correlation products while katsdpimager works on baselines, so some
conversion is needed.
"""

from __future__ import division, print_function, absolute_import
import argparse
import logging
import itertools
import math
import re
import six.moves.cPickle as pickle
import katsdpimager.loader_core
import katdal
import katpoint
import scipy.interpolate
import numpy as np
import astropy.units as units
from . import polarization
from six.moves import range, urllib


_logger = logging.getLogger(__name__)


class CalibrationReadError(RuntimeError):
    """An error occurred in loading calibration values from file"""
    pass


class ComplexInterpolate1D(object):
    """Interpolator that separates magnitude and phase of complex values.

    The phase interpolation is done by first linearly interpolating the
    complex values, then normalising. This is not perfect because the angular
    velocity changes (slower at the ends and faster in the middle), but it
    avoids the loss of amplitude that occurs without normalisation.

    The parameters are the same as for :func:`scipy.interpolate.interp1d`,
    except that fill values other than nan and "extrapolate" should not be
    used.
    """
    def __init__(self, x, y, *args, **kwargs):
        mag = np.abs(y)
        phase = y / mag
        self._mag = scipy.interpolate.interp1d(x, mag, *args, **kwargs)
        self._phase = scipy.interpolate.interp1d(x, phase, *args, **kwargs)

    def __call__(self, x):
        mag = self._mag(x)
        phase = self._phase(x)
        return phase / np.abs(phase) * mag


def _unique(seq):
    """Like np.unique, but operates in pure Python.

    This ensures that tuples and other objects preserve their identity instead
    of being coerced into numpy arrays.

    Parameters
    ----------
    seq : iterable
        Sequence of comparable objects

    Returns
    -------
    unique : list
        Sorted list of the unique objects from seq
    """
    data = sorted(seq)
    return [key for key, group in itertools.groupby(data)]


class LoaderKatdal(katsdpimager.loader_core.LoaderBase):
    def _find_target(self, target):
        """Find and return the target index based on the argument.

        Parameters
        ----------
        target : str or int or NoneType
            If None, autoselect the target. If an integer, treat it as a
            catalogue index. If a string, match it to a name in the
            catalogue.

        Raises
        ------
        ValueError
            If the selected target did not match the catalogue
        """
        if not self._file.catalogue:
            raise ValueError('The file does not contain any targets')
        if target is None:
            # Find first target with 'target' tag. If that fails, first without
            # bpcal or gaincal tag. If that fails too, first target.
            for i, trg in enumerate(self._file.catalogue):
                if 'target' in trg.tags:
                    return i
            for i, trg in enumerate(self._file.catalogue):
                if 'bpcal' not in trg.tags and 'gaincal' not in trg.tags:
                    return i
            return 0
        else:
            try:
                idx = int(target)
            except ValueError:
                for i, trg in enumerate(self._file.catalogue):
                    if target in [trg.name] + trg.aliases:
                        return i
                raise ValueError('Target {} not found in catalogue'.format(target))
            else:
                if idx < 0 or idx >= len(self._file.catalogue):
                    raise ValueError('Target index {} is out of range'.format(idx))
                return idx

    def _load_cal_attribute(self, key):
        """Load a fixed attribute from file.

        If the attribute is presented as a sensor, it is checked to ensure that
        all the values are the same.

        Raises
        ------
        CalibrationReadError
            if there was a problem reading the value from file (sensor does not exist,
            does not unpickle correctly, inconsistent values etc)
        """
        try:
            value = self._file.file['TelescopeState/{}'.format(key)]['value']
            if len(value) == 0:
                raise ValueError('empty sensor')
            value = [pickle.loads(x) for x in value]
        except (NameError, SyntaxError):
            raise
        except Exception as e:
            raise CalibrationReadError('Could not read {}: {}'.format(key, e))
        if not all(np.array_equal(value[0], x) for x in value):
            raise CalibrationReadError('Could not read {}: inconsistent values'.format(key))
        return value[0]

    def _load_cal_antlist(self):
        """Load antenna list used for calibration.

        If the value does not match the antenna list in the katdal dataset,
        a :exc:`CalibrationReadError` is raised. Eventually this could be
        extended to allow for an antenna list that doesn't match by permuting
        the calibration solutions.
        """
        cal_antlist = self._load_cal_attribute('cal_antlist')
        if cal_antlist != [ant.name for ant in self._file.ants]:
            raise CalibrationReadError('cal_antlist does not match katdal antenna list')
        return cal_antlist

    def _load_cal_pol_ordering(self):
        """Load polarization ordering used by calibration solutions.

        Returns
        -------
        dict
            Keys are 'h' and 'v' and values are 0 and 1, in some order
        """
        cal_pol_ordering = self._load_cal_attribute('cal_pol_ordering')
        try:
            cal_pol_ordering = np.array(cal_pol_ordering)
        except (NameError, SyntaxError):
            raise
        except Exception as e:
            raise CalibrationReadError(str(e))
        if cal_pol_ordering.shape != (4, 2):
            raise CalibrationReadError('cal_pol_ordering does not have expected shape')
        if cal_pol_ordering[0, 0] != cal_pol_ordering[0, 1]:
            raise CalibrationReadError('cal_pol_ordering[0] is not consistent')
        if cal_pol_ordering[1, 0] != cal_pol_ordering[1, 1]:
            raise CalibrationReadError('cal_pol_ordering[1] is not consistent')
        order = [cal_pol_ordering[0, 0], cal_pol_ordering[1, 0]]
        if set(order) != set('vh'):
            raise CalibrationReadError('cal_pol_ordering does not contain h and v')
        return {order[0]: 0, order[1]: 1}

    def _load_cal_product(self, key, start_channel=None, stop_channel=None, **kwargs):
        """Loads calibration solutions from a katdal file.

        If an error occurs while loading the data, a warning is printed and the
        return value is ``None``. Any keyword args are passed to
        :func:`scipy.interpolate.interp1d` or `ComplexInterpolate1D`.

        Solutions that contain non-finite values are discarded.

        Parameters
        ----------
        key : str
            Name of the telescope state sensor

        Returns
        -------
        interp : callable
            Interpolation function which accepts timestamps and returns
            interpolated data with shape (time, channel, pol, antenna). If the
            solution is channel-independent, that axis will be present with
            size 1.
        """
        try:
            ds = self._file.file['TelescopeState/' + key]
            timestamps = ds['timestamp']
            values = []
            good_timestamps = []
            for i, ts in enumerate(timestamps):
                solution = pickle.loads(ds['value'][i])
                if solution.ndim == 2:
                    # Insert a channel axis
                    solution = solution[np.newaxis, ...]
                elif solution.ndim == 3 and stop_channel is not None:
                    solution = solution[start_channel:stop_channel, ...]
                else:
                    raise ValueError('wrong number of dimensions')
                if np.all(np.isfinite(solution)):
                    good_timestamps.append(ts)
                    values.append(solution)
            if not good_timestamps:
                raise ValueError('no finite solutions')
            values = np.array(values)
            kind = kwargs.get('kind', 'linear')
            if np.iscomplexobj(values) and kind not in ['zero', 'nearest']:
                interp = ComplexInterpolate1D
            else:
                interp = scipy.interpolate.interp1d
            return interp(
                good_timestamps, values, axis=0, fill_value='extrapolate',
                assume_sorted=True, **kwargs)
        except (NameError, SyntaxError):
            raise
        except Exception as e:
            _logger.warn('Could not load %s: %s', key, e)
            return None

    def __init__(self, filename, options):
        super(LoaderKatdal, self).__init__(filename, options)
        parser = argparse.ArgumentParser(
            prog='katdal options',
            usage='katdal options: [-i subarray=N] [-i spw=M] ...')
        parser.add_argument('--subarray', type=int, default=0,
                            help='Subarray index within file [%(default)s]')
        parser.add_argument('--spw', type=int, default=0,
                            help='Spectral window index within file [%(default)s]')
        parser.add_argument('--target', type=str,
                            help='Target to image (index or name) [auto]')
        parser.add_argument('--ref-ant', type=str, default='',
                            help='Reference antenna for identifying scans [first in file]')
        parser.add_argument('--apply-cal', type=str, default='all',
                            help='Calibration solutions to pre-apply, from K, B, G, or '
                                 '"all" or "none" [%(default)s]')
        args = parser.parse_args(options)
        if args.apply_cal == 'all':
            args.apply_cal = 'KBG'
        elif args.apply_cal == 'none':
            args.apply_cal = ''
        if not re.match('^[KBG]*$', args.apply_cal):
            parser.error('apply-cal must be some combination of K, B, G, or all')
        self._apply_cal = frozenset(args.apply_cal)

        self._file = katdal.open(filename, ref_ant=args.ref_ant)
        if args.subarray < 0 or args.subarray >= len(self._file.subarrays):
            raise ValueError('Subarray {} is out of range', args.subarray)
        if args.spw < 0 or args.spw >= len(self._file.spectral_windows):
            raise ValueError('Spectral window {} is out of range', args.spw)
        target_idx = self._find_target(args.target)
        self._file.select(subarray=args.subarray, spw=args.spw,
                          targets=[target_idx], scans=['track'],
                          corrprods='cross')
        self._target = self._file.catalogue.targets[target_idx]
        _logger.info('Selected target %r', self._target.description)
        if self._target.body_type != 'radec':
            raise ValueError('Target does not have fixed RA/DEC')

        # Identify polarizations present in the file. Note that _unique
        # returns a sorted list.
        pols = _unique(a[-1] + b[-1] for a, b in self._file.corr_products)
        self._polarizations = pols

        # Compute permutation of correlation products to place all
        # polarizations for one baseline together, and identify missing
        # correlation product (so that they can be flagged). Note that if a
        # baseline is totally absent it is completely ignored, rather than
        # marked as missing.
        corr_product_inverse = {
            tuple(corr_product): i for i, corr_product in enumerate(self._file.corr_products)
        }
        # Find baselines represented by corr_products.
        baselines = _unique((a[:-1], b[:-1]) for a, b in self._file.corr_products)
        corr_product_permutation = []
        missing_corr_products = []
        for a, b in baselines:
            for pol in pols:
                corr_product = (a + pol[0], b + pol[1])
                corr_product_permutation.append(corr_product_inverse.get(corr_product, None))
        for i in range(len(corr_product_permutation)):
            if corr_product_permutation[i] is None:
                # Has to have some valid index for advanced indexing
                corr_product_permutation[i] = 0
                missing_corr_products.append(i)
        self._corr_product_permutation = corr_product_permutation
        self._missing_corr_products = missing_corr_products

        # Turn baselines from pairs of names into pairs of antenna indices
        ant_inverse = {antenna.name: i for i, antenna in enumerate(self._file.ants)}
        try:
            self._baselines = [(ant_inverse[a], ant_inverse[b]) for a, b in baselines]
        except KeyError:
            raise ValueError('File does not contain antenna specifications for all antennas')

        # Set up a reference antenna for the array centre. katdal's reference
        # antenna is the first antenna in the file, which is not as useful.
        # This determines the reference frame for UVW coordinates.
        self._ref_ant = katpoint.Antenna('', *self._file.ants[0].ref_position_wgs84)

        # Load frequency-independent calibration solutions
        self._K = None
        self._G = None
        self._cal_pol_ordering = None
        if self._apply_cal != '':
            try:
                self._load_cal_antlist()
                self._cal_pol_ordering = self._load_cal_pol_ordering()
            except CalibrationReadError as e:
                _logger.warn('%s', e)
                _logger.warn('No calibration solutions will be applied')
            else:
                if 'K' in self._apply_cal:
                    try:
                        self._K = self._load_cal_product('cal_product_K', kind='zero')
                    except CalibrationReadError as e:
                        _logger.warn('%s', e)
                if 'G' in self._apply_cal:
                    # TODO: ideally this should interpolate magnitude and phase
                    # separately.
                    try:
                        self._G = self._load_cal_product('cal_product_G', kind='linear')
                    except CalibrationReadError as e:
                        _logger.warn('%s', e)

    @classmethod
    def match(cls, filename):
        if filename.lower().endswith('.h5') or filename.lower().endswith('.rdb'):
            return True
        # katdal also supports URLs with query parameters, in which case the
        # URL string won't end with .rdb.
        try:
            url = urllib.parse.urlsplit(filename)
            return url.path.endswith('.rdb')
        except ValueError:
            # Invalid URL, but could still be valid for another loader
            return False

    def antenna_diameters(self):
        diameters = [ant.diameter for ant in self._file.ants]
        return units.Quantity(diameters, unit=units.m, dtype=np.float32)

    def antenna_positions(self):
        positions = [ant.position_ecef for ant in self._file.ants]
        return units.Quantity(positions, unit=units.m)

    def num_channels(self):
        return self._file.shape[1]

    def frequency(self, channel):
        return self._file.freqs[channel] * units.Hz

    def phase_centre(self):
        ra, dec = self._target.astrometric_radec()
        return units.Quantity([ra, dec], unit=units.rad)

    def polarizations(self):
        out_map = {
            'hh': polarization.STOKES_XX,
            'hv': polarization.STOKES_XY,
            'vh': polarization.STOKES_YX,
            'vv': polarization.STOKES_YY
        }
        return [out_map[pol] for pol in self._polarizations]

    def has_feed_angles(self):
        return True

    def data_iter(self, start_channel, stop_channel, max_chunk_vis=None, max_load_vis=None):
        n_file_times, n_file_chans, n_file_cp = self._file.shape
        assert 0 <= start_channel < stop_channel <= n_file_chans
        n_chans = stop_channel - start_channel
        n_pols = len(self._polarizations)
        n_ants = len(self._file.ants)
        if max_load_vis is None:
            load_times = n_file_times
        else:
            load_times = max(1, max_load_vis // (n_chans * n_file_cp))
        # timestamps is a property, so ensure it's only evaluated once
        timestamps = self._file.timestamps
        antenna_uvw = [None] * n_ants
        baseline_idx = np.arange(len(self._baselines)).astype(np.int32)
        if self._K is not None:
            # K contains delays. This turns them into complex numbers to
            # correct phase, per channel. The phases are negated so that the
            # resulting values can be multiplied rather than divided.
            freqs = self._file.freqs[start_channel:stop_channel]
            delay_to_phase = (-2j * np.pi * freqs)[np.newaxis, :, np.newaxis, np.newaxis]
        if 'B' in self._apply_cal:
            B = self._load_cal_product('cal_product_B', start_channel, stop_channel, kind='zero')
        else:
            B = None
        for start in range(0, n_file_times, load_times):
            end = min(n_file_times, start + load_times)
            # Load a chunk from the lazy indexer, then reindex to order
            # the baselines as desired.
            select = np.s_[start:end, start_channel:stop_channel, :]
            fix = np.s_[:, :, self._corr_product_permutation]
            vis = self._file.vis[select][fix]
            weights = self._file.weights[select][fix]
            flags = self._file.flags[select][fix]
            # Flag missing correlation products
            if self._missing_corr_products:
                flags[:, :, self._missing_corr_products] = True
            # Apply flags to weights
            weights *= np.logical_not(flags)

            # Apply calibration corrections
            if self._K is not None:
                K = np.exp(self._K(timestamps[start:end]) * delay_to_phase)
            if self._G is not None:
                G = self._G(timestamps[start:end])
            if B is not None:
                B_sample = B(timestamps[start:end])
            if self._K is not None or self._G is not None or B is not None:
                idx = 0
                for a, b in self._baselines:
                    for pol in self._polarizations:
                        cpol = (self._cal_pol_ordering[pol[0]], self._cal_pol_ordering[pol[1]])
                        if self._K is not None:
                            vis[:, :, idx] *= K[:, :, cpol[0], a] * K[:, :, cpol[1], b].conj()
                        if B is not None:
                            scale = B_sample[:, :, cpol[0], a] * B_sample[:, :, cpol[1], b].conj()
                            vis[:, :, idx] /= scale
                            # Weight is inverse variance, so scale by squared
                            # magnitude. This ignores the uncertainty in B,
                            # since we don't know it.
                            weights[:, :, idx] *= scale.real**2 + scale.imag**2
                        if self._G is not None:
                            scale = G[:, :, cpol[0], a] * G[:, :, cpol[1], b].conj()
                            # The np.reciprocal ensures that the expensive
                            # division part is done prior to broadcasting over
                            # channels.
                            vis[:, :, idx] *= np.reciprocal(scale)
                            weights[:, :, idx] *= scale.real**2 + scale.imag**2
                        idx += 1

            # Compute per-antenna UVW coordinates and parallactic angles. The
            # tensor product yields arrays of shape 3xN, which we transpose to
            # Nx3.
            basis = self._target.uvw_basis(timestamp=timestamps[start:end], antenna=self._ref_ant)
            for i, antenna in enumerate(self._file.ants):
                enu = np.array(self._ref_ant.baseline_toward(antenna))
                ant_uvw = np.tensordot(basis, enu, ([1], [0]))
                antenna_uvw[i] = units.Quantity(
                    ant_uvw.transpose(), unit=units.m, dtype=np.float32)
            # parangle converts to degree before returning, so we have to
            # convert back to radians.
            antenna_pa = units.Quantity(
                self._file.parangle[start:end, :].transpose(),
                unit=units.deg, dtype=np.float32, copy=False).to(units.rad)
            # We've mapped H to x and V to y, so we need the angle from x to H
            # rather than from x to V.
            antenna_pa -= math.pi / 2 * units.rad
            # Combine these into per-baseline UVW coordinates and feed angles
            uvw = np.empty((end - start, len(self._baselines), 3), np.float32)
            feed_angle1 = np.empty((end - start, len(self._baselines)), np.float32)
            feed_angle2 = np.empty_like(feed_angle1)
            for i, (a, b) in enumerate(self._baselines):
                uvw[:, i, :] = antenna_uvw[b] - antenna_uvw[a]
                feed_angle1[:, i] = antenna_pa[a]
                feed_angle2[:, i] = antenna_pa[b]

            # reshape everything into the target formats
            yield dict(
                uvw=uvw.reshape(-1, 3),
                weights=weights.swapaxes(0, 1).reshape(n_chans, -1, n_pols),
                baselines=np.tile(baseline_idx, end - start),
                vis=vis.swapaxes(0, 1).reshape(n_chans, -1, n_pols),
                feed_angle1=feed_angle1.reshape(-1),
                feed_angle2=feed_angle2.reshape(-1),
                progress=end,
                total=n_file_times)

    def close(self):
        # katdal does not provide a way to close an input file. The best we can
        # do is leave it to the garbage collector to sort out.
        self._file = None
