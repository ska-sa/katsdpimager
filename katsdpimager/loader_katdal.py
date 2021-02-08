"""Data loading backend for katdal.

To avoid confusion, the term "baseline" is used for a pair of antennas, while
"correlation product" is used for a pair of single-pol inputs. katdal works
on correlation products while katsdpimager works on baselines, so some
conversion is needed.
"""

import argparse
import logging
import itertools
import bisect
import math
import urllib

import dask.array as da
import katdal
import katsdpmodels.fetch.requests
import katsdpmodels.rfi_mask
import katsdpmodels.band_mask
import numpy as np
from astropy import units
from astropy.time import Time
import astropy.io.fits

from . import polarization, loader_core, sky_model, arguments
from .profiling import profile, profile_function, profile_generator


_logger = logging.getLogger(__name__)


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


def _timestamp_to_fits(timestamp):
    """Convert :class:`katdal.Timestamp` or UNIX time to FITS header format."""
    return Time(float(timestamp), format='unix').utc.isot


def _compute(*arrays):
    """Like da.compute, but work around https://github.com/dask/dask/issues/7187."""
    outputs = [np.empty(array.shape, array.dtype) for array in arrays]
    da.store(arrays, outputs, lock=False)
    return outputs


class LoaderKatdal(loader_core.LoaderBase):
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
                    if target in [trg.name, trg.description] + trg.aliases:
                        return i
                raise ValueError('Target {} not found in catalogue'.format(target))
            else:
                if idx < 0 or idx >= len(self._file.catalogue):
                    raise ValueError('Target index {} is out of range'.format(idx))
                return idx

    @profile_function()
    def __init__(self, filename, options, start_channel, stop_channel):
        super().__init__(filename, options, start_channel, stop_channel)
        parser = argparse.ArgumentParser(
            prog='katdal options',
            usage='katdal options: [-i subarray=N] [-i spw=M] ...')
        parser.add_argument('--target', type=str,
                            help='Target to image (index or name) [auto]')
        parser.add_argument('--ref-ant', type=str, default='',
                            help='Override reference antenna for identifying scans [array]')
        parser.add_argument('--apply-cal', type=str, default='all',
                            help='Comma-separated calibration solutions to pre-apply [%(default)s]')
        parser.add_argument('--rfi-mask', type=str, default='none',
                            choices=('none', 'fixed', 'config'),
                            help='Use RFI and band mask from the observation to skip channels')
        parser.add_argument('--access-key', type=str, help='S3 access key')
        parser.add_argument('--secret-key', type=str, help='S3 secret key')
        args = parser.parse_args(options, namespace=arguments.SmartNamespace())

        open_args = dict(
            ref_ant=args.ref_ant,
            applycal=args.apply_cal,
            preselect=dict(channels=np.s_[start_channel:stop_channel])
        )
        if (args.access_key is not None) != (args.secret_key is not None):
            raise ValueError('access-key and secret-key must be used together')
        if args.access_key is not None:
            open_args['credentials'] = (args.access_key, args.secret_key)

        with profile('katdal.open'):
            self._file = katdal.open(filename, **open_args)
        # Frequency-dependent private attributes contain data only for
        # start_channel:stop_channel. The public interface adjusts by
        # _start_channel to present channels 0:stop_channel.
        self._start_channel = start_channel
        self._spectral_window = self._file.spectral_windows[0]
        target_idx = self._find_target(args.target)
        with profile('katdal.DataSet.select'):
            self._file.select(targets=[target_idx], scans=['track'],
                              corrprods='cross')
        self._target = self._file.catalogue.targets[target_idx]
        _logger.info('Selected target %r', self._target.description)
        if self._target.body_type != 'radec':
            raise ValueError('Target does not have fixed RA/DEC')
        self._frequencies = self._file.freqs * units.Hz

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
        for a, b in baselines:
            for pol in pols:
                corr_product = (a + pol[0], b + pol[1])
                corr_product_permutation.append(corr_product_inverse.get(corr_product, None))
        missing_corr_products_mask = np.zeros(len(corr_product_permutation), np.bool_)
        for i in range(len(corr_product_permutation)):
            if corr_product_permutation[i] is None:
                # Has to have some valid index for advanced indexing
                corr_product_permutation[i] = 0
                missing_corr_products_mask[i] = 1
        self._corr_product_permutation = corr_product_permutation
        if np.any(missing_corr_products_mask):
            self._missing_corr_products_mask = da.from_array(
                missing_corr_products_mask[np.newaxis, np.newaxis, :],
                chunks=-1
            )
        else:
            self._missing_corr_products_mask = None

        # Turn baselines from pairs of names into pairs of antenna indices
        ant_inverse = {antenna.name: i for i, antenna in enumerate(self._file.ants)}
        try:
            self._baselines = [(ant_inverse[a], ant_inverse[b]) for a, b in baselines]
        except KeyError:
            raise ValueError('File does not contain antenna specifications for all antennas')

        # Set up a reference antenna for the array centre. katdal's reference
        # antenna is the first antenna in the file, which is not as useful.
        # This determines the reference frame for UVW coordinates.
        self._ref_ant = self._file.sensor.get('Antennas/array/antenna')[0]
        corrections = ', '.join(self._file.applycal_products)
        if not corrections:
            corrections = 'none'
        _logger.info('Calibration corrections applied: %s', corrections)

        # Determine RFI and band mask, if any. Using the band mask isn't
        # entirely necessary because those channels will have no un-flagged
        # data and hence will be skipped anyway, but this causes them to appear
        # as "masked" rather than "no data" in the MeerKAT pipeline report.
        # TODO: eventually katdal should do some of this.
        if args.rfi_mask != 'none':
            telstate = self._file.source.telstate
            with katsdpmodels.fetch.requests.TelescopeStateFetcher(telstate) as fetcher:
                rfi_mask_key = telstate.join('model', 'rfi_mask', args.rfi_mask)
                rfi_mask = fetcher.get(rfi_mask_key, katsdpmodels.rfi_mask.RFIMask)
                # Step vis -> baseline-correlation-products -> antenna-channelised_voltage
                bcp_stream = telstate['src_streams'][0]
                acv_stream = telstate.view(bcp_stream, exclusive=True)['src_streams'][0]
                band_mask_key = telstate.join('model', 'band_mask', args.rfi_mask)
                telstate_acv = telstate.view(acv_stream)
                band_mask = fetcher.get(band_mask_key, katsdpmodels.band_mask.BandMask,
                                        telstate=telstate_acv)
                spw = katsdpmodels.band_mask.SpectralWindow(
                    telstate_acv['bandwidth'] * units.Hz,
                    telstate_acv['center_freq'] * units.Hz
                )
                channel_width = spw.bandwidth / telstate_acv['n_chans']
                max_length = rfi_mask.max_baseline_length(self._frequencies, channel_width)
                self._channel_mask = (
                    (max_length > 0) | band_mask.is_masked(spw, self._frequencies, channel_width)
                )
        else:
            self._channel_mask = None

        # Record command-line options for FITS HISTORY
        unparsed = arguments.unparse_args(args, {'access_key', 'secret_key'})
        self._command_line_options = []
        for arg in unparsed:
            self._command_line_options.append('-i')
            self._command_line_options.append(arg[2:])   # Strip "--"

    def command_line_options(self):
        return self._command_line_options

    @classmethod
    def match(cls, filename):
        if filename.lower().endswith('.rdb'):
            return True
        # katdal also supports URLs with query parameters, in which case the
        # URL string won't end with .rdb.
        try:
            url = urllib.parse.urlsplit(filename)
            return url.scheme == 'redis' or url.path.endswith('.rdb')
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
        return self._file.shape[1] + self._start_channel

    def frequency(self, channel):
        return self._frequencies[channel - self._start_channel]

    def band(self):
        return self._spectral_window.band

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

    def weight_scale(self):
        # A katdal complex visibility with weight 1 has an expected squared
        # magnitude of 1, so the real and imaginary parts each have variance
        # 1/2.
        return math.sqrt(0.5)

    def channel_enabled(self, channel):
        return self._channel_mask is None or not self._channel_mask[channel - self._start_channel]

    def _permute(self, array):
        # Ensure only a single chunk on the baseline axis, so that we can do
        # the permutation on a chunk-by-chunk basis. This is more efficient
        # than using dask to do the permutation.
        #
        # Currently katdal already does this step, but the code is left here
        # in case that changes in future.
        if array.numblocks[2] != 1:
            array = da.rechunk(array, chunks={2: -1})     # pragma: nocover
        index = np.s_[:, :, self._corr_product_permutation]
        return da.map_blocks(
            lambda block: block[index],
            array,
            chunks=(array.chunks[0], array.chunks[1], (len(self._corr_product_permutation),)),
            dtype=array.dtype)

    @profile_generator(labels=('start_channel', 'stop_channel'))
    def data_iter(self, start_channel, stop_channel, max_chunk_vis=None):
        start_channel -= self._start_channel
        stop_channel -= self._start_channel
        self._file.select(reset='F')
        n_file_times, n_file_chans, n_file_cp = self._file.shape
        self._file.select(channels=np.s_[start_channel : stop_channel])
        assert 0 <= start_channel < stop_channel <= n_file_chans
        n_chans = stop_channel - start_channel
        n_pols = len(self._polarizations)
        if max_chunk_vis is None:
            load_times = n_file_times
        else:
            load_times = max(1, max_chunk_vis // (n_chans * n_file_cp))
        # timestamps is a property, so ensure it's only evaluated once
        with profile('katdal.DataSet.timestamps'):
            timestamps = self._file.timestamps
        dask_vis = self._file.vis.dataset
        dask_weights = self._file.weights.dataset
        dask_flags = self._file.flags.dataset

        # Determine chunking scheme
        chunk_sizes = dask_vis.chunks[0]
        chunk_boundaries = [0] + list(itertools.accumulate(chunk_sizes))
        if chunk_sizes and chunk_sizes[0] > load_times:
            _logger.warning('Chunk size is %d dumps but only %d loaded at a time. '
                            'Consider increasing --vis-load',
                            chunk_sizes[0], load_times)

        # Do some modifications:
        # - reorder baselines to the desired order
        # - flag baselines that are missing in the original data
        # - apply the channel mask
        # - apply flags to weights
        dask_vis = self._permute(dask_vis)
        dask_weights = self._permute(dask_weights)
        dask_flags = self._permute(dask_flags)
        if self._channel_mask is not None:
            channel_mask = da.from_array(
                self._channel_mask[np.newaxis, start_channel:stop_channel, np.newaxis],
                chunks=((1,), dask_flags.chunks[1], (1,))
            )
            dask_flags |= channel_mask
        if self._missing_corr_products_mask is not None:
            dask_flags |= self._missing_corr_products_mask
        dask_weights *= np.logical_not(dask_flags)

        start = 0
        while start < n_file_times:
            end = min(n_file_times, start + load_times)
            # Align to chunking if possible
            aligned_end = chunk_boundaries[bisect.bisect(chunk_boundaries, end) - 1]
            if aligned_end > start:
                end = aligned_end
            # Load a chunk from dask
            _logger.debug('Loading dumps %d:%d', start, end)
            select = np.s_[start:end, :, :]
            with profile('dask.array.store',
                         {'start_dump': start, 'end_dump': end}):
                vis, weights, flags = _compute(
                    dask_vis[select], dask_weights[select], dask_flags[select])
            _logger.debug('Dumps %d:%d loaded', start, end)

            # Compute per-antenna UVW coordinates and parallactic angles.
            with profile('katpoint.Target.uvw'):
                antenna_uvw = np.asarray(self._target.uvw(
                    self._file.ants, timestamp=timestamps[start:end], antenna=self._ref_ant))
            antenna_uvw = antenna_uvw.T   # Switch from (uvw, time, ant) to (ant, time, uvw)
            # parangle converts to degrees before returning, so we have to
            # convert back to radians.
            with profile('katdal.DataSet.parangle'):
                antenna_pa = units.Quantity(
                    self._file.parangle[start:end, :].transpose(),
                    unit=units.deg, dtype=np.float32, copy=False).to(units.rad)
            # We've mapped H to x and V to y, so we need the angle from x to H
            # rather than from x to V.
            antenna_pa -= math.pi / 2 * units.rad
            # Combine these into per-baseline UVW coordinates and feed angles
            uvw = np.empty((len(self._baselines), end - start, 3), np.float32)
            feed_angle1 = np.empty((len(self._baselines), end - start), np.float32)
            feed_angle2 = np.empty_like(feed_angle1)
            for i, (a, b) in enumerate(self._baselines):
                uvw[i, :, :] = antenna_uvw[b] - antenna_uvw[a]
                feed_angle1[i, :] = antenna_pa[a]
                feed_angle2[i, :] = antenna_pa[b]

            def reorder(data):
                # Split baseline axis into an antenna baseline and polarization. This
                # shouldn't require any copying.
                data = data.reshape(data.shape[0], data.shape[1], len(self._baselines), n_pols)
                # Move time axis after the baseline axis, so that we follow tracks.
                data = np.moveaxis(data, 0, 2)
                # Flatten time and baseline axes together (which will require copying)
                return data.reshape(data.shape[0], -1, n_pols)

            # reshape everything into the target formats
            yield dict(
                uvw=uvw.reshape(-1, 3) << units.m,
                weights=reorder(weights),
                vis=reorder(vis),
                feed_angle1=feed_angle1.reshape(-1),
                feed_angle2=feed_angle2.reshape(-1),
                progress=end,
                total=n_file_times)
            start = end

    def sky_model(self):
        source = self._file.source
        return sky_model.KatpointSkyModel(sky_model.catalogue_from_telstate(
            source.telstate, source.capture_block_id, None, self._target))

    def extra_fits_headers(self):
        headers = astropy.io.fits.Header()
        timestamps = self._file.timestamps
        if not len(timestamps):
            avg = self._file.start_time.secs
        else:
            avg = np.mean(timestamps)

        headers['OBJECT'] = self._target.name
        headers['SPECSYS'] = 'TOPOCENT'
        # SSYSOBS is not needed because it defaults to TOPOCENT
        headers['DATE-OBS'] = _timestamp_to_fits(self._file.start_time)
        headers['DATE-AVG'] = _timestamp_to_fits(avg)
        headers['ONTIME'] = (len(timestamps) * self._file.dump_period,
                             '[s] Time tracking the target')
        if self._file.observer:
            headers['OBSERVER'] = self._file.observer
        if self._spectral_window.product:
            headers['INSTRUME'] = self._spectral_window.product

        # Exception checks below are to handle pre-MVF v4 files
        # (if they become supported in future), post-MVF v4 which maybe
        # don't define the necessary attributes, or broken files. They
        # are not currently expected to be reachable.
        try:
            array_ant = self._file.sensor['Antennas/array/antenna'][0]
            array_pos = array_ant.position_ecef
            headers['OBSGEO-X'] = array_pos[0]
            headers['OBSGEO-Y'] = array_pos[1]
            headers['OBSGEO-Z'] = array_pos[2]
        except (KeyError, IndexError):    # pragma: nocover
            pass

        try:
            headers['HISTORY'] = f'Capture block id: {self._file.source.capture_block_id}'
        except AttributeError:            # pragma: nocover
            pass
        try:
            headers['HISTORY'] = f'Stream name: {self._file.source.stream_name}'
        except AttributeError:            # pragma: nocover
            pass

        return headers

    @property
    def raw_data(self):
        return self._file

    def close(self):
        # katdal does not provide a way to close an input file. The best we can
        # do is leave it to the garbage collector to sort out.
        self._file = None

    @property
    def raw_target(self):
        """Target as a katpoint object"""
        return self._target
