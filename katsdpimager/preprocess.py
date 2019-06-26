# -*- coding: utf-8 -*-

"""Pre-process visibilities into a form that is more suitable for gridding.

The following transformations are done:

- Autocorrelations are removed
- Visibilities with w < 0 are flipped
- UVW coordinates are converted to grid coordinates and quantized
- Weights are multiplied into the visibilities (but are still retained)
- Visibilities are sorted by channel then W slice
- Visibilities are partially sorted by baseline, then time
- Visibilities with the same UVW coordinates (after quantisation) are merged
  ("compressed").

The resulting visibilities are stored in HDF5
(:class:`VisibilityCollectorHDF5`) or numpy arrays
(:class:`VisibilityCollectorMem`). The latter is used mainly for testing.

Partial sorting by baseline is implemented by buffering up a reasonably large
number of visibilities and sorting them. The merging is also partial, since
only adjacent visibilities are candidates for merging.

The core pieces are implemented in C++ for speed (see preprocess.cpp).
"""

import sys
import math
import logging

import h5py
import numpy as np
from astropy import units

from katsdpimager import _preprocess


logger = logging.getLogger(__name__)


def _make_dtype(num_polarizations):
    """Creates a numpy structured dtype to hold a preprocessed visibility with
    associated metadata.

    Parameters
    ----------
    num_polarizations : int
        Number of polarizations in the visibility.
    """
    fields = [
        ('uv', np.int16, (2,)),
        ('sub_uv', np.int16, (2,)),
        ('weights', np.float32, (num_polarizations,)),
        ('vis', np.complex64, (num_polarizations,)),
        ('w_plane', np.int16)
    ]
    return np.dtype(fields)


def _make_fapl(cache_entries, cache_size, w0):
    """Create a File Access Properties List for h5py with a specified number
    of cache entries and cache size. This is based around the internal
    make_fapl function in h5py.
    """

    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    cache_settings = list(fapl.get_cache())
    fapl.set_cache(cache_settings[0], cache_entries, cache_size, w0)
    fapl.set_fclose_degree(h5py.h5f.CLOSE_STRONG)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    return fapl


class VisibilityCollector(_preprocess.VisibilityCollector):
    """Base class that accepts a stream of visibility data and stores it. The
    subclasses provide the storage backends. Multiple channels are supported.

    Parameters
    ----------
    image_parameters : list of :class:`katsdpimager.parameters.ImageParameters`
        The image parameters for each channel. They must all have the same set
        of polarizations.
    grid_parameters : list of :class:`katsdpimager.parameters.GridParameters`
        Gridding parameters, with one entry per entry in `image_parameters`
    buffer_size : int
        Number of visibilities to buffer, prior to compression
    """
    def __init__(self, image_parameters, grid_parameters, buffer_size):
        num_polarizations = len(image_parameters[0].polarizations)
        if len(image_parameters) != len(grid_parameters):
            raise ValueError('Inconsistent lengths of image_parameters and grid_parameters')
        num_channels = len(image_parameters)
        config = np.zeros(num_channels, _preprocess.CHANNEL_CONFIG_DTYPE)
        for i, grid_p in enumerate(grid_parameters):
            config[i]["max_w"] = grid_p.max_w.to(units.m).value
            config[i]["w_slices"] = grid_p.w_slices
            config[i]["w_planes"] = grid_p.w_planes
            config[i]["oversample"] = grid_p.oversample
            config[i]["cell_size"] = image_parameters[i].cell_size.to(units.m).value
        super().__init__(
            num_polarizations, config, self._emit, buffer_size)
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self.store_dtype = _make_dtype(num_polarizations)

    @property
    def num_channels(self):
        return len(self.image_parameters)

    def _emit(self, elements):
        """Write an array of compressed elements with the same channel and w
        slice to the backing store. The caller must provide a non-empty
        array.

        Overrides of this function *must not* hold a reference to `elements`.
        The memory underneath is not reference-counted and is invalid once
        this function returns.
        """
        raise NotImplementedError()

    def add(self, uvw, weights, baselines, vis,
            feed_angle1, feed_angle2, mueller_stokes, mueller_circular):
        """Add a set of visibilities to the collector. Each of the provided
        arrays must have the same size on the N axis.

        Parameters
        ----------
        uvw : Quantity array, float32
            N×3 array of UVW coordinates.
        weights : array, float32
            C×N×Q array of weights, where C is the number of channels and
            Q is the number of input polarizations. Flags must be folded into
            the weights.
        baselines : array, int
            1D array of integer, indicating baseline IDs. The IDs are
            arbitrary and need not be contiguous, and are used only to
            associate visibilities from the same baseline together.
            Negative baseline IDs indicate autocorrelations, which will
            be discarded.
        vis : array, complex64
            C×N×Q array of visibilities.
        feed_angle1, feed_angle2 : array, float32
            Feed angles in radians for the two antennas in the baseline,
            for rotating from feed-relative to celestial frame.
        mueller_stokes : matrix
            Converts from circular polarization to output Stokes parameters (4×Q)
        mueller_circular : matrix
            Converts from input polarizations to circular frame (P×4, where P
            is the number of output polarizations)
        """

        # Ensure metres
        uvw = units.Quantity(uvw, unit=units.m, copy=False)
        uvw = np.require(uvw.value, np.float32, 'C')
        super().add(
            uvw, weights, baselines, vis, feed_angle1, feed_angle2,
            mueller_stokes, mueller_circular)

    def reader(self):
        """Create and return a reader object that can be used to iterate over
        visibility data. This may only be called *after* :meth:`close`.
        """
        raise NotImplementedError()


def _is_prime(n):
    for i in range(2, int(math.sqrt(n) + 1)):
        if n % i == 0:
            return False
    return True


class VisibilityCollectorHDF5(VisibilityCollector):
    """Visibility collector that stores data in an HDF5 file. All the
    visibilities are stored in one dataset, with axes for channel and W-slice.
    The visibilities for one channel and W-slice are strung along the third
    axis. This creates a ragged array, which HDF5 does not directly support
    (it supports variable-length arrays, but as far as I can tell each VL
    array is stored and retrieved as a unit, rather than chunked). However,
    chunked storage allocates chunks only as needed, so we store a regular
    array and use an attribute to indicate the length of each row.

    An alternative is to use a separate dataset for each channel and W-slice.
    This will work, but (again, as far as I can tell), libhdf5 uses a separate
    cache for each dataset, making it difficult to bound the total memory
    usage in this scenario.

    Chunks are shaped so that each chunk only contains data for one channel
    and W-slice, to allow this data to be efficiently read back.

    Parameters
    ----------
    filename : str
        Filename for HDF5 file to write
    max_cache_size : int, optional
        Maximum bytes to use for the HDF5 chunk cache. The default is
        unbounded. The actual size will not be larger than one chunk per
        channel/w-slice pair.
    chunk_elements : int, optional
        Number of visibilities per chunk. The default corresponds to about
        1MB per chunk for full-Stokes. Reducing this will save memory, but may
        reduce performance (particularly for spinning drives with high
        latency).
    args,kwargs
        Passed to base class constructor
    """

    def __init__(self, filename, *args, **kwargs):
        max_cache_size = kwargs.pop('max_cache_size', None)
        chunk_elements = kwargs.pop('chunk_elements', 16384)
        super().__init__(*args, **kwargs)
        chunk_size = chunk_elements * self.store_dtype.itemsize
        # We will be jumping between channels and W slices, so to avoid
        # evicting a chunk and then reloading it, we ideally have a big
        # enough cache to hold one chunk for each channel+w-slice.
        cache_chunks = sum(grid_p.w_slices for grid_p in self.grid_parameters)
        max_w_slices = max(grid_p.w_slices for grid_p in self.grid_parameters)
        cache_size = chunk_size * cache_chunks
        if max_cache_size is not None and cache_size > max_cache_size:
            cache_size = max_cache_size
            cache_chunks = cache_size // chunk_size
        # HDF5 recommendation for max performance is 100 slots per chunk, and
        # a prime number
        slots = cache_chunks * 100 + 1
        while not _is_prime(slots):
            slots += 2
        logger.debug('Setting cache size to %d slots, %d bytes', slots, cache_size)
        if isinstance(filename, str):
            filename = filename.encode(sys.getfilesystemencoding())
        self.filename = filename
        self._file = h5py.File(h5py.h5f.create(filename, fapl=_make_fapl(slots, cache_size, 1.0)))
        self._length = np.zeros((self.num_channels, max_w_slices), np.int64)
        self._dataset = self._file.create_dataset(
            "vis", (self.num_channels, max_w_slices, 0),
            maxshape=(self.num_channels, max_w_slices, None),
            dtype=self.store_dtype,
            compression='gzip',
            chunks=(1, 1, chunk_elements))

    def _emit(self, elements):
        N = elements.shape[0]
        channel = elements[0]['channel']
        w_slice = elements[0]['w_slice']
        old_length = self._length[channel, w_slice]
        self._length[channel, w_slice] += N
        if self._length[channel, w_slice] > self._dataset.shape[2]:
            self._dataset.resize(self._length[channel, w_slice], axis=2)
        # Work around FutureWarning in numpy 1.12 by first creating a view of
        # elements that contains only the fields we want.
        elements = np.ndarray(elements.shape, self.store_dtype, elements, 0, elements.strides)
        # This slightly contorted access is for performance reasons: see
        # https://github.com/h5py/h5py/issues/492
        self._dataset[channel : channel+1,
                      w_slice : w_slice+1,
                      old_length : self._length[channel, w_slice]] = \
            elements.astype(self.store_dtype)[np.newaxis, np.newaxis, :]

    def close(self):
        super().close()
        self._dataset.attrs.create('length', self._length)
        if logger.isEnabledFor(logging.INFO):
            filesize = self._file.id.get_filesize()
            n_compressed = np.sum(self._length)
            expected = self.store_dtype.itemsize * n_compressed
            if expected > 0:
                logger.info("Wrote %d visibilities in %d bytes to %s (%.2f%% compression ratio)",
                            n_compressed, filesize, self._file.filename,
                            100.0 * filesize / expected)
            else:
                logger.info("Wrote %d bytes to %s (no visibilities)",
                            filesize, self._file.filename)
        self._file.close()
        self._file = None

    def reader(self):
        return VisibilityReaderHDF5(self)


class VisibilityCollectorMem(VisibilityCollector):
    """Visibility collector that stores data in memory. Each dataset is stored
    as a list of numpy arrays.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = [
            [[] for w_slice in range(self.grid_parameters[channel].w_slices)]
            for channel in range(self.num_channels)]

    def _emit(self, elements):
        dataset = self.datasets[elements[0]['channel']][elements[0]['w_slice']]
        # Work around FutureWarning in numpy 1.12 by first creating a view of
        # elements that contains only the fields we want.
        elements = np.ndarray(elements.shape, self.store_dtype, elements, 0, elements.strides)
        dataset.append(elements.astype(self.store_dtype))

    def reader(self):
        return VisibilityReaderMem(self)


class VisibilityReader:
    """Abstract base class for visibility readers.

    Parameters
    ----------
    collector : subclass of :class:`VisibilityCollector`
        Closed collector that collected the visibilities.
    """
    def __init__(self, collector):
        pass

    def iter_slice(self, channel, w_slice, block_size=None):
        """A generator that iterates over the visibilities in blocks. Each
        iteration yields an array of some number of visibilities, in the type
        returned by :func:`_make_dtype`.

        If `block_size` is specified, the visibilities will be batched into
        groups of this size (except possibly the final batch, which will be
        shorter). If not specified, batches are of arbitrary length,
        depending on the collector.

        .. warning:: An implementation may recycle a single buffer for this
           purpose. The returned array must be consumed (or copied) before
           requesting the next one.
        """
        raise NotImplementedError()

    def len(self, channel, w_slice):
        """Number of visibilities that will be enumerated by :meth:`iter_slice`"""
        raise NotImplementedError()

    def close(self):
        """Free any resources associated with the reader."""
        pass


class VisibilityReaderHDF5(VisibilityReader):
    def __init__(self, collector):
        super().__init__(collector)
        # We're doing a linear read over the file, so we don't need to increase
        # the cache size.
        # TODO: experiment with setting a tiny cache so that blocks bypass the
        # cache entirely.
        self._file = h5py.File(h5py.h5f.open(collector.filename, flags=h5py.h5f.ACC_RDONLY))
        self._dataset = self._file["vis"]
        self._length = self._dataset.attrs['length']

    def len(self, channel, w_slice):
        return self._length[channel, w_slice]

    def iter_slice(self, channel, w_slice, block_size=None):
        if block_size is None:
            block_size = self._dataset.chunks[2]
        buf3d = np.rec.recarray((1, 1, block_size), self._dataset.dtype)
        buf = buf3d[0, 0, :]
        N = self.len(channel, w_slice)
        for start in range(0, N - block_size + 1, block_size):
            self._dataset.read_direct(
                buf3d,
                np.s_[channel : channel+1, w_slice : w_slice+1, start : start+block_size],
                np.s_[:, :, 0 : block_size])
            yield buf
        last = N % block_size
        if last > 0:
            self._dataset.read_direct(
                buf3d,
                np.s_[channel : channel+1, w_slice : w_slice+1, N - last : N],
                np.s_[:, :, 0 : last])
            yield buf[:last]

    @property
    def num_channels(self):
        return self._length.shape[0]

    def num_w_slices(self, channel):
        return self._length.shape[1]

    def close(self):
        super().close()
        self._file.close()
        self._file = None


class VisibilityReaderMem(VisibilityReader):
    def __init__(self, collector):
        super().__init__(collector)
        self.datasets = collector.datasets

    def _iter_slice_blocked(self, channel, w_slice, block_size):
        dataset = self.datasets[channel][w_slice]
        if not dataset:
            return
        buf = np.rec.recarray((block_size,), dataset[0].dtype)
        buf_pos = 0
        for dset in dataset:
            dset_pos = 0
            while len(dset) - dset_pos > block_size - buf_pos:
                buf[buf_pos:] = dset[dset_pos : dset_pos + (block_size - buf_pos)]
                yield buf
                dset_pos += block_size - buf_pos
                buf_pos = 0
            buf[buf_pos : buf_pos + len(dset) - dset_pos] = dset[dset_pos:]
            buf_pos += len(dset) - dset_pos
        if buf_pos > 0:
            yield buf[:buf_pos]

    def iter_slice(self, channel, w_slice, block_size=None):
        if block_size is None:
            return iter(self.datasets[channel][w_slice])
        else:
            return self._iter_slice_blocked(channel, w_slice, block_size)

    def len(self, channel, w_slice):
        return sum(len(x) for x in self.datasets[channel][w_slice])

    @property
    def num_channels(self):
        return len(self.datasets)

    def num_w_slices(self, channel):
        return len(self.datasets[channel])

    def close(self):
        super().close()
        self.datasets = None
