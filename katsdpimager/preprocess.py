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

The resulting visibilities are stored in HDF5 (:class:`VisibilityCollectorHDF5`)
or numpy arrays (:class:`VisibilityCollectorMem`).

Partial sorting by baseline is implemented by buffering up a reasonably large
number of visibilities and sorting them. The merging is also partial, since
only adjacent visibilities are candidates for merging.

There is also a memory-based backend to simplify testing.
"""

from __future__ import print_function, division
import h5py
import numpy as np
from katsdpimager import numba
import math
import logging
from katsdpimager import polarization, grid, types, _preprocess
import astropy.units as units


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


@numba.jit(nopython=True)
def _convert_to_buffer(
        channel, uvw, weights, baselines, vis, out,
        pixels, cell_size, max_w, w_slices, w_planes, oversample):
    """Implementation of :meth:`VisibilityCollector._convert_to_buffer`,
    split out as a function for numba acceleration.
    """
    N = uvw.shape[0]
    P = vis.shape[1]
    offset = np.float32(pixels // 2)
    uv_scale = np.float32(1 / cell_size)
    w_scale = float((w_slices - 0.5) * w_planes / max_w)
    max_slice_plane = w_slices * w_planes - 1
    for row in range(N):
        u = uvw[row, 0]
        v = uvw[row, 1]
        w = uvw[row, 2]
        if w < 0:
            u = -u
            v = -v
            w = -w
            for p in range(P):
                out[row].vis[p] = np.conj(vis[row, p])
        else:
            for p in range(P):
                out[row].vis[p] = vis[row, p]
        for p in range(P):
            out[row].vis[p] *= weights[row, p]
        u = u * uv_scale + offset
        v = v * uv_scale + offset
        # The plane number is biased by half a slice, because the first slice
        # is half-width and centered at w=0.
        w = np.trunc(w * w_scale + w_planes / 2)
        w_slice_plane = int(min(w, max_slice_plane))
        w_slice = w_slice_plane // w_planes
        w_plane = w_slice_plane % w_planes
        u0, sub_u = grid.subpixel_coord(u, oversample)
        v0, sub_v = grid.subpixel_coord(v, oversample)
        out[row].channel = channel
        out[row].uv[0] = u0
        out[row].uv[1] = v0
        out[row].sub_uv[0] = sub_u
        out[row].sub_uv[1] = sub_v
        out[row].w_plane = w_plane
        out[row].w_slice = w_slice
        for p in range(P):
            out[row].weights[p] = weights[row, p]
        out[row].baseline = baselines[row]


@numba.jit(nopython=True)
def _compress_buffer(buffer):
    """Core loop of :meth:`VisibilityCollector._process_buffer, split into a
    free function for numba acceleration.
    """
    out_pos = 0
    last = buffer[0]    # Value irrelevant, but needed for type inference
    last_valid = False
    P = buffer[0].vis.shape[0]
    for element in buffer:
        if element.baseline < 0:
            continue       # Autocorrelation
        # Here uv and sub_uv are converted to lists because == gives
        # elementwise results on arrays.
        key = (element.channel, element.w_slice,
               element.uv[0], element.uv[1],
               element.sub_uv[0], element.sub_uv[1],
               element.w_plane)
        if (last_valid and
                element.channel == last.channel and
                element.w_slice == last.w_slice and
                element.uv[0] == last.uv[0] and
                element.uv[1] == last.uv[1] and
                element.sub_uv[0] == last.sub_uv[0] and
                element.sub_uv[1] == last.sub_uv[1] and
                element.w_plane == last.w_plane):
            for p in range(P):
                last.vis[p] += element.vis[p]
            for p in range(P):
                last.weights[p] += element.weights[p]
        else:
            if last_valid:
                buffer[out_pos] = last
                out_pos += 1
            prev_key = key
            last = element
            last_valid = True
    if last_valid:
        buffer[out_pos] = last
        out_pos += 1
    return out_pos


class VisibilityCollector(_preprocess.VisibilityCollector):
    """Base class that accepts a stream of visibility data and stores it. The
    subclasses provide the storage backends. Multiple channels are supported.

    Parameters
    ----------
    image_parameters : list of :class:`katsdpimager.parameters.ImageParameters`
        The image parameters for each channel. They must all have the same set
        of polarizations.
    grid_parameters : :class:`katsdpimager.parameters.GridParameters`
        Gridding parameters
    buffer_size : int
        Number of visibilities to buffer, prior to compression
    """
    def __init__(self, image_parameters, grid_parameters, buffer_size):
        num_polarizations = len(image_parameters[0].polarizations)
        super(VisibilityCollector, self).__init__(
            num_polarizations,
            grid_parameters.max_w.to(units.m).value,
            grid_parameters.w_slices,
            grid_parameters.w_planes,
            grid_parameters.oversample,
            self._emit, buffer_size)
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self.store_dtype = _make_dtype(num_polarizations)

    @property
    def num_channels(self):
        return len(self.image_parameters)

    @property
    def num_w_slices(self):
        return self.grid_parameters.w_slices

    def _emit(self, elements):
        """Write an array of compressed elements with the same channel and w
        slice to the backing store. The caller must provide a non-empty
        array."""
        raise NotImplementedError()

    def add(self, channel, uvw, weights, baselines, vis, polarization_matrix=None):
        """Add a set of visibilities to the collector. Each of the provided
        arrays must have the same size on the first axis.

        Parameters
        ----------
        channel : int
            A channel ID, which indexes the `image_parameters` array passed to
            the constructor.
        uvw : Quantity array, Quantity
            N×3 array of UVW coordinates.
        weights : array, float32
            N×P array of weights, where P is the number of polarizations.
            Flags must be folded into the weights.
        baselines : array, int
            1D array of integer, indicating baseline IDs. The IDs are
            arbitrary and need not be contiguous, and are used only to
            associate visibilities from the same baseline together.
            Negative baseline IDs indicate autocorrelations, which will
            be discarded.
        vis : array, complex64
            N×P array of visibilities.
        polarization_matrix : matrix, optional
            If specified, the input visibilities are transformed by this
            matrix.
        """

        if polarization_matrix is not None:
            weights = polarization.apply_polarization_matrix_weights(weights, polarization_matrix)
            vis = polarization.apply_polarization_matrix(vis, polarization_matrix)

        ip = self.image_parameters[channel]
        super(VisibilityCollector, self).add(
            channel,
            ip.pixels, ip.cell_size.to(units.m).value,
            uvw, weights, baselines, vis)

    def reader(self):
        """Create and return a reader object that can be used to iterate over
        visibility data. This may only be called *after* :meth:`close`.
        """
        raise NotImplementedError()


def _is_prime(n):
    for i in xrange(2, int(math.sqrt(n) + 1)):
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
        super(VisibilityCollectorHDF5, self).__init__(*args, **kwargs)
        chunk_size = chunk_elements * self.store_dtype.itemsize
        # We will be jumping between channels and W slices, so to avoid
        # evicting a chunk and then reloading it, we ideally have a big
        # enough cache to hold one chunk for each channel+w-slice.
        cache_chunks = self.num_channels * self.num_w_slices
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
        self.filename = filename
        self._file = h5py.File(h5py.h5f.create(filename, fapl=_make_fapl(slots, cache_size, 1.0)))
        self._length = np.zeros((self.num_channels, self.num_w_slices), np.int64)
        self._dataset = self._file.create_dataset(
            "vis", (self.num_channels, self.num_w_slices, 0),
            maxshape=(self.num_channels, self.num_w_slices, None),
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
        # This slightly contorted access is for performance reasons: see
        # https://github.com/h5py/h5py/issues/492
        self._dataset[channel : channel+1, w_slice : w_slice+1, old_length : self._length[channel, w_slice]] = \
            elements.astype(self.store_dtype)[np.newaxis, np.newaxis, :]

    def close(self):
        super(VisibilityCollectorHDF5, self).close()
        self._dataset.attrs.create('length', self._length)
        if logger.isEnabledFor(logging.INFO):
            filesize = self._file.id.get_filesize()
            n_compressed = np.sum(self._length)
            expected = self.store_dtype.itemsize * n_compressed
            if expected > 0:
                logger.info("Wrote %d visibilities in %d bytes to %s (%.2f%% compression ratio)",
                            n_compressed, filesize, self._file.filename, 100.0 * filesize / expected)
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
        super(VisibilityCollectorMem, self).__init__(*args, **kwargs)
        self.datasets = [
            [[] for w_slice in range(self.num_w_slices)] for channel in range(self.num_channels)]

    def _emit(self, elements):
        dataset = self.datasets[elements[0]['channel']][elements[0]['w_slice']]
        dataset.append(elements.astype(self.store_dtype))

    def reader(self):
        return VisibilityReaderMem(self)


class VisibilityReader(object):
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
        super(VisibilityReaderHDF5, self).__init__(collector)
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
        for start in xrange(0, N - block_size + 1, block_size):
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

    @property
    def num_w_slices(self):
        return self._length.shape[1]

    def close(self):
        super(VisibilityReaderHDF5, self).close()
        self._file.close()
        self._file = None


class VisibilityReaderMem(VisibilityReader):
    def __init__(self, collector):
        super(VisibilityReaderMem, self).__init__(collector)
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

    @property
    def num_w_slices(self):
        return len(self.datasets[0])

    def close(self):
        super(VisibilityReaderMem, self).close()
        self.datasets = None
