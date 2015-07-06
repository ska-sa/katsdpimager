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

The resulting visibilities are stored in HDF5. A separate dataset is created
for each channel and W slice, because the data needs to be fully sorted by W
slice. HDF5 variable-length arrays are not suitable because the whole
variable-length array has to be loaded as a unit.

Partial sorting by baseline is implemented by buffering up a reasonably large
number of visibilities and sorting them. The merging is also partial, since
only adjacent visibilities are candidates for merging.

There is also a memory-based backend to simplify testing.
"""

from __future__ import print_function, division
import h5py
import numpy as np
import numba
import katsdpimager.polarization as polarization
import katsdpimager.grid as grid
import astropy.units as units


def _make_dtype(num_polarizations, internal):
    """Creates a numpy structured dtype to hold a preprocessed visibility with
    associated metadata.

    Parameters
    ----------
    num_polarizations : int
        Number of polarizations in the visibility.
    internal : bool
        If True, the structure will include extra fields for baseline, w slice
        and channel.
    """
    fields = [
        ('uv', np.int16, (2,)),
        ('sub_uv', np.int16, (2,)),
        ('weights', np.float32, (num_polarizations,)),
        ('vis', np.complex64, (num_polarizations,)),
        ('w_plane', np.int16)
    ]
    if internal:
        fields += [('w_slice', np.int16), ('channel', np.int32), ('baseline', np.int32)]
    return np.dtype(fields)


def _make_fapl(cache_entries, cache_size):
    """Create a File Access Properties List for h5py with a specified number
    of cache entries and cache size. This is based around the internal
    make_fapl function in h5py.
    """

    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    cache_settings = list(fapl.get_cache())
    fapl.set_cache(cache_settings[0], cache_entries, cache_size, cache_settings[3])
    fapl.set_fclose_degree(h5py.h5f.CLOSE_STRONG)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    return fapl


@numba.jit(nopython=True)
def _convert_to_buffer(
    channel, uvw, weights, baselines, vis, out,
    pixels, cell_size, max_w, w_slices, w_planes, kernel_width, oversample):
    """Implementation of :meth:`VisibilityCollector._convert_to_buffer`,
    split out as a function for numba acceleration.
    """
    N = uvw.shape[0]
    P = vis.shape[1]
    offset = np.float32(pixels // 2 - (kernel_width - 1) // 2)
    uv_scale = np.float32(1 / cell_size)
    w_scale = float(w_slices * w_planes / max_w)
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
        w = np.trunc(w * w_scale)
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
        if (last_valid
                and element.channel == last.channel
                and element.w_slice == last.w_slice
                and element.uv[0] == last.uv[0]
                and element.uv[1] == last.uv[1]
                and element.sub_uv[0] == last.sub_uv[0]
                and element.sub_uv[1] == last.sub_uv[1]
                and element.w_plane == last.w_plane):
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


class VisibilityCollector(object):
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
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        num_polarizations = len(image_parameters[0].polarizations)
        self.store_dtype = _make_dtype(num_polarizations, False)
        self.buffer_dtype = _make_dtype(num_polarizations, True)
        self._buffer = np.rec.recarray(buffer_size, self.buffer_dtype)
        #: Number of valid elements in buffer
        self._used = 0
        self.num_input = 0
        self.num_output = 0

    @property
    def num_channels(self):
        return len(self.image_parameters)

    @property
    def num_w_slices(self):
        return self.grid_parameters.w_slices

    def _process_buffer(self):
        """Sort and compress the buffer, and write the results to file.

        The buffer is sorted, then compressed in-place. It is then split into
        regions that have the same channel and w slice, each of which is
        appended to the relevant dataset in the file.
        """
        if self._used == 0:
            return
        buffer = self._buffer[:self._used]
        # mergesort is stable, so will preserve ordering by time
        buffer.sort(kind='mergesort', order=['channel', 'w_slice', 'baseline'])
        N = _compress_buffer(buffer)
        # Write regions to file
        if N > 0:
            key = buffer[:N][['channel', 'w_slice']]
            prev = 0
            for split in np.nonzero(key[1:] != key[:-1])[0]:
                end = split + 1
                self._emit(buffer[prev:end])
                prev = end
            self._emit(buffer[prev:N])
        self.num_input += self._used
        self.num_output += N
        self._used = 0

    def _emit(self, elements):
        """Write an array of compressed elements with the same channel and w
        slice to the backing store. The caller must provide a non-empty
        array."""
        raise NotImplementedError()

    def _convert_to_buffer(self, channel, uvw, weights, baselines, vis, out):
        """Apply element-wise preprocessing to a number of elements and write
        the results to `out`, which will be a view of a range from the buffer.

        Parameters
        ----------
        channel : int
            Channel number for all the provided visibilities
        uvw, weights, baselines, vis : array
            See :meth:`add`
        out : record array
            N-element view of portion of the buffer to write
        """
        _convert_to_buffer(
            channel, uvw.to(units.m).value, weights, baselines, vis, out,
            self.image_parameters[channel].pixels,
            self.image_parameters[channel].cell_size.to(units.m).value,
            self.grid_parameters.max_w.to(units.m).value,
            self.grid_parameters.w_slices,
            self.grid_parameters.w_planes,
            self.grid_parameters.kernel_width,
            self.grid_parameters.oversample)

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
        N = uvw.shape[0]
        if len(uvw.shape) != 2:
            raise ValueError('uvw has wrong number of dimensions')
        if len(weights.shape) != 2:
            raise ValueError('weights has wrong number of dimensions')
        if len(baselines.shape) != 1:
            raise ValueError('baselines has wrong number of dimensions')
        if len(vis.shape) != 2:
            raise ValueError('vis has wrong number of dimensions')
        if weights.shape[0] != N or baselines.shape[0] != N or vis.shape[0] != N:
            raise ValueError('Arrays have different shapes')
        if uvw.shape[1] != 3:
            raise ValueError('uvw has invalid shape')
        ip = self.image_parameters[channel]

        if polarization_matrix is not None:
            weights = polarization.apply_polarization_matrix_weights(weights, polarization_matrix)
            vis = polarization.apply_polarization_matrix(vis, polarization_matrix)
        if weights.shape[1] != len(ip.polarizations):
            raise ValueError('weights has invalid shape')
        if vis.shape[1] != len(ip.polarizations):
            raise ValueError('vis has invalid shape')

        # Copy data to the buffer
        start = 0
        while start < N:
            M = min(N - start, len(self._buffer) - self._used)
            end = start + M
            fill = self._buffer[self._used : self._used + M]
            self._convert_to_buffer(
                channel,
                uvw[start : end],
                weights[start : end],
                baselines[start : end],
                vis[start : end], 
                self._buffer[self._used : self._used + M])
            self._used += M
            start += M
            if self._used == len(self._buffer):
                self._process_buffer()

    def close(self):
        """Finalize processing and close any resources. This must be called before
        :meth:`reader`. Subclasses may overload this method.
        """
        self._process_buffer()

    def reader(self):
        """Create and return a reader object that can be used to iterate over
        visibility data. This may only be called *after* :meth:`close`.
        """
        raise NotImplementedError()


class VisibilityCollectorHDF5(VisibilityCollector):
    """Visibility collector that stores data in an HDF5 file.

    Parameters
    ----------
    filename : str
        Filename for HDF5 file to write
    args,kwargs
        Passed to base class constructor
    """

    def __init__(
            self, filename, *args, **kwargs):
        super(VisibilityCollectorHDF5, self).__init__(*args, **kwargs)
        # TODO: need a more intelligent manner to select cache sizes
        # TODO: investigate adding compression. Should work fairly well on UVW.
        self.filename = filename
        self.file = h5py.File(h5py.h5f.create(filename, fapl=_make_fapl(100003, 128 * 1024**2)))
        self.datasets = []
        for channel in range(self.num_channels):
            group = self.file.create_group("channel_{}".format(channel))
            group_datasets = []
            for w_slice in range(self.num_w_slices):
                # TODO: set chunk size more sensibly
                # TODO: ensure that fill is with undefined rather than zeros
                group_datasets.append(group.create_dataset(
                    "slice_{}".format(w_slice),
                    (0,), maxshape=(None,), dtype=self.store_dtype,
                    chunks=(65536,)))
            self.datasets.append(group_datasets)

    def _emit(self, elements):
        N = elements.shape[0]
        dataset = self.datasets[elements[0].channel][elements[0].w_slice]
        dataset.resize(dataset.shape[0] + N, axis=0)
        dataset[-N:] = elements.astype(self.store_dtype)

    def close(self):
        super(VisibilityCollectorHDF5, self).close()
        self.file.close()
        self.file = None

    def reader(self):
        return VisibilityReaderHDF5(self)


class VisibilityCollectorMem(VisibilityCollector):
    """Visibility collector that stores data in memory. Each dataset is stored
    as a list of numpy arrays.
    """

    def __init__(self, *args, **kwargs):
        super(VisibilityCollectorMem, self).__init__(*args, **kwargs)
        self.datasets = [
            [ [] for w_slice in range(self.num_w_slices) ] for channel in range(self.num_channels)]

    def _emit(self, elements):
        dataset = self.datasets[elements[0].channel][elements[0].w_slice]
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

    def iter_slice(self, channel, w_slice):
        """A generator that iterates over the visibilities in blocks. Each
        iteration yields an array of some number of visibilities, in the type
        returned by :func:`_make_dtype`.

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
        # TODO: think about what to set cache size to
        self.file = h5py.File(h5py.h5f.open(collector.filename, flags=h5py.h5f.ACC_RDONLY, fapl=_make_fapl(100003, 128 * 1024**2)))
        self.num_w_slices = collector.num_w_slices

    def _get_dataset(self, channel, w_slice):
        return self.file['channel_{}'.format(channel)]['slice_{}'.format(w_slice)]

    def iter_slice(self, channel, w_slice):
        dataset = self._get_dataset(channel, w_slice)
        block_size = dataset.chunks[0]
        if block_size is None:
            block_size = 65536
        buf = np.empty((block_size,), dataset.dtype)
        N = dataset.shape[0]
        for start in xrange(0, N - block_size, block_size):
            dataset.read_direct(buf, np.s_[start : start+block_size], np.s_[0 : block_size])
            yield buf
        last = N % block_size
        if last > 0:
            buf = np.empty((last,), dataset.dtype)
            dataset.read_direct(buf, np.s_[N - last : N], np.s_[0 : last])
            yield buf

    def len(self, channel, w_slice):
        return len(self._get_dataset(channel, w_slice))

    def close(self):
        super(VisibilityReaderHDF5, self).close()
        self.file.close()
        self.file = None


class VisibilityReaderMem(VisibilityReader):
    def __init__(self, collector):
        super(VisibilityReaderMem, self).__init__(collector)
        self.datasets = collector.datasets

    def iter_slice(self, channel, w_slice):
        return iter(self.datasets[channel][w_slice])

    def len(self, channel, w_slice):
        return sum(len(x) for x in self.datasets[channel][w_slice])

    @property
    def num_w_slices(self):
        return len(self.datasets[0])

    def close(self):
        super(VisibilityReaderMem, self).close()
        self.datasets = None
