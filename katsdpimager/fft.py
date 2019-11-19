"""Utilities to wrap skcuda.fft for imaging purposes

.. include:: macros.rst
"""

import numpy as np
import pkg_resources
import skcuda.fft
from katsdpsigproc import accel

import katsdpimager.types

#: Forward FFT
FFT_FORWARD = 0
#: Inverse FFT, without normalization factor of 1/N
FFT_INVERSE = 1


class _Gpudata:
    """Adapter to allow skcuda.fft to work with managed memory
    allocations. Add it as a `gpudata` member on an arbitrary object, to allow
    that object to be passed instead of a :py:class`pycuda.gpuarray.GPUArray`.
    """
    def __init__(self, array):
        # .buffer gives the ndarray created by PyCUDA, and .base the ManagedAllocation
        self._allocation = array.buffer.base

    def __int__(self):
        return self._allocation.get_device_pointer()

    def __eq__(self, other):
        return int(self) == int(other)

    def __ne__(self, other):
        return int(self) != int(other)


class _GpudataWrapper:
    """Forwarding wrapper around a :py:class:`katsdpsigproc.accel.SVMArray` or
    :py:class:`katsdpsigproc.accel.DeviceArray` that allows it to be passed
    to skcuda.fft.
    """
    def __init__(self, wrapped):
        self._wrapped = wrapped
        try:
            # Handle DeviceArray case
            self.gpudata = wrapped.buffer.gpudata
        except AttributeError:
            # SVMArray case
            self.gpudata = _Gpudata(wrapped)

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)


class FftshiftTemplate:
    """Operation template for the equivalent of :py:meth:`np.fft.fftshift` on
    the device, in-place. The last two dimensions are shifted, and these
    dimensions must have even size. Because the size is even, this operation
    is its own inverse.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    dtype : numpy dtype
        Data type being stored
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, context, dtype, tuning=None):
        self.context = context
        self.dtype = np.dtype(dtype)
        # TODO: autotune
        self.wgsx = 16
        self.wgsy = 8
        self.program = accel.build(
            context, "imager_kernels/fftshift.mako",
            {
                'ctype': katsdpimager.types.dtype_to_ctype(dtype),
                'wgsx': self.wgsx,
                'wgsy': self.wgsy
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return Fftshift(self, *args, **kwargs)


class Fftshift(accel.Operation):
    """Instantiation of :py:class:`FftshiftTemplate`.

    .. rubric:: Slots

    **data**
        Input and output array, transformed in-place

    Parameters
    ----------
    template : :class:`FftshiftTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    shape : tuple of int
        Shape of the data.
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Raises
    ------
    ValueError
        if the first two dimensions in `shape` are not even
    """
    def __init__(self, template, command_queue, shape, allocator=None):
        super().__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel('fftshift')
        if shape[-1] % 2 != 0 or shape[-2] % 2 != 0:
            raise ValueError('First two dimensions of shape must be even')
        self.slots['data'] = accel.IOSlot(shape, template.dtype)

    def _run(self):
        data = self.buffer('data')
        items_x = data.shape[-1] // 2
        items_y = data.shape[-2] // 2
        if len(data.shape) == 2:
            items_z = 1
        else:
            items_z = data.shape[0] * int(np.product(data.padded_shape[1:-2]))
        row_stride = data.padded_shape[-1]
        slice_stride = data.padded_shape[-2] * data.padded_shape[-1]
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                data.buffer,
                np.int32(row_stride),
                np.int32(slice_stride),
                np.int32(items_x), np.int32(items_y),
                np.int32(items_y * row_stride)
            ],
            global_size=(accel.roundup(items_x, self.template.wgsx),
                         accel.roundup(items_y, self.template.wgsy),
                         items_z),
            local_size=(self.template.wgsx, self.template.wgsy, 1)
        )


class FftTemplate:
    r"""Operation template for a forward or reverse FFT. The transformation is
    done over the last N dimensions, with the remaining dimensions for batching
    multiple arrays to be transformed. Dimensions before the first N must have
    consistent padding between the source and destination, and it is
    recommended to have no padding at all since the padding arrays are also
    transformed.

    This template bakes in more information than most (command queue and data
    shapes), which is due to constraints in CUFFT.

    The template can specify real->complex, complex->real, or
    complex->complex. In the last case, the same template can be used to
    instantiate forward or inverse transforms. Otherwise, real->complex
    transforms must be forward, and complex->real transforms must be inverse.

    For real<->complex transforms, the final dimension of the padded shape
    need only be :math:`\lfloor\frac{L}{2}\rfloor + 1`, where :math:`L` is the
    last element of `shape`.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue`
        Command queue for the operation
    N : int
        Number of dimensions for the transform
    shape : tuple
        Shape of the input (N or more dimensions)
    dtype_src : {`np.float32, `np.float64`, `np.complex64`, `np.complex128`}
        Data type for input
    dtype_dest : {`np.float32, `np.float64`, `np.complex64`, `np.complex128`}
        Data type for output
    padded_shape_src : tuple
        Padded shape of the input
    padded_shape_dest : tuple
        Padded shape of the output
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, command_queue, N, shape, dtype_src, dtype_dest,
                 padded_shape_src, padded_shape_dest, tuning=None):
        if padded_shape_src[:-N] != padded_shape_dest[:-N]:
            raise ValueError('Source and destination padding does not match on batch dimensions')
        self.command_queue = command_queue
        self.shape = shape
        self.dtype_src = np.dtype(dtype_src)
        self.dtype_dest = np.dtype(dtype_dest)
        self.padded_shape_src = padded_shape_src
        self.padded_shape_dest = padded_shape_dest
        # CUDA 7.0 CUFFT has a bug where kernels are run in the default stream
        # instead of the requested one, if dimensions are up to 1920. There is
        # a patch, but there is no query to detect whether it has been
        # applied.
        self.needs_synchronize_workaround = any(x <= 1920 for x in shape[:N])
        batches = int(np.product(padded_shape_src[:-N]))
        with command_queue.context:
            self.plan = skcuda.fft.Plan(
                shape[-N:], dtype_src, dtype_dest, batches,
                stream=command_queue._pycuda_stream,
                inembed=np.array(padded_shape_src[-N:], np.int32),
                istride=1,
                idist=int(np.product(padded_shape_src[-N:])),
                onembed=np.array(padded_shape_dest[-N:], np.int32),
                ostride=1,
                odist=int(np.product(padded_shape_dest[-N:])))

    def instantiate(self, *args, **kwargs):
        return Fft(self, *args, **kwargs)


class Fft(accel.Operation):
    """Forward or inverse Fourier transformation.

    .. rubric:: Slots

    **src**
        Input data
    **dest**
        Output data

    Parameters
    ----------
    template : :class:`FftTemplate`
        Operation template
    mode : {:data:`FFT_FORWARD`, :data:`FFT_INVERSE`}
        FFT direction
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, mode, allocator=None):
        super().__init__(template.command_queue, allocator)
        self.template = template
        src_shape = list(template.shape)
        dest_shape = list(template.shape)
        if template.dtype_src.kind != 'c':
            if mode != FFT_FORWARD:
                raise ValueError('R2C transform must use FFT_FORWARD')
            dest_shape[-1] = template.shape[-1] // 2 + 1
        if template.dtype_dest.kind != 'c':
            if mode != FFT_INVERSE:
                raise ValueError('C2R transform must use FFT_INVERSE')
            src_shape[-1] = template.shape[-1] // 2 + 1
        src_dims = [accel.Dimension(d[0], min_padded_size=d[1], exact=True)
                    for d in zip(src_shape, template.padded_shape_src)]
        dest_dims = [accel.Dimension(d[0], min_padded_size=d[1], exact=True)
                     for d in zip(dest_shape, template.padded_shape_dest)]
        self.slots['src'] = accel.IOSlot(src_dims, template.dtype_src)
        self.slots['dest'] = accel.IOSlot(dest_dims, template.dtype_dest)
        self.mode = mode

    def _run(self):
        src_buffer = self.buffer('src')
        dest_buffer = self.buffer('dest')
        context = self.template.command_queue.context
        with context:
            if self.mode == FFT_FORWARD:
                skcuda.fft.fft(_GpudataWrapper(src_buffer), _GpudataWrapper(dest_buffer),
                               self.template.plan)
            else:
                skcuda.fft.ifft(_GpudataWrapper(src_buffer), _GpudataWrapper(dest_buffer),
                                self.template.plan)
            if self.template.needs_synchronize_workaround:
                context._pycuda_context.synchronize()
