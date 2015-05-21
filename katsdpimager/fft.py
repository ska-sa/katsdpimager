"""Utilities to wrap scikits.cuda.fft for imaging purposes"""

from __future__ import print_function, division
import numpy as np
import pkg_resources
import scikits.cuda.fft
from katsdpsigproc import accel
import katsdpimager.types

#: Forward FFT
FFT_FORWARD = 0
#: Inverse FFT, without normalization factor of 1/N
FFT_INVERSE = 1


class _Gpudata(object):
    """Adapter to allow scikits.cuda.fft to work with managed memory
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


class _GpudataWrapper(object):
    """Forwarding wrapper around a :py:class:`katsdpsigproc.accel.SVMArray` or
    :py:class:`katsdpsigproc.accel.DeviceArray` that allows it to be passed
    to scikits.cuda.fft.
    """
    def __init__(self, wrapped):
        self._wrapped = wrapped
        try:
            # Handle DeviceArray case
            self.gpudata = wrapper.buffer.gpudata
        except:
            # SVMArray case
            self.gpudata = _Gpudata(wrapped)

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)


class FftshiftTemplate(object):
    """Operation template for the equivalent of :py:meth:`np.fft.fftshift` on
    the device, in-place. The first two dimensions are shifted, and these
    dimensions must have even size. Because the size is even, this operation
    is its own inverse.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
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
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
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
        super(Fftshift, self).__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel('fftshift')
        if shape[0] % 2 != 0 or shape[1] % 2 != 0:
            raise ValueError('First two dimensions of shape must be even')
        self.slots['data'] = accel.IOSlot(shape, template.dtype)

    def _run(self):
        data = self.buffer('data')
        minor = int(np.product(data.padded_shape[2:]))
        stride = minor * data.padded_shape[1]
        items_x = data.shape[1] // 2 * minor
        items_y = data.shape[0] // 2
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                data.buffer, np.int32(stride),
                np.int32(items_x), np.int32(items_y),
                np.int32(items_y * stride)
            ],
            global_size=(accel.roundup(items_x, self.template.wgsx),
                         accel.roundup(items_y, self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy)
        )


class FftTemplate(object):
    """Operation template for a forward or reverse FFT, complex to complex.
    The transformation is done over the first N dimensions, with the remaining
    dimensions for interleaving multiple arrays to be transformed. Dimensions
    beyond the first N must have consistent padding between the source and
    destination, and it is recommended to have no padding at all since the
    padding arrays are also transformed.

    This template bakes in more information than most (command queue and data
    shapes), which is due to constraints in CUFFT.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue`
        Command queue for the operation
    N : int
        Number of dimensions for the transform
    shape : tuple
        Shape of the input (N or more dimensions)
    dtype : {`np.complex64`, `np.complex128`}
        Data type, for both input and output
    padded_shape_src : tuple
        Padded shape of the input
    padded_shape_dest : tuple
        Padded shape of the output
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, command_queue, N, shape, dtype,
                 padded_shape_src, padded_shape_dest, tuning=None):
        if padded_shape_src[N:] != padded_shape_dest[N:]:
            raise ValueError('Source and destination padding does not match on batch dimensions')
        self.command_queue = command_queue
        self.shape = shape
        self.dtype = dtype
        self.padded_shape_src = padded_shape_src
        self.padded_shape_dest = padded_shape_dest
        batches = int(np.product(padded_shape_src[N:]))
        with command_queue.context:
            self.plan = scikits.cuda.fft.Plan(
                shape[:N], dtype, dtype, batches,
                stream=command_queue._pycuda_stream,
                inembed=np.array(padded_shape_src[:N], np.int32),
                istride=batches,
                idist=1,
                onembed=np.array(padded_shape_dest[:N], np.int32),
                ostride=batches,
                odist=1)

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
        super(Fft, self).__init__(template.command_queue, allocator)
        self.template = template
        src_dims = [accel.Dimension(d[0], min_padded_size=d[1], exact=True)
                    for d in zip(template.shape, template.padded_shape_src)]
        dest_dims = [accel.Dimension(d[0], min_padded_size=d[1], exact=True)
                     for d in zip(template.shape, template.padded_shape_dest)]
        self.slots['src'] = accel.IOSlot(src_dims, template.dtype)
        self.slots['dest'] = accel.IOSlot(dest_dims, template.dtype)
        self.mode = mode

    def _run(self):
        src_buffer = self.buffer('src')
        dest_buffer = self.buffer('dest')
        with self.template.command_queue.context:
            if self.mode == FFT_FORWARD:
                scikits.cuda.fft.fft(_GpudataWrapper(src_buffer), _GpudataWrapper(dest_buffer),
                                     self.template.plan)
            else:
                scikits.cuda.fft.ifft(_GpudataWrapper(src_buffer), _GpudataWrapper(dest_buffer),
                                      self.template.plan)


class ComplexToRealTemplate(object):
    """Extracts just the real part of a complex array. This could probably be
    done more efficiently using rectangle copy operations, but this class will
    form the basis for more complicated transformations in w stacking.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    real_dtype : {`np.float32`, `np.float64`}
        Output type
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, context, real_dtype, tuning=None):
        self.real_dtype = real_dtype
        self.wgs = 256  # TODO: autotuning
        self.program = accel.build(
            context, "imager_kernels/complex_to_real.mako",
            {
                'real_type': katsdpimager.types.dtype_to_ctype(real_dtype),
                'wgs': self.wgs,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return ComplexToReal(self, *args, **kwargs)


class ComplexToReal(accel.Operation):
    """Instantiation of :class:`ComplexToRealTemplate`

    .. rubric:: Slots

    **src** : array of complex values
        Input
    **dest** : array of real values
        Output

    Parameters
    ----------
    template : :class:`ComplexToRealTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    shape : tuple of int
        Shape of the data.
    """
    def __init__(self, template, command_queue, shape, allocator=None):
        super(ComplexToReal, self).__init__(command_queue, allocator)
        self.template = template
        dims = [accel.Dimension(size) for size in shape]
        complex_dtype = katsdpimager.types.real_to_complex(template.real_dtype)
        self.slots['src'] = accel.IOSlot(dims, complex_dtype)
        self.slots['dest'] = accel.IOSlot(dims, template.real_dtype)
        self.kernel = template.program.get_kernel('complex_to_real')

    def _run(self):
        src = self.buffer('src')
        dest = self.buffer('dest')
        elements = int(np.product(src.padded_shape))
        self.command_queue.enqueue_kernel(
            self.kernel,
            [dest.buffer, src.buffer, np.int32(elements)],
            global_size=(accel.roundup(elements, self.template.wgs),),
            local_size=(self.template.wgs,))


class GridToImageTemplate(object):
    """Template for a combined operation that converts from a complex grid to
    a real image. The grid need not be conjugate symmetric: it is put through
    a complex-to-complex transformation, and the real part of the result is
    returned. Both the grid and the image have the DC term in the middle.

    This operation is destructive: the grid is modified in-place.

    Because it uses :class:`FftTemplate`, most of the parameters are baked
    into the template rather than the instance.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue`
        Command queue for the operation
    shape : tuple of int
        Shape for both the source and destination, as (height, width, polarizations)
    padded_shape_src : tuple of int
        Padded shape of the grid
    padded_shape_dest : tuple of int
        Padded shape of the image
    real_dtype : {`np.float32`, `np.float64`}
        Precision
    """

    def __init__(self, command_queue, shape, padded_shape_src, padded_shape_dest, real_dtype):
        complex_dtype = katsdpimager.types.real_to_complex(real_dtype)
        self.shift_real = FftshiftTemplate(command_queue.context, real_dtype)
        self.shift_complex = FftshiftTemplate(command_queue.context, complex_dtype)
        self.fft = FftTemplate(command_queue, 2, shape, complex_dtype,
                               padded_shape_src, padded_shape_dest)
        self.complex_to_real = ComplexToRealTemplate(command_queue.context, real_dtype)

    def instantiate(self, *args, **kwargs):
        return GridToImage(self, *args, **kwargs)


class GridToImage(accel.OperationSequence):
    """Instantiation of :class:`GridToImageTemplate`

    Parameters
    ----------
    template : :class:`GridToImageTemplate`
        Operation template
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, allocator=None):
        command_queue = template.fft.command_queue
        self._shift_grid = template.shift_complex.instantiate(
            command_queue, template.fft.shape, allocator)
        self._ifft = template.fft.instantiate(FFT_INVERSE, allocator)
        self._complex_to_real = template.complex_to_real.instantiate(
            command_queue, template.fft.shape, allocator)
        self._shift_image = template.shift_real.instantiate(
            command_queue, template.fft.shape, allocator)
        operations = [
            ('shift_grid', self._shift_grid),
            ('ifft', self._ifft),
            ('complex_to_real', self._complex_to_real),
            ('shift_image', self._shift_image),
        ]
        compounds = {
            'grid': ['shift_grid:data', 'ifft:src'],
            'layer': ['ifft:dest', 'complex_to_real:src'],
            'image': ['complex_to_real:dest', 'shift_image:data']
        }
        super(GridToImage, self).__init__(command_queue, operations, compounds, allocator=allocator)


class GridToImageHost(object):
    """CPU-only equivalent to :class:`GridToHost`.

    The parameters specify which buffers the operation runs on, but the
    contents at construction time are irrelevant. The operation is
    performed at call time.

    Parameters
    ----------
    grid : ndarray, complex
        Input grid (unmodified)
    layer : ndarray, complex
        Intermediate structure holding the complex FFT
    """

    def __init__(self, grid, layer, image):
        self.grid = grid
        self.layer = layer
        self.image = image

    def __call__(self):
        self.layer[:] = np.fft.ifft2(np.fft.fftshift(self.grid), axes=(0, 1))
        # Scale factor is to match behaviour of CUFFT, which is unnormalized
        scale = self.layer.shape[0] * self.layer.shape[1]
        self.image[:] = np.fft.fftshift(self.layer.real * scale)
