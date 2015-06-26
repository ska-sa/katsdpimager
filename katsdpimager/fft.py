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
        # CUDA 7.0 CUFFT has a bug where kernels are run in the default
        # stream instead of the requested one, if dimensions are up to
        # 1920.
        self.needs_synchronize_workaround = any(x <= 1920 for x in shape[:N])
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
        context = self.template.command_queue.context
        with context:
            if self.mode == FFT_FORWARD:
                scikits.cuda.fft.fft(_GpudataWrapper(src_buffer), _GpudataWrapper(dest_buffer),
                                     self.template.plan)
            else:
                scikits.cuda.fft.ifft(_GpudataWrapper(src_buffer), _GpudataWrapper(dest_buffer),
                                      self.template.plan)
            if self.template.needs_synchronize_workaround:
                context._pycuda_context.synchronize()


class TaperDivideTemplate(object):
    r"""Extracts the real part of a complex image and scales by a tapering
    function.

    The function is the combination of a separable antialiasing function and
    the 3rd direction cosine (n). Specifically, for input pixel coordinates x,
    y, we calculate

    .. math::


       l(x) &= \text{lm_scale}\cdot x + \text{lm_bias}\\
       m(y) &= \text{lm_scale}\cdot y + \text{lm_bias}\\
       f(x, y) &= \frac{\text{kernel1d}[x] \cdot \text{kernel1d}[y]}{\sqrt{1-l(x)^2-m(y)^2}}

    and divide :math:`f` from the image.

    Further dimensions are supported e.g. for Stokes parameters.

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
        self.real_dtype = np.dtype(real_dtype)
        self.wgs_x = 16  # TODO: autotuning
        self.wgs_y = 16
        self.program = accel.build(
            context, "imager_kernels/taper_divide.mako",
            {
                'real_type': katsdpimager.types.dtype_to_ctype(real_dtype),
                'wgs_x': self.wgs_x,
                'wgs_y': self.wgs_y,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return TaperDivide(self, *args, **kwargs)


class TaperDivide(accel.Operation):
    """Instantiation of :class:`TaperDivideTemplate`

    .. rubric:: Slots

    **src** : array of complex values
        Input
    **dest** : array of real values
        Output
    **kernel1d** : array of real values, 1D
        Antialiasing function

    Parameters
    ----------
    template : :class:`TaperDivideTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    shape : tuple of int
        Shape of the data (must be square)
    lm_scale : float
        Scale factor from pixel coordinates to l/m coordinates
    lm_bias : float
        Bias from scaled pixel coordinates to l/m coordinates
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Raises
    ------
    ValueError
        if the first two elements of shape are not equal and even
    """
    def __init__(self, template, command_queue, shape, lm_scale, lm_bias, allocator=None):
        if len(shape) < 2 or shape[0] != shape[1]:
            raise ValueError('shape must be square, not {}'.format(shape))
        if shape[0] % 2 != 0:
            raise ValueError('image size must be even, not {}'.format(shape[0]))
        super(TaperDivide, self).__init__(command_queue, allocator)
        self.template = template
        complex_dtype = katsdpimager.types.real_to_complex(template.real_dtype)
        dims = [accel.Dimension(x) for x in shape]
        self.slots['src'] = accel.IOSlot(dims, complex_dtype)
        self.slots['dest'] = accel.IOSlot(dims, template.real_dtype)
        self.slots['kernel1d'] = accel.IOSlot((shape[0],), template.real_dtype)
        self.kernel = template.program.get_kernel('taper_divide')
        self.lm_scale = lm_scale
        self.lm_bias = lm_bias

    def _run(self):
        src = self.buffer('src')
        dest = self.buffer('dest')
        assert(src.padded_shape[1] == dest.padded_shape[1])
        half_size = src.shape[0] // 2
        x_stride = int(np.product(src.padded_shape[2:]))
        y_stride = int(np.product(src.padded_shape[1:]))
        kernel1d = self.buffer('kernel1d')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                dest.buffer,
                src.buffer,
                np.int32(x_stride),
                np.int32(y_stride),
                kernel1d.buffer,
                self.template.real_dtype.type(self.lm_scale),
                self.template.real_dtype.type(self.lm_bias),
                np.int32(half_size),
                np.int32(half_size * x_stride),
                np.int32(half_size * y_stride),
                self.template.real_dtype.type(self.lm_scale * half_size)
            ],
            global_size=(x_stride,
                         accel.roundup(half_size, self.template.wgs_x),
                         accel.roundup(half_size, self.template.wgs_y)),
            local_size=(x_stride, self.template.wgs_x, self.template.wgs_y)
        )


class GridToImageTemplate(object):
    """Template for a combined operation that converts from a complex grid to a
    real image, including tapering correction.  The grid need not be conjugate
    symmetric: it is put through a complex-to-complex transformation, and the
    real part of the result is returned. Both the grid and the image have the
    DC term in the middle.

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
        self.shift_complex = FftshiftTemplate(command_queue.context, complex_dtype)
        self.fft = FftTemplate(command_queue, 2, shape, complex_dtype,
                               padded_shape_src, padded_shape_dest)
        self.taper_divide = TaperDivideTemplate(command_queue.context, real_dtype)

    def instantiate(self, *args, **kwargs):
        return GridToImage(self, *args, **kwargs)


class GridToImage(accel.OperationSequence):
    """Instantiation of :class:`GridToImageTemplate`

    Parameters
    ----------
    template : :class:`GridToImageTemplate`
        Operation template
    lm_scale, lm_bias : float
        See :class:`TaperDivide`
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, lm_scale, lm_bias, allocator=None):
        command_queue = template.fft.command_queue
        self._shift_grid = template.shift_complex.instantiate(
            command_queue, template.fft.shape, allocator)
        self._ifft = template.fft.instantiate(FFT_INVERSE, allocator)
        self._taper_divide = template.taper_divide.instantiate(
            command_queue, template.fft.shape, lm_scale, lm_bias, allocator)
        operations = [
            ('shift_grid', self._shift_grid),
            ('ifft', self._ifft),
            ('taper_divide', self._taper_divide)
        ]
        compounds = {
            'grid': ['shift_grid:data', 'ifft:src'],
            'layer': ['ifft:dest', 'taper_divide:src'],
            'image': ['taper_divide:dest'],
            'kernel1d': ['taper_divide:kernel1d']
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
    image : ndarray, real
        Output image
    kernel1d : ndarray, real
        Tapering function
    lm_scale, lm_bias : real
        Linear transformation from pixel coordinates to l/m values
    """

    def __init__(self, grid, layer, image, kernel1d, lm_scale, lm_bias):
        assert image.shape[0] == image.shape[1]
        assert image.shape[0] % 2 == 0
        self.grid = grid
        self.layer = layer
        self.image = image
        self.kernel1d = kernel1d
        self.lm_scale = lm_scale
        self.lm_bias = lm_bias

    def __call__(self):
        self.layer[:] = np.fft.ifft2(np.fft.ifftshift(self.grid), axes=(0, 1))
        # Scale factor is to match behaviour of CUFFT, which is unnormalized
        scale = self.layer.shape[0] * self.layer.shape[1]
        self.image[:] = np.fft.fftshift(self.layer.real * scale)
        lm = np.arange(self.image.shape[0]).astype(self.image.dtype) * self.lm_scale + self.lm_bias
        lm2 = lm * lm
        n = np.sqrt(1 - (lm2[:, np.newaxis] + lm2[np.newaxis, :]))
        self.image *= n[..., np.newaxis]   # newaxis maps to polarization
        self.image /= np.outer(self.kernel1d, self.kernel1d)[..., np.newaxis]
