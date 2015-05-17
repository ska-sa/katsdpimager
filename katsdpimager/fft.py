"""Utilities to wrap scikits.cuda.fft for imaging purposes"""

from __future__ import print_function, division
import numpy as np
import pkg_resources
import scikits.cuda.fft
from katsdpsigproc import accel

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
    ctype : str
        OpenCL/CUDA C name for the type
    """
    def __init__(self, context, dtype, ctype, tuning=None):
        self.context = context
        self.dtype = np.dtype(dtype)
        # TODO: autotune
        self.wgsx = 16
        self.wgsy = 8
        self.program = accel.build(context, "imager_kernels/fftshift.mako",
            {
                'ctype': ctype,
                'wgsx': self.wgsx,
                'wgsy': self.wgsy
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return Fftshift(self, *args, **kwargs)


class Fftshift(accel.Operation):
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
    def __init__(self, command_queue, N, shape, dtype, padded_shape_src, padded_shape_dest, tuning=None):
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


class GridToImageTemplate(object):
    def __init__(self, command_queue, shape, padded_shape_src, padded_shape_dest, dtype):
        if dtype == np.complex64:
            ctype = 'float2'
        elif dtype == np.complex128:
            ctype = 'double2'
        else:
            raise ValueError('Unhandled data type {}'.format(dtype))
        self.shift_template = FftshiftTemplate(command_queue.context, dtype, ctype)
        self.fft_template = FftTemplate(command_queue, 2, shape, dtype,
                                        padded_shape_src, padded_shape_dest)

    def instantiate(self, *args, **kwargs):
        return GridToImage(self, *args, **kwargs)

class GridToImage(accel.OperationSequence):
    def __init__(self, template, allocator=None):
        command_queue = template.fft_template.command_queue
        self.shift_grid = template.shift_template.instantiate(
            command_queue, template.fft_template.shape, allocator)
        self.ifft = template.fft_template.instantiate(FFT_INVERSE, allocator)
        self.shift_image = template.shift_template.instantiate(
            command_queue, template.fft_template.shape, allocator)
        operations = [
            ('shift_grid', self.shift_grid),
            ('ifft', self.ifft),
            ('shift_image', self.shift_image)
        ]
        compounds = {
            'grid': ['shift_grid:data', 'ifft:src'],
            'image': ['ifft:dest', 'shift_image:data']
        }
        super(GridToImage, self).__init__(command_queue, operations, compounds)

class GridToImageHost(object):
    def __init__(self, grid, image):
        self.grid = grid
        self.image = image

    def __call__(self):
        self.image[:] = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(self.grid), axes=(0, 1)).real)
        # Scale factor is to match behaviour of CUFFT, which is unnormalized
        self.image *= self.image.shape[0] * self.image.shape[1]
