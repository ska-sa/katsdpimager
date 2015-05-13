"""Utilities to wrap scikits.cuda.fft for imaging purposes"""

from __future__ import print_function, division
import numpy as np
import pkg_resources
import scikits.cuda.fft
from katsdpsigproc import accel

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
    def __init__(self, wrapped):
        self._wrapped = wrapped
        if not hasattr(wrapped, 'gpudata'):
            self.gpudata = _Gpudata(wrapped)

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)


class FftshiftTemplate(object):
    """Operation template for the equivalent of :py:meth:`np.fft.fftshift` on
    the device, in-place. The first two dimensions are shifted, and these
    dimensions must have even size.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    dtype : numpy dtype
        Data type being stored
    alignment : int, optional
        If specified, indicates that all values will move by a multiple of this
        many bytes, which must be a power of two. The default is the largest
        power of two dividing the itemsize of `dtype`.
    """
    def __init__(self, context, dtype, alignment=None, tuning=None):
        self.context = context
        self.dtype = np.dtype(dtype)
        if alignment is None:
            alignment = 1
            while self.dtype.itemsize % (alignment * 2) == 0:
                alignment *= 2
        if alignment <= 0 or (alignment & (alignment - 1)):
            raise ValueError('alignment is not a power of 2')
        if alignment == 1:
            ctype = 'char'
        elif alignment == 2:
            ctype = 'short'
        elif alignment == 4:
            ctype = 'float'
        elif alignment == 8:
            ctype = 'float2'
        else:
            ctype = 'float4'
            alignment = 16
        self.alignment = alignment
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
        bytes_item = self.template.dtype.itemsize
        bytes_x = data.shape[1] // 2 * int(np.product(data.padded_shape[2:])) * bytes_item
        if bytes_x % self.template.alignment != 0:
            raise ValueError('Data array is not aligned as promised')
        items_x = bytes_x // self.template.alignment
        items_y = data.shape[0] // 2
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                data.buffer, np.int32(data.padded_shape[1]),
                np.int32(items_x), np.int32(items_y),
                np.int32(items_y * data.padded_shape[1])
            ],
            global_size=(accel.roundup(items_x, self.template.wgsx),
                         accel.roundup(items_y, self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy)
        )

class IfftTemplate(object):
    """Operation template for an inverse FFT from complex to complex.

    This template bakes in more information than most (command queue and data
    shapes), which is due to constraints in CUFFT.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue`
        Command queue for the operation
    shape : 3-tuple
        Shape of the input, as (rows, columns, polarizations)
    padded_shape_src : 3-tuple
        Padded shape of the input
    padded_shape_dest : 3-tuple
        Padded shape of the output
    """
    def __init__(self, command_queue, shape, padded_shape_src, padded_shape_dest, tuning=None):
        self.command_queue = command_queue
        self.shape = shape
        self.padded_shape_src = padded_shape_src
        self.padded_shape_dest = padded_shape_dest
        with command_queue.context:
            self.plan = scikits.cuda.fft.Plan(
                shape[:2], np.complex64, np.complex64, shape[2],
                stream=command_queue._pycuda_stream,
                inembed=np.array(padded_shape_src[:2], np.int32),
                istride=padded_shape_src[2],
                idist=1,
                onembed=np.array(padded_shape_dest[:2], np.int32),
                ostride=padded_shape_dest[2],
                odist=1)

    def instantiate(self, allocator=None):
        return Ifft(self, allocator)


class Ifft(accel.Operation):
    def __init__(self, template, allocator=None):
        super(Ifft, self).__init__(template.command_queue, allocator)
        self.template = template
        src_dims = [accel.Dimension(d[0], min_padded_size=d[1])
                    for d in zip(template.shape, template.padded_shape_src)]
        dest_dims = [accel.Dimension(d[0], min_padded_size=d[1])
                     for d in zip(template.shape, template.padded_shape_dest)]
        self.slots['src'] = accel.IOSlot(src_dims, np.complex64)
        self.slots['dest'] = accel.IOSlot(dest_dims, np.complex64)

    def _run(self):
        src_buffer = self.buffer('src')
        dest_buffer = self.buffer('dest')
        # accel.Dimension doesn't currently have a way to enforce an
        # exact but non-zero amount of padding, so we need to fall back
        # on this check.
        if src_buffer.padded_shape != self.template.padded_shape_src:
            raise ValueError('Source buffer is incorrectly padded for plan')
        if dest_buffer.padded_shape != self.template.padded_shape_dest:
            raise ValueError('Output buffer is incorrectly padded for plan')
        with self.template.command_queue.context:
            scikits.cuda.fft.ifft(_GpudataWrapper(src_buffer), _GpudataWrapper(dest_buffer),
                                  self.template.plan)


class GridToImageTemplate(object):
    def __init__(self, command_queue, shape, padded_shape_src, padded_shape_dest):
        self.shift_template = FftshiftTemplate(command_queue.context, np.complex64)
        self.ifft_template = IfftTemplate(command_queue, shape, padded_shape_src, padded_shape_dest)

    def instantiate(self, *args, **kwargs):
        return GridToImage(self, *args, **kwargs)

class GridToImage(accel.OperationSequence):
    def __init__(self, template, allocator=None):
        command_queue = template.ifft_template.command_queue
        self.shift_grid = template.shift_template.instantiate(
            command_queue, template.ifft_template.shape, allocator)
        self.ifft = template.ifft_template.instantiate(allocator)
        self.shift_image = template.shift_template.instantiate(
            command_queue, template.ifft_template.shape, allocator)
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
