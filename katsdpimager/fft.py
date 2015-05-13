"""Utilities to wrap scikits.cuda.fft for imaging purposes"""

import numpy as np
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


class IfftTemplate(object):
    """Operation template for an inverse FFT from complex to complex. The
    operation includes an FFT shift (ala :py:meth:`np.fft.fftshift`) on both
    the input and the output.

    This template bakes in more information than most (command queue and data
    shapes), which is due to constraints in CUFFT.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue`
        Command queue for the operation
    shape : 3-tuple
        Shape of the input, as (rows, columns, polarizations)
    padded_shape_in : 3-tuple
        Padded shape of the input
    padded_shape_out : 3-tuple
        Padded shape of the output
    """
    def __init__(self, command_queue, shape, padded_shape_in, padded_shape_out, tuning=None):
        self.command_queue = command_queue
        self.shape = shape
        self.padded_shape_in = padded_shape_in
        self.padded_shape_out = padded_shape_out
        with command_queue.context:
            self.plan = scikits.cuda.fft.Plan(
                shape[:2], np.complex64, np.complex64, shape[2],
                stream=command_queue._pycuda_stream,
                inembed=np.array(padded_shape_in[:2], np.int32),
                istride=padded_shape_in[2],
                idist=1,
                onembed=np.array(padded_shape_out[:2], np.int32),
                ostride=padded_shape_out[2],
                odist=1)

    def instantiate(self, allocator=None):
        return Ifft(self, allocator)


class Ifft(accel.Operation):
    def __init__(self, template, allocator=None):
        super(Ifft, self).__init__(template.command_queue, allocator)
        self.template = template
        in_dims = [accel.Dimension(d[0], min_padded_size=d[1])
                   for d in zip(template.shape, template.padded_shape_in)]
        out_dims = [accel.Dimension(d[0], min_padded_size=d[1])
                    for d in zip(template.shape, template.padded_shape_out)]
        self.slots['in'] = accel.IOSlot(in_dims, np.complex64)
        self.slots['out'] = accel.IOSlot(out_dims, np.complex64)

    def _run(self):
        in_buffer = self.buffer('in')
        out_buffer = self.buffer('out')
        # accel.Dimension doesn't currently have a way to enforce an
        # exact but non-zero amount of padding, so we need to fall back
        # on this check.
        if in_buffer.padded_shape != self.template.padded_shape_in:
            raise ValueError('Input buffer is incorrectly padded for plan')
        if out_buffer.padded_shape != self.template.padded_shape_out:
            raise ValueError('Output buffer is incorrectly padded for plan')
        # TODO: do these shifts with a kernel
        in_buffer[:] = np.fft.fftshift(in_buffer)
        with self.template.command_queue.context:
            scikits.cuda.fft.ifft(_GpudataWrapper(in_buffer), _GpudataWrapper(out_buffer),
                                  self.template.plan)
            self.template.command_queue.finish()
        out_buffer[:] = np.fft.fftshift(out_buffer)
