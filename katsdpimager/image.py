"""Kernels for image-domain processing, and conversion between visibility and
image planes."""

from __future__ import print_function, division
import numpy as np
import math
import pkg_resources
from katsdpsigproc import accel
from . import fft
import katsdpimager.types


class _LayerImageTemplate(object):
    r"""Base class for :class:`LayerToImageTemplate` and :class:`ImageToLayerTemplate`.

    These operations convert between a "layer" (the raw Fourier Transform from
    the UV plane) and the "image". :class:`LayerToImageTemplate` converts from
    layer to image (accumulating into the stacked image), and
    :class:`ImageToLayerTemplate` is the inverse. :class:`LayerToImageTemplate`
    applies the following operations:

    - It reorders the elements to put the image centre at the centre (in the layer,
      it is in the corners). This is the equivalent of `np.fft.fftshift`.

    - It divides out an image tapering function. The function is the
      combination of a separable antialiasing function and the 3rd direction
      cosine (n). Specifically, for input pixel coordinates x, y, we
      calculate

      .. math::

         l(x) &= \text{lm_scale}\cdot x + \text{lm_bias}\\
         m(y) &= \text{lm_scale}\cdot y + \text{lm_bias}\\
         f(x, y) &= \frac{\text{kernel1d}[x] \cdot \text{kernel1d}[y]}{\sqrt{1-l(x)^2-m(y)^2}}

    - It multiplies by a W correction term, namely

      .. math::

         e^{2\pi i w\left(\sqrt{1-l(x)^2-m(y)^2} - 1\right)}

         Only the real component of the result is kept in the image.

    Further dimensions are supported e.g. for Stokes parameters, which occur
    before the l/m dimensions.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    real_dtype : {`np.float32`, `np.float64`}
        Image type
    kernel_filename : str
        Mako template containing the kernel
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, context, real_dtype, kernel_filename, tuning=None):
        self.real_dtype = np.dtype(real_dtype)
        self.wgs_x = 16  # TODO: autotuning
        self.wgs_y = 16
        self.program = accel.build(
            context, kernel_filename,
            {
                'real_type': katsdpimager.types.dtype_to_ctype(real_dtype),
                'wgs_x': self.wgs_x,
                'wgs_y': self.wgs_y,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])


class _LayerImage(accel.Operation):
    """Base class for instantiation of subclasses of :class:`_LayerImageTemplate`

    .. rubric:: Slots

    **layer** : array of complex values
        Input (for divide) / output (for multiply)
    **image** : array of real values
        Output (for divide) / input (for multiply)
    **kernel1d** : array of real values, 1D
        Antialiasing function

    Parameters
    ----------
    template : :class:`_LayerImageTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    shape : tuple of int
        Shape of the data (must be square)
    lm_scale : float
        Scale factor from pixel coordinates to l/m coordinates
    lm_bias : float
        Bias from scaled pixel coordinates to l/m coordinates
    kernel_name : str
        Name of the kernel function
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Raises
    ------
    ValueError
        if the last two elements of shape are not equal and even
    """
    def __init__(self, template, command_queue, shape, lm_scale, lm_bias, kernel_name, allocator=None):
        if len(shape) < 2 or shape[-1] != shape[-2]:
            raise ValueError('shape must be square, not {}'.format(shape))
        if shape[-1] % 2 != 0:
            raise ValueError('image size must be even, not {}'.format(shape[-1]))
        super(_LayerImage, self).__init__(command_queue, allocator)
        self.template = template
        complex_dtype = katsdpimager.types.real_to_complex(template.real_dtype)
        dims = [accel.Dimension(x) for x in shape]
        self.slots['layer'] = accel.IOSlot(dims, complex_dtype)
        self.slots['image'] = accel.IOSlot(dims, template.real_dtype)
        self.slots['kernel1d'] = accel.IOSlot((shape[-1],), template.real_dtype)
        self.kernel = template.program.get_kernel(kernel_name)
        self.lm_scale = lm_scale
        self.lm_bias = lm_bias
        self.w = 0

    def set_w(self, w):
        """Set w (in wavelengths)."""
        self.w = w

    def _run(self):
        layer = self.buffer('layer')
        image = self.buffer('image')
        assert(layer.padded_shape == image.padded_shape)
        half_size = layer.shape[-1] // 2
        row_stride = layer.padded_shape[-1]
        slice_stride = row_stride * layer.padded_shape[-2]
        batches = int(np.product(layer.padded_shape[:-2]))
        kernel1d = self.buffer('kernel1d')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                image.buffer,
                layer.buffer,
                np.int32(row_stride),
                np.int32(slice_stride),
                kernel1d.buffer,
                self.template.real_dtype.type(self.lm_scale),
                self.template.real_dtype.type(self.lm_bias),
                np.int32(half_size),
                np.int32(half_size * row_stride),
                self.template.real_dtype.type(half_size * self.lm_scale),
                self.template.real_dtype.type(2 * self.w)
            ],
            global_size=(accel.roundup(half_size, self.template.wgs_x),
                         accel.roundup(half_size, self.template.wgs_y),
                         batches),
            local_size=(self.template.wgs_x, self.template.wgs_y, 1)
        )


class LayerToImageTemplate(_LayerImageTemplate):
    """Convert layer to image. See :class:`_LayerImageTemplate` for details.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    real_dtype : {`np.float32`, `np.float64`}
        Image type
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, context, real_dtype, tuning=None):
        super(LayerToImageTemplate, self).__init__(
            context, real_dtype, 'imager_kernels/layer_to_image.mako', tuning)

    def instantiate(self, *args, **kwargs):
        return LayerToImage(self, *args, **kwargs)


class LayerToImage(_LayerImage):
    """Instantiation of :class:`LayerToImageTemplate`. See :class:`_LayerImage` for
    details.

    Parameters
    ----------
    template : :class:`LayerToImageTemplate`
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
        if the last two elements of shape are not equal and even
    """
    def __init__(self, template, command_queue, shape, lm_scale, lm_bias, allocator=None):
        super(LayerToImage, self).__init__(
            template, command_queue, shape, lm_scale, lm_bias, "layer_to_image", allocator)


class ImageToLayerTemplate(_LayerImageTemplate):
    """Convert image to layer. See :class:`_LayerImageTemplate` for details.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    real_dtype : {`np.float32`, `np.float64`}
        Image type
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, context, real_dtype, tuning=None):
        super(ImageToLayerTemplate, self).__init__(
            context, real_dtype, 'imager_kernels/image_to_layer.mako', tuning)

    def instantiate(self, *args, **kwargs):
        return ImageToLayer(self, *args, **kwargs)


class ImageToLayer(_LayerImage):
    """Instantiation of :class:`ImageToLayerTemplate`. See :class:`_LayerImage` for
    details.

    Parameters
    ----------
    template : :class:`ImageToLayerTemplate`
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
        if the last two elements of shape are not equal and even
    """
    def __init__(self, template, command_queue, shape, lm_scale, lm_bias, allocator=None):
        super(ImageToLayer, self).__init__(
            template, command_queue, shape, lm_scale, lm_bias, "image_to_layer", allocator)


class ScaleTemplate(object):
    """Scale an image by a fixed amount per polarization.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    dtype : {`np.float32`, `np.float64`}
        Image precision
    num_polarizations : int
        Number of polarizations stored in the image
    tuning : dict, optional
        Tuning parameters (currently unused)
    """

    def __init__(self, context, dtype, num_polarizations, tuning=None):
        # TODO: autotuning
        self.context = context
        self.dtype = np.dtype(dtype)
        self.num_polarizations = num_polarizations
        self.wgsx = 16
        self.wgsy = 16
        self.program = accel.build(
            context, "imager_kernels/scale.mako",
            {
                'real_type': katsdpimager.types.dtype_to_ctype(dtype),
                'wgsx': self.wgsx,
                'wgsy': self.wgsy,
                'num_polarizations': num_polarizations
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return Scale(self, *args, **kwargs)


class Scale(accel.Operation):
    """Instantiation of :class:`ScaleTemplate`.

    .. rubric:: Slots

    **data** : array
        Image, indexed by polarization, y, x

    Parameters
    ----------
    template : :class:`ScaleTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    shape : tuple of int
        Shape of the data.
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, shape, allocator=None):
        super(Scale, self).__init__(command_queue, allocator)
        self.template = template
        if len(shape) != 3:
            raise ValueError('Wrong number of dimensions in shape')
        if shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        self.slots['data'] = accel.IOSlot(
            (accel.Dimension(shape[0]),
             accel.Dimension(shape[1], template.wgsy),
             accel.Dimension(shape[2], template.wgsx)), template.dtype)
        self.kernel = template.program.get_kernel('scale')
        self.scale_factor = np.zeros((shape[0],), template.dtype)

    def set_scale_factor(self, scale_factor):
        self.scale_factor[:] = scale_factor

    def _run(self):
        data = self.buffer('data')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                data.buffer,
                np.int32(data.padded_shape[2]),
                np.int32(data.padded_shape[1] * data.padded_shape[2]),
                self.scale_factor
            ],
            global_size=(accel.roundup(data.shape[2], self.template.wgsx),
                         accel.roundup(data.shape[1], self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy)
        )


class GridToImageTemplate(object):
    """Template for a combined operation that converts from a complex grid to a
    real image, including layer-to-image conversion.  The grid need not be
    conjugate symmetric: it is put through a complex-to-complex transformation,
    and the real part of the result is returned. Both the grid and the image
    have the DC term in the middle.

    This operation is destructive: the grid is modified in-place.

    Because it uses :class:`~katsdpimager.fft.FftTemplate`, most of the
    parameters are baked into the template rather than the instance.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue`
        Command queue for the operation
    shape : tuple of int
        Shape for both the source and destination, as (polarizations, height, width)
    padded_shape_src : tuple of int
        Padded shape of the grid
    padded_shape_dest : tuple of int
        Padded shape of the image
    real_dtype : {`np.float32`, `np.float64`}
        Precision
    """

    def __init__(self, command_queue, shape, padded_shape_src, padded_shape_dest, real_dtype):
        complex_dtype = katsdpimager.types.real_to_complex(real_dtype)
        self.shift_complex = fft.FftshiftTemplate(command_queue.context, complex_dtype)
        self.fft = fft.FftTemplate(command_queue, 2, shape, complex_dtype,
                                   padded_shape_src, padded_shape_dest)
        self.layer_to_image = LayerToImageTemplate(command_queue.context, real_dtype)

    def instantiate(self, *args, **kwargs):
        return GridToImage(self, *args, **kwargs)


class GridToImage(accel.OperationSequence):
    """Instantiation of :class:`GridToImageTemplate`.

    Parameters
    ----------
    template : :class:`GridToImageTemplate`
        Operation template
    lm_scale, lm_bias : float
        See :class:`LayerToImage`
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, lm_scale, lm_bias, allocator=None):
        command_queue = template.fft.command_queue
        self._shift_grid = template.shift_complex.instantiate(
            command_queue, template.fft.shape, allocator)
        self._ifft = template.fft.instantiate(fft.FFT_INVERSE, allocator)
        self._layer_to_image = template.layer_to_image.instantiate(
            command_queue, template.fft.shape, lm_scale, lm_bias, allocator)
        operations = [
            ('shift_grid', self._shift_grid),
            ('ifft', self._ifft),
            ('layer_to_image', self._layer_to_image)
        ]
        compounds = {
            'grid': ['shift_grid:data', 'ifft:src'],
            'layer': ['ifft:dest', 'layer_to_image:layer'],
            'image': ['layer_to_image:image'],
            'kernel1d': ['layer_to_image:kernel1d']
        }
        super(GridToImage, self).__init__(command_queue, operations, compounds, allocator=allocator)

    def set_w(self, w):
        self._layer_to_image.set_w(w)


class ImageToGridTemplate(object):
    """Convert from a real image to a complex grid, for a single W layer.
    Both the grid and the image have the DC term in the middle.

    Because it uses :class:`~katsdpimager.fft.FftTemplate`, most of the
    parameters are baked into the template rather than the instance.
    """
    def __init__(self, command_queue, shape, padded_shape_src, padded_shape_dest, real_dtype):
        complex_dtype = katsdpimager.types.real_to_complex(real_dtype)
        self.shift_complex = fft.FftshiftTemplate(command_queue.context, complex_dtype)
        self.fft = fft.FftTemplate(command_queue, 2, shape, complex_dtype,
                                   padded_shape_src, padded_shape_dest)
        self.image_to_layer = ImageToLayerTemplate(command_queue.context, real_dtype)

    def instantiate(self, *args, **kwargs):
        return ImageToGrid(self, *args, **kwargs)


class ImageToGrid(accel.OperationSequence):
    """Instantiation of :class:`ImageToGridTemplate`.

    Parameters
    ----------
    template : :class:`ImageToGridTemplate`
        Operation template
    lm_scale, lm_bias : float
        See :class:`ImageToLayer`
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, lm_scale, lm_bias, allocator=None):
        command_queue = template.fft.command_queue
        self._shift_grid = template.shift_complex.instantiate(
            command_queue, template.fft.shape, allocator)
        self._fft = template.fft.instantiate(fft.FFT_FORWARD, allocator)
        self._image_to_layer = template.image_to_layer.instantiate(
            command_queue, template.fft.shape, lm_scale, lm_bias, allocator)
        operations = [
            ('image_to_layer', self._image_to_layer),
            ('fft', self._fft),
            ('shift_grid', self._shift_grid)
        ]
        compounds = {
            'grid': ['shift_grid:data', 'fft:dest'],
            'layer': ['fft:src', 'image_to_layer:layer'],
            'image': ['image_to_layer:image'],
            'kernel1d': ['image_to_layer:kernel1d']
        }
        super(ImageToGrid, self).__init__(command_queue, operations, compounds, allocator=allocator)


class GridToImageHost(object):
    """CPU-only equivalent to :class:`GridToImage`.

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
        assert image.shape[-1] == image.shape[-2]
        assert image.shape[-1] % 2 == 0
        self.grid = grid
        self.layer = layer
        self.image = image
        self.kernel1d = kernel1d
        self.lm_scale = lm_scale
        self.lm_bias = lm_bias
        self.w = 0

    def set_w(self, w):
        self.w = w

    def clear(self):
        self.image.fill(0)

    def __call__(self):
        self.layer[:] = np.fft.ifft2(np.fft.ifftshift(self.grid, axes=(1, 2)), axes=(1, 2))
        # Scale factor is to match behaviour of CUFFT, which is unnormalized
        scale = self.layer.shape[1] * self.layer.shape[2]
        lm = np.arange(self.image.shape[1]).astype(self.image.dtype) * self.lm_scale + self.lm_bias
        # We do calculations involving the complex values prior to applying
        # fftshift to save memory, so we have to apply the inverse shift to lm.
        lm = np.fft.ifftshift(lm)
        lm2 = lm * lm
        n = np.sqrt(1 - (lm2[:, np.newaxis] + lm2[np.newaxis, :]))
        w_correct = np.exp(2j * math.pi * self.w * (n - 1))
        self.layer *= w_correct
        # TODO: most of the calculations here would be more efficient with numba
        image = self.layer.real.copy()
        image *= scale
        image *= n[np.newaxis, ...]
        image = np.fft.fftshift(image, axes=(1, 2))
        image /= np.outer(self.kernel1d, self.kernel1d)[np.newaxis, ...]
        self.image += image


class ImageToGridHost(object):
    """CPU-only equivalent to :class:`ImageToGrid`.

    The parameters specify which buffers the operation runs on, but the
    contents at construction time are irrelevant. The operation is
    performed at call time.

    Parameters
    ----------
    grid : ndarray, complex
        Output grid
    layer : ndarray, complex
        Intermediate structure holding the complex FFT
    image : ndarray, real
        Input image
    kernel1d : ndarray, real
        Tapering function
    lm_scale, lm_bias : real
        Linear transformation from pixel coordinates to l/m values
    """
    def __init__(self, grid, layer, image, kernel1d, lm_scale, lm_bias):
        assert image.shape[-1] == image.shape[-2]
        assert image.shape[-1] % 2 == 0
        self.grid = grid
        self.layer = layer
        self.image = image
        self.kernel1d = kernel1d
        self.lm_scale = lm_scale
        self.lm_bias = lm_bias
        self.w = 0

    def set_w(self, w):
        self.w = w

    def __call__(self):
        # TODO: rewrite most of this using numba
        lm = np.arange(self.image.shape[1]).astype(self.image.dtype) * self.lm_scale + self.lm_bias
        lm2 = lm * lm
        n = np.sqrt(1 - (lm2[:, np.newaxis] + lm2[np.newaxis, :]))[np.newaxis, ...]
        w_correct = np.exp(-2j * math.pi * self.w * (n - 1))
        kernel = np.outer(self.kernel1d, self.kernel1d)[np.newaxis, ...]
        self.layer[:] = self.image * kernel / n * w_correct
        self.grid[:] = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(self.layer, axes=(1, 2)),
                axes=(1, 2)),
            axes=(1, 2))
