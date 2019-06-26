"""Kernels for image-domain processing

It also handles conversion between visibility and image planes.

.. include:: macros.rst
"""

import math

import numpy as np
import pkg_resources
from katsdpsigproc import accel

import katsdpimager.types
from . import fft


class _LayerImageTemplate:
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

      This function is also *divided* in the reverse direction, making the two
      operations not actually inverses of each other. This is necessary because
      the transform of the tapering function is convolved with visibilities in
      both directions.

    - It multiplies by a W correction term, namely

      .. math::

         e^{2\pi i w\left(\sqrt{1-l(x)^2-m(y)^2} - 1\right)}

         Only the real component of the result is kept in the image.

    Further dimensions are supported e.g. for Stokes parameters, which occur
    before the l/m dimensions.

    A note on sign conventions: the measurement equation is taken to be

    .. math::

       V(u, v, w) = \int \frac{I(l, m)}{n} e^{-2\pi i(ul + vm + w(n-1))}\ dl\ dm.

    This is consistent with the sign conventions for the phase of :math:`V` and
    for :math:`(u, v, w)` that are documented in
    :ref:`katsdpimager.loader_core.LoaderBase.data_iter`.

    Parameters
    ----------
    context : |Context|
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
    command_queue : |CommandQueue|
        Command queue for the operation
    shape : tuple of int
        Shape of the image data (polarizations, height, width) - must be square
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
    def __init__(self, template, command_queue, shape, lm_scale, lm_bias, kernel_name,
                 allocator=None):
        if len(shape) != 3 or shape[-1] != shape[-2]:
            raise ValueError('shape must be square, not {}'.format(shape))
        if shape[-1] % 2 != 0:
            raise ValueError('image size must be even, not {}'.format(shape[-1]))
        super().__init__(command_queue, allocator)
        self.template = template
        complex_dtype = katsdpimager.types.real_to_complex(template.real_dtype)
        dims = [accel.Dimension(x) for x in shape]
        self.slots['layer'] = accel.IOSlot(dims[-2:], complex_dtype)
        self.slots['image'] = accel.IOSlot(dims, template.real_dtype)
        self.slots['kernel1d'] = accel.IOSlot((shape[-1],), template.real_dtype)
        self.kernel = template.program.get_kernel(kernel_name)
        self.lm_scale = lm_scale
        self.lm_bias = lm_bias
        self.w = 0
        self.polarization = 0

    def set_w(self, w):
        """Set w (in wavelengths)."""
        self.w = w

    def set_polarization(self, polarization):
        """Set polarization index in the image"""
        if polarization < 0 or polarization >= self.slots['image'].shape[0]:
            raise IndexError('polarization index out of range')
        self.polarization = polarization

    def _run(self):
        layer = self.buffer('layer')
        image = self.buffer('image')
        assert layer.padded_shape == image.padded_shape[-2:]
        half_size = image.shape[-1] // 2
        row_stride = image.padded_shape[-1]
        slice_stride = row_stride * image.padded_shape[-2]
        kernel1d = self.buffer('kernel1d')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                image.buffer,
                layer.buffer,
                np.int32(self.polarization * slice_stride),
                np.int32(row_stride),
                kernel1d.buffer,
                self.template.real_dtype.type(self.lm_scale),
                self.template.real_dtype.type(self.lm_bias),
                np.int32(half_size),
                np.int32(half_size * row_stride),
                self.template.real_dtype.type(half_size * self.lm_scale),
                self.template.real_dtype.type(2 * self.w)
            ],
            global_size=(accel.roundup(half_size, self.template.wgs_x),
                         accel.roundup(half_size, self.template.wgs_y)),
            local_size=(self.template.wgs_x, self.template.wgs_y)
        )


class LayerToImageTemplate(_LayerImageTemplate):
    """Convert layer to image. See :class:`_LayerImageTemplate` for details.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    real_dtype : {`np.float32`, `np.float64`}
        Image type
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, context, real_dtype, tuning=None):
        super().__init__(
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
    command_queue : |CommandQueue|
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
        super().__init__(
            template, command_queue, shape, lm_scale, lm_bias, "layer_to_image", allocator)


class ImageToLayerTemplate(_LayerImageTemplate):
    """Convert image to layer. See :class:`_LayerImageTemplate` for details.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    real_dtype : {`np.float32`, `np.float64`}
        Image type
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, context, real_dtype, tuning=None):
        super().__init__(
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
    command_queue : |CommandQueue|
        Command queue for the operation
    shape : tuple of int
        Shape of the image as (polarizations, height, width) - must be square
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
        super().__init__(
            template, command_queue, shape, lm_scale, lm_bias, "image_to_layer", allocator)


class ScaleTemplate:
    """Scale an image by a fixed amount per polarization.

    Parameters
    ----------
    context : |Context|
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
    command_queue : |CommandQueue|
        Command queue for the operation
    shape : tuple of int
        Shape of the data.
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, shape, allocator=None):
        super().__init__(command_queue, allocator)
        self.template = template
        if len(shape) != 3:
            raise ValueError('Wrong number of dimensions in shape')
        if shape[0] != template.num_polarizations:
            raise ValueError('Mismatch in number of polarizations')
        self.slots['data'] = accel.IOSlot(shape, template.dtype)
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
                np.int32(data.shape[2]),
                np.int32(data.shape[1]),
                self.scale_factor
            ],
            global_size=(accel.roundup(data.shape[2], self.template.wgsx),
                         accel.roundup(data.shape[1], self.template.wgsy)),
            local_size=(self.template.wgsx, self.template.wgsy)
        )


class GridImageTemplate:
    """Template for a combined operation that converts from a complex grid to a
    real image or vice versa, including layer-to-image/image-to-layer
    conversion.  The grid need not be conjugate symmetric: it is put through a
    complex-to-complex transformation, and the real part of the result is
    returned. Both the grid and the image have the DC term in the middle.

    Because it uses :class:`~katsdpimager.fft.FftTemplate`, most of the
    parameters are baked into the template rather than the instance.

    .. warning::
        Instances created from the same template cannot be executed
        simultaneously, due to limitations of CUFFT (work area is part of the
        plan).

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue`
        Command queue for the operation
    shape_layer : tuple of int
        Shape for the intermediate layer, as (height, width)
    padded_shape_layer : tuple of int
        Padded shape of the intermediate layer
    real_dtype : {`np.float32`, `np.float64`}
        Precision
    """

    def __init__(self, command_queue, shape_layer, padded_shape_layer, real_dtype):
        complex_dtype = katsdpimager.types.real_to_complex(real_dtype)
        self.fft = fft.FftTemplate(command_queue, 2, shape_layer, complex_dtype, complex_dtype,
                                   padded_shape_layer, padded_shape_layer)
        self.layer_to_image = LayerToImageTemplate(command_queue.context, real_dtype)
        self.image_to_layer = ImageToLayerTemplate(command_queue.context, real_dtype)

    def instantiate_grid_to_image(self, *args, **kwargs):
        return GridToImage(self, *args, **kwargs)

    def instantiate_image_to_grid(self, *args, **kwargs):
        return ImageToGrid(self, *args, **kwargs)


class GridToImage(accel.OperationSequence):
    """Instantiation of :class:`GridToImageTemplate`.

    Parameters
    ----------
    template : :class:`GridImageTemplate`
        Operation template
    shape_grid : tuple of int
        Shape of the grid, as (polarizations, height, width)
    lm_scale,lm_bias : float
        See :class:`LayerToImage`
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, shape_grid, lm_scale, lm_bias, allocator=None):
        command_queue = template.fft.command_queue
        self._ifft = template.fft.instantiate(fft.FFT_INVERSE, allocator)
        polarizations = shape_grid[0]
        shape_image = (polarizations,) + tuple(template.fft.shape)
        self._layer_to_image = template.layer_to_image.instantiate(
            command_queue, shape_image, lm_scale, lm_bias, allocator)
        operations = [
            ('ifft', self._ifft),
            ('layer_to_image', self._layer_to_image)
        ]
        compounds = {
            'layer': ['ifft:src', 'ifft:dest', 'layer_to_image:layer'],
            'image': ['layer_to_image:image'],
            'kernel1d': ['layer_to_image:kernel1d']
        }
        super().__init__(command_queue, operations, compounds, allocator=allocator)
        self.slots['grid'] = accel.IOSlot(shape_grid, template.fft.dtype_src)

    def set_w(self, w):
        self._layer_to_image.set_w(w)

    def _run(self):
        grid = self.buffer('grid')
        layer = self.buffer('layer')
        polarizations, width, height = grid.shape
        # This could probably be made to work correctly with odd width/height,
        # but it would need to be done carefully to avoid off-by-one bugs.
        assert width % 2 == 0
        assert height % 2 == 0
        half_width = width // 2
        half_height = height // 2
        for pol in range(polarizations):
            layer.zero(self.command_queue)
            # Copy one polarization from the grid to the layer, also switching
            # the centre to the corners. The two slices in each of src/dest_x/y
            # are the two halves of the data.
            src_y = np.s_[:half_height, -half_height:]
            dest_y = np.s_[-half_height:, :half_height]
            src_x = np.s_[:half_width, -half_width:]
            dest_x = np.s_[-half_width:, :half_width]
            for sy, dy in zip(src_y, dest_y):
                for sx, dx in zip(src_x, dest_x):
                    grid.copy_region(
                        self.command_queue, layer, (pol, sy, sx), (dy, dx))
            self._layer_to_image.set_polarization(pol)
            super()._run()


class ImageToGrid(accel.OperationSequence):
    """Instantiation of :class:`ImageToGridTemplate`.

    Parameters
    ----------
    template : :class:`GridImageTemplate`
        Operation template
    shape_grid : tuple of int
        Shape of the grid, as (polarizations, height, width)
    lm_scale,lm_bias : float
        See :class:`ImageToLayer`
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, shape_grid, lm_scale, lm_bias, allocator=None):
        command_queue = template.fft.command_queue
        polarizations = shape_grid[0]
        shape_image = (polarizations,) + tuple(template.fft.shape)
        self._fft = template.fft.instantiate(fft.FFT_FORWARD, allocator)
        self._image_to_layer = template.image_to_layer.instantiate(
            command_queue, shape_image, lm_scale, lm_bias, allocator)
        operations = [
            ('image_to_layer', self._image_to_layer),
            ('fft', self._fft),
        ]
        compounds = {
            'layer': ['fft:src', 'fft:dest', 'image_to_layer:layer'],
            'image': ['image_to_layer:image'],
            'kernel1d': ['image_to_layer:kernel1d']
        }
        super().__init__(command_queue, operations, compounds, allocator=allocator)
        self.slots['grid'] = accel.IOSlot(shape_grid, template.fft.dtype_dest)

    def set_w(self, w):
        self._image_to_layer.set_w(w)

    def _run(self):
        # TODO: unify with GridToImage._run
        grid = self.buffer('grid')
        layer = self.buffer('layer')
        polarizations, width, height = grid.shape
        # This could probably be made to work correctly with odd width/height,
        # but it would need to be done carefully to avoid off-by-one bugs.
        assert width % 2 == 0
        assert height % 2 == 0
        half_width = width // 2
        half_height = height // 2
        for pol in range(polarizations):
            self._image_to_layer.set_polarization(pol)
            super()._run()
            # Copy one polarization from the grid to the layer, also switching
            # the centre to the corners. The two slices in each of src/dest_x/y
            # are the two halves of the data.
            src_y = np.s_[:half_height, -half_height:]
            dest_y = np.s_[-half_height:, :half_height]
            src_x = np.s_[:half_width, -half_width:]
            dest_x = np.s_[-half_width:, :half_width]
            for sy, dy in zip(src_y, dest_y):
                for sx, dx in zip(src_x, dest_x):
                    layer.copy_region(
                        self.command_queue, grid, (sy, sx), (pol, dy, dx))


class GridToImageHost:
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
    lm_scale,lm_bias : real
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


class ImageToGridHost:
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
    lm_scale,lm_bias : real
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
        self.layer[:] = self.image / (kernel * n) * w_correct
        self.grid[:] = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(self.layer, axes=(1, 2)),
                axes=(1, 2)),
            axes=(1, 2))
