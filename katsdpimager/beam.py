r"""Fitting of the synthesised beam, and convolution of this beam with the
model.

Convolution implementation details
----------------------------------
This is done by FFT-based convolution, which causes the restoring beam to
wrap at the edges of the image. Since the edges are usually suspect anyway,
this isn't considered an issue.

The Fourier transform of the restoring beam is computed analytically,
rather than applying an FFT to a sampled beam. Once GPU-acceleration is
being done, this will save memory since the transform can be multiplied
in-place as it is computed.

Consider a 1D Gaussian of the form :math:`e^{-\frac12 x^2}`: it has a
Fourier Transform of :math:`\sqrt{2\pi}e^{-2\pi^2 k^2}` (see Mathworld_).
Since a Gaussian is separable, the transform of the 2D Gaussian
:math:`Ae^{-\frac12 \lVert x\rVert^2}` is
:math:`2\pi A e^{-2\pi^2\lVert k\rVert^2}`. Next, using variable
substitution, the transform of
:math:`Ae^{-\frac12 \lVert M^{-1}x\rVert^2}` is
:math:`2\pi \lvert M\rvert A e^{-2\pi^2\lVert Mk\rVert^2}` for a matrix
:math:`M`.

Astropy represents a 2D Gaussian by the standard deviation along the major
and minor axes and the angle of the axes, so we need to reconstruct M,
which is the square root of the covariance matrix. For standard deviations
:math:`\sigma_1` and :math:`\sigma_2` and :math:`R` being the rotation
matrix for the stored angle, the covariance matrix is
:math:`R\left(\begin{smallmatrix}\sigma_1^2 & 0\\0 & \sigma_2^2\end{smallmatrix}\right)R^T`
and its square root is
:math:`M = R\left(\begin{smallmatrix}\sigma_1 & 0\\0 & \sigma_2\end{smallmatrix}\right)R^T`.

.. _Mathworld: http://mathworld.wolfram.com/FourierTransformGaussian.html

.. include:: macros.rst
"""

import math

import numpy as np
import pkg_resources
from astropy.modeling import models, fitting
from astropy import units
from katsdpsigproc import accel
import katsdpimager.types

from katsdpimager import fft


class Beam:
    """Gaussian synthesised beam model.

    Parameters
    ----------
    model : :class:`astropy.modeling.models.Gaussian2D`
        Fitted 2D Gaussian model

    Attributes
    ----------
    model : :class:`astropy.modeling.models.Gaussian2D`
        Model passed to the constructor
    major : float
        Full-width half-maximum for the major axis
    minor : float
        Full-width half-maximum for the minor axis
    theta : angle Quantity
        Angle of the ellipse, measured from positive axis 0 of the PSF to the
        major axis of the ellipse, in the direction of positive axis 1 of the
        PSF.
    """
    def __init__(self, model):
        self.model = model
        # Scale factor between stddev and FWHM
        scale = math.sqrt(8 * math.log(2))
        self.major = model.x_stddev.value * scale
        self.minor = model.y_stddev.value * scale
        theta = model.theta.value
        if self.major < self.minor:
            self.minor, self.major = self.major, self.minor
            theta += math.pi / 2
        theta %= math.pi
        self.theta = theta * units.rad

    def __str__(self):
        return "Beam({0.major}, {0.minor}, {0.theta})".format(self)

    def __repr__(self):
        return "Beam({0.major!r}, {0.minor!r}, {0.theta!r})".format(self)


def fit_beam(psf, step=1.0, threshold=0.01, init_threshold=0.5):
    """Fit a 2D Gaussian to the point spread function. The function is not
    truncated to the central region: the caller should do this.

    Values far from the origin are troublesome because they are not even
    vaguely Gaussian. We restrict the match to values that exceed
    `threshold`.

    Parameters
    ----------
    psf : array-like, 2D and real
        Point spead function, with the origin at the central pixel (rounded up)
    step : float
        Spacing between sample points. The units are arbitrary, and the return
        values will be in the same units.
    threshold : float
        Minimum PSF value to be used in the fit.
    init_threshold : float
        Minimum PSF value used to compute the initial estimate.

    Returns
    -------
    major : float
        Full-width half-maximum for the major axis
    minor : float
        Full-width half-maximum for the minor axis
    theta : float
        Angle of the ellipse, measured from positive axis 0 of the PSF to the
        major axis of the ellipse, in the direction of positive axis 1 of the
        PSF.
    """
    def extract(psf, threshold):
        """Get coordinates and values that are > threshold"""
        mask = psf > threshold
        indices = list(np.nonzero(mask))
        indices[0] = (indices[0] - psf.shape[0] // 2) * step
        indices[1] = (indices[1] - psf.shape[1] // 2) * step
        picked = psf[mask]
        return picked, indices

    # Initial guess: compute the second moment about the origin, but keeping
    # only the values above a (fairly high) threshold.
    picked, indices = extract(psf, init_threshold)
    cov = np.empty((2, 2))
    total = np.sum(picked)
    cov[0, 0] = np.sum(picked * indices[0]**2) / total
    cov[0, 1] = np.sum(picked * indices[0] * indices[1]) / total
    cov[1, 0] = cov[0, 1]
    cov[1, 1] = np.sum(picked * indices[1]**2) / total
    # Because we truncated at init_threshold, we will have an underestimate
    # of the radius. We can correct for this with Mathematics (tm). The
    # variance of a standard 2D Gaussian truncated at radius R is
    # 1 - (1 + R^2/2) * exp(-R*2 / 2).
    R2 = -2 * np.log(init_threshold)
    scale = 1 - (1 + 0.5 * R2) * np.exp(-0.5 * R2)
    cov /= scale
    init = models.Gaussian2D(
        amplitude=1.0,
        cov_matrix=cov,
        fixed={'x_mean': True, 'y_mean': True, 'amplitude': True})

    # Compute a more accurate model by fitting to more of the data
    picked, indices = extract(psf, threshold)
    fit = fitting.LevMarLSQFitter()
    model = fit(init, indices[0], indices[1], picked)
    return Beam(model)


def beam_covariance_sqrt(beam):
    model = beam.model
    # Rotation matrix for theta
    c = np.cos(model.theta)
    s = np.sin(model.theta)
    Q = np.matrix([[c, -s], [s, c]])
    # Diagonalisation of square root of covariance matrix
    D = np.asmatrix(np.diag([model.x_stddev.value, model.y_stddev.value]))
    return Q * D * Q.T


def convolve_beam(model, beam, out=None):
    """Convolve a model image with a restoring beam.

    Parameters
    ----------
    model : array-like, float
        Model image as a 3D array indexed by polarization, m, l
    beam : :class:`Beam`
        Restoring beam model, created with step size equal to the pixel size of
        `model`.
    out : array-like, optional
        If specified, it will contain the output. It is safe to pass `model`.
    """
    if out is None:
        out = np.empty_like(model)
    model_ft = np.fft.fftn(model, axes=[1, 2])
    M = beam_covariance_sqrt(beam)
    # Due to https://github.com/astropy/astropy/issues/1105, the determinant
    # can be negative; hence, the np.abs is necessary.
    amplitude = 2 * np.pi * beam.model.amplitude * np.abs(np.linalg.det(M))
    u = np.fft.fftfreq(model.shape[1])
    v = np.fft.fftfreq(model.shape[2])
    # Coords is shape (model.shape[1], model.shape[2], 2) - pairs of coordinates
    coords = np.stack(np.meshgrid(u, v, indexing='ij'), axis=-1)
    # Compute matrix-vector product M * uv, for every uv coordinate
    rotated_coords = np.inner(coords, M.A)
    rotated_coords_square = np.sum(rotated_coords**2, axis=-1)
    beam_ft = amplitude * np.exp(-2.0 * np.pi**2 * rotated_coords_square)
    out[:] = np.fft.ifftn(model_ft * beam_ft[np.newaxis, ...], axes=[1, 2]).real
    return out


class FourierBeamTemplate:
    """Multiply the Fourier transform of an image by the Fourier transform of a
    Gaussian beam model. The beam parameters must be in units of image pixels.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    dtype : {`numpy.float32`, `numpy.float64`}
        Real data type
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, context, dtype, tuning=None):
        # TODO: autotune
        self.wgs_x = 16
        self.wgs_y = 16
        self.dtype = np.dtype(dtype)
        parameters = {
            'wgs_x': self.wgs_x,
            'wgs_y': self.wgs_y,
            'real_type': katsdpimager.types.dtype_to_ctype(self.dtype)
        }
        self.program = accel.build(
            context, "imager_kernels/fourier_beam.mako", parameters,
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    def instantiate(self, *args, **kwargs):
        return FourierBeam(self, *args, **kwargs)


class FourierBeam(accel.Operation):
    """Instantiation of :class:`FourierBeamTemplate`.

    .. rubric:: Slots

    **data** : array of shape (height, width // 2 + 1), complex
        Fourier transform (real-to-complex) of the model image. On output, the
        Fourier transform of the restored image.

    Attributes
    ----------
    beam : :class:`Beam`
        Restoring beam. It must be set before invoking the operation.

    Parameters
    ----------
    template : :class:`FourierBeamTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    image_shape : tuple
        (height, width) of the image. Note that the buffer on which this operation works
        will have shape (height, width // 2 + 1).
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, image_shape, allocator=None):
        if len(image_shape) != 2:
            raise ValueError('image_shape must be 2D')
        super().__init__(command_queue, allocator=allocator)
        self.template = template
        self.image_shape = image_shape
        output_shape = (image_shape[0], image_shape[1] // 2 + 1)
        complex_type = katsdpimager.types.real_to_complex(template.dtype)
        self.slots['data'] = accel.IOSlot(output_shape, complex_type)
        self.beam = None
        self._kernel = template.program.get_kernel('fourier_beam')

    def _run(self):
        if self.beam is None:
            raise ValueError('Must set beam')
        M = beam_covariance_sqrt(self.beam)
        # Due to https://github.com/astropy/astropy/issues/1105, the determinant
        # can be negative; hence, the np.abs is necessary.
        amplitude = 2 * np.pi * self.beam.model.amplitude * np.abs(np.linalg.det(M))
        # CUFFT, unlikely numpy.fft, does not normalize the inverse transform.
        # We fold the normalization into the amplitude
        amplitude /= self.image_shape[0] * self.image_shape[1]
        # The kernel has integral coordinates, which we need to convert to
        # the range [-1, 1). We fold this into the matrix.
        M *= np.asmatrix(np.diag([1.0 / self.image_shape[0],
                                  1.0 / self.image_shape[1]]))
        # Compute overall matrix for the exponent
        C = -2 * np.pi**2 * M.T * M
        real = self.template.dtype.type
        data = self.buffer('data')
        self.command_queue.enqueue_kernel(
            self._kernel,
            [
                data.buffer,
                np.int32(data.padded_shape[1]),
                real(amplitude),
                real(C[0, 0]),
                real(2 * C[0, 1]),
                real(C[1, 1]),
                np.int32(data.shape[1]),
                np.int32(data.shape[0])
            ],
            global_size=(accel.roundup(data.shape[1], self.template.wgs_x),
                         accel.roundup(data.shape[0], self.template.wgs_y)),
            local_size=(self.template.wgs_x, self.template.wgs_y)
        )


class ConvolveBeamTemplate:
    """Template for device convolution of a model image with a Gaussian
    restoring beam, for a single polarization. Due to limitations in CUFFT,
    many parameters are folded into the template.

    Due to limitations in katsdpsigproc, convolving a multi-polarization image
    currently requires each polarization to be copied into a
    single-polarization image first.

    Parameters
    ----------
    command_queue : |CommandQueue|
        Command queue for the operation
    shape : tuple
        (height, width) of the image
    dtype : {`numpy.float32`, `numpy.float64`}
        Type of the image data
    padded_shape_image : tuple, optional
        Padded shape for the image. Defaults to no padding.
    padded_shape_fourier : tuple, optional
        Padded shape for the Fourier transform of the image. Defaults to no
        padding. Note that because the image is real, the Fourier transform has
        shape (height, width // 2 + 1).
    tuning : dict, optional
        Tuning parameters (currently unused)
    """
    def __init__(self, command_queue, shape, dtype,
                 padded_shape_image=None, padded_shape_fourier=None, tuning=None):
        if padded_shape_image is None:
            padded_shape_image = tuple(shape)
        if padded_shape_fourier is None:
            padded_shape_fourier = tuple(shape[:-1]) + (shape[-1] // 2 + 1,)
        if len(shape) != 2 or len(padded_shape_image) != 2 or len(padded_shape_fourier) != 2:
            raise ValueError('wrong number of dimensions')
        self.dtype = np.dtype(dtype)
        self.shape = shape
        complex_dtype = katsdpimager.types.real_to_complex(self.dtype)
        self.fft = fft.FftTemplate(command_queue, 2, shape, dtype, complex_dtype,
                                   padded_shape_image, padded_shape_fourier)
        self.ifft = fft.FftTemplate(command_queue, 2, shape, complex_dtype, dtype,
                                    padded_shape_fourier, padded_shape_image)
        self.fourier_beam = FourierBeamTemplate(command_queue.context, dtype, tuning)

    def instantiate(self, *args, **kwargs):
        return ConvolveBeam(self, *args, **kwargs)


class ConvolveBeam(accel.OperationSequence):
    """Instantiation of :class:`ConvolveBeamTemplate`.

    .. rubric:: Slots

    **image** : array of shape (height, width), real
        Image to be convolved, as well as the output.
    **fourier** : array of shape (height, width // 2 + 1), real
        Fourier transform of the image. It should be treated as an opaque
        scratch buffer.

    Attributes
    ----------
    beam : :class:`Beam`
        Restoring beam. It must be set before invoking the operation.
    """
    def __init__(self, template, allocator=None):
        self._fft = template.fft.instantiate(fft.FFT_FORWARD, allocator=allocator)
        self._ifft = template.ifft.instantiate(fft.FFT_INVERSE, allocator=allocator)
        self._fourier_beam = template.fourier_beam.instantiate(
            template.fft.command_queue, template.shape, allocator=allocator)
        operations = [
            ('fft', self._fft),
            ('fourier_beam', self._fourier_beam),
            ('ifft', self._ifft)
        ]
        compounds = {
            'image': ['fft:src', 'ifft:dest'],
            'fourier': ['fft:dest', 'ifft:src', 'fourier_beam:data']
        }
        super().__init__(
            template.fft.command_queue, operations, compounds, allocator=allocator)

    @property
    def beam(self):
        return self._fourier_beam.beam

    @beam.setter
    def beam(self, value):
        self._fourier_beam.beam = value
