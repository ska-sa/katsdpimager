"""Fitting of the synthesised beam, and convolution of this beam with the
model."""

from __future__ import print_function, division
import math
import numpy as np
from astropy.modeling import models, fitting
from astropy import units


class Beam(object):
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


def fit_beam(psf, step=1.0, threshold=0.05):
    """Fit a 2D Gaussian to the point spread function. The function is not
    truncated to the central region: the caller should do this.

    Values far from the origin are troublesome because they have an overly
    large impact on the initial estimate, and may also be negative. We restrict
    the match to values that exceed `threshold`.

    Parameters
    ----------
    psf : array-like, 2D and real
        Point spead function, with the origin at the central pixel (rounded up)
    step : float
        Spacing between sample points. The units are arbitrary, and the return
        values will be in the same units.
    threshold : float
        Minimum PSF value to be used in the fit.

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

    # Initial guess: compute the second moment about the origin.
    mask = psf > threshold
    indices = list(np.nonzero(mask))
    indices[0] = (indices[0] - psf.shape[0] // 2) * step
    indices[1] = (indices[1] - psf.shape[1] // 2) * step
    picked = psf[mask]
    cov = np.empty((2, 2))
    total = np.sum(picked)
    cov[0, 0] = np.sum(picked * indices[0]**2) / total
    cov[0, 1] = np.sum(picked * indices[0] * indices[1]) / total
    cov[1, 0] = cov[0, 1]
    cov[1, 1] = np.sum(picked * indices[1]**2) / total
    init = models.Gaussian2D(
        amplitude=1.0,
        cov_matrix=cov,
        fixed={'x_mean': True, 'y_mean': True, 'amplitude': True})
    # Compute a more accurate model by fitting to the data
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
    r"""Convolve a model image with a restoring beam.

    This is done by FFT-based convolution, which causes the restoring beam to
    wrap at the edges of the image. Since the edges are usually suspect anyway,
    this isn't considered an issue.

    .. rubric:: Implementation details

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
    amplitude = 2 * np.pi * beam.model.amplitude * np.linalg.det(M)
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
