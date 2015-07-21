"""Fitting of the synthesised beam, and convolution of this beam with the
model."""

from __future__ import print_function, division
import math
import numpy as np
from astropy.modeling import models, fitting
from astropy import units
import scipy.signal


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
    # Sample the synthesized beam
    window = min(model.shape[-1], int(math.ceil(beam.major * 5)))
    lm = np.arange(-window, window + 1)
    beam_pixels = beam.model(*np.meshgrid(lm, lm, indexing='ij'))
    if out is None:
        out = np.empty_like(model)
    for pol in range(model.shape[0]):
        out[pol, ...] = scipy.signal.fftconvolve(model[pol, ...], beam_pixels, 'same')
    return out
