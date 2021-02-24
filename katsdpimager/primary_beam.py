"""Primary beam models and correction."""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Mapping, Any

import numpy as np
import numba
import h5py
import pkg_resources
import astropy.units as units

from .profiling import profile_function


class Parameter:
    """A parameter on which a `BeamModelSet` is parametrised.

    Some parameters are standardized: refer to the constants in this class.

    Parameters
    ----------
    name
        Short name of the parameter, which must be a valid Python identifier.
    description
        Human-readable description of the parameter.
    unit
        Units in which the parameter should be specified (or a compatible
        unit). If ``None``, the parameter is not numeric (for a unit-less
        numeric quantity, use ``units.dimensionless_unscaled``).
    """

    def __init__(self, name: str, description: str, unit: Optional[units.Unit] = None) -> None:
        self.name = name
        self.description = description
        self.unit = unit

    ANTENNA: 'Parameter'
    ELEVATION: 'Parameter'


Parameter.ANTENNA = Parameter('antenna', 'Name of the antenna')
Parameter.ELEVATION = Parameter('elevation', 'Elevation at which the antenna is pointing',
                                units.rad)


class BeamModel(ABC):
    r"""Model of an antenna primary beam.

    This is a model that describes the response of an antenna in a particular
    direction relative to nominal pointing direction, for each polarization
    and for a range of frequencies.

    It is most likely specific to an antenna, elevation, or other variables.
    However, it is specified in a frame oriented to the dish rather than the
    sky, so remains valid with parallactic angle rotation. Description of the
    mount type (e.g. Alt-Az versus equatorial or 3-axis) is currently beyond
    the scope of this class.

    The phase of the electric field at a point increases with time i.e., the
    phasor is

    .. math:: e^{(\omega t - kz)i}

    The "ideal" complex voltage is multiplied by the beam model to obtain the
    measured voltage.

    If you stick your right arm out and left hand up, then pretend to be an
    antenna (lean back and look at the sky) then they correspond to the
    directions of positive horizontal and vertical polarization respectively
    (the absolute signs don't matter but the signs relative to each other do
    for cross-hand terms).

    The grid is regularly sampled in azimuth and elevation, but may be
    irregularly sampled in frequency. The samples are ordered in increasing
    order of azimuth (north to east) and elevation (horizon to zenith). Note
    that this system has the opposite handedness to RA/Dec.
    """

    @property
    @abstractmethod
    def parameter_values(self) -> Mapping[str, Any]:
        """Get the parameter values for which this model has been specialised."""

    @abstractmethod
    def sample(self,
               az_start: float, az_step: float, az_samples: int,
               el_start: float, el_step: float, el_samples: int,
               frequencies: units.Quantity) -> np.ndarray:
        """Generate a grid of samples.

        Samples that fall outside the support of the beam model will be filled
        with NaNs.

        Parameters
        ----------
        az_start
            Relative azimuth of the first sample (in the SIN projection).
        az_step
            Step between samples in the azimuth direction (in the SIN projection).
        az_samples
            Number of samples in the azimuth direction.
        el_start
            Relative elevation of the first sample (in the SIN projection).
        el_step
            Step between samples in the elevation direction (in the SIN projection).
        el_samples
            Number of samples in the elevation direction.
        frequencies
            1D array of frequencies.

        Returns
        -------
        beam
            A 5D array. The axes are:

              1. The feed (0=horizontal, 1=vertical).
              2. The polarization of the signal (0=horizontal, 1=vertical).
              3. Frequency (indexed as per `frequencies`).
              4. Elevation (of the source relative to the dish).
              5. Azimuth (of the source relative to the dish).
        """
        pass


class BeamModelSet(ABC):
    """Generator for beam models.

    The interface is relatively general and allows models to be specialised for
    on variety of parameters. It is intended that a single instance can
    describe a whole telescope for an observation.
    """

    @property
    @abstractmethod
    def parameters(self) -> Sequence[Parameter]:
        """Describes the variables by which the general model is parametrized."""

    @abstractmethod
    def sample(self, **kwargs) -> BeamModel:
        """Obtain a specific model.

        It is expected that this may be an expensive function and the caller
        should cache the result if appropriate.

        Where parameters are omitted, the implementation should return a
        model that is suitable for the typical range of the parameter. Extra
        parameters are ignored. Thus, calling code can always safely supply
        the parameters is knows about, without needing to check
        :attr:`parameters`.

        Parameters
        ----------
        **kwargs
            Parameter values to specialize on.

        Raises
        ------
        ValueError
            if the value of any parameter is out of range of the model (optional).
        TypeError
            if the type of any parameter is inappropriate (optional).
        """


@profile_function()
@numba.njit(parallel=True, nogil=True)
def _sample_impl(az, el, beam, step, out):
    """Do linear interpolation of a beam on a 2D grid.

    Parameters
    ----------
    az
        1D array of direction cosines in the azimuth axis
    el
        1D array of direction cosines in the elevation axis
    beam
        2D array of beam values, with axes for frequency and radius
    step
        Increase in radius for each step along the radius axis
    out
        Output array, with shape (2, 2, frequency, el, az). Only the
        [0, 0] and [1, 1] elements (co-pol) are filled in.
    """
    pos = 0
    deltas = np.empty(beam.shape[1] - 1, beam.dtype)
    for i in range(beam.shape[0]):
        for j in range(len(deltas)):
            deltas[j] = beam[i, j + 1] - beam[i, j]
        for j in numba.prange(len(el)):
            for k in range(len(az)):
                radius = np.sqrt(el[j] * el[j] + az[k] * az[k])
                radius_steps = radius / step
                pos = np.int_(radius_steps)
                if pos < len(deltas):
                    value = beam[i, pos] + deltas[pos] * (radius_steps - pos)
                else:
                    value = np.nan
                out[0, 0, i, j, k] = value
                out[1, 1, i, j, k] = value


class TrivialBeamModel(BeamModel):
    """The simplest possible beam model.

    It is
    - radially symmetric
    - antenna-independent
    - elevation-independent
    - real-valued
    - polarization-independent (assumes no leakage and same response in H and V)
    """

    def __init__(self, filename: str) -> None:
        group = h5py.File(filename, 'r')
        self._frequencies = group['frequencies'][:] << units.Hz
        self._step = group['beam'].attrs['step']
        self._beam = group['beam'][:]

    @property
    def parameter_values(self) -> Mapping[str, Any]:
        return {}

    @profile_function()
    def sample(self,
               az_start: float, az_step: float, az_samples: int,
               el_start: float, el_step: float, el_samples: int,
               frequencies: units.Quantity) -> np.ndarray:
        # Compute coordinates for az and el
        az = np.arange(az_samples) * az_step + az_start
        el = np.arange(el_samples) * el_step + el_start
        # Ensure we have matching units, so that versions of numpy prior to NEP 18
        # will do the right thing in np.interp.
        frequencies = units.Quantity(frequencies, copy=False)
        frequencies = frequencies.to(self._frequencies.unit, equivalencies=units.spectral())

        # Reorder arguments to match what apply_along_axis will use
        def interp(fp, x, xp):
            return np.interp(x, xp, fp, left=np.nan, right=np.nan)

        # Interpolate the original data to the new frequencies
        beam = np.apply_along_axis(interp, 0, self._beam, frequencies, self._frequencies)
        # Add polarization dimensions
        out = np.zeros((2, 2, len(frequencies), len(el), len(az)), self._beam.dtype)
        _sample_impl(az, el, beam, self._step, out)
        return out


class MeerkatBeamModelSet1(BeamModelSet):
    """A very simple model of the MeerKAT dish beam."""

    BANDS = {'L', 'UHF'}

    def __init__(self, band: str) -> None:
        if band not in self.BANDS:
            raise ValueError(f'band ({band}) must be one of {self.BANDS}')
        filename = pkg_resources.resource_filename(
            'katsdpimager',
            f'models/beams/meerkat/v1/beam_{band}.h5')
        self._model = TrivialBeamModel(filename)

    @property
    def parameters(self) -> Sequence[Parameter]:
        return []

    def sample(self, **kwargs) -> BeamModel:
        return self._model
