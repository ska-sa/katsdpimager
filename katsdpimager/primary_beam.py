"""Primary beam models and correction."""

from abc import ABC, abstractmethod
import math
from typing import Optional, Sequence, Mapping, Any

import numpy as np
import h5py
import pkg_resources
import astropy.units as units


class BeamRangeError(ValueError):
    """Requested data lies outside the beam model."""


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

    .. math:: e^{\omega t - kz)i}

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
    that this system has the opposite handedness to RA/Dec. The samples are
    placed symmetrically around the direction of pointing; if there are an
    even number of samples then there will be no sample at the centre.
    """

    @property
    @abstractmethod
    def parameter_values(self) -> Mapping[str, Any]:
        """Get the parameter values for which this model has been specialised."""

    @abstractmethod
    @units.quantity_input(frequencies=units.Hz, equivalencies=units.spectral())
    def sample(self,
               az_step: float, az_samples: int,
               el_step: float, el_samples: int,
               frequencies: units.Quantity) -> np.ndarray:
        """Generate a grid of samples.

        Parameters
        ----------
        az_step
            Step between samples in the azimuth direction (in the SIN projection).
        az_samples
            Number of samples in the azimuth direction.
        el_step
            Step between samples in the elevation direction (in the SIN projection).
        el_samples
            Number of samples in the elevation direction.
        frequencies
            1D array of frequencies.

        Raises
        ------
        BeamRangeError
            If the requested sampling falls beyond the range of the model.

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


class MeerkatBeamModel1(BeamModel):
    BANDS = {'L'}

    def __init__(self, band: str) -> None:
        if band not in self.BANDS:
            raise ValueError(f'band ({band}) must be one of {self.BANDS}')
        filename = pkg_resources.resource_filename(
            'katsdpimager',
            f'models/meerkat/v1/beam_{band}.hdf5')
        group = h5py.File(filename, 'r')
        self._frequencies = group['frequencies'] << units.Hz
        self._step = group.attrs['step']
        self._beam = group.attrs['beam']

    @property
    def parameter_values(self) -> Mapping[str, Any]:
        return {}

    @units.quantity_input(frequencies=units.Hz, equivalencies=units.spectral())
    def sample(self,
               az_step: float, az_samples: int,
               el_step: float, el_samples: int,
               frequencies: units.Quantity) -> np.ndarray:
        # Compute coordinates for az and el
        az = (np.arange(az_samples) - (az_samples - 1) / 2) * az_step
        el = (np.arange(el_samples) - (el_samples - 1) / 2) * el_step
        # Check ranges
        max_radius = math.sqrt(az[0] * az[0] + el[0] * el[0])
        allowed_sin = self._step * self._beam.shape[0]
        if max_radius > allowed_sin:
            allowed_angle = (math.asin(allowed_sin) * units.rad).to(units.deg)
            raise BeamRangeError(f'Requested grid is more than {allowed_angle} from the centre')
        if min(frequencies) < self._frequencies[0] or max(frequencies) > self._frequencies[-1]:
            raise BeamRangeError(f'Requested frequencies lie outside '
                                 f'[{self._frequencies[0]}, {self._frequencies[-1]}]')

        # Reorder arguments to match what apply_along_axis will use
        def interp(fp, x, xp):
            return np.interp(x, xp, fp)

        # Interpolate the original data to the new frequencies
        beam = np.apply_along_axis(interp, 0, self._beam, frequencies, self._frequencies)
        # Interpolate by radius
        radii = np.arange(beam.shape[1]) * self._step
        az = az[np.newaxis, :]
        el = el[:, np.newaxis]
        eval_radii = np.sqrt(np.square(az) + np.square(el))
        # Create model with axes for frequency, el, az
        model = np.apply_along_axis(interp, 1, beam, eval_radii, radii)

        # Add polarization dimensions
        out = np.zeros((2, 2) + model.shape, model.dtype)
        out[0, 0] = model
        out[1, 1] = model
        return out


class MeerkatBeamModelSet1(BeamModelSet):
    """A very simple model of the MeerKAT dish beam.

    It is
    - radially symmetric
    - antenna-independent
    - elevation-independent
    - real-valued
    - polarization-independent (assumes no leakage and same response in H and V)
    """

    def __init__(self, band: str) -> None:
        self._model = MeerkatBeamModel1(band)

    @property
    def parameters(self) -> Sequence[Parameter]:
        return []

    def sample(self, **kwargs) -> BeamModel:
        return self._model
