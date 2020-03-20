"""Primary beam models and correction."""

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
import h5py
import pkg_resources
import astropy.units as units


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

    ANTENNA = Parameter('antenna', 'Name of the antenna')
    ELEVATION = Parameter('elevation', 'Elevation at which the antenna is pointing', units.rad)


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

    @abstractmethod
    @property
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

    @abstractmethod
    @property
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
    def __init__(self) -> None:
        filename = pkg_resources.resource_filename('katsdpimager', 'models/meerkat/v1/beam.hdf5')
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
        # TODO: resample the beam according to the above.
        pass


class MeerkatBeamModelSet1(BeamModelSet):
    """A very simple model of the MeerKAT dish beam.

    It is
    - radially symmetric
    - antenna-independent
    - elevation-independent
    - real-valued
    - polarization-independent (assumes no leakage and same response in H and V)
    """

    def __init__(self) -> None:
        self._model = MeerkatBeamModel1()

    @property
    def parameters(self) -> Sequence[Parameter]:
        return []

    def sample(self, **kwargs) -> BeamModel:
        return self._model
