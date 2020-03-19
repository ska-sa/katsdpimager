"""Primary beam models and correction."""

from typing import Optional

import numpy as np
import astropy.units as units


class BeamModel:
    r"""Model of an antenna primary beam.

    This is a model that describes the response of an antenna in a particular
    direction relative to nominal pointing direction, for each polarization.

    It is most likely frequency-dependent and may also be specific
    to an antenna, elevation, or other variables. However, it is specified in
    a frame oriented to the dish rather than the sky, so remains valid with
    parallactic angle rotation. Description of the mount type (e.g. Alt-Az versus
    equatorial or 3-axis) is currently beyond the scope of this class.

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
    """

    def sample(self, spacing: float, samples: int) -> np.ndarray:
        """Generate a 2D grid of samples.

        Parameters
        ----------
        spacing
            Distance between samples, in SIN projection. Note that if `spacing`
            is even then there won't be a sample for the centre of the beam,
            but rather a middle two samples places just either side.
        samples
            Number of samples along each dimension. The samples will be placed
            symmetrically. The beam is normalised so that the the centre of the
            beam has a value of 1.

        Returns
        -------
        beam
            A 4D array. The first axis selects the feed (0=horizontal,
            1=vertical). The second axis selects the polarization of the
            signal (again, 0=horizontal). The third axis is vertical angle,
            going upwards as one looks at the sky. The last axis is the
            horizontal angle, going left-to-right as one looks at the sky
            (which is the *opposite* to the direction in which right ascension
            increases).
        """
        pass


class BeamModelSet:
    """Generator for beam models.

    The interface is relatively general and allows models to be specialised for
    on variety of parameters. It is intended that a single instance can
    describe a whole telescope for an observation.
    """

    @units.quantity_input(frequency=units.Hz, equivalencies=units.spectral())
    def sample(self, frequency: units.Quantity,
               antenna: Optional[str] = None,
               elevation: Optional[units.Quantity] = None) -> BeamModel:
        """Obtain a specific model.

        It is expected that this may be an expensive function and the caller
        should cache the result if appropriate.

        If either `antenna` or `elevation` is not specified, returns a
        compromise value that should give reasonable results across all
        antennas / a range of typical elevations.

        Raises
        ------
        ValueError
            if `antenna` is not a known antenna (implementations that do not
            model individual antennas may instead return an array average).
        """
        pass
