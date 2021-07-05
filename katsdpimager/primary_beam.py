"""Primary beam models and correction."""

from typing import Tuple, Optional, Union, Any

import numpy as np
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore
import scipy.interpolate
import numba
import h5py
import pkg_resources
import astropy.units as u
from katsdpmodels.primary_beam import PrimaryBeam, AltAzFrame, RADecFrame, OutputType

from .profiling import profile_function


@profile_function()
@numba.njit(parallel=True, nogil=True)
def _sample_impl(l, m, beam, step, out):
    """Do linear interpolation of a beam on a 2D grid.

    Parameters
    ----------
    l
        1D array of direction cosines in one axis
    m
        1D array of direction cosines in the orthogonal axis
    beam
        array of beam values, the last axis for radius and the rest for frequency
    step
        Increase in radius for each step along the radius axis
    out
        Output array, with shape (frequency, el/az).
    """
    pos = 0
    deltas = np.empty(beam.shape[-1] - 1, beam.dtype)
    for freq_idx in np.ndindex(beam.shape[:-1]):
        fbeam = beam[freq_idx]
        fout = out[freq_idx]
        for i in range(len(deltas)):
            deltas[i] = fbeam[i + 1] - fbeam[i]
        for i in numba.prange(len(l)):
            radius = np.sqrt(l[i] * l[i] + m[i] * m[i])
            radius_steps = radius / step
            pos = np.int_(radius_steps)
            if pos < len(deltas):
                value = fbeam[pos] + deltas[pos] * (radius_steps - pos)
            else:
                value = np.nan
            fout[i] = value * value  # Convert voltage scale to power


class TrivialPrimaryBeam(PrimaryBeam):
    """The simplest possible beam model.

    It is
    - radially symmetric
    - antenna-independent
    - elevation-independent
    - real-valued
    - polarization-independent (assumes no leakage and same response in H and V)

    It only supports the UNPOLARIZED_POWER output type. There may be other
    areas where it falls short of the katsdpmodels implementation; it is only
    intended to be used within katsdpimager.

    Parameters
    ----------
    step
        Distance (in sine projection) between samples
    frequency
        1D array of frequencies for which samples are available
    samples
        2D array of samples along a radial slice, with axes for frequency and
        position. The samples start at boresight and have step `step`.
    band
        Name of the receiver band
    """

    def __init__(self, step: float, frequency: u.Quantity, samples: np.ndarray, *,
                 band: str) -> None:
        if len(samples) != len(frequency):
            raise ValueError('frequency and samples have inconsistent shape')
        self._step = step
        self._samples = samples
        self._frequency = frequency
        self._band = band
        self._interp_samples = scipy.interpolate.interp1d(
            frequency.to_value(u.Hz), samples,
            axis=0, copy=False, bounds_error=False, fill_value=np.nan,
            assume_sorted=True)

    def spatial_resolution(self, frequency: u.Quantity) -> np.ndarray:
        return self._step

    def frequency_range(self) -> Tuple[u.Quantity, u.Quantity]:
        return self._frequency[0], self._frequency[-1]

    def frequency_resolution(self) -> u.Quantity:
        if len(self._frequency) <= 1:
            return 0 * u.Hz
        else:
            return np.min(np.diff(self._frequency))

    def inradius(self, frequency: u.Quantity) -> float:
        return self._step * (self._samples.shape[1] - 1)

    def circumradius(self, frequency: u.Quantity) -> float:
        return self.inradius(frequency)

    @property
    def is_circular(self) -> bool:
        return True

    @property
    def is_unpolarized(self) -> bool:
        return True

    @property
    def antenna(self) -> None:
        return None

    @property
    def receiver(self) -> None:
        return None

    @property
    def band(self) -> str:
        return self._band

    def sample(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,
               frame: Union[AltAzFrame, RADecFrame],
               output_type: OutputType, *,
               out: Optional[np.ndarray] = None) -> np.ndarray:
        l_ = np.asarray(l)
        m_ = np.asarray(m)
        if output_type != OutputType.UNPOLARIZED_POWER:
            raise NotImplementedError('Only UNPOLARIZED_POWER is currently implemented')
        in_shape = np.broadcast_shapes(l_.shape, m_.shape, frame.shape)
        out_shape = frequency.shape + in_shape
        if out is not None:
            if out.shape != out_shape:
                raise ValueError(f'out must have shape {out_shape}, not {out.shape}')
            if out.dtype != np.float32:
                raise TypeError(f'out must have dtype float32, not {out.dtype}')
            if not out.flags.c_contiguous:
                raise ValueError('out must be C contiguous')
        else:
            out = np.empty(out_shape, np.float32)
        frequency_Hz = frequency.to_value(u.Hz).astype(np.float32, copy=False, casting='same_kind')
        samples = self._interp_samples(frequency_Hz)
        # Create view with l/m axis flattened to 1D for benefit of numba
        l_view = np.broadcast_to(l_, in_shape).ravel()
        m_view = np.broadcast_to(m_, in_shape).ravel()
        out_view = out.view()
        out_view.shape = frequency.shape + l_view.shape
        _sample_impl(l_view, m_view, samples, self._step, out_view)
        return out

    def sample_grid(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,
                    frame: Union[AltAzFrame, RADecFrame],
                    output_type: OutputType, *,
                    out: Optional[np.ndarray] = None) -> np.ndarray:
        l_ = np.asarray(l)
        m_ = np.asarray(m)
        return self.sample(
            l_[np.newaxis, :], m_[:, np.newaxis], frequency,
            frame, output_type, out=out
        )


BANDS = {'L', 'UHF'}


def meerkat_v1_beam(band: str) -> TrivialPrimaryBeam:
    if band not in BANDS:
        raise ValueError(f'band ({band}) must be one of {BANDS}')
    filename = pkg_resources.resource_filename(
        'katsdpimager',
        f'models/beams/meerkat/v1/beam_{band}.h5')
    group = h5py.File(filename, 'r')
    frequencies = group['frequencies'][:] << u.Hz
    step = group['beam'].attrs['step']
    beam = group['beam'][:]
    return TrivialPrimaryBeam(step, frequencies, beam, band=band)
